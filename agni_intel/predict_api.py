from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn, os, uuid, numpy as np, matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import imageio
import rasterio
from skimage.measure import label, regionprops
from network.swin_unet_v2 import SwinTransformerSys

# configuration 
ROOT_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(ROOT_DIR, "static")
PREPROC_DIR = os.path.join(STATIC_DIR, "preprocessed")
OUTPUT_DIR = os.path.join(STATIC_DIR, "output")
WEIGHTS_PATH = os.path.join(ROOT_DIR, "best_swin_v2.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REFLECT_SCALE = 1e4
IMG_SIZE = (224, 224)
THRESHOLD = 0.55
PIXEL_AREA_KM2 = (30*30)/1e6

os.makedirs(PREPROC_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = SwinTransformerSys(
    img_size=IMG_SIZE[0], patch_size=4, in_chans=23, num_classes=23,
    embed_dim=96, depths=[2,2,2,2], depths_decoder=[1,2,2,2],
    num_heads=[3,6,12,24], window_size=7, mlp_ratio=4.0,
    drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
    norm_layer=torch.nn.LayerNorm, final_upsample="expand_first"
).to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.eval()

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def preprocess_tiff(input_path: str, output_npy: str):
    with rasterio.open(input_path) as src:
        arr = src.read(list(range(1,24)))
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    np.save(output_npy, arr)


def infer_and_metrics(npy_path: str):
    arr0 = np.load(npy_path)                # (23, H, W)
    _, H0, W0 = arr0.shape

    # Day t true fire
    mask_t = (arr0[0] / REFLECT_SCALE > 0)
    n0 = mask_t.sum()
    area0 = n0 * PIXEL_AREA_KM2

    # Model inference
    x0 = torch.from_numpy(arr0/REFLECT_SCALE).unsqueeze(0).float().to(DEVICE)
    x  = F.interpolate(x0, size=IMG_SIZE, mode='bilinear', align_corners=False)
    with torch.no_grad():
        out = model(x)                      # (1,23,h,w)

    # false-color prediction (bands 4,3,2)
    pred_ref = out[0].cpu().numpy() * REFLECT_SCALE  # (23, h_small, w_small)
    pred_up = F.interpolate(
        torch.from_numpy(pred_ref).unsqueeze(0),
        size=(H0, W0), mode='bilinear', align_corners=False
    )[0].numpy()
    rgb_pred = np.clip(np.moveaxis(pred_up[[3,2,1]]/REFLECT_SCALE,0,-1),0,1)

    # Probability & binary mask
    fire_prob = torch.sigmoid(out[:,0:1])[0,0]
    fire_prob = F.interpolate(
        fire_prob.unsqueeze(0).unsqueeze(0),
        size=(H0, W0), mode='bilinear', align_corners=False
    )[0,0].cpu().numpy()
    mask_pred = (fire_prob > THRESHOLD)

    # Predicted Day t+1 fire area
    n1 = mask_pred.sum()
    area1 = n1 * PIXEL_AREA_KM2
    delta = area1 - area0

    # Top-5 hotspots
    idxs = np.argpartition(-fire_prob.flatten(), 5)[:5]
    ys, xs = np.unravel_index(idxs, fire_prob.shape)
    hotspots = [
        {"x": int(x), "y": int(y), "p": float(fire_prob[y,x])}
        for x,y in zip(xs, ys)
    ]

    # Growth region & propagation direction
    growth = mask_pred & ~mask_t
    def centroid(mask):
        props = regionprops(label(mask))
        if not props: return None
        y,x = props[0].centroid
        return np.array([x,y])
    c0 = centroid(mask_t)
    c1 = centroid(growth)
    if c0 is not None and c1 is not None:
        dx, dy = c1 - c0
        direction = ("S" if dy>0 else "N") + ("E" if dx>0 else "W")
    else:
        direction = None

    # Burn corridor overlay on rgb_pred
    diff = rgb_pred.mean(axis=2) - rgb_pred.mean(axis=2)  
    rgb0 = np.clip(np.moveaxis(arr0[[3,2,1]]/REFLECT_SCALE,0,-1),0,1)
    diff = rgb0.mean(2) - rgb_pred.mean(2)
    th = diff.mean() + diff.std()
    corridor_mask = diff > th

    return {
        "area0": area0, "area1": area1, "delta": delta,
        "hotspots": hotspots, "direction": direction,
        "rgb_pred": rgb_pred, "corridor_mask": corridor_mask
    }


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.tif', '.tiff')):
        raise HTTPException(400, "Only TIFF files accepted")
    uid = uuid.uuid4().hex

    # Save & preprocess
    tiff_path = os.path.join(PREPROC_DIR, f"{uid}.tif")
    npy_path = os.path.join(PREPROC_DIR, f"{uid}.npy")
    with open(tiff_path, 'wb') as f:
        f.write(await file.read())
    preprocess_tiff(tiff_path, npy_path)

    # Inference & metrics
    res = infer_and_metrics(npy_path)

    rgb_png = os.path.join(OUTPUT_DIR, f"{uid}_rgb_pred.png")
    corridor_png = os.path.join(OUTPUT_DIR, f"{uid}_corridor_rgb.png")

    # Predicted false-color
    imageio.imwrite(rgb_png, (res["rgb_pred"]*255).astype(np.uint8))

    # corridor as RGB overlay
    rgb_corr = rgb_pred = res["rgb_pred"].copy()
    overlay = np.zeros_like(rgb_corr)
    overlay[corridor_png!=None] = 0 

    mask = res["corridor_mask"]
    rgb_corr[mask, :] = [1.0, 0.0, 0.0] 
    imageio.imwrite(corridor_png, (rgb_corr*255).astype(np.uint8))

    return JSONResponse({
        "dayt_true_area_km2": res["area0"],
        "dayt1_pred_area_km2": res["area1"],
        "delta_area_km2": res["delta"],
        "hotspots": res["hotspots"],
        "direction": res["direction"],
        "rgb_pred_png": f"/static/output/{uid}_rgb_pred.png",
        "corridor_rgb_png": f"/static/output/{uid}_corridor_rgb.png"
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
