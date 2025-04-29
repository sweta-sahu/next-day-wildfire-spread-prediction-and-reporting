from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import uuid, os, numpy as np, torch, imageio, rasterio
import torch.nn.functional as F
from skimage.measure import label, regionprops
import uvicorn, os, uuid, numpy as np, torch
from network.swin_unet_v2 import SwinTransformerSys

ROOT_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(ROOT_DIR, "static")
OUTPUT_DIR = os.path.join(STATIC_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REFLECT_SCALE = 1e4
IMG_SIZE = (224, 224)
THRESHOLD = 0.55
PIXEL_AREA_KM2 = (30 * 30) / 1e6

TARGET_SHAPE = (351, 303)
model = SwinTransformerSys(
    img_size=IMG_SIZE[0], patch_size=4, in_chans=23, num_classes=23,
    embed_dim=96, depths=[2,2,2,2], depths_decoder=[1,2,2,2],
    num_heads=[3,6,12,24], window_size=7, mlp_ratio=4.0,
    drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
    norm_layer=torch.nn.LayerNorm, final_upsample="expand_first"
).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(ROOT_DIR, "best_swin_v2.pth"), map_location=DEVICE))
model.eval()

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def preprocess_tiff(input_tif: str, output_npy: str):
    # read 23 bands of reflectance
    with rasterio.open(input_tif) as src:
        if src.count != 23:
            raise HTTPException(400, f"Expected 23 bands, got {src.count}")
        img = src.read()            # shape (23, H, W)

    # drop if >50% NaNs
    nan_ratio = np.isnan(img).sum() / img.size
    if nan_ratio > 0.5:
        raise HTTPException(400, f"Too many NaNs: {nan_ratio:.1%}")

    # replace NaNs & clamp negatives
    img = np.nan_to_num(img, nan=0.0)
    img = np.clip(img, a_min=0.0, a_max=None)

    # drop if entirely zero
    if np.all(img == 0):
        raise HTTPException(400, "Image is entirely zero")

    # inline resize to TARGET_SHAPE
    b, h, w = img.shape
    th, tw = TARGET_SHAPE

    # center‐crop
    top = max((h - th)//2, 0)
    left = max((w - tw)//2, 0)
    img2 = img[:, top:top+th, left:left+tw]

    # pad if needed
    pad_h = max(th - img2.shape[1], 0)
    pad_w = max(tw - img2.shape[2], 0)
    pt, pb = pad_h//2, pad_h - pad_h//2
    pl, pr = pad_w//2, pad_w - pad_w//2
    aligned = np.pad(
      img2,
      ((0,0), (pt,pb), (pl,pr)),
      mode='constant', constant_values=0
    )  

    np.save(output_npy, aligned)

def infer_and_metrics_npy(npy_path: str):
    arr = np.load(npy_path)     

    # Burned‐area 
    pixel_area_km2 = (30 * 30) / 1e6

    # Day t true mask & area
    mask_t = (arr[0] / REFLECT_SCALE > 0).astype(np.uint8)
    n_t = mask_t.sum()
    area_t = n_t * pixel_area_km2

    # Model inference 
    refl = arr 
    H, W = refl.shape[1:]
    x0 = torch.from_numpy(refl / REFLECT_SCALE)\
                      .unsqueeze(0).float().to(DEVICE) # (1,23,H,W)
    x_small = F.interpolate(x0, size=IMG_SIZE, mode="bilinear", align_corners=False)
    with torch.no_grad():
        out_small = model(x_small)  # (1,23,h_small,w_small)
        out_up = F.interpolate(
                       out_small,
                       size=(H, W),
                       mode="bilinear",
                       align_corners=False
                   )[0]                           

    # Day t+1 predicted mask & area
    fire_prob = torch.sigmoid(out_up[0]).cpu().numpy()
    mask_pred = (fire_prob > 0.5).astype(np.uint8)
    n_pred = mask_pred.sum()
    area_pred = n_pred * pixel_area_km2

    # Increase in fire
    delta_n = n_pred - n_t
    delta_area = area_pred - area_t

    # Hotspots 
    idxs = np.argpartition(-fire_prob.flatten(), 5)[:5]
    ys, xs = np.unravel_index(idxs, fire_prob.shape)
    hotspots = [{"x":int(x),"y":int(y),"p":float(fire_prob[y,x])}
                for x,y in zip(xs,ys)]

    # False-color + corridor 
    rgb0 = np.clip(arr[[3,2,1]].transpose(1,2,0)/REFLECT_SCALE,0,1)
    pred_ref = out_up.cpu().numpy() * REFLECT_SCALE
    rgb_pred = np.clip(pred_ref[[3,2,1]].transpose(1,2,0)/REFLECT_SCALE,0,1)
    diff = rgb0.mean(2) - rgb_pred.mean(2)
    th = diff.mean() + diff.std()
    corridor = diff > th

    # Direction 
    # Growth & propagation direction (mean‐of‐all‐new‐pixels)
    ys_t, xs_t = np.nonzero(mask_t) # original fire pixels
    ys_c, xs_c = np.nonzero(corridor) # corridor pixels

    if len(xs_t) and len(xs_c):
        c0 = np.array([xs_t.mean(), ys_t.mean()])
        c1 = np.array([xs_c.mean(), ys_c.mean()])
        dx, dy = c1 - c0
        direction = ("S" if dy > 0 else "N") + ("E" if dx > 0 else "W")
    else:
        direction = None

    return {
      "n_t": n_t, "area_t": area_t,
      "n_pred": n_pred, "area_pred": area_pred,
      "delta_n": delta_n, "delta_area": delta_area,
      "hotspots": hotspots, "direction": direction,
      "rgb_pred": rgb_pred, "corridor": corridor
    }


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.tif','.tiff')):
        raise HTTPException(400, "Please upload a TIFF file.")

    uid = uuid.uuid4().hex
    tiff_path = os.path.join(OUTPUT_DIR, f"{uid}.tif")
    npy_path = os.path.join(OUTPUT_DIR, f"{uid}.npy") 

    with open(tiff_path,'wb') as f:
        f.write(await file.read())

    preprocess_tiff(tiff_path, npy_path)

    res = infer_and_metrics_npy(npy_path)

    rgb_png = os.path.join(OUTPUT_DIR, f"{uid}_rgb_pred.png")
    corridor_png = os.path.join(OUTPUT_DIR, f"{uid}_corridor.png")

    imageio.imwrite(rgb_png, (res["rgb_pred"] * 255).astype(np.uint8))
    rgb_corr = res["rgb_pred"].copy()
    rgb_corr[res["corridor"],:] = [1, 0, 0]
    imageio.imwrite(corridor_png, (rgb_corr * 255).astype(np.uint8))

    return JSONResponse({
      "predicted_rgb_png": f"/static/output/{uid}_rgb_pred.png",
      "corridor_rgb_png": f"/static/output/{uid}_corridor.png",
      "day_t_true_fire_area": f"{res['area_t']:.2f} km² ({res['n_t']} px)",
      "day_t1_predicted_fire_area": f"{res['area_pred']:.2f} km² ({res['n_pred']} px)",
      "increase_in_fire_area": f"{res['delta_area']:.2f} km² ({res['delta_n']} px)",
      "top_5_hotspots": res["hotspots"],
      "direction_of_spread": res["direction"]
    })

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)