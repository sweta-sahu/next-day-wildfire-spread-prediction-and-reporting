import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import uuid
from datetime import datetime
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import torch
import torch.nn.functional as F
import imageio
import rasterio
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, BitsAndBytesConfig
)

from network.swin_unet_v2 import SwinTransformerSys

# Configuration 
ROOT_DIR = os.path.dirname(__file__)
CHECKPOINT_PATH = os.path.join(ROOT_DIR, "best_swin_v2.pth")
STATIC_DIR = os.path.join(ROOT_DIR, "static")
OUTPUT_DIR = os.path.join(STATIC_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REFLECT_SCALE = 1e4
IMG_SIZE = (224, 224)
TARGET_SHAPE = (351, 303)
PIXEL_AREA_KM2= (30*30)/1e6

# Swin-u-net model setup 
model = SwinTransformerSys(
    img_size=IMG_SIZE[0], patch_size=4, in_chans=23, num_classes=23,
    embed_dim=96, depths=[2,2,2,2], depths_decoder=[1,2,2,2],
    num_heads=[3,6,12,24], window_size=7, mlp_ratio=4.0,
    drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
    norm_layer=torch.nn.LayerNorm, final_upsample="expand_first"
).to(DEVICE)
model.load_state_dict(
    torch.load(CHECKPOINT_PATH, map_location=DEVICE)
)
model.eval()
model.cpu()

# LLM Setup
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN","HUGGINGFACEHUB_API_TOKEN")
LLM_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(LLM_NAME, use_auth_token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
quant_cfg = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)
llm = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    quantization_config=quant_cfg,
    device_map="auto",
    offload_folder="offload",
    offload_state_dict=True,
    torch_dtype=torch.float16,
    use_auth_token=HF_TOKEN
)
pipeline = TextGenerationPipeline(model=llm, tokenizer=tokenizer)

# FastAPI App 
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def preprocess_tiff(input_tif: str, output_npy: str):
    with rasterio.open(input_tif) as src:
        if src.count != 23:
            raise HTTPException(400, f"Expected 23 bands, got {src.count}")
        img = src.read()
    if np.isnan(img).sum()/img.size > 0.5:
        raise HTTPException(400,"Too many NaNs")
    img = np.nan_to_num(img, nan=0.0)
    img = np.clip(img, 0, None)
    if img.sum() == 0:
        raise HTTPException(400,"Image is entirely zero")

    b, h, w = img.shape
    th, tw = TARGET_SHAPE
    top,left = max((h-th)//2,0), max((w-tw)//2,0)
    img2 = img[:, top:top+th, left:left+tw]
    ph,pw = max(th-img2.shape[1],0), max(tw-img2.shape[2],0)
    pt,pb = ph//2, ph-ph//2
    pl,pr = pw//2, pw-pw//2
    aligned = np.pad(img2, ((0,0),(pt,pb),(pl,pr)), constant_values=0)
    np.save(output_npy, aligned)

def infer_and_metrics_npy(npy_path: str):
    arr = np.load(npy_path)
    mask_t = (arr[0]/REFLECT_SCALE > 0).astype(np.uint8)
    area_t = mask_t.sum() * PIXEL_AREA_KM2

    H,W = arr.shape[1:]
    device = next(model.parameters()).device
    x0 = torch.from_numpy(arr/REFLECT_SCALE).unsqueeze(0).float().to(device)
    x_small = F.interpolate(x0, size=IMG_SIZE, mode="bilinear", align_corners=False)
    with torch.no_grad():
        out = model(x_small)
    out_up = F.interpolate(out, size=(H,W), mode="bilinear", align_corners=False)[0]

    fire_prob = torch.sigmoid(out_up[0]).cpu().numpy()
    mask_p = (fire_prob>0.5).astype(np.uint8)
    area_p = mask_p.sum() * PIXEL_AREA_KM2
    delta_a = area_p - area_t

    idxs = np.argpartition(-fire_prob.flatten(),5)[:5]
    ys, xs = np.unravel_index(idxs, fire_prob.shape)
    hotspots = [{"x":int(x),"y":int(y),"p":float(fire_prob[y,x])}
                for x,y in zip(xs,ys)]

    # burn corridor
    rgb0 = np.clip(arr[[3,2,1]].transpose(1,2,0)/REFLECT_SCALE,0,1)
    rgbp = np.clip((out_up.cpu().numpy()*REFLECT_SCALE)[[3,2,1]].transpose(1,2,0)/REFLECT_SCALE,0,1)
    diff = rgb0.mean(2) - rgbp.mean(2)
    th = diff.mean()+diff.std()
    corridor = diff > th

    ys_t, xs_t = np.nonzero(mask_t)
    ys_c, xs_c = np.nonzero(corridor)
    if len(xs_t) and len(xs_c):
        c0 = np.array([xs_t.mean(),ys_t.mean()])
        c1 = np.array([xs_c.mean(),ys_c.mean()])
        dx, dy = c1-c0
        direction = ("S" if dy>0 else "N")+("E" if dx>0 else "W")
    else:
        direction = None

    return {
      "area_t": area_t, "area_p": area_p, "delta_a": delta_a,
      "hotspots": hotspots, "direction": direction,
      "rgb_pred": rgbp, "corridor": corridor
    }

def build_prompt_fixed(m: dict, pct: float, speed: float, rpt_date: str) -> str:
    rows = "\n".join(
        f"| {i+1} | {h['x']} | {h['y']} | {h['p']:.3f} |"
        for i,h in enumerate(m["top_5_hotspots"])
    )
    return f"""
You are an expert wildfire incident analyst generating an operational briefing.
**Report Date:** {rpt_date}

## 0. Report Date
- **{rpt_date}**

## 1. Executive Summary
- Yesterday’s fire burned **{m['day_t_true_fire_area']}**, expected to grow to **{m['day_t1_predicted_fire_area']}**.
- Area ↑ **{m['increase_in_fire_area']}** (**{pct:.2f}%**), spread **{speed:.2f} km²/hr**, direction **{m['direction_of_spread']}**.

## 2. Key Metrics
| Metric | Value |
|---|---|
| Yesterday’s burn | {m['day_t_true_fire_area']} |
| Tomorrow’s predicted | {m['day_t1_predicted_fire_area']} |
| Increase | {m['increase_in_fire_area']} |
| Spread direction | {m['direction_of_spread']} |

## 3. Top‑5 Hotspots
| # | X | Y | Confidence |
|---|---|---|---|
{rows}

## 4. Actionable Recommendations
- Establish defensive lines → **{m['direction_of_spread']}** perimeter.
- Evacuate zones **{m['direction_of_spread'].lower()}** of fire front.
- Pre‑position medical teams near growth corridor.
- Focus aerial water drops on high‑confidence hotspots.

*End of report.*
""".strip()

@app.post("/predict/")
async def predict(request: Request, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.tif','.tiff')):
        raise HTTPException(400, "Please upload a TIFF file.")
    uid = uuid.uuid4().hex
    tiff_path = os.path.join(OUTPUT_DIR, f"{uid}.tif")
    npy_path = os.path.join(OUTPUT_DIR, f"{uid}.npy")
    with open(tiff_path,'wb') as f:
        f.write(await file.read())
    preprocess_tiff(tiff_path, npy_path)
    res = infer_and_metrics_npy(npy_path)

    rgb_path = os.path.join(OUTPUT_DIR, f"{uid}_rgb_pred.png")
    corridor_path = os.path.join(OUTPUT_DIR, f"{uid}_corridor.png")
    imageio.imwrite(rgb_path, (res["rgb_pred"]*255).astype(np.uint8))
    # overlay corridor in red
    rgb_corr = res["rgb_pred"].copy()
    rgb_corr[res["corridor"],:] = [1,0,0]
    imageio.imwrite(corridor_path, (rgb_corr*255).astype(np.uint8))

    base = str(request.base_url).rstrip('/')
    return JSONResponse({
      "predicted_rgb_png": f"{base}/static/output/{uid}_rgb_pred.png",
      "corridor_rgb_png": f"{base}/static/output/{uid}_corridor.png",
      "day_t_true_fire_area": f"{res['area_t']:.2f} km² ({int(res['area_t']/PIXEL_AREA_KM2)} px)",
      "day_t1_predicted_fire_area": f"{res['area_p']:.2f} km² ({int(res['area_p']/PIXEL_AREA_KM2)} px)",
      "increase_in_fire_area": f"{res['delta_a']:.2f} km² ({int(res['delta_a']/PIXEL_AREA_KM2)} px)",
      "top_5_hotspots": res["hotspots"],
      "direction_of_spread": res["direction"]
    })

@app.post("/report/pdf/")
async def report_pdf(file: UploadFile = File(...)):
    # reuse same preprocess+infer
    uid = uuid.uuid4().hex
    tiff_p = os.path.join(OUTPUT_DIR, f"{uid}.tif")
    npy_p = os.path.join(OUTPUT_DIR, f"{uid}.npy")
    with open(tiff_p,'wb') as f:
        f.write(await file.read())
    preprocess_tiff(tiff_p, npy_p)
    res = infer_and_metrics_npy(npy_p)

    # numeric metrics
    pct = res["delta_a"] / res["area_t"] * 100
    speed = res["delta_a"] / 24
    rpt_dt = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    api_m = {
      "day_t_true_fire_area": f"{res['area_t']:.2f} km²",
      "day_t1_predicted_fire_area": f"{res['area_p']:.2f} km²",
      "increase_in_fire_area": f"{res['delta_a']:.2f} km²",
      "top_5_hotspots": res["hotspots"],
      "direction_of_spread": res["direction"] or "N/A"
    }
    prompt = build_prompt_fixed(api_m, pct, speed, rpt_dt)
    narrative = pipeline(
      prompt,
      max_new_tokens=400,
      do_sample=True,
      temperature=0.7,
      top_p=0.9,
      return_full_text=False
    )[0]["generated_text"].strip()

    # write images 
    rgb_png = os.path.join(OUTPUT_DIR, f"{uid}_rgb.png")
    cor_png = os.path.join(OUTPUT_DIR, f"{uid}_cor.png")
    imageio.imwrite(rgb_png, (res["rgb_pred"]*255).astype(np.uint8))
    ci = res["rgb_pred"].copy()
    ci[res["corridor"],:] = [1,0,0]
    imageio.imwrite(cor_png, (ci*255).astype(np.uint8))

    # build PDF
    pdf_path = os.path.join(OUTPUT_DIR, f"{uid}_report.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
        leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleStyle', fontSize=18, leading=22, spaceAfter=12))
    styles.add(ParagraphStyle(name='Heading', fontSize=14, leading=18, spaceAfter=6))

    elems = []
    elems.append(Paragraph("Wildfire Incident Report", styles['TitleStyle']))
    elems.append(Paragraph(f"Report Date: {rpt_dt}", styles['Normal']))
    elems.append(Spacer(1,12))

    # metrics table
    data = [
      ["Metric","Value"],
      ["Yesterday’s burn area", api_m["day_t_true_fire_area"]],
      ["Predicted tomorrow", api_m["day_t1_predicted_fire_area"]],
      ["Increase in area", api_m["increase_in_fire_area"]],
      ["Percent increase", f"{pct:.2f}%"],
      ["Avg. spread speed", f"{speed:.2f} km²/hr"],
      ["Spread direction", api_m["direction_of_spread"]]
    ]
    tbl = Table(data, colWidths=[240,200])
    tbl.setStyle(TableStyle([
      ('BACKGROUND',(0,0),(-1,0), colors.HexColor('#4F81BD')),
      ('TEXTCOLOR',(0,0),(-1,0), colors.white),
      ('ALIGN',(0,0),(-1,-1),'LEFT'),
      ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
      ('GRID',(0,0),(-1,-1), 0.5, colors.gray),
    ]))
    elems.extend([tbl, Spacer(1,12)])

    # narrative
    for p in narrative.split('\n'):
      elems.extend([Paragraph(p, styles['Normal']), Spacer(1,4)])
    elems.append(Spacer(1,12))

    # hotspots table
    hr = [["#", "X","Y","Confidence"]] + [
      [str(i+1), str(h['x']), str(h['y']), f"{h['p']:.3f}"] 
      for i,h in enumerate(res["hotspots"])
    ]
    htbl = Table(hr, colWidths=[30,70,70,100])
    htbl.setStyle(TableStyle([
      ('BACKGROUND',(0,0),(-1,0), colors.HexColor('#4F81BD')),
      ('TEXTCOLOR',(0,0),(-1,0), colors.white),
      ('ALIGN',(0,0),(-1,-1),'CENTER'),
      ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
      ('GRID',(0,0),(-1,-1),0.5, colors.gray),
    ]))
    elems.extend([htbl, Spacer(1,12)])

    # recommendations
    dir_map = {
      "N":"North","S":"South","E":"East","W":"West",
      "NE":"North-East","NW":"North-West",
      "SE":"South-East","SW":"South-West"
    }
    full_dir = dir_map.get(res['direction'],res['direction'])
    elems.append(Paragraph("Actionable Recommendations", styles['Heading']))
    recs = [
      f"• Establish defensive lines along the <b>{full_dir}</b> perimeter.",
      f"• Evacuate at-risk zones immediately <b>{full_dir}</b> of the fire front.",
      "• Pre-position medical teams at staging area α.",
      "• Deploy aerial water drops on hotspots."
    ]
    for r in recs:
      elems.extend([Paragraph(r, styles['Normal']), Spacer(1,4)])
    elems.append(Spacer(1,12))

    # images
    elems.append(Paragraph("Predicted Next-Day Fire", styles['Heading']))
    elems.append(RLImage(rgb_png,width=400, height=240))
    elems.append(Spacer(1,12))
    elems.append(Paragraph("Burn Corridor", styles['Heading']))
    elems.append(RLImage(cor_png,width=400, height=240))

    doc.build(elems)
    return FileResponse(pdf_path, media_type="application/pdf")


if __name__=="__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

