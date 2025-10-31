import cv2
import numpy as np
import os
import pandas as pd
import random
import torch
import yaml
from typing import Dict, List, Tuple

from models.transunet import TransUNet
from models.unet import UNet
from training.dataset import SegCSV

"""
Visualize model predictions versus ground-truth masks.

Loads a trained segmentation checkpoint and displays, for a random sample:
    • Input grayscale image
    • Ground-truth mask (colored)
    • Predicted mask (argmax or probability)

Supports --pred_mode binary|multiclass for visualization style.

Useful for sanity checks and qualitative validation.

Clinician preview panels:
[ ORIGINAL grayscale | GROUND TRUTH (raw as-is) | PREDICTION (grayscale or argmax) ]

- Left column reads the ultrasound image directly from CSV path (no “magic”).
- Middle column shows the GT mask exactly as stored on disk (color/grayscale).
- Right column shows model prediction:
    prob    -> RD probability grayscale (0..255), optional contrast stretching
                only shows the retinal-detachment probability map (a single class channel)
    binary  -> thresholded RD (use --th)
    argmax  -> per-pixel class argmax mask colorized

Usage:
  python -m utils.preview_predictions \
      --config   configs/config_usg.yaml \
      --ckpt     work_dir/runs/seg_transunet/best.ckpt \
      --eval_csv work_dir/data/test.csv \
      --out_dir  work_dir/preview_predictions \
      --num_samples 6 \
      --pred_mode prob|binary|argmax \
      --th 0.50 \
      --prob_contrast none|auto \
      --gt_mode raw|colorized
"""

# ---------------------- helpers ----------------------

def _make_palette(n: int) -> List[Tuple[int, int, int]]:
    base = [(0,0,0), (0,255,255), (0,255,0)]  # bg, retina/sclera, RD
    if n <= len(base): return base[:n]
    extra = [(255,0,0),(255,0,255),(255,255,0),(255,128,0),
             (128,0,255),(0,128,255),(128,255,0),(0,255,128)]
    out = base[:]
    i = 0
    while len(out) < n:
        out.append(extra[i % len(extra)])
        i += 1
    return out

def colorize_indices(mask_idx: np.ndarray, palette: List[Tuple[int,int,int]]) -> np.ndarray:
    h, w = mask_idx.shape
    out = np.zeros((h, w, 3), np.uint8)
    for k, bgr in enumerate(palette):
        out[mask_idx == k] = bgr
    return out

def normalize_mask_to_indices(mask_raw: np.ndarray, num_classes: int, rd_id: int) -> np.ndarray:
    """Map arbitrary grayscale GT to indices. Used only when gt_mode=colorized."""
    u = np.unique(mask_raw)
    if np.all(np.isin(u, np.arange(num_classes))):
        return mask_raw.astype(np.uint8)
    if len(u) == 2 and u[0] == 0:
        idx = np.zeros_like(mask_raw, np.uint8)
        idx[mask_raw > 0] = rd_id
        return idx
    vals = sorted(list(u))
    lut = {v: (i if i < num_classes else rd_id) for i, v in enumerate(vals)}
    return np.vectorize(lut.get)(mask_raw).astype(np.uint8)

def resize_like(img, size_hw, is_mask=False, is_color=False):
    H, W = size_hw
    inter = cv2.INTER_NEAREST if is_mask and not is_color else cv2.INTER_LINEAR
    return cv2.resize(img, (W, H), interpolation=inter)

def pad_to_same_height(cols: List[np.ndarray]) -> List[np.ndarray]:
    Hmax = max(c.shape[0] for c in cols)
    out = []
    for c in cols:
        h, w = c.shape[:2]
        if h == Hmax: out.append(c); continue
        pad = Hmax - h
        out.append(cv2.copyMakeBorder(c, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0)))
    return out

def as_bgr(img_raw: np.ndarray) -> np.ndarray:
    if img_raw.ndim == 2:
        return cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
    if img_raw.shape[2] == 4:
        return cv2.cvtColor(img_raw, cv2.COLOR_BGRA2BGR)
    return img_raw

def contrast_stretch(img_gray_u8: np.ndarray) -> np.ndarray:
    p5, p95 = np.percentile(img_gray_u8, (5, 95))
    if p95 <= p5:
        return img_gray_u8
    return np.clip((img_gray_u8 - p5) * (255.0/(p95 - p5)), 0, 255).astype(np.uint8)


# --------------------------- model helpers ---------------------------

def load_model(cfg: Dict, ckpt_path: str, device: torch.device):
    name = cfg["model"]["name"].lower()
    in_ch = 1
    num_classes = int(cfg["data"]["num_classes"])
    base = int(cfg["model"].get("base", 32))
    if "trans" in name:
        model = TransUNet(
            in_ch=in_ch, num_classes=num_classes, base=base,
            embed_dim=int(cfg["model"].get("embed_dim", 256)),
            depth=int(cfg["model"].get("depth", 4)),
            heads=int(cfg["model"].get("heads", 8)),
        )
    else:
        model = UNet(in_ch=in_ch, num_classes=num_classes, base=base)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)
    return model

def _rd_class_id(cfg: Dict, default: int = 2) -> int:
    try:
        return int(cfg["data"]["labels"]["retinal_detachment"])
    except Exception:
        return default


# ------------------------------- main --------------------------------

def preview_predictions(cfg_path: str, ckpt: str, eval_csv: str, out_dir: str,
                        num_samples: int = 6, pred_mode: str = "prob",
                        th: float = 0.5, prob_contrast: str = "none", gt_mode: str = "raw"):
    os.makedirs(out_dir, exist_ok=True)
    cfg = yaml.safe_load(open(cfg_path, "r"))
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    # dataset (for preprocessing sizes to model), but display uses raw files
    ds = SegCSV(eval_csv, cfg, augment=None, is_train=False)
    df = pd.read_csv(eval_csv)
    assert {"image_path","mask_path"}.issubset(df.columns), "CSV must have image_path, mask_path"

    model = load_model(cfg, ckpt, device)
    palette = _make_palette(int(cfg["data"]["num_classes"]))
    rd_id = _rd_class_id(cfg, default=2)

    idxs = random.sample(range(len(df)), min(num_samples, len(df)))
    for i, idx in enumerate(idxs, start=1):
        # ---- read raw originals (preserve as-is) ----
        img_path = str(df.iloc[idx]["image_path"])
        msk_path = str(df.iloc[idx]["mask_path"])

        img_raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_raw is None:
            print(f"[WARN] Could not read image: {img_path}"); continue
        H0, W0 = img_raw.shape[:2]

        msk_raw = cv2.imread(msk_path, cv2.IMREAD_UNCHANGED)
        if msk_raw is None:
            print(f"[WARN] Could not read mask: {msk_path}"); continue

        # ---- middle column (GT): RAW or colorized indices ----
        if gt_mode == "raw":
            # show file exactly as stored; resize to match image if needed
            is_color = (msk_raw.ndim == 3 and msk_raw.shape[2] in (3,4))
            gt_disp = as_bgr(msk_raw)
            if gt_disp.shape[:2] != (H0, W0):
                gt_disp = resize_like(gt_disp, (H0, W0), is_mask=True, is_color=is_color)
        else:  # colorized indices from grayscale labels
            gray = msk_raw if msk_raw.ndim == 2 else cv2.cvtColor(msk_raw, cv2.COLOR_BGR2GRAY)
            idx_mask = normalize_mask_to_indices(gray, int(cfg["data"]["num_classes"]), rd_id)
            if idx_mask.shape[:2] != (H0, W0):
                idx_mask = resize_like(idx_mask, (H0, W0), is_mask=True, is_color=False)
            gt_disp = colorize_indices(idx_mask, palette)

        # ---- model inference on dataset-preprocessed tensor ----
        img_t, _ = ds[idx]  # [C,H,W] float normalized to [0,1]
        x = img_t.unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = model(x)              # [1,K,h,w]
            probs  = torch.softmax(logits, dim=1)[0]  # [K,h,w]

        # --- get argmax segmentation and per-class pixel counts ---
        pred_idx = torch.argmax(logits, dim=1).cpu().numpy()[0]  # [H,W]
        unique, counts = np.unique(pred_idx, return_counts=True)
        print(f"[{os.path.basename(img_path)}] Predicted pixel counts:")
        for u, c in zip(unique, counts):
            print(f"  class {u}: {c} pixels")
        import matplotlib.pyplot as plt
        plt.bar(unique, counts)
        plt.title(f"{os.path.basename(img_path)} class pixel counts")
        plt.xlabel("Class ID")
        plt.ylabel("Pixel count")
        plt.savefig(os.path.join(out_dir, f"class_hist_{i:02d}.png"))
        plt.close()


        # RD prob map (or fallback)
        if rd_id < probs.shape[0]:
            rd_prob = probs[rd_id].detach().cpu().numpy()
        else:
            pred_idx = torch.argmax(probs, dim=0).cpu().numpy().astype(np.uint8)
            rd_prob = (pred_idx > 0).astype(np.float32)

        # resize prediction to original image size
        rd_prob = resize_like(rd_prob, (H0, W0), is_mask=False)

        # compose prediction display
        if pred_mode == "binary":
            pred_gray = (rd_prob >= th).astype(np.uint8) * 255
        elif pred_mode == "argmax":
            pred_idx = torch.argmax(probs, dim=0).cpu().numpy().astype(np.uint8)
            pred_idx = resize_like(pred_idx, (H0, W0), is_mask=True)
            pred_gray = colorize_indices(pred_idx, palette)  # color panel for argmax
        else:  # prob
            pred_gray = (rd_prob * 255.0).astype(np.uint8)
            if prob_contrast == "auto":
                pred_gray = contrast_stretch(pred_gray)

        # ---- build columns ----
        left  = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
        mid   = gt_disp
        right = cv2.cvtColor(pred_gray, cv2.COLOR_GRAY2BGR) if pred_mode != "argmax" else pred_gray

        # titles
        cv2.putText(left,  "input (grayscale)",       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.putText(mid,   "mask (ground truth RAW)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2) if gt_mode=="raw" \
            else cv2.putText(mid, "mask (colorized)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.putText(right, f"prediction ({pred_mode})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        # pad heights and concat
        left, mid, right = pad_to_same_height([left, mid, right])
        panel = np.concatenate([left, mid, right], axis=1)

        out_path = os.path.join(out_dir, f"pred_panel_{i:02d}.png")
        cv2.imwrite(out_path, panel)
        print(f"Saved: {out_path}")

    print(f"✅ Saved {len(idxs)} preview panels to {out_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--eval_csv", required=True)
    p.add_argument("--out_dir", default="work_dir/preview_predictions")
    p.add_argument("--num_samples", type=int, default=6)
    p.add_argument("--pred_mode", choices=["prob","binary","argmax"], default="prob")
    p.add_argument("--th", type=float, default=0.50, help="threshold for binary mode")
    p.add_argument("--prob_contrast", choices=["none","auto"], default="none")
    p.add_argument("--gt_mode", choices=["raw","colorized"], default="raw")
    args = p.parse_args()

    preview_predictions(
        args.config, args.ckpt, args.eval_csv, args.out_dir,
        args.num_samples, args.pred_mode, args.th, args.prob_contrast, args.gt_mode
    )