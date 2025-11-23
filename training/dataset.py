# training/dataset.py
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.paths import resolve_under_root_cfg
from pathlib import Path
import json
"""
Dataset loader for segmentation training and validation.
Reads image/mask paths from CSV.
Always returns:
  img: FloatTensor [1,H,W]
  msk: LongTensor  [H,W] with class ids in [0..num_classes-1]
"""

# --- add near the top of dataset.py ---

def _norm_key(s: str) -> str:
    return (
        s.strip().lower()
         .replace("-", " ")
         .replace("/", " ")
         .replace(".", " ")
         .replace("__", "_")
         .replace(" ", "_")
    )

def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

def _letterbox_pair(img_gray, mask, size_hw):
    """
    Resize with preserved aspect ratio and pad to exactly size_hw (H,W).
    Pads with 0 (black) for image and 0 (background) for mask.
    Returns: img_lb, msk_lb
    """
    Ht, Wt = size_hw
    h, w = img_gray.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"Bad image shape: {img_gray.shape}")

    scale = min(Ht / float(h), Wt / float(w))
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    img_r = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    msk_r = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    top    = (Ht - new_h) // 2
    bottom = Ht - new_h - top
    left   = (Wt - new_w) // 2
    right  = Wt - new_w - left

    img_lb = cv2.copyMakeBorder(img_r, top, bottom, left, right,
                                borderType=cv2.BORDER_CONSTANT, value=0)
    msk_lb = cv2.copyMakeBorder(msk_r, top, bottom, left, right,
                                borderType=cv2.BORDER_CONSTANT, value=0)
    return img_lb, msk_lb

def _read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def _resize_pair(img_gray, mask, size_hw):
    H, W = size_hw
    img_r = cv2.resize(img_gray, (W, H), interpolation=cv2.INTER_LINEAR)
    msk_r = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return img_r, msk_r

def _normalize(img_f, mode: str):
    mode = (mode or "zscore").lower()
    if mode == "zscore":
        mean = float(img_f.mean())
        std  = float(img_f.std())
        if std < 1e-6:
            return img_f - mean
        return (img_f - mean) / std
    elif mode == "minmax":
        return img_f
    elif mode == "log":
        return np.log1p(img_f)
    else:
        return img_f


class SegCSV(Dataset):
    def __init__(self, csv_path, cfg, augment=None, is_train=False):
        self.cfg = cfg
        self.df  = pd.read_csv(csv_path)
        self.augment = augment
        self.is_train = is_train
        self.size_hw = tuple(self.cfg["data"].get("resize") or (0, 0))  # allow null/None
        self.labels  = cfg["data"].get("labels", {})

    def __len__(self):
        return len(self.df)


    def __getitem__(self, i):
        row = self.df.iloc[i]
        ip = resolve_under_root_cfg(self.cfg, str(row["image_path"]))
        mp = resolve_under_root_cfg(self.cfg, str(row["mask_path"]))

        img = _read_gray(ip)  # HxW uint8 (raw, no squeeze)
        msk = load_mask_as_ids(str(mp), self.cfg)  # HxW int64 ids (raw, no squeeze)


        # ENABLE for 1280 * 800
        # H, W = img.shape[:2]
        # target = self.size_hw
        # if target and target != (0, 0):
        #     th, tw = int(target[0]), int(target[1])
        #     if (H, W) != (th, tw):
        #         img, msk = _resize_pair(img, msk, (th, tw))



        # Albumentations: operate on raw HxW (variable size)
        img_hwc = img[:, :, None]
        if self.augment is not None:
            out = self.augment(image=img_hwc, mask=msk)
            img_hwc, msk = out["image"], out["mask"]

        # Letterbox (preserve AR) to the model's fixed input size
        img_fixed, msk_fixed = _letterbox_pair(img_hwc[:, :, 0], msk, self.size_hw)

        # Normalize -> tensor
        img_f = img_fixed.astype(np.float32) / 255.0
        img_f = _normalize(img_f, self.cfg["data"].get("normalize", "zscore"))
        img_f = np.expand_dims(img_f, 0)  # 1xHxW

        # force ids valid
        num_classes = int(self.cfg["data"]["num_classes"])
        msk_fixed = np.clip(msk_fixed.astype(np.int64), 0, num_classes - 1)
        # Optional: return raw path for preview panels
        if self.cfg.get("data", {}).get("return_path", False):
            return torch.from_numpy(img_f), torch.from_numpy(msk_fixed), str(ip)
        return torch.from_numpy(img_f), torch.from_numpy(msk_fixed), str(ip)

def load_mask_as_ids(mask_path: str, cfg: dict) -> np.ndarray:
    """
    Convert a color (Supervisely-style) or grayscale mask to a 2D id map.
    Uses meta.json to map class name -> hex color -> id, so that training,
    eval, and visualization share the exact same palette.

    Returns int64 array with ids in [0..C-1].
    """
    dm = cfg.get("data", {})
    mode = (dm.get("mask_mode", "auto") or "auto").lower()

    raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(mask_path)

    # Decide path based on channels
    if mode == "auto":
        mode = "color" if raw.ndim == 3 and raw.shape[2] >= 3 else "grayscale"
    labels_map = dm.get("labels", {})  # {"retina":2, ...}
    # print("labels_map", labels_map)

    # --- Grayscale ids (kept for completeness) ---
    if mode == "grayscale":
        if raw.ndim == 3:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        ids = raw.astype(np.int64)
        uniq = np.unique(ids)
        if set(uniq.tolist()) <= {0, 255}:
            ids = (ids > 127).astype(np.int64)

        drop_classes = (cfg.get("data", {}).get("drop_classes") or [])
        if drop_classes:
            labels = cfg["data"]["labels"]
            drop_ids = [int(labels[nm]) for nm in drop_classes if nm in labels]
            print(drop_ids)
            for did in drop_ids:
                ids[ids == did] = 0  # send to background

        return ids

    # --- Color path: build BGR->id map from meta.json + labels ---
    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    bgr = raw[:, :, :3]




    # load meta.json sitting under work_root
    meta_path = Path(cfg.get("work_root", ".")) / "meta.json"
    meta = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = None

    bgr_to_id = {}
    if isinstance(meta, dict) and "classes" in meta and labels_map:
        # name -> id via labels_map with normalized keys
        norm_to_id = {_norm_key(k): int(v) for k, v in labels_map.items()}
        for cls in meta.get("classes", []):
            title = cls.get("title", "")
            hexcol = cls.get("color", None)
            nk = _norm_key(title)
            if nk in norm_to_id and isinstance(hexcol, str):
                r, g, b = _hex_to_rgb(hexcol)          # RGB from meta
                bgr_to_id[(b, g, r)] = norm_to_id[nk]  # store as BGR for cv2 image
        # ensure background maps to 0
        if 0 not in set(bgr_to_id.values()):
            bgr_to_id[(0, 0, 0)] = 0
    else:
        # fallback to config palette if meta.json missing
        pal_cfg = dm.get("color_palette", {}) or {"background": [0, 0, 0]}
        for name, cid in labels_map.items():
            col = pal_cfg.get(name, [0, 0, 0])  # expecting BGR here
            bgr_to_id[tuple(int(c) for c in col)] = int(cid)

    # vectorized paint (loop over known colors)
    H, W, _ = bgr.shape
    ids = np.zeros((H, W), np.int64)
    for (bb, gg, rr), cid in bgr_to_id.items():
        mask = (bgr[:, :, 0] == bb) & (bgr[:, :, 1] == gg) & (bgr[:, :, 2] == rr)
        if mask.any():
            ids[mask] = cid

    # clip to valid range
    C = int(dm.get("num_classes", max(labels_map.values()) + 1))
    ids = np.clip(ids, 0, C - 1).astype(np.int64)

    # --- (Optional) tiny sanity check while you debug mapping ---
    # uniq = np.unique(ids)
    # print(f"[GT ids] {Path(mask_path).name}: " + " ".join(f"{u}:{(ids==u).sum()}" for u in uniq))
    drop_classes = (cfg.get("data", {}).get("drop_classes") or [])
    if drop_classes:
        labels = cfg["data"]["labels"]
        drop_ids = [int(labels[nm]) for nm in drop_classes if nm in labels]
        for did in drop_ids:
            ids[ids == did] = 0  # send to background
    return ids

