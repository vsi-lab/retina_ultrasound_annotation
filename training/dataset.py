# training/dataset.py
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

"""
Dataset loader for segmentation training and validation.

Implements SegCSV, which reads image–mask paths from a CSV, loads grayscale
ultrasound frames, applies preprocessing (resize, normalization, despeckle),
and optional Albumentations augmentations.

Ensures that geometric transforms are applied synchronously to image and mask.
"""

def _read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def _resize_pair(img_gray, mask, size_hw):
    H, W = size_hw
    img_r = cv2.resize(img_gray, (W, H), interpolation=cv2.INTER_LINEAR)
    msk_r = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return img_r, msk_r

def _normalize(img_f, mode: str):
    # img_f is float32 in [0,1] after /255
    mode = (mode or "zscore").lower()
    if mode == "zscore":
        mean = float(img_f.mean())
        std  = float(img_f.std())
        if std < 1e-6:
            # constant image: zero-center only
            return img_f - mean
        return (img_f - mean) / std
    elif mode == "minmax":
        # already [0,1]
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
        self.size_hw = tuple(cfg["data"]["resize"])  # [H,W]
        self.labels  = cfg["data"].get("labels", {})
        # reader mode ignored here; we always read grayscale

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        ip = str(row["image_path"]); mp = str(row["mask_path"])
        img = _read_gray(ip)           # HxW uint8
        # msk = _read_gray(mp)           # HxW uint8 (indices or grayscale)
        msk = load_mask_as_ids(row.mask_path, self.cfg)  # 2-D int map
        # resize to configured size
        img, msk = _resize_pair(img, msk, self.size_hw)
        # Albumentations expects HWC; convert and apply with mask
        img_hwc = img[:, :, None]      # HxWx1
        if self.augment is not None:
            out = self.augment(image=img_hwc, mask=msk)
            img_hwc, msk = out["image"], out["mask"]
        # back to CHW float, mask long
        img = img_hwc[:, :, 0].astype(np.float32) / 255.0
        img = _normalize(img, self.cfg["data"].get("normalize", "zscore"))
        img = np.expand_dims(img, 0)   # 1xHxW
        msk = msk.astype(np.int64)
        return torch.from_numpy(img), torch.from_numpy(msk)


def load_mask_as_ids(mask_path: str, cfg: dict) -> np.ndarray:
    """
    Load a mask image (color or grayscale) and return a 2-D int array of class IDs in [0..C-1].

    - If the file is grayscale: values should match cfg['data']['labels'] (e.g., 0/1/2/3).
      If your binary masks are {0,255}, they will be mapped to {0,1}.
    - If the file is color: pixels are matched to the palette in cfg['data']['color_palette'] (BGR).
      Unmatched colors fall back to background (0) with a warning.
    """
    dm = cfg.get('data', {})
    mode = dm.get('mask_mode', 'auto')

    # read raw
    raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(mask_path)

    # Detect mode if needed
    if mode == 'auto':
        mode = 'color' if raw.ndim == 3 and raw.shape[2] >= 3 else 'grayscale'

    if mode == 'grayscale':
        if raw.ndim == 3:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        ids = raw.astype(np.int64)

        # Map common binary case {0,255} -> {0,1}
        uniq = np.unique(ids)
        if set(uniq.tolist()) <= {0, 255}:
            ids = (ids > 127).astype(np.int64)

        # Optional: enforce expected values via a lookup
        gs = dm.get('labels', None)
        if gs is not None:
            # Make a reverse table from raw value -> id index by class order in labels
            labels = dm.get('labels', {})  # {'background':0, 'retina_sclera':1, ...}
            lut = np.full(256, 0, dtype=np.int64)
            for name, cid in labels.items():
                v = int(gs.get(name, cid))
                lut[v] = cid
            # For values not listed, leave as-is (best effort)
            ids = lut[np.clip(ids, 0, 255)]
        return ids

    else:  # 'color'
        # Ensure 3 channels BGR
        if raw.ndim == 2:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        bgr = raw[:, :, :3]

        # Build palette map: BGR tuple -> class id
        labels = dm.get('labels', {})
        pal_cfg = dm.get('color_palette', None)
        if pal_cfg is None:
            # Safe fallback palette: background black, classes bright hues
            pal_cfg = {'background': [0,0,0]}
            for name, cid in labels.items():
                if name == 'background': continue
                pal_cfg[name] = [0, 255, 0]  # replace with your real colors

        bgr_to_id = {}
        for name, cid in labels.items():
            col = pal_cfg.get(name, [0,0,0])
            bgr_to_id[tuple(int(c) for c in col)] = int(cid)

        # Vectorized matching
        H, W, _ = bgr.shape
        ids = np.zeros((H, W), np.int64)
        # Build mask per class
        for col, cid in bgr_to_id.items():
            mask = (bgr[:, :, 0] == col[0]) & (bgr[:, :, 1] == col[1]) & (bgr[:, :, 2] == col[2])
            ids[mask] = cid

        # Warn if there are pixels not matched (i.e., unknown color)
        if (ids == 0).sum() != (bgr[:, :, 0] == 0).sum() or \
           (ids == 0).sum() != (bgr[:, :, 1] == 0).sum() or \
           (ids == 0).sum() != (bgr[:, :, 2] == 0).sum():
            # Optional: add a proper logger; here we’re quiet to avoid spam.
            pass

        return ids