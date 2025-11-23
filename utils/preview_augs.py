# utils/preview_augs.py
import argparse
import cv2
import numpy as np
import os
import pandas as pd
import pathlib
import yaml
from utils.paths import resolve_under_root_cfg

from utils.usg_transforms import representative_params, apply_geom, apply_photo, AugParams

"""
Preview panel now includes ALL training augs:
original, despeckle, rotate, shear, hflip, scale, translate, speckle, gamma.
Magnitudes are picked from the SAME config ranges (representative values).
Outputs to: work_dir/aug_previews/<stem>__preview_panel.png
"""

"""
Visualize Albumentations augmentations on sample ultrasound images.

Generates a grid showing:
    • Original grayscale image and mask
    • Multiple random augmentation variants (same transform applied to both)
Each row corresponds to one image; each column to a specific transformation.

Intended for clinical/annotation review of augmentation realism.
"""


FONT = cv2.FONT_HERSHEY_SIMPLEX

import math

def stack_grid(tiles, cols=2, pad=12, bg=(0,0,0)):
    """
    Arrange equally-sized tiles into a grid with `cols` columns.
    If tiles differ slightly in size, we pad them to the max tile W/H.
    """
    # find max tile size
    max_h = max(t.shape[0] for t in tiles)
    max_w = max(t.shape[1] for t in tiles)
    norm = []
    for t in tiles:
        # pad to (max_h, max_w)
        canvas = np.full((max_h, max_w, 3), bg, np.uint8)
        canvas[:t.shape[0], :t.shape[1]] = t
        norm.append(canvas)

    rows = math.ceil(len(norm) / cols)
    H = rows * max_h + (rows - 1) * pad
    W = cols * max_w + (cols - 1) * pad
    out = np.full((H, W, 3), bg, np.uint8)

    i = 0
    for r in range(rows):
        for c in range(cols):
            if i >= len(norm): break
            y = r * (max_h + pad)
            x = c * (max_w + pad)
            out[y:y+max_h, x:x+max_w] = norm[i]
            i += 1
    return out

def _with(p: AugParams, **kw) -> AugParams:
    d = p.__dict__.copy(); d.update(kw); return AugParams(**d)

def _rng(lo_hi, default=(0.0, 0.0)):
    if lo_hi is None: return default
    lo, hi = lo_hi if isinstance(lo_hi, (list, tuple)) else (lo_hi, lo_hi)
    return float(lo), float(hi)

def _rep_rotate(cfg):
    lo, hi = _rng(cfg['aug'].get('rotate_deg', [-5, 5]))
    return hi if abs(hi) >= abs(lo) else lo

def _rep_shear(cfg):
    lo, hi = _rng(cfg['aug'].get('shear_deg', [-2, 2]))
    return hi if abs(hi) >= abs(lo) else lo

def _rep_scale(cfg):
    lo, hi = _rng(cfg['aug'].get('scale', [1.0, 1.0]))
    # pick the factor farther from 1.0
    return hi if abs(hi - 1.0) >= abs(1.0 - lo) else lo

def _rep_translate(cfg):
    lo, hi = _rng(cfg['aug'].get('translate', [0.0, 0.0]))
    # pick larger abs shift; apply to both x & y for the preview row
    t = hi if abs(hi) >= abs(lo) else lo
    return float(t)

def _rep_gamma(cfg):
    gmin, gmax = _rng(cfg['aug'].get('brightness', [1.0, 1.0]))
    gmid = 0.5 * (gmin + gmax)
    return gmin, gmax, gmid

def _rep_sigma(cfg):
    return float(cfg['aug'].get('gaussian_noise_std', 0.0))

def _hflip_p(cfg):
    return float(cfg['aug'].get('hflip', 0.0))


def put_title(img_bgr, text):
    out = img_bgr.copy()
    cv2.putText(out, text, (8, 28), FONT, 1.2, (255,255,255), 3, cv2.LINE_AA)
    return out

def load_gray_unchanged(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.dtype != np.uint8:
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return im

def load_mask_png(path):
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 2:
        # grayscale label map → preview as yellow on black (purely visual)
        vis = np.zeros((m.shape[0], m.shape[1], 3), np.uint8)
        vis[m > 0] = (0,255,255)  # yellow in BGR
        return vis
    if m.shape[2] == 4:
        bgr = m[:, :, :3].astype(np.float32)
        a   = (m[:, :, 3:4].astype(np.float32)/255.0)
        out = (bgr * a).astype(np.uint8)
        return out
    return m

def to_bgr(gray_u8): return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

def row_pair(left_bgr, right_bgr, pad=8):
    if left_bgr.ndim == 2:  left_bgr = to_bgr(left_bgr)
    if right_bgr.ndim == 2: right_bgr = to_bgr(right_bgr)
    if left_bgr.shape[2]==4:  left_bgr = cv2.cvtColor(left_bgr,  cv2.COLOR_BGRA2BGR)
    if right_bgr.shape[2]==4: right_bgr = cv2.cvtColor(right_bgr, cv2.COLOR_BGRA2BGR)
    h = max(left_bgr.shape[0], right_bgr.shape[0])
    w = left_bgr.shape[1] + pad + right_bgr.shape[1]
    out = np.zeros((h, w, 3), np.uint8)
    out[:left_bgr.shape[0], :left_bgr.shape[1]] = left_bgr
    out[:right_bgr.shape[0], left_bgr.shape[1]+pad:left_bgr.shape[1]+pad+right_bgr.shape[1]] = right_bgr
    return out

def stack_rows(rows, pad=12):
    h = sum(r.shape[0] for r in rows) + pad*(len(rows)-1)
    w = max(r.shape[1] for r in rows)
    out = np.zeros((h, w, 3), np.uint8)
    y = 0
    for r in rows:
        out[y:y+r.shape[0], :r.shape[1]] = r
        y += r.shape[0] + pad
    return out

def _with(p: AugParams, **kw) -> AugParams:
    d = p.__dict__.copy()
    d.update(kw)
    return AugParams(**d)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--csv', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--n', type=int, default=5)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r'))
    csv_path = args.csv or (cfg['data']['train_csv']).format(**cfg)
    out_dir  = args.out or os.path.join(cfg['data'].get('work_dir','.'), 'aug_previews').format(**cfg)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Deterministic params from config
    base_p = representative_params(cfg)
    rot_rep = _rep_rotate(cfg)
    shear_rep = _rep_shear(cfg)
    scale_rep = _rep_scale(cfg)
    t_rep = _rep_translate(cfg)
    gmin, gmax, gmid = _rep_gamma(cfg)
    sigma = _rep_sigma(cfg)
    hfp = _hflip_p(cfg)

    for i in range(min(args.n, len(df))):
        ipath = str(resolve_under_root_cfg(cfg,df.iloc[i].image_path).as_posix())
        mpath = str(resolve_under_root_cfg(cfg,df.iloc[i].mask_path).as_posix())
        stem  = pathlib.Path(ipath).stem

        img = load_gray_unchanged(ipath)
        msk_png = load_mask_png(mpath)

        # Row 1: ORIGINAL (unaltered)
        row1 = row_pair(
            put_title(to_bgr(img), "original (grayscale)"),
            put_title(msk_png, "mask (from file)")
        )

        # Row 2: DESPECKLE
        if cfg['data'].get('despeckle', 'none') == 'median3':
            img_d = cv2.medianBlur(img, 3)
            row2 = row_pair(
                put_title(to_bgr(img_d), "despeckle: median3"),
                put_title(msk_png, "mask (unchanged)")
            )
        else:
            row2 = row_pair(
                put_title(to_bgr(img), "despeckle: none"),
                put_title(msk_png, "mask (unchanged)")
            )

        # Row 3: ROTATE (geom on image+mask)
        p_rot = _with(base_p, rot_deg=float(rot_rep), shear_deg=0.0, scale=1.0, tx_frac=0.0, ty_frac=0.0, hflip=False)
        img_r, msk_r = apply_geom(img, msk_png, p_rot)
        row3 = row_pair(
            put_title(to_bgr(img_r), f"rotate: {rot_rep:+.1f}°"),
            put_title(msk_r, "mask (rotated)")
        )

        # Row 4: SHEAR (geom on image+mask)
        p_sh = _with(base_p, rot_deg=0.0, shear_deg=float(shear_rep), scale=1.0, tx_frac=0.0, ty_frac=0.0, hflip=False)
        img_s, msk_s = apply_geom(img, msk_png, p_sh)
        row4 = row_pair(
            put_title(to_bgr(img_s), f"shear: {shear_rep:+.1f}°"),
            put_title(msk_s, "mask (sheared)")
        )

        # Row 5: HFLIP (geom on image+mask)
        p_hf = _with(base_p, hflip=True, rot_deg=0.0, shear_deg=0.0, scale=1.0, tx_frac=0.0, ty_frac=0.0)
        img_hf, msk_hf = apply_geom(img, msk_png, p_hf)
        row5 = row_pair(
            put_title(to_bgr(img_hf), f"hflip: applied (p={hfp:.2f})"),
            put_title(msk_hf, "mask (hflipped)")
        )

        # Row 6: SCALE (geom on image+mask)
        p_sc = _with(base_p, scale=float(scale_rep), rot_deg=0.0, shear_deg=0.0, tx_frac=0.0, ty_frac=0.0, hflip=False)
        img_sc, msk_sc = apply_geom(img, msk_png, p_sc)
        row6 = row_pair(
            put_title(to_bgr(img_sc), f"scale: ×{scale_rep:.3f}"),
            put_title(msk_sc, "mask (scaled)")
        )

        # Row 7: TRANSLATE (geom on image+mask)
        p_tr = _with(base_p, tx_frac=float(t_rep), ty_frac=float(t_rep), rot_deg=0.0, shear_deg=0.0, scale=1.0,
                     hflip=False)
        img_tr, msk_tr = apply_geom(img, msk_png, p_tr)
        row7 = row_pair(
            put_title(to_bgr(img_tr), f"translate: {t_rep:+.3f}W, {t_rep:+.3f}H"),
            put_title(msk_tr, "mask (translated)")
        )

        # Row 8: SPECKLE (photo on image only)
        # Use sigma from config and fix gamma=1 so only speckle is shown in this row
        p_sp = _with(base_p, speckle_sigma=sigma, gamma_min=1.0, gamma_max=1.0)
        img_sp = apply_photo(img, p_sp)
        row8 = row_pair(
            put_title(to_bgr(img_sp), f"speckle σ={sigma:.3f} (multiplicative)"),
            put_title(msk_png, "mask (unchanged)")
        )

        # Row 9: GAMMA (photo on image only)
        # Use the mid-gamma for determinism, and print the full range
        p_g = _with(base_p, gamma_min=gmid, gamma_max=gmid, speckle_sigma=0.0)
        img_g = apply_photo(img, p_g)
        row9 = row_pair(
            put_title(to_bgr(img_g), f"gamma: {gmid:.3f}  (range [{gmin:.3f},{gmax:.3f}])"),
            put_title(msk_png, "mask (unchanged)")
        )

        tiles = [row1, row2, row3, row4, row5, row6, row7, row8, row9]
        panel = stack_grid(tiles, cols=2, pad=16, bg=(0, 0, 0))
        cv2.imwrite(os.path.join(out_dir, f"{stem}__preview_panel.png"), panel)

    print(f"Saved previews to {out_dir}")

if __name__ == '__main__':
    main()


