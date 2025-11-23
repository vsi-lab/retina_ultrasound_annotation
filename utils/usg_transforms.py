# utils/usg_transforms.py

# Preview behavior:
# - Shows representative examples for rotate, shear, hflip, scale, translate, speckle, gamma.
# - Uses the same AugParams and apply_* functions as training.

from dataclasses import dataclass

import cv2
import numpy as np

"""
Shared ultrasound augmentations used by BOTH training and preview.

Order of operations (training)
------------------------------
1) Despeckle (if enabled in config; image only)
2) Geometric: rotate, shear, scale, translate, hflip (image & mask)
   - Image: bilinear sampling
   - Mask: nearest sampling (label-safe)
3) Photometric: speckle (multiplicative), gamma jitter, optional contrast (image only)

Preview behavior
----------------
- Uses `representative_params(cfg)` to pick deterministic values derived from the same ranges.
- Shows rows: original, despeckle, rotate, shear, speckle, gamma.
- Scale/translate/hflip are exercised during training but omitted as separate rows for clarity.

Config keys (configs/config_usg.yaml)
-------------------------------------
aug:
  hflip: float           # probability
  rotate_deg: [min, max]
  shear_deg:  [min, max]
  scale:      [min, max]
  translate:  [min, max] # fractions of W/H
  brightness: [gmin, gmax] # gamma
  contrast:   [cmin, cmax] # contrast gain; [1,1] disables
  gaussian_noise_std: float
  p: float               # block-level apply prob (dataset wrapper)
"""



# ------- physics/primitive ops we already use --------
def add_speckle(img01: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply multiplicative speckle noise to an image in [0,1].

    Parameters
    ----------
    img01 : np.ndarray
        Float32 image in [0,1].
    sigma : float
        Standard deviation for multiplicative speckle. If <=0, no-op.

    Returns
    -------
    np.ndarray
        Float32 image in [0,1] after speckle.
    """

    if sigma <= 0:
        return img01
    noise = np.random.normal(0.0, sigma, img01.shape).astype(np.float32)
    out = img01 * (1.0 + noise)
    return np.clip(out, 0.0, 1.0)

def gamma_jitter(img01: np.ndarray, gmin: float, gmax: float) -> np.ndarray:
    """
    Apply gamma correction with gamma ~ U[gmin, gmax].

    Notes
    -----
    Using gamma around 1.0 acts like brightness variation while avoiding
    additive bias that can distort speckle statistics.
    """
    if gmin == 1.0 and gmax == 1.0:
        return img01
    g = np.random.uniform(gmin, gmax)
    # avoid zero^gamma
    return np.power(np.clip(img01, 1e-6, 1.0), g)

def affine_geom(img01: np.ndarray, mask_u8: np.ndarray,
                rot_deg: float, scale: float, tx_frac: float, ty_frac: float, shear_deg: float):
    """Apply the same 2D affine to image and mask.

    Image warp uses bilinear (INTER_LINEAR), mask uses nearest (INTER_NEAREST).

    Parameters
    ----------
    img01 : np.ndarray
        Float32 image in [0,1].
    mask_u8 : np.ndarray
        Grayscale label map (uint8).
    rot_deg, scale, tx_frac, ty_frac, shear_deg : float
        Rotation (deg), isotropic scale, translations as fraction of W/H,
        and x-shear (deg).

    Returns
    -------
    (np.ndarray, np.ndarray)
        (img01_out, mask_u8_out) after warp.
    """
    H, W = img01.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    # rotation
    a = np.deg2rad(rot_deg)
    R = np.array([[np.cos(a), -np.sin(a), 0.0],
                  [np.sin(a),  np.cos(a), 0.0],
                  [0.0,        0.0,       1.0]], dtype=np.float32)
    # shear x
    s = np.tan(np.deg2rad(shear_deg))
    Sh = np.array([[1.0, s,   0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float32)
    # scale
    Sc = np.array([[scale, 0.0,   0.0],
                   [0.0,   scale, 0.0],
                   [0.0,   0.0,   1.0]], dtype=np.float32)
    # translate in pixels
    tx = tx_frac * W
    ty = ty_frac * H
    T = np.array([[1.0, 0.0, tx],
                  [0.0, 1.0, ty],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    # center shift
    C1 = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]], dtype=np.float32)
    C2 = np.array([[1,0, cx],[0,1, cy],[0,0,1]], dtype=np.float32)

    A = C2 @ T @ Sc @ Sh @ R @ C1
    A2x3 = A[:2, :]

    img_u8 = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    img_u8 = cv2.warpAffine(img_u8, A2x3, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask_u8 = cv2.warpAffine(mask_u8, A2x3, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    img01_out = img_u8.astype(np.float32) / 255.0
    return img01_out, mask_u8

def hflip_if_allowed(img01: np.ndarray, mask_u8: np.ndarray, allow_flip: bool):
    if not allow_flip:
        return img01, mask_u8
    return img01[:, ::-1, ...], mask_u8[:, ::-1]

# --------------- unified param struct -----------------
@dataclass
class AugParams:
    rot_deg: float = 0.0
    shear_deg: float = 0.0
    scale: float = 1.0
    tx_frac: float = 0.0
    ty_frac: float = 0.0
    hflip: bool = False
    speckle_sigma: float = 0.0
    gamma_min: float = 1.0
    gamma_max: float = 1.0

def _range_from_cfg(cfg, key, default):
    v = cfg['aug'].get(key, default)
    return (float(v[0]), float(v[1])) if isinstance(v, (list, tuple)) else (float(v), float(v))

def sample_params(cfg, rng=np.random):
    rot_lo, rot_hi   = _range_from_cfg(cfg, 'rotate_deg', [-5, 5])
    shear_lo, shear_hi = _range_from_cfg(cfg, 'shear_deg', [-2, 2])
    sc_lo, sc_hi     = _range_from_cfg(cfg, 'scale', [0.98, 1.02])
    tr_lo, tr_hi     = _range_from_cfg(cfg, 'translate', [-0.02, 0.02])
    b_lo, b_hi       = _range_from_cfg(cfg, 'brightness', [0.98, 1.02])  # gamma range around 1.0
    sigma            = float(cfg['aug'].get('gaussian_noise_std', 0.0))
    hflip_p          = float(cfg['aug'].get('hflip', 0.2))

    return AugParams(
        rot_deg   = rng.uniform(rot_lo, rot_hi),
        shear_deg = rng.uniform(shear_lo, shear_hi),
        scale     = rng.uniform(sc_lo, sc_hi),
        tx_frac   = rng.uniform(tr_lo, tr_hi),
        ty_frac   = rng.uniform(tr_lo, tr_hi),
        hflip     = rng.rand() < hflip_p,
        speckle_sigma = sigma,
        gamma_min = b_lo,
        gamma_max = b_hi,
    )

def representative_params(cfg) -> AugParams:
    """Deterministic params derived from cfg (for preview rows)."""
    rot_lo, rot_hi   = _range_from_cfg(cfg, 'rotate_deg', [-5, 5])
    shear_lo, shear_hi = _range_from_cfg(cfg, 'shear_deg', [-2, 2])
    sc_lo, sc_hi     = _range_from_cfg(cfg, 'scale', [0.98, 1.02])
    tr_lo, tr_hi     = _range_from_cfg(cfg, 'translate', [-0.02, 0.02])
    b_lo, b_hi       = _range_from_cfg(cfg, 'brightness', [0.98, 1.02])
    sigma            = float(cfg['aug'].get('gaussian_noise_std', 0.0))

    return AugParams(
        rot_deg   = np.sign(rot_hi if abs(rot_hi)>=abs(rot_lo) else rot_lo) * max(abs(rot_lo), abs(rot_hi)),
        shear_deg = np.sign(shear_hi if abs(shear_hi)>=abs(shear_lo) else shear_lo) * max(abs(shear_lo), abs(shear_hi)),
        scale     = 1.0,
        tx_frac   = 0.0,
        ty_frac   = 0.0,
        hflip     = False,
        speckle_sigma = sigma,
        gamma_min = b_lo,
        gamma_max = b_hi,
    )

# ----------------- apply pipeline ---------------------
def apply_geom(img_u8: np.ndarray, mask_u8: np.ndarray, p: AugParams):
    img01 = img_u8.astype(np.float32) / 255.0
    img01, mask_u8 = affine_geom(img01, mask_u8, p.rot_deg, p.scale, p.tx_frac, p.ty_frac, p.shear_deg)
    img01, mask_u8 = hflip_if_allowed(img01, mask_u8, p.hflip)
    return (img01 * 255).astype(np.uint8), mask_u8

def apply_photo(img_u8: np.ndarray, p: AugParams):
    img01 = img_u8.astype(np.float32) / 255.0
    img01 = add_speckle(img01, p.speckle_sigma)
    img01 = gamma_jitter(img01, p.gamma_min, p.gamma_max)
    return (np.clip(img01, 0, 1) * 255).astype(np.uint8)

def augment_pair(img_u8: np.ndarray, mask_u8: np.ndarray, cfg, rng=None):
    """Random full augmentation for training.

    Steps
    -----
    1) Sample params via `sample_params(cfg, rng)`.
    2) `apply_geom` to image & mask.
    3) `apply_photo` to image only.
    """
    rng = np.random if rng is None else rng
    p = sample_params(cfg, rng)
    img_u8, mask_u8 = apply_geom(img_u8, mask_u8, p)
    img_u8 = apply_photo(img_u8, p)
    return img_u8, mask_u8
