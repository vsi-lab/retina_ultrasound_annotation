# utils/vis_usg.py
import numpy as np
import cv2
import torch
from pathlib import Path
from training.metrics import per_class_dice_from_logits

def _rgb_to_bgr(col):
    # col is (R,G,B) from meta.json; OpenCV needs (B,G,R)
    return (int(col[2]), int(col[1]), int(col[0]))

def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0,2,4))

def _norm_key(s: str) -> str:
    return (
        s.strip().lower()
         .replace("-", " ")
         .replace("/", " ")
         .replace(".", " ")
         .replace("__", "_")
         .replace(" ", "_")
    )

def build_id2color_from_meta(meta_json: dict, labels_map: dict):
    """
    Robust mapping so all foreground classes (not background) get a color.
    If meta titles differ in style (e.g., 'Vitreous Humor' vs 'vitreous_humor'),
    we normalize names before matching.
    """
    norm_to_id = {_norm_key(k): int(v) for k, v in labels_map.items()}
    id2color = {}

    classes = meta_json.get("classes", []) if isinstance(meta_json, dict) else []
    for c in classes:
        title = c.get("title", "")
        hexcol = c.get("color")
        nk = _norm_key(title)
        if nk in norm_to_id and isinstance(hexcol, str):
            id2color[norm_to_id[nk]] = _hex_to_rgb(hexcol)

    # stable fallback palette for anything missing
    palette = [
        (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0),
        (255, 128, 0), (0, 128, 255), (128, 0, 255), (255, 0, 0)
    ]
    for _, cid in norm_to_id.items():
        if cid not in id2color:
            id2color[cid] = (0,0,0) if cid == 0 else palette[(cid - 1) % len(palette)]
    return id2color

def colorize_mask(mask_hw: np.ndarray, id2color: dict):
    h, w = mask_hw.shape
    out = np.zeros((h, w, 3), np.uint8)
    for cid, col_rgb in id2color.items():
        if cid == 0:
            continue  # leave background black
        out[mask_hw == cid] = _rgb_to_bgr(col_rgb)
    return out

# Colorize (BGR expected by OpenCV)
def colorize_ids(ids_hw: np.ndarray, id2color: dict) -> np.ndarray:
    """
    Convert a 2-D class-id map -> BGR color image for OpenCV display.
    - Background id 0 stays black (unless explicitly given in id2color).
    - id2color is assumed RGB tuples (or lists); we convert to BGR for cv2.
    - Robust to missing entries: any missing cid gets a stable fallback color.
    """
    h, w = ids_hw.shape
    out = np.zeros((h, w, 3), np.uint8)

    # stable fallback palette for any missing foreground ids (RGB)
    fallback = [
        (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0),
        (255, 128, 0), (0, 128, 255), (128, 0, 255), (255, 0, 0)
    ]

    # build RGB map with fallbacks (leave 0 as black unless provided)
    rgb_map = {}
    max_cid = int(ids_hw.max()) if ids_hw.size else 0
    for cid in range(1, max(1, max_cid + 1)):
        if cid in id2color:
            col = id2color[cid]
            rgb = (int(col[0]), int(col[1]), int(col[2]))  # ensure RGB
        else:
            rgb = fallback[(cid - 1) % len(fallback)]
        rgb_map[cid] = rgb

    # paint (convert RGB->BGR for OpenCV)
    for cid, rgb in rgb_map.items():
        mask = (ids_hw == cid)
        if mask.any():
            out[mask] = (rgb[2], rgb[1], rgb[0])

    # optional: if you explicitly provided a color for background id 0
    if 0 in id2color:
        r, g, b = [int(x) for x in id2color[0]]
        out[ids_hw == 0] = (b, g, r)

    return out

@torch.no_grad()
def save_preview_panel(out_path: Path, img_t, gt_t, logits_t, cfg, meta_json=None, raw_img_path: str | None = None):
    """
    Left: original raw image (no resize).
    Middle: ground-truth mask, mapped back to raw resolution (un-letterboxed).
    Right: prediction, mapped back to raw resolution (un-letterboxed).
    Legend: color + Dice and Present-Dice per foreground class (no background row).

    img_t:     [1,Hm,Wm] float tensor (letterboxed model input)
    gt_t:      [Hm,Wm]   long  tensor (letterboxed target)
    logits_t:  [C,Hm,Wm] float tensor (letterboxed logits)
    raw_img_path: path to original image file; if None, left column uses img_t
    """
    labels_map  = cfg["data"]["labels"]
    num_classes = max(int(v) for v in labels_map.values()) + 1
    class_names = cfg["data"].get("class_names", [f"class_{i}" for i in range(num_classes)])
    # class_names may be order-based; rely on ids instead:
    id2name = {int(v): k for k, v in labels_map.items()}

    # Colors
    if meta_json and "classes" in meta_json:
        id2color = build_id2color_from_meta(meta_json, labels_map)
    else:
        print("Warning: no meta json file")
        # stable fallback
        palette = [
            (255,255,0),(0,255,255),(255,0,255),(0,255,0),
            (255,128,0),(0,128,255),(128,0,255),(255,0,0)
        ]
        id2color = {0:(0,0,0)}
        for cid in range(1, num_classes):
            id2color[cid] = palette[(cid-1) % len(palette)]

    # Load raw (left) as-is
    if raw_img_path:
        left_gray = cv2.imread(str(raw_img_path), cv2.IMREAD_GRAYSCALE)
        if left_gray is None:
            left_gray = (img_t[0].cpu().numpy() * 255).astype(np.uint8)
            raw_img_path = None  # don't print bad path
    else:
        left_gray = (img_t[0].cpu().numpy() * 255).astype(np.uint8)

    left_bgr = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2BGR)
    H0, W0   = left_bgr.shape[:2]

    # Letterboxed tensors at model input res
    gt_lb   = gt_t.cpu().numpy().astype(np.int64)                      # [Hm,Wm]
    pred_lb = torch.argmax(logits_t, dim=0).cpu().numpy().astype(np.int64)  # [Hm,Wm]
    Hm, Wm  = gt_lb.shape

    # --- Un-letterbox back to original raw resolution (crop padded area, upscale) ---
    # Forward letterbox used: scale = min(Hm/H0, Wm/W0); pad equally on both sides.
    scale = min(Hm / float(H0), Wm / float(W0))
    new_h = int(round(H0 * scale))
    new_w = int(round(W0 * scale))
    top   = (Hm - new_h) // 2
    left  = (Wm - new_w) // 2
    box   = slice(top, top + new_h), slice(left, left + new_w)

    gt_crop   = gt_lb[box]
    pred_crop = pred_lb[box]

    gt_uhw   = cv2.resize(gt_crop,   (W0, H0), interpolation=cv2.INTER_NEAREST)
    pred_uhw = cv2.resize(pred_crop, (W0, H0), interpolation=cv2.INTER_NEAREST)



    mid_bgr   = colorize_ids(gt_uhw, id2color)
    right_bgr = colorize_ids(pred_uhw, id2color)

    # Metrics (Dice + present-only Dice), ignore background
    overall_d, present_d, _ = per_class_dice_from_logits(
        logits_t.unsqueeze(0), gt_t.unsqueeze(0), num_classes=num_classes, ignore_index=0
    )

    # Legend panel (rightmost)
    fg_ids = [cid for cid in sorted(id2color.keys()) if cid != 0]
    legend_w, pad, row_h = 360, 12, 26
    legend_h = pad*2 + row_h * (len(fg_ids) + 2)
    legend = np.zeros((max(H0, legend_h), legend_w, 3), np.uint8)

    cv2.putText(legend, "Dice (overall)  |  P-Dice (present)",
                (pad, pad + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1, cv2.LINE_AA)

    y = pad + 16 + 8
    for cid in fg_ids:
        y += row_h
        name = id2name.get(cid, f"class_{cid}")
        # id2color is RGB; convert once to BGR for drawing
        rgb = id2color.get(cid, (200, 200, 200))
        col_bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        d    = overall_d.get(cid, 0.0)
        po   = present_d.get(cid, None)
        po_s = f"{po:.3f}" if po is not None else "N/A"

        cv2.rectangle(legend, (pad, y - 18), (pad + 18, y), col_bgr, thickness=-1)
        cv2.putText(legend, f"{name:<16}  {d:0.3f}  |  {po_s}",
                    (pad + 26, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240,240,240), 1, cv2.LINE_AA)

    # Header: print (short) path
    if raw_img_path:
        # Try to print relative to work_root if applicable
        try:
            root = Path(cfg.get("work_root", "."))
            rel  = Path(raw_img_path).resolve().relative_to(root.resolve())
            title = rel.as_posix()
        except Exception:
            title = str(Path(raw_img_path).as_posix())
        cv2.putText(left_bgr, title, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    # Equalize heights for concat
    H = max(H0, legend.shape[0])
    def pad_h(img):
        if img.shape[0] == H: return img
        return cv2.copyMakeBorder(img, 0, H - img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0)

    panel = np.concatenate([pad_h(left_bgr), pad_h(mid_bgr), pad_h(right_bgr), legend], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)