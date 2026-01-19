# # utils/region_features.py
# from __future__ import annotations
# import math
# from typing import Dict, List, Tuple
# import numpy as np
# import cv2
#
# def _ensure_uint8(mask_bool: np.ndarray) -> np.ndarray:
#     return (mask_bool.astype(np.uint8) * 255)
#
# def _connected_stats(mask_bool: np.ndarray) -> Tuple[int, int, float, Tuple[int,int,int,int], Tuple[float,float], float]:
#     h, w = mask_bool.shape[:2]
#     N = float(h * w)
#     if not mask_bool.any():
#         return 0, 0, 0.0, (0,0,0,0), (0.0,0.0), 0.0
#
#     m8 = _ensure_uint8(mask_bool)
#     n_lbl, _, stats, centroids = cv2.connectedComponentsWithStats(m8, connectivity=8)
#     if n_lbl <= 1:
#         return 0, 0, 0.0, (0,0,0,0), (0.0,0.0), 0.0
#
#     areas = stats[1:, cv2.CC_STAT_AREA]
#     idx   = int(np.argmax(areas)) + 1
#     area_max = int(areas[idx-1])
#     frac_max = area_max / N
#     x, y, bw, bh, _ = stats[idx, :]
#     cx, cy = centroids[idx]
#
#     cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     perim = float(sum(cv2.arcLength(c, True) for c in cnts))
#     perim_norm = perim / math.hypot(h, w)
#
#     return (int(n_lbl - 1), area_max, float(frac_max), (int(x), int(y), int(bw), int(bh)), (float(cx), float(cy)), float(perim_norm))
#
# def compute_region_features(
#     img_1hw: np.ndarray,
#     ids_hw: np.ndarray,
#     labels_map: Dict[str, int],
#     class_names: List[str] | None = None,
#     ignore_background_id: int = 0,
# ) -> Dict[str, float]:
#     """
#     Per-class geometry & intensity features (background ignored).
#     Keys look like: {<cls>_area_frac, _area_px, _n_comp, _largest_frac, _bbox_ar, _centroid_x/_y, _perim_norm, _int_mean/_std/_med}
#     """
#     img = img_1hw[0] if img_1hw.ndim == 3 else img_1hw  # [H,W]
#     H, W = ids_hw.shape
#     N    = float(H * W)
#
#     # stable order
#     if class_names:
#         ordered = [(n, labels_map.get(n, None)) for n in class_names]
#         ordered = [(n, i) for (n, i) in ordered if i is not None]
#     else:
#         ordered = sorted(((n, int(i)) for n, i in labels_map.items()), key=lambda t: t[1])
#
#     feats: Dict[str, float] = {}
#     feats["img_mean"] = float(np.mean(img))
#     feats["img_std"]  = float(np.std(img))
#
#     for n, cid in ordered:
#         if cid == ignore_background_id:
#             continue
#         mask_c = (ids_hw == cid)
#         area   = int(mask_c.sum())
#         frac   = float(area / N)
#
#         feats[f"{n}_area_frac"] = frac
#         feats[f"{n}_area_px"]   = float(area)
#
#         n_comp, area_max, frac_max, (bx,by,bw,bh), (cx,cy), perim_n = _connected_stats(mask_c)
#         feats[f"{n}_n_comp"]        = float(n_comp)
#         feats[f"{n}_largest_frac"]  = float(frac_max)
#         feats[f"{n}_bbox_ar"]       = (float(bw) / float(bh)) if bh > 0 else 0.0
#         feats[f"{n}_centroid_x"]    = float(cx / max(W,1))
#         feats[f"{n}_centroid_y"]    = float(cy / max(H,1))
#         feats[f"{n}_perim_norm"]    = perim_n
#
#         if area > 0:
#             vals = img[mask_c]
#             feats[f"{n}_int_mean"] = float(vals.mean())
#             feats[f"{n}_int_std"]  = float(vals.std())
#             feats[f"{n}_int_med"]  = float(np.median(vals))
#         else:
#             feats[f"{n}_int_mean"] = 0.0
#             feats[f"{n}_int_std"]  = 0.0
#             feats[f"{n}_int_med"]  = 0.0
#
#     return feats
#
# def derive_labels_from_gt(
#     gt_ids_hw: np.ndarray,
#     labels_map: Dict[str,int],
#     task: str = "rd_binary",
# ) -> Tuple[int, str]:
#     """
#     task:
#       - 'rd_binary'        -> y in {0: Normal, 1: RD}
#       - 'rd_vh_normal'     -> y in {0: Normal, 1: VH-only, 2: RD}  (RD wins if both present)
#     returns (y, y_name)
#     """
#     rid = labels_map.get("retina", None) or labels_map.get("retinal_detachment", None) or labels_map.get("rd", None)
#     vid = labels_map.get("vitreous_humor", None) or labels_map.get("vh", None)
#
#     has_rd = (rid is not None) and (gt_ids_hw == rid).any()
#     has_vh = (vid is not None) and (gt_ids_hw == vid).any()
#
#     if task == "rd_binary":
#         y = 1 if has_rd else 0
#         return y, ("rd" if y==1 else "n")
#
#     # rd_vh_normal
#     if has_rd:
#         return 2, "rd"
#     if has_vh:
#         return 1, "vh"
#     return 0, "n"
