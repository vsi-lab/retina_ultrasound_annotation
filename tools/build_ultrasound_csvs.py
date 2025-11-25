# # tools/build_ultrasound_csvs.py
# """
# Build patient-aware CSVs from Supervisely export.
#
# Assumptions (as per your latest clarification):
# - Folder structure:
#     work_dir/
#       images/PatientX/*.png|jpg
#       masks/PatientX/*.png|jpg
#       meta.json
#       obj_class_to_machine_color.json
#
# - Filename inside a patient folder can be arbitrary
#   (e.g., Patient3/Subject 13.6.png). We DO NOT parse filenames
#   to infer patient_id.
#
# Outputs:
#   work_dir/metadata/
#     labels.csv   : all pairs with patient_id + scan_id + placeholders
#     train.csv
#     val.csv
#     test.csv
#     stats.csv    : distribution by split/patient and simple counts
#
# Clinical diagnosis labels (RD/VH/Normal) are NOT available yet.
# We keep placeholders for later merge.
#
# python -m tools.build_ultrasound_csvs --work_dir work_dir --config configs/config_usg.yaml
#
# python -m tools.build_ultrasound_csvs --work_dir work_dir --config configs/config_usg.yaml --labels_csv work_dir/metadata/disease_labels.csv
#
# """
#
# import argparse, json, os, glob
# from typing import Dict, List, Tuple
# import cv2
# import yaml
#
# from pathlib import Path
# import json, glob
# import pandas as pd
#
# IMG_EXTS = (".png", ".jpg", ".jpeg")
#
#
# def _load_yaml(p: str) -> Dict:
#     with open(p, "r") as f:
#         return yaml.safe_load(f)
#
#
# def _read_json(p: Path) -> Dict:
#     if not p.exists():
#         return {}
#     with open(p, "r") as f:
#         return json.load(f)
#
#
# def _find_patient_folders(root: Path) -> List[Path]:
#     if not root.exists():
#         return []
#     return sorted([p for p in root.iterdir() if p.is_dir()])
#
#
# def _find_images(folder: Path) -> List[Path]:
#     files = []
#     for ext in IMG_EXTS:
#         files.extend(folder.glob(f"*{ext}"))
#     return sorted(files)
#
#
# def _match_mask(image_path: Path, mask_folder: Path) -> Path | None:
#     """
#     Match by same stem (case-sensitive).
#     If not found, return None.
#     """
#     stem = image_path.stem
#     for ext in IMG_EXTS:
#         mp = mask_folder / f"{stem}{ext}"
#         if mp.exists():
#             return mp
#     return None
#
#
# def _convert_color_mask_to_gray(mask_bgr, color2id: Dict[Tuple[int,int,int], int]):
#     """
#     Convert Supervisely color mask -> single channel class-id mask.
#
#     color2id keys are (B,G,R).
#     Unknown colors become 0 (background).
#     """
#     h, w, _ = mask_bgr.shape
#     gray = (mask_bgr[...,0].astype("uint32") << 16) | \
#            (mask_bgr[...,1].astype("uint32") << 8)  | \
#            (mask_bgr[...,2].astype("uint32"))
#
#     out = pd.Series(gray.ravel()).map(
#         lambda v: color2id.get(((v>>16)&255, (v>>8)&255, v&255), 0)
#     ).to_numpy().reshape(h, w).astype("uint8")
#     return out
#
#
# def _maybe_load_mask(mask_path: Path, color2id: Dict):
#     """
#     Load mask.
#     If 3-channel, map colors via obj_class_to_machine_color.json.
#     If already 1-channel, use as is.
#     """
#     m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
#     if m is None:
#         raise FileNotFoundError(mask_path)
#
#     if m.ndim == 3 and m.shape[2] == 3:
#         # convert using mapping
#         return _convert_color_mask_to_gray(m, color2id)
#     if m.ndim == 2:
#         return m
#     # handle weird alpha channel 4th
#     if m.ndim == 3 and m.shape[2] == 4:
#         return m[...,0]
#     return m
#
#
#
#
# def build_labels(work_dir: Path, splits_cfg: dict) -> pd.DataFrame:
#     images_root = work_dir / "images"
#     masks_root  = work_dir / "masks"
#
#     # -------------------------
#     # local helpers (self-contained)
#     # -------------------------
#     def _read_json(p: Path):
#         if not p.exists():
#             return None
#         with open(p, "r") as f:
#             return json.load(f)
#
#     def _hex_to_rgb(h: str):
#         if not isinstance(h, str):
#             return None
#         h = h.lstrip("#")
#         if len(h) != 6:
#             return None
#         try:
#             r = int(h[0:2], 16)
#             g = int(h[2:4], 16)
#             b = int(h[4:6], 16)
#             return (r, g, b)
#         except Exception:
#             return None
#
#     def _iter_classes(obj):
#         """
#         Supports:
#           - meta.json: {"classes": [ {...}, ... ]}
#           - obj_class_to_machine_color.json:
#                 * dict: { "Retina": {...}, ... }
#                 * list: [ {...}, ... ]
#         """
#         if obj is None:
#             return []
#         if isinstance(obj, dict):
#             # meta.json-like
#             if "classes" in obj and isinstance(obj["classes"], list):
#                 return obj["classes"]
#             # obj_map-like dict-of-dicts
#             return [{"title": k, **v} if isinstance(v, dict) else {"title": k}
#                     for k, v in obj.items()]
#         if isinstance(obj, list):
#             return obj
#         return []
#
#     def _extract_name_and_rgb(d, fallback_name="class"):
#         # name
#         cname = None
#         if isinstance(d, dict):
#             cname = d.get("title") or d.get("name") or d.get("class_name") or fallback_name
#
#             # color may be hex string in meta.json
#             col = d.get("color") or d.get("machine_color")
#             rgb = None
#             if isinstance(col, str):
#                 rgb = _hex_to_rgb(col)
#             elif isinstance(col, (list, tuple)) and len(col) == 3:
#                 rgb = tuple(int(x) for x in col)
#             else:
#                 rgb = None
#             return cname, rgb
#         return fallback_name, None
#
#     def _find_patient_folders(root: Path):
#         if not root.exists():
#             return []
#         return sorted([p for p in root.iterdir() if p.is_dir()])
#
#     def _find_images(pf: Path):
#         exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
#         imgs = []
#         for e in exts:
#             imgs.extend(glob.glob(str(pf / e)))
#         return sorted([Path(x) for x in imgs])
#
#     def _match_mask(ip: Path, mask_pf: Path):
#         # same stem, any image extension
#         stem = ip.stem
#         exts = ("png","jpg","jpeg","bmp","tif","tiff")
#         for e in exts:
#             cand = mask_pf / f"{stem}.{e}"
#             if cand.exists():
#                 return cand
#         return None
#
#     # -------------------------
#     # parse class colors (kept for later steps, not required to build rows)
#     # -------------------------
#     meta = _read_json(work_dir / "meta.json")
#     obj_colors = _read_json(work_dir / "obj_class_to_machine_color.json")
#
#     class_entries = _iter_classes(obj_colors) if obj_colors is not None else _iter_classes(meta)
#
#     class_rgb = {}
#     for i, d in enumerate(class_entries):
#         cname, rgb = _extract_name_and_rgb(d, fallback_name=f"class_{i}")
#         if cname is None:
#             continue
#         if rgb is not None:
#             class_rgb[cname] = rgb
#     # (class_rgb currently unused here; leave it for downstream pixel counting / sanity checks)
#
#     # -------------------------
#     # build labels rows
#     # -------------------------
#     rows = []
#     patient_folders = _find_patient_folders(images_root)
#
#     for pf in patient_folders:
#         patient_id = pf.name  # folder name defines patient
#         mask_pf = masks_root / patient_id
#         if not mask_pf.exists():
#             print(f"[WARN] No mask folder for {patient_id}: {mask_pf}")
#             continue
#
#         imgs = _find_images(pf)
#         if not imgs:
#             print(f"[WARN] No images in {pf}")
#             continue
#
#         for ip in imgs:
#             mp = _match_mask(ip, mask_pf)
#             if mp is None:
#                 print(f"[WARN] No mask for image {ip.name} under {mask_pf}")
#                 continue
#
#             rows.append({
#                 # store paths relative to work_dir for portability
#                 "image_path": str(ip.relative_to(work_dir)),
#                 "mask_path": str(mp.relative_to(work_dir)),
#                 "patient_id": patient_id,
#                 "scan_id": ip.stem,   # e.g. "Subject 13.6"
#                 "diagnosis": "",      # placeholder: Normal/RD/VH later
#             })
#
#     df = pd.DataFrame(rows)
#     return df
#
#
# def apply_splits(df: pd.DataFrame, splits_cfg: Dict) -> Dict[str, pd.DataFrame]:
#     """
#     Split by patient lists provided in config.
#     """
#     splits = splits_cfg.get("splits", {})
#     out = {}
#
#     all_patients = sorted(df["patient_id"].unique().tolist())
#     for split_name in ["train", "val", "test"]:
#         want = splits.get(split_name, [])
#         want_set = set(want)
#
#         found = [p for p in want if p in all_patients]
#         missing = [p for p in want if p not in all_patients]
#
#         if missing:
#             print(f"[WARN] {split_name}: patients not found in images/: {missing}")
#         if not want:
#             print(f"[WARN] {split_name}: empty patient list in config.")
#
#         sdf = df[df["patient_id"].isin(want_set)].reset_index(drop=True)
#         out[split_name] = sdf
#
#         print(f"[INFO] {split_name}: {len(sdf)} rows from patients {found}")
#
#     return out
#
#
# def write_outputs(work_dir: Path, df: pd.DataFrame, split_dfs: Dict[str, pd.DataFrame]):
#     meta_dir = work_dir / "metadata"
#     meta_dir.mkdir(parents=True, exist_ok=True)
#
#     labels_csv = meta_dir / "labels.csv"
#     df.to_csv(labels_csv, index=False)
#     print("[OK] wrote", labels_csv)
#
#     for name, sdf in split_dfs.items():
#         p = meta_dir / f"{name}.csv"
#         sdf.to_csv(p, index=False)
#         print("[OK] wrote", p)
#
#     # Simple stats
#     stats = []
#     for name, sdf in split_dfs.items():
#         stats.append({
#             "split": name,
#             "num_scans": len(sdf),
#             "num_patients": sdf["patient_id"].nunique()
#         })
#         by_pat = sdf.groupby("patient_id").size().reset_index(name="num_scans")
#         for _, r in by_pat.iterrows():
#             stats.append({
#                 "split": f"{name}:{r['patient_id']}",
#                 "num_scans": int(r["num_scans"]),
#                 "num_patients": 1
#             })
#
#     stats_df = pd.DataFrame(stats)
#     stats_csv = meta_dir / "stats.csv"
#     stats_df.to_csv(stats_csv, index=False)
#     print("[OK] wrote", stats_csv)
#
#
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--work_dir", required=False, help="root data folder")
#     ap.add_argument("--config", required=True, help="yaml with splits")
#     args = ap.parse_args()
#     cfg = _load_yaml(args.config)
#
#     if args.work_dir:
#         work_dir = Path(args.work_dir)
#     else:
#         work_dir = Path(cfg.get("data")['work_dir'])
#     print(f"[INFO] work_dir: {work_dir}")
#     df = build_labels(work_dir, cfg)
#     if df.empty:
#         raise RuntimeError("No image-mask pairs found. Check folder structure / naming.")
#
#     split_dfs = apply_splits(df, cfg)
#     write_outputs(work_dir, df, split_dfs)
#
#     print("[DONE] total pairs:", len(df))
#
#
# if __name__ == "__main__":
#     main()