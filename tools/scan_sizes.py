# # tools/scan_sizes.py
# import argparse
# from collections import Counter, defaultdict
# from pathlib import Path
#
# import cv2
# import numpy as np
# import pandas as pd
# import yaml
#
# """
# Print sizes + class presence per frame, and summarize class presence overall & by split.
#
# Usage:
# python -m tools.scan_sizes \
#   --config configs/config_usg.yaml \
#   --out work_dir/sizes.csv \
#   --csv work_dir/metadata/train.csv \
#   --csv work_dir/metadata/val.csv \
#   --csv work_dir/metadata/test.csv
#
#
# # python -m tools.scan_sizes --config  configs/config_usg.yaml --out work_dir/sizes.csv --csv work_dir/metadata/train.csv --csv work_dir/metadata/test.csv --csv work_dir/metadata/val.csv
#
# """
#
# # --- optional: use your project resolver if available ---
# try:
#     from utils.paths import resolve_under_root_cfg as _resolve
# except Exception:
#     _resolve = None
#
#
# def resolve_under_root_cfg(cfg, p: str) -> Path:
#     if _resolve is not None:
#         return Path(_resolve(cfg, p))
#     root = Path(cfg.get("work_root", "."))
#     pp = Path(p)
#     return pp if pp.is_absolute() else (root / pp)
#
#
# def hw_of(img):
#     """Return (H, W) ignoring channels."""
#     if img is None:
#         return None
#     return img.shape[0], img.shape[1]
#
#
# def _labels_from_cfg(cfg):
#     dm = cfg.get("data", {})
#     labels_map = dm.get("labels", {}) or {}
#     # normalize to {int_id: name} and keep a stable order (by id)
#     id2name = {}
#     for name, cid in labels_map.items():
#         try:
#             id2name[int(cid)] = str(name)
#         except Exception:
#             continue
#     if not id2name:
#         # fallback: assume {0:"background",1:"class_1",...} if nothing provided
#         id2name = {0: "background", 1: "class_1", 2: "class_2", 3: "class_3", 4: "class_4"}
#     return dict(sorted(id2name.items(), key=lambda kv: kv[0]))
#
#
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config", required=True)
#     ap.add_argument("--csv", action="append", required=True,
#                     help="One or more CSV files with image_path,mask_path")
#     ap.add_argument("--out", default=None, help="Optional CSV report path")
#     ap.add_argument("--include_background", action="store_true",
#                     help="Include background (id 0) in presence summary")
#     args = ap.parse_args()
#
#     cfg = yaml.safe_load(open(args.config, "r"))
#     id2name = _labels_from_cfg(cfg)
#     class_ids_sorted = list(id2name.keys())
#
#     rows = []
#     mismatches = []
#     img_sizes = Counter()
#     msk_sizes = Counter()
#
#     for csv_path in args.csv:
#         df = pd.read_csv(csv_path)
#         split = Path(csv_path).stem.lower()  # e.g., train/val/test from filename
#         for _, r in df.iterrows():
#             ip = resolve_under_root_cfg(cfg, str(r["image_path"]))
#             mp = resolve_under_root_cfg(cfg, str(r["mask_path"]))
#
#             img = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
#             msk = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
#
#             img_hw = hw_of(img)
#             msk_hw = hw_of(msk)
#
#             row = {
#                 "split": split,
#                 "csv": Path(csv_path).name,
#                 "image_path": str(ip),
#                 "mask_path": str(mp),
#                 "img_H": None if img_hw is None else img_hw[0],
#                 "img_W": None if img_hw is None else img_hw[1],
#                 "msk_H": None if msk_hw is None else msk_hw[0],
#                 "msk_W": None if msk_hw is None else msk_hw[1],
#                 "img_exists": img is not None,
#                 "msk_exists": msk is not None,
#             }
#
#             # quick unique id sample (for sanity)
#             if msk is not None:
#                 if msk.ndim == 2:
#                     samp = msk[::64, ::64]
#                     row["mask_unique_sample"] = ",".join(map(str, sorted(pd.unique(samp.reshape(-1))[:10])))
#                 else:
#                     row["mask_unique_sample"] = "color_mask"
#
#             # per-class presence flags (only reliable for id-masks: 2D)
#             presence_valid = (msk is not None and msk.ndim == 2)
#             row["presence_valid"] = bool(presence_valid)
#             if presence_valid:
#                 vals = np.unique(msk).astype(int)
#                 present_set = set(vals.tolist())
#                 for cid in class_ids_sorted:
#                     cname = id2name[cid]
#                     row[f"has_{cname}"] = int(cid in present_set)
#             else:
#                 for cid in class_ids_sorted:
#                     cname = id2name[cid]
#                     row[f"has_{cname}"] = np.nan  # unknown for color masks / missing masks
#
#             rows.append(row)
#
#             if img_hw is not None:
#                 img_sizes[(img_hw[0], img_hw[1])] += 1
#             if msk_hw is not None:
#                 msk_sizes[(msk_hw[0], msk_hw[1])] += 1
#
#             if img_hw is None or msk_hw is None or (img_hw != msk_hw):
#                 mismatches.append((str(ip), img_hw, str(mp), msk_hw))
#
#     report = pd.DataFrame(rows)
#
#     # --- console size summary ---
#     print("\n=== Scan Summary ===")
#     print(f"Total pairs scanned: {len(report)}")
#
#     print("\nTop image sizes (H x W):")
#     for (h, w), cnt in img_sizes.most_common(10):
#         print(f"  {h}x{w} : {cnt}")
#
#     print("\nTop mask sizes (H x W):")
#     for (h, w), cnt in msk_sizes.most_common(10):
#         print(f"  {h}x{w} : {cnt}")
#
#     if mismatches:
#         print(f"\nMismatched / unreadable pairs: {len(mismatches)}")
#         for k, (ip, ihw, mp, mhw) in enumerate(mismatches[:25]):
#             print(f"  [{k:02d}] IMG {ihw} <- {ip}")
#             print(f"       MSK {mhw} <- {mp}")
#         if len(mismatches) > 25:
#             print(f"  ... and {len(mismatches) - 25} more")
#     else:
#         print("\nNo size mismatches found.")
#
#     # --- class presence summary (overall + by split) ---
#     print("\n=== Class Presence Summary (mask id-maps only) ===")
#     valid_mask = (report["msk_exists"] == True) & (report["presence_valid"] == True)
#     denom_overall = int(valid_mask.sum())
#     splits = sorted(report["split"].dropna().unique().tolist())
#
#     summary_rows = []
#     for cid in class_ids_sorted:
#         if cid == 0 and not args.include_background:
#             continue
#         cname = id2name[cid]
#         col = f"has_{cname}"
#
#         cnt_overall = int(pd.to_numeric(report.loc[valid_mask, col], errors="coerce").fillna(0).sum())
#         pct_overall = (100.0 * cnt_overall / denom_overall) if denom_overall else np.nan
#
#         row = {
#             "class_id": cid,
#             "class": cname,
#             "overall_presence_count": cnt_overall,
#             "overall_presence_pct": round(pct_overall, 2) if pct_overall == pct_overall else np.nan,
#             "overall_images": denom_overall,
#         }
#
#         for sp in splits:
#             sp_mask = valid_mask & (report["split"].astype(str).str.lower() == sp)
#             denom_sp = int(sp_mask.sum())
#             cnt_sp = int(pd.to_numeric(report.loc[sp_mask, col], errors="coerce").fillna(0).sum())
#             pct_sp = (100.0 * cnt_sp / denom_sp) if denom_sp else np.nan
#             row[f"{sp}_count"] = cnt_sp
#             row[f"{sp}_pct"] = round(pct_sp, 2) if pct_sp == pct_sp else np.nan
#             row[f"{sp}_images"] = denom_sp
#
#         summary_rows.append(row)
#
#     summary = pd.DataFrame(summary_rows).sort_values(["class_id"]).reset_index(drop=True)
#
#     # pretty print to console
#     if not summary.empty:
#         print(summary.drop(columns=[c for c in summary.columns if c.endswith("_images")]).to_string(index=False))
#     else:
#         print("No valid id-masks found to compute presence.")
#
#     # --- write CSVs ---
#     if args.out:
#         out_path = Path(args.out)
#         out_path.parent.mkdir(parents=True, exist_ok=True)
#         report.to_csv(out_path, index=False)
#         sum_path = out_path.with_name(out_path.stem + "_presence_summary.csv")
#         summary.to_csv(sum_path, index=False)
#         print(f"\nSaved detailed report to: {out_path}")
#         print(f"Saved class presence summary to: {sum_path}")
#
#
# if __name__ == "__main__":
#     main()