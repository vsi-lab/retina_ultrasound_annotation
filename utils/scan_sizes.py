# tools/scan_sizes.py
import argparse
from pathlib import Path
import sys
import json
import yaml
import pandas as pd
import cv2
from collections import Counter

# --- optional: use your project resolver if available ---
try:
    from utils.paths import resolve_under_root_cfg as _resolve
except Exception:
    _resolve = None

def resolve_under_root_cfg(cfg, p: str) -> Path:
    if _resolve is not None:
        return Path(_resolve(cfg, p))
    root = Path(cfg.get("work_root", "."))
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp)

def hw_of(img):
    """Return (H, W) ignoring channels."""
    if img is None:
        return None
    if img.ndim == 2:
        return img.shape[0], img.shape[1]
    return img.shape[0], img.shape[1]  # CHW/ HWC -> first two dims

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--csv", action="append", required=True,
                    help="One or more CSV files with image_path,mask_path")
    ap.add_argument("--out", default=None, help="Optional CSV report path")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    rows = []
    mismatches = []
    img_sizes = Counter()
    msk_sizes = Counter()

    for csv_path in args.csv:
        df = pd.read_csv(csv_path)
        for i, r in df.iterrows():
            ip = resolve_under_root_cfg(cfg, str(r["image_path"]))
            mp = resolve_under_root_cfg(cfg, str(r["mask_path"]))

            img = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
            msk = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)

            img_hw = hw_of(img)
            msk_hw = hw_of(msk)

            row = {
                "csv": Path(csv_path).name,
                "image_path": str(ip),
                "mask_path": str(mp),
                "img_H": None if img_hw is None else img_hw[0],
                "img_W": None if img_hw is None else img_hw[1],
                "msk_H": None if msk_hw is None else msk_hw[0],
                "msk_W": None if msk_hw is None else msk_hw[1],
                "img_exists": img is not None,
                "msk_exists": msk is not None,
            }

            # quick unique id sample for masks (fast, but skip if huge)
            # comment out if you want it faster
            if msk is not None:
                if msk.ndim == 2:
                    # id mask
                    # take a small grid sample to avoid huge cost
                    samp = msk[::64, ::64]
                    row["mask_unique_sample"] = ",".join(map(str, sorted(pd.unique(samp.reshape(-1))[:10])))
                else:
                    row["mask_unique_sample"] = "color_mask"

            rows.append(row)

            if img_hw is not None:
                img_sizes[(img_hw[0], img_hw[1])] += 1
            if msk_hw is not None:
                msk_sizes[(msk_hw[0], msk_hw[1])] += 1

            if img_hw is None or msk_hw is None or (img_hw != msk_hw):
                mismatches.append((str(ip), img_hw, str(mp), msk_hw))

    report = pd.DataFrame(rows)
    # --- console summary ---
    print("\n=== Scan Summary ===")
    print(f"Total pairs scanned: {len(report)}")

    # Top image sizes
    print("\nTop image sizes (H x W):")
    for (h, w), cnt in img_sizes.most_common(10):
        print(f"  {h}x{w} : {cnt}")

    print("\nTop mask sizes (H x W):")
    for (h, w), cnt in msk_sizes.most_common(10):
        print(f"  {h}x{w} : {cnt}")

    if mismatches:
        print(f"\nMismatched / unreadable pairs: {len(mismatches)}")
        for k, (ip, ihw, mp, mhw) in enumerate(mismatches[:25]):
            print(f"  [{k:02d}] IMG {ihw} <- {ip}")
            print(f"       MSK {mhw} <- {mp}")
        if len(mismatches) > 25:
            print(f"  ... and {len(mismatches) - 25} more")
    else:
        print("\nNo size mismatches found.")

    # write CSV if requested
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(out_path, index=False)
        print(f"\nSaved detailed report to: {out_path}")

if __name__ == "__main__":
    main()