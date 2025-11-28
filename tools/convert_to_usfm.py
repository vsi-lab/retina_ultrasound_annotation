#!/usr/bin/env python3
import argparse, csv, json, re, shutil, sys
from pathlib import Path
from collections import defaultdict, Counter

SPLITS = [("train","trainning_set"), ("val","val_set"), ("test","test_set")]
CLS_FOLDERS = {"n": "n", "normal": "normal", "vh":"vh", "rd":"rd"}  # case-insensitive map

BOLD = "\033[1m"
END  = "\033[0m"

"""
# save as tools/convert_to_usfm.py (or anywhere)
python tools/convert_to_usfm.py \
  --work_dir "/Users/saurav1/python/masters/arizona/2nd/fall/retina_ultrasound_annotation/work_dir" \
  --dataset retina_usg \
  --out_dir "/Users/saurav1/python/masters/arizona/2nd/fall/retina_ultrasound_annotation/work_dir/USFM_datasets_Cls"
"""


def bold(msg): return f"{BOLD}{msg}{END}"

def read_csv(p: Path):
    import pandas as pd
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(bold(f"[WARN] Failed reading {p}: {e}")); return None

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def parse_patient_from_rel(rel: str) -> str:
    m = re.search(r"(Patient\d+)", rel)
    return m.group(1) if m else "PatientX"

def sanitize_name(name: str) -> str:
    # keep stem-friendly characters; replace spaces and weird chars with underscores
    s = name.replace(" ", "_")
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s

def build_dest_name(img_rel: str) -> str:
    ip = Path(img_rel)
    patient = parse_patient_from_rel(img_rel)
    stem = sanitize_name(ip.stem)
    return f"{patient}__{stem}{ip.suffix.lower()}"

def copy_with_dedupe(src: Path, dst: Path):
    if not dst.exists():
        shutil.copy2(src, dst)
        return dst.name
    # append numbered suffix if collision
    base = dst.stem
    suf  = dst.suffix
    parent = dst.parent
    k = 1
    while True:
        cand = parent / f"{base}__dup{k}{suf}"
        if not cand.exists():
            shutil.copy2(src, cand)
            return cand.name
        k += 1

def normalize_rel(work_dir: Path, rel: str) -> Path:
    # CSVs store paths relative to work_dir; keep that convention
    rel = rel.strip().replace("\\", "/")
    return (work_dir / rel).resolve()

def load_labels_table(labels_csv: Path, work_dir: Path) -> dict:
    """Return {images/PatientX/Scan.png: class_folder_name}"""
    lab = read_csv(labels_csv)
    if lab is None or lab.empty:
        print(bold(f"[WARN] labels.csv empty or unreadable: {labels_csv}")); return {}
    if "image_path" not in lab.columns:
        raise ValueError("labels.csv must contain 'image_path'")
    col = None
    for c in ["diagnosis", "label", "class"]:
        if c in lab.columns: col = c; break
    if col is None:
        raise ValueError("labels.csv must have a diagnosis/label/class column")

    m = {}
    for _, r in lab.iterrows():
        ip = str(r["image_path"]).strip().replace("\\","/")
        raw = str(r[col]).strip().lower()
        if raw not in CLS_FOLDERS:
            print(bold(f"[WARN] Unknown label '{r[col]}' for {ip}; skipping in classification."))
            continue
        m[ip] = CLS_FOLDERS[raw]
    return m

def convert_split(work_dir: Path, out_dir: Path, dataset: str,
                  split_name: str, split_dirname: str,
                  csv_path: Path, labels_map: dict,
                  manifests, counters, skips):

    df = read_csv(csv_path)
    if df is None or df.empty:
        print(bold(f"[WARN] Empty CSV: {csv_path}"))
        return

    # Expected columns
    if not {"image_path","mask_path"}.issubset(set(df.columns)):
        raise ValueError(f"{csv_path} must contain image_path,mask_path")

    # USFM Seg targets
    seg_img_dir = ensure_dir(out_dir / "Seg" / dataset / split_dirname / "image")
    seg_msk_dir = ensure_dir(out_dir / "Seg" / dataset / split_dirname / "mask")
    # USFM Cls targets
    cls_split_dir = out_dir / "Cls" / dataset / split_dirname
    for sub in ("n","vh","rd"):
        ensure_dir(cls_split_dir / sub)

    seg_manifest_rows = []
    cls_manifest_rows = []

    for i, r in df.iterrows():
        img_rel = str(r["image_path"]).strip().replace("\\","/")
        msk_rel = str(r["mask_path"]).strip().replace("\\","/")
        img_abs = normalize_rel(work_dir, img_rel)
        msk_abs = normalize_rel(work_dir, msk_rel)

        if not img_abs.exists():
            msg = f"[{split_name}] missing image: {img_abs}"
            print(bold("[SKIP] "+msg)); skips.append(msg); continue
        if not msk_abs.exists():
            msg = f"[{split_name}] missing mask: {msk_abs}"
            print(bold("[SKIP] "+msg)); skips.append(msg); continue

        dest_name = build_dest_name(img_rel)

        # ---- Segmentation copy ----
        img_dst = seg_img_dir / dest_name
        msk_dst = seg_msk_dir / dest_name  # keep same name/suffix as image
        img_final = copy_with_dedupe(img_abs, img_dst)
        msk_final = copy_with_dedupe(msk_abs, msk_dst)

        seg_manifest_rows.append({
            "split": split_name,
            "src_image": str(img_abs),
            "src_mask": str(msk_abs),
            "dst_image": str(seg_img_dir / img_final),
            "dst_mask": str(seg_msk_dir / msk_final),
        })
        counters["seg"][split_name] += 1

        # ---- Classification copy (optional if label exists) ----
        cls_key = img_rel  # labels.csv uses rel paths like images/PatientX/Scan.png
        cls_folder = labels_map.get(cls_key)
        if cls_folder is None:
            # try also with normalized rel (in case of path sanitize differences)
            cls_folder = labels_map.get(img_rel.replace(" ", "%20"))  # unlikely, but harmless
        if cls_folder is None:
            print(bold(f"[WARN] No diagnosis label for {img_rel}; skipping in Cls."))
        else:
            cls_dir = cls_split_dir / cls_folder
            cls_dst = cls_dir / dest_name
            cls_final = copy_with_dedupe(img_abs, cls_dst)
            cls_manifest_rows.append({
                "split": split_name,
                "label": cls_folder,
                "src_image": str(img_abs),
                "dst_image": str(cls_dir / cls_final),
            })
            counters["cls"][split_name, cls_folder] += 1

    # write manifests
    man_dir = ensure_dir(out_dir / "manifests")
    seg_csv = man_dir / f"seg_{split_name}.csv"
    cls_csv = man_dir / f"cls_{split_name}.csv"
    with seg_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split","src_image","src_mask","dst_image","dst_mask"])
        w.writeheader(); w.writerows(seg_manifest_rows)
    with cls_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split","label","src_image","dst_image"])
        w.writeheader(); w.writerows(cls_manifest_rows)

    manifests["seg"][split_name] = str(seg_csv)
    manifests["cls"][split_name] = str(cls_csv)

def main():
    ap = argparse.ArgumentParser(description="Convert current dataset → USFM layout (Seg + Cls).")
    ap.add_argument("--work_dir", required=True, help="Your current work_dir (has images/, masks/, metadata/)")
    ap.add_argument("--dataset", default="retina_usg", help="USFM dataset name")
    ap.add_argument("--out_dir", default=None, help="Where to write USFM datasets (default: <work_dir>/USFM_datasets)")
    ap.add_argument("--labels_csv", default=None, help="Override labels CSV (default: work_dir/metadata/labels.csv)")
    ap.add_argument("--strict", action="store_true", help="Fail on missing files/labels (default: skip with warnings)")
    args = ap.parse_args()

    work_dir = Path(args.work_dir).resolve()
    out_dir  = Path(args.out_dir or (work_dir / "USFM_datasets")).resolve()
    meta_dir = work_dir / "metadata"

    csv_by_split = {
        "train": meta_dir / "train.csv",
        "val":   meta_dir / "val.csv",
        "test":  meta_dir / "test.csv",
    }
    for k, p in csv_by_split.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing split CSV: {p}")

    labels_csv = Path(args.labels_csv) if args.labels_csv else (meta_dir / "labels.csv")
    if not labels_csv.exists():
        print(bold(f"[WARN] labels.csv not found at {labels_csv}. Classification export will be sparse."))
    labels_map = load_labels_table(labels_csv, work_dir) if labels_csv.exists() else {}

    # Prepare containers
    manifests = {"seg": {}, "cls": {}}
    counters  = {"seg": Counter(), "cls": Counter()}
    skips = []

    # Do conversion per split
    for split_name, split_dirname in SPLITS:
        convert_split(work_dir, out_dir, args.dataset,
                      split_name, split_dirname,
                      csv_by_split[split_name],
                      labels_map, manifests, counters, skips)

    # Optional alias (some repos use 'trainning_set' typo) → create a symlink if possible
    try:
        for task in ("Seg","Cls"):
            base = out_dir / task / args.dataset
            src  = base / "training_set"
            dst  = base / "trainning_set"
            if src.exists() and not dst.exists():
                dst.symlink_to(src, target_is_directory=True)
    except Exception:
        pass  # non-fatal on macOS if symlink perms blocked

    # Summaries
    summary = {
        "out_dir": str(out_dir),
        "manifests": manifests,
        "counts": {
            "seg": {k: int(v) for k, v in counters["seg"].items()},
            "cls": {f"{k[0]}:{k[1]}": int(v) for k, v in counters["cls"].items()},
        },
        "skipped": len(skips),
    }
    ensure_dir(out_dir / "manifests")
    (out_dir / "manifests" / "summary.json").write_text(json.dumps(summary, indent=2))
    if skips:
        with (out_dir / "manifests" / "skipped.txt").open("w") as f:
            f.write("\n".join(skips))

    # Pretty print
    print("\n=== Done: USFM export ===")
    print("Seg counts:", dict(counters["seg"]))
    print("Cls counts:", dict(counters["cls"]))
    if skips:
        print(bold(f"Skipped items: {len(skips)} → see {out_dir}/manifests/skipped.txt"))
    print(f"Manifests in: {out_dir}/manifests")
    print("You can now copy the whole folder to Google Drive and point USFM configs to it.")

if __name__ == "__main__":
    sys.exit(main())
