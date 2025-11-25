# tools/build_ultrasound_index.py
import argparse, json, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import pandas as pd
import yaml
import glob
from collections import Counter, defaultdict

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

"""
	1.	scans work_dir/images & work_dir/masks,
	2.	reads meta.json / obj_class_to_machine_color.json (for color→class mapping),
	3.	converts color masks → id masks (if needed),
	4.	computes sizes + class-presence stats (overall & by split),
	5.	auto-selects a patient-wise split (or uses config if provided), and
	6.	writes work_dir/metadata/{labels,train,val,test}.csv + sizes.csv + presence_summary.csv + patient_split_manifest.csv.



# auto-pick a good test/val split (patient-wise), then write CSVs + stats
python -m tools.build_ultrasound_index \
  --config configs/config_usg.yaml \
  --out work_dir \
  --auto_split \
  --target_test 10 \
  --target_val 1

OR specify test / val split yourself 
python -m tools.build_ultrasound_index --config configs/config_usg.yaml --test 3,8 --val 5 --out work_dir
  
  
"""
# -----------------------
# Helpers: IO + parsing
# -----------------------

def _normalize_patient_token(t: str) -> str:
    t = t.strip()
    if not t:
        return ""
    if t.lower().startswith("patient"):
        # keep exact casing for consistency
        return "Patient" + t[len("patient"):]
    return f"Patient{t}"

def _parse_patient_list(arg: str | None, available: set[str]) -> list[str] | None:
    if not arg:
        return None
    pts = [_normalize_patient_token(tok) for tok in arg.split(",") if tok.strip()]
    missing = [p for p in pts if p not in available]
    if missing:
        raise ValueError(f"Unknown patient(s) in override: {missing}. "
                         f"Available: {sorted(available)}")
    return pts

def load_yaml(p: str) -> Dict:
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}

def read_json(p: Path) -> Optional[Dict]:
    if not p.exists():
        return None
    with open(p, "r") as f:
        return json.load(f)

def first_component_after(root: Path, path: Path, anchor: str) -> str:
    """
    Return the first folder name after anchor (e.g., 'images') relative to root.
    work_dir/images/Patient3/Subject 13.6.png -> Patient3
    """
    try:
        rel = path.relative_to(root)
    except Exception:
        return "unknown"
    parts = rel.parts
    if anchor in parts:
        idx = parts.index(anchor)
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"

def find_images(folder: Path) -> List[Path]:
    files = []
    for ext in IMG_EXTS:
        files.extend(folder.glob(f"*{ext}"))
    return sorted(files)

def match_mask(image_path: Path, mask_folder: Path) -> Optional[Path]:
    stem = image_path.stem
    for ext in IMG_EXTS:
        mp = mask_folder / f"{stem}{ext}"
        if mp.exists():
            return mp
    return None

def hex_to_rgb(h: str) -> Optional[Tuple[int, int, int]]:
    if not isinstance(h, str):
        return None
    h = h.lstrip("#")
    if len(h) != 6:
        return None
    try:
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        return (r, g, b)
    except Exception:
        return None

def iter_classes(meta_or_map) -> List[Dict]:
    """
    Supports:
      - meta.json: {"classes":[{"title":"Retina","color":"#RRGGBB"}, ...]}
      - obj_class_to_machine_color.json: dict or list forms
    """
    if meta_or_map is None:
        return []
    if isinstance(meta_or_map, dict):
        if "classes" in meta_or_map and isinstance(meta_or_map["classes"], list):
            return meta_or_map["classes"]
        return [{"title": k, **v} if isinstance(v, dict) else {"title": k}
                for k, v in meta_or_map.items()]
    if isinstance(meta_or_map, list):
        return meta_or_map
    return []

def canon(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")

# -----------------------
# Color mask -> id mask
# -----------------------
def build_color_to_id_map(
    work_dir: Path,
    cfg: Dict
) -> Dict[Tuple[int,int,int], int]:
    """
    Build mapping from BGR tuple -> class id using:
      1) obj_class_to_machine_color.json (preferred) OR meta.json
      2) cfg['data']['labels'] name->id to assign consistent ids

    Only classes present in cfg['data']['labels'] are mapped.
    Unmapped colors default to 0 (background).
    """
    meta = read_json(work_dir / "meta.json")
    objmap = read_json(work_dir / "obj_class_to_machine_color.json")
    classes = iter_classes(objmap) or iter_classes(meta)

    labels_map = (cfg.get("data", {}) or {}).get("labels", {}) or {}
    # normalize labels to canonical names -> id
    name2id = {canon(k): int(v) for k, v in labels_map.items()}

    color2id = {}
    for i, d in enumerate(classes):
        title = d.get("title") or d.get("name") or f"class_{i}"
        cname = canon(title)
        # color may be hex (#RRGGBB) or explicit
        col = d.get("machine_color") or d.get("color")
        rgb = None
        if isinstance(col, str):
            rgb = hex_to_rgb(col)
        elif isinstance(col, (list, tuple)) and len(col) == 3:
            rgb = tuple(int(x) for x in col)
        if rgb is None:
            continue
        # cv2 loads as BGR
        bgr = (rgb[2], rgb[1], rgb[0])
        if cname in name2id:
            color2id[bgr] = name2id[cname]
    return color2id

def convert_color_mask_to_id(mask_bgr: np.ndarray, color2id: Dict[Tuple[int,int,int], int]) -> np.ndarray:
    """
    Vectorized conversion of 3-channel color mask to integer id mask.
    Unknown colors -> 0.
    """
    h, w, _ = mask_bgr.shape
    # pack BGR into uint32
    packed = (mask_bgr[...,0].astype(np.uint32) << 16) | \
             (mask_bgr[...,1].astype(np.uint32) << 8)  | \
             (mask_bgr[...,2].astype(np.uint32))
    # dict of packed(BGR)->id
    lut = {}
    for (b,g,r), cid in color2id.items():
        pk = (int(b) << 16) | (int(g) << 8) | int(r)
        lut[pk] = int(cid)
    flat = packed.reshape(-1)
    out = np.fromiter((lut.get(int(v), 0) for v in flat), dtype=np.uint8, count=flat.size)
    return out.reshape(h, w)

# -----------------------
# Dataset building
# -----------------------
def build_index(work_dir: Path) -> pd.DataFrame:
    images_root = work_dir / "images"
    masks_root  = work_dir / "masks"
    rows = []
    # iterate each patient folder in images/
    for pf in sorted([p for p in images_root.iterdir() if p.is_dir()]):
        patient_id = pf.name
        mask_pf = masks_root / patient_id
        if not mask_pf.exists():
            print(f"[WARN] No mask folder for {patient_id}: {mask_pf}")
            continue
        imgs = find_images(pf)
        if not imgs:
            print(f"[WARN] No images in {pf}")
            continue
        for ip in imgs:
            mp = match_mask(ip, mask_pf)
            if mp is None:
                print(f"[WARN] No mask for image {ip.name} under {mask_pf}")
                continue
            rows.append({
                "image_path": str(ip.relative_to(work_dir)),
                "mask_path":  str(mp.relative_to(work_dir)),
                "patient_id": patient_id,
                "scan_id":    ip.stem,
            })
    return pd.DataFrame(rows)

def load_mask_id(msk_path: Path, color2id: Dict[Tuple[int,int,int], int]) -> Optional[np.ndarray]:
    m = cv2.imread(str(msk_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 2:
        return m
    if m.ndim == 3 and m.shape[2] in (3,4):
        bgr = m[..., :3]  # drop alpha if present
        return convert_color_mask_to_id(bgr, color2id)
    return None

def size_hw(img_path: Path) -> Tuple[Optional[int], Optional[int]]:
    im = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if im is None: return None, None
    return int(im.shape[0]), int(im.shape[1])

# -----------------------
# Presence & sizes
# -----------------------
def compute_sizes_and_presence(work_dir: Path, df: pd.DataFrame, cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      report_df: per-row details (sizes, presence flags)
      presence_summary: overall + by planned split (if split column exists)
    """
    # label id->name map from config (stable)
    labels_map = (cfg.get("data", {}) or {}).get("labels", {}) or {}
    id2name = {int(v): k for k, v in labels_map.items()}
    id_list = sorted(id2name.keys())

    color2id = build_color_to_id_map(work_dir, cfg)

    rows = []
    img_sizes = Counter()
    msk_sizes = Counter()
    mismatches = 0

    for _, r in df.iterrows():
        ip = work_dir / r["image_path"]
        mp = work_dir / r["mask_path"]

        ih, iw = size_hw(ip)
        m = load_mask_id(mp, color2id)
        mh, mw = (None, None) if m is None else (int(m.shape[0]), int(m.shape[1]))

        row = {
            **r.to_dict(),
            "img_H": ih, "img_W": iw,
            "msk_H": mh, "msk_W": mw,
            "img_exists": ih is not None,
            "msk_exists": mh is not None,
        }

        # class presence flags
        if m is not None:
            vals = np.unique(m).astype(int).tolist()
            present = set(vals)
            for cid in id_list:
                row[f"has_{id2name[cid]}"] = int(cid in present)
        else:
            for cid in id_list:
                row[f"has_{id2name[cid]}"] = np.nan

        rows.append(row)

        if ih is not None: img_sizes[(ih, iw)] += 1
        if mh is not None: msk_sizes[(mh, mw)] += 1
        if (ih is None) or (mh is None) or ((ih, iw) != (mh, mw)):
            mismatches += 1

    report = pd.DataFrame(rows)

    # console summary
    print("\n=== Scan Summary ===")
    print(f"Total pairs scanned: {len(report)}")
    print("\nTop image sizes (H x W):")
    for (h, w), cnt in img_sizes.most_common(10):
        print(f"  {h}x{w} : {cnt}")
    print("\nTop mask sizes (H x W):")
    for (h, w), cnt in msk_sizes.most_common(10):
        print(f"  {h}x{w} : {cnt}")
    if mismatches:
        print(f"\nMismatched / unreadable pairs: {mismatches}")
    else:
        print("\nNo size mismatches found.")

    # presence summary (overall; by split printed after splitting)
    valid_mask = (report["msk_exists"] == True)
    denom_overall = int(valid_mask.sum())

    rows_s = []
    for cid in id_list:
        cname = id2name[cid]
        col = f"has_{cname}"
        cnt = int(pd.to_numeric(report.loc[valid_mask, col], errors="coerce").fillna(0).sum())
        pct = (100.0 * cnt / max(1, denom_overall))
        rows_s.append({
            "class_id": cid, "class": cname,
            "overall_presence_count": cnt,
            "overall_presence_pct": round(pct, 2),
            "overall_images": denom_overall
        })
    summary = pd.DataFrame(rows_s).sort_values("class_id")
    print("\n=== Class Presence Summary (overall) ===")
    if not summary.empty:
        print(summary[["class_id","class","overall_presence_count","overall_presence_pct"]].to_string(index=False))
    else:
        print("No valid masks found.")
    return report, summary

# -----------------------
# Splitting (patient-wise)
# -----------------------
def has_any(df: pd.DataFrame, col: str) -> int:
    return int(pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).sum())

def auto_split_patients(
    report: pd.DataFrame, cfg: Dict,
    target_test: int = 10, target_val: int = 1
) -> Dict[str, List[str]]:
    """
    Greedy heuristic:
      - choose TEST patients to maximize coverage of rare classes
        (optic_nerve + vitreous_humor) near target_test images
      - choose one VAL patient with both ON/VH if possible
      - remainder -> TRAIN
    """
    labels_map = (cfg.get("data", {}) or {}).get("labels", {}) or {}
    id2name = {int(v): k for k, v in labels_map.items()}
    need_cols = []
    for key in ("vitreous_humor", "optic_nerve"):
        # only if present in labels map
        if any(canon(key) == canon(nm) for nm in id2name.values()):
            need_cols.append(f"has_{key}")

    # per-patient rollups
    pats = sorted(report["patient_id"].unique().tolist())
    pat_rows = []
    for p in pats:
        dfp = report[report["patient_id"] == p]
        entry = {"patient": p, "n": len(dfp)}
        for col in need_cols:
            entry[col] = has_any(dfp, col)
        pat_rows.append(entry)
    ptab = pd.DataFrame(pat_rows).sort_values("patient")

    # score patients by rare-class hits normalized by size
    def score_row(r):
        rare = sum(int(r[c] > 0) for c in need_cols)
        return 4*rare + min(r["n"], target_test) / max(1, target_test)
    ptab["score"] = ptab.apply(score_row, axis=1)
    ptab = ptab.sort_values(["score","n"], ascending=[False, False])

    test, total = [], 0
    for _, rr in ptab.iterrows():
        if total >= target_test:
            break
        test.append(rr["patient"])
        total += int(rr["n"])

    # choose val: best remaining with ON or VH if possible
    remaining = [p for p in pats if p not in test]
    cand = []
    for p in remaining:
        rr = ptab[ptab["patient"] == p].iloc[0].to_dict()
        rr["patient"] = p
        rr["rare_hits"] = sum(int(rr.get(c,0)) > 0 for c in need_cols)
        cand.append(rr)
    cand = sorted(cand, key=lambda d: (d["rare_hits"], d["n"]), reverse=True)
    val = [cand[0]["patient"]] if cand else []
    train = [p for p in pats if p not in set(test+val)]
    return {"train": train, "val": val, "test": test}

def splits_from_cfg_or_auto(report: pd.DataFrame, cfg: Dict, auto: bool, target_test: int, target_val: int) -> Dict[str, List[str]]:
    if auto:
        return auto_split_patients(report, cfg, target_test, target_val)
    splits = (cfg.get("data", {}) or {}).get("splits", {}) or {}
    out = {k: list(v) for k, v in splits.items()}
    for k in ("train","val","test"):
        out.setdefault(k, [])
    return out

def apply_splits_to_rows(report: pd.DataFrame, split_pats: Dict[str, List[str]]) -> pd.DataFrame:
    def lab(p):
        if p in split_pats["test"]: return "test"
        if p in split_pats["val"]:  return "val"
        if p in split_pats["train"]:return "train"
        return "train"
    out = report.copy()
    out["split"] = out["patient_id"].map(lab)
    return out

# -----------------------
# Writing outputs
# -----------------------
def write_all_outputs(work_dir: Path, df_index: pd.DataFrame, report: pd.DataFrame, summary_overall: pd.DataFrame):
    meta_dir = work_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # labels = minimal index (image_path/mask_path/patient_id/scan_id)
    labels_csv = meta_dir / "labels.csv"
    df_index.to_csv(labels_csv, index=False)
    print("[OK] wrote", labels_csv)

    # split CSVs
    for name in ["train","val","test"]:
        sdf = report[report["split"] == name][["image_path","mask_path"]]
        p = meta_dir / f"{name}.csv"
        sdf.to_csv(p, index=False)
        print("[OK] wrote", p)

    # sizes + presence details
    sizes_csv = meta_dir / "sizes.csv"
    report.to_csv(sizes_csv, index=False)
    print("[OK] wrote", sizes_csv)

    # presence summary overall + by split
    # add split-wise presence
    pres_rows = []
    # start with overall that we already computed
    pres_rows.append(summary_overall.assign(scope="overall"))
    # by split
    for split in ["train","val","test"]:
        sdf = report[report["split"] == split]
        if sdf.empty:
            continue
        # recompute per split
        cols = [c for c in report.columns if c.startswith("has_")]
        class_names = [c[len("has_"):] for c in cols]
        cnts = {nm: has_any(sdf, f"has_{nm}") for nm in class_names}
        denom = len(sdf)
        rows = []
        # need id mapping from report columns -> unknown; we only have names
        # so store name only here
        for nm in class_names:
            rows.append({
                "class": nm,
                "overall_presence_count": cnts[nm],
                "overall_presence_pct": round(100.0 * cnts[nm] / max(1, denom), 2),
            })
        pres_rows.append(pd.DataFrame(rows).assign(scope=split))
    presence_summary = pd.concat(pres_rows, ignore_index=True, sort=False)
    presence_csv = meta_dir / "presence_summary.csv"
    presence_summary.to_csv(presence_csv, index=False)
    print("[OK] wrote", presence_csv)

    # patient manifest
    pats = sorted(report["patient_id"].unique().tolist())
    manifest = pd.DataFrame({
        "patient": pats,
        "split": [ report[report["patient_id"]==p]["split"].iloc[0] for p in pats ]
    })
    man_csv = meta_dir / "patient_split_manifest.csv"
    manifest.to_csv(man_csv, index=False)
    print("[OK] wrote", man_csv)

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Build index, compute stats, auto-split patients, and write CSVs.")
    ap.add_argument("--config", required=True, help="YAML config (must include data.labels; optional data.splits)")
    ap.add_argument("--out", default=None, help="Root folder with images/, masks/, meta.json")
    ap.add_argument("--auto_split", action="store_true", help="Auto pick patient-wise splits")
    ap.add_argument("--target_test", type=int, default=10, help="Target #images for test (auto mode)")
    ap.add_argument("--target_val", type=int, default=1,  help="Target #patients for val (auto mode, typically 1)")
    ap.add_argument("--test", default=None,
                    help="Comma-separated patient ids, e.g. '3,8' or 'Patient3,Patient8'. Overrides config.")
    ap.add_argument("--val", default=None,
                    help="Comma-separated patient ids, e.g. '5' or 'Patient5'. Overrides config.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    work_dir = Path(args.out or cfg.get("data", {}).get("work_dir", ".")).resolve()
    print(f"[INFO] work_dir: {work_dir}")


    # 1) Build index from folder structure
    df_index = build_index(work_dir)
    if df_index.empty:
        raise RuntimeError("No image-mask pairs found. Check folder structure / naming.")
    print(f"[INFO] indexed {len(df_index)} pairs across {df_index['patient_id'].nunique()} patients.")

    # ... (args/cfg/work_dir/df_index built above)

    all_patients = set(df_index["patient_id"].unique())
    test_override = _parse_patient_list(args.test, all_patients)
    val_override = _parse_patient_list(args.val, all_patients)

    # 2) Compute sizes + class presence
    report, summary_overall = compute_sizes_and_presence(work_dir, df_index, cfg)

    # ensure patient_id is present on report for splitting
    if "patient_id" not in report.columns:
        report["patient_id"] = report["image_path"].str.extract(r'(Patient\d+)', expand=False)

    # 3) Derive splits (config/auto or CLI override) → ALWAYS end with split_pats
    split_pats = splits_from_cfg_or_auto(report, cfg, args.auto_split, args.target_test, args.target_val)

    if test_override or val_override:
        cfg_splits = dict(cfg.get("splits", {}))  # shallow copy
        if test_override is not None:
            cfg_splits["test"] = test_override
        if val_override is not None:
            cfg_splits["val"] = val_override

        test_pat = set(cfg_splits.get("test", []))
        val_pat = set(cfg_splits.get("val", []))
        overlap = test_pat & val_pat
        if overlap:
            raise ValueError(f"Patient(s) in both TEST and VAL: {sorted(overlap)}")

        train_pat = sorted(all_patients - test_pat - val_pat)
        # override the auto/config choice
        split_pats = {
            "train": train_pat,
            "val": sorted(val_pat),
            "test": sorted(test_pat),
        }
        print(f"[OVERRIDE] train patients: {split_pats['train']}")
        print(f"[OVERRIDE] val   patients: {split_pats['val']}")
        print(f"[OVERRIDE] test  patients: {split_pats['test']}")

    print("\n=== Patient-wise split ===")
    for k in ("train", "val", "test"):
        print(f"{k.upper()}: {sorted(split_pats.get(k, []))}")

    # 🔑 Single, consistent path: tag rows in `report`
    report = apply_splits_to_rows(report, split_pats)

    # 4) Class presence by split (now works for both branches)
    print("\n=== Class Presence by Split ===")
    for name in ["train", "val", "test"]:
        sdf = report[report["split"] == name]
        if sdf.empty:
            print(f"{name.upper()}: (empty)")
            continue
        n = len(sdf)
        cols = [c for c in report.columns if c.startswith("has_")]
        line = [f"{name.upper()} N={n}"] + [
            f"{c[4:]}={int(pd.to_numeric(sdf[c], errors='coerce').fillna(0).sum())}"
            for c in cols
        ]
        print("  " + " | ".join(line))

    # 5) Write outputs (unchanged)
    write_all_outputs(work_dir, df_index, report, summary_overall)
    print("\n[DONE] End-to-end build complete.")

if __name__ == "__main__":
    main()