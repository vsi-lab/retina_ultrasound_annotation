# training/feature_clf.py
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from models.model_factory import build_seg_model_from_cfg
from training.augments import build_val_augs
from training.dataset import SegCSV
from utils import environment
from features.region_features import compute_region_features, derive_labels_from_gt
from utils.paths import resolve_under_root_cfg
from utils.vis_usg import save_preview_panel
"""
All-in-one (extract → train → test):
python -m training.feature_clf --mode all --config configs/config_usg.yaml --ckpt  work_dir/runs/seg_transunet/best.ckpt --train_csv work_dir/metadata/train.csv --val_csv   work_dir/metadata/val.csv --test_csv  work_dir/metadata/test.csv --out_dir work_dir/runs/cls_rd --task rd_vh_normal --models lr,rf
  
  
Just extract features (any split):
python -m training.feature_clf --mode extract --config configs/config_usg.yaml --ckpt work_dir/runs/seg_transunet/best.ckpt --train_csv work_dir/metadata/train.csv --val_csv   work_dir/metadata/val.csv --test_csv  work_dir/metadata/test.csv --out_dir  work_dir/features --save_panels
  
  
Train+eval later from saved features:
python -m training.feature_clf \
  --mode train \
  --out_dir work_dir/runs/cls_rd \
  --train_feats work_dir/features/features_train.parquet \
  --val_feats   work_dir/features/features_val.parquet \
  --models lr,rf

python -m training.feature_clf \
  --mode eval \
  --out_dir work_dir/runs/cls_rd \
  --test_feats work_dir/features/features_test.parquet    
  
  
3-class classification (Normal / VH-only / RD):
python -m training.feature_clf \
  --mode all \
  --config configs/config_usg.yaml \
  --ckpt   work_dir/runs/seg_transunet/best.ckpt \
  --train_csv work_dir/metadata/train.csv \
  --val_csv   work_dir/metadata/val.csv \
  --test_csv  work_dir/metadata/test.csv \
  --out_dir   work_dir/runs/cls_rd_vh \
  --task rd_vh_normal \
  --models lr,rf
"""
# --- optional sklearn
SK_OK = False
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
SK_OK = True

from pathlib import Path

def _normalize_rel_path(p: str, work_dir: Path) -> str:
    """Return path relative to work_dir if possible, with forward slashes."""
    p = str(p).replace("\\", "/")
    try:
        rp = Path(p).resolve()
        wd = work_dir.resolve()
        try:
            return str(rp.relative_to(wd)).replace("\\", "/")
        except Exception:
            return str(rp).replace("\\", "/")
    except Exception:
        return p

def _load_labels_table(labels_csv: str, work_dir: Path, label_col: str = "diagnosis") -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    if "image_path" not in df.columns:
        raise ValueError("labels_csv must contain 'image_path'")
    if label_col not in df.columns:
        raise ValueError(f"labels_csv must contain '{label_col}'")

    # normalize
    lab_str = df[label_col].astype(str).str.strip().str.lower()
    ymap = {"n": 0, "rd": 1, "vh": 2}
    y = lab_str.map(ymap)
    if y.isna().any():
        bad = sorted(df.loc[y.isna(), label_col].unique().tolist())
        raise ValueError(f"Unrecognized labels in '{label_col}': {bad}")

    df = df.assign(y=y, y_name=lab_str)
    df["image_path"] = df["image_path"].apply(lambda s: _normalize_rel_path(s, work_dir))
    return df[["image_path", "y", "y_name"]]

def _rel_from_workdir(cfg: dict, p: str) -> str:
    """Return path relative to cfg['data']['work_dir']; else from images/ or masks/ anchor."""
    root = Path(cfg["data"]["work_dir"]).resolve()
    pp = Path(p)
    try:
        return str(pp.resolve().relative_to(root)).replace("\\", "/")
    except Exception:
        parts = pp.resolve().parts  # tuple of path components
        for anchor in ("images", "masks"):
            if anchor in parts:
                i = parts.index(anchor)
                return "/".join(parts[i:])  # e.g. images/PatientX/Subject 1.1.png
        return pp.name  # last resort


def _merge_labels_into_feats(df_feats: pd.DataFrame,
                             df_labels: pd.DataFrame,
                             work_dir: Path,
                             split_name: str) -> pd.DataFrame:
    if "image_path" not in df_feats.columns:
        raise ValueError(f"[{split_name}] features missing 'image_path'; ensure extraction stored it.")

    df_feats = df_feats.copy()
    df_feats["image_path"] = df_feats["image_path"].apply(lambda s: _normalize_rel_path(s, work_dir))
    out = df_feats.merge(df_labels, on="image_path", how="left", suffixes=("", "_lab"))

    if out["y"].isna().any():
        missing = out.loc[out["y"].isna(), "image_path"].tolist()
        raise ValueError(f"[{split_name}] {len(missing)} images missing diagnosis in labels_csv. First few: {missing[:5]}")

    out["y"] = out["y"].astype(int)

    # ensure y_name matches supervised y (ignore any rule-based yname_rule)
    if "y_na    me" not in out.columns or out["y_name"].isna().any():
        # fill from numeric map
        inv = {0: "n", 1: "rd", 2: "vh"}
        out["y_name"] = out["y"].map(inv)
    else:
        # overwrite to be safe
        inv = {0: "n", 1: "rd", 2: "vh"}
        out["y_name"] = out["y"].map(inv)

    # print(f"[debug:{split_name}] label counts:", out["y_name"].value_counts(dropna=False).to_dict())
    # print(out[["image_path", "y", "y_name"]].head(5).to_string(index=False))

    return out

def _read_df(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(path)

def _write_df(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

@torch.no_grad()
def _extract_one_split(cfg: Dict, ckpt: str, csv_path: str, out_path: Path, save_panels: bool=False) -> Path:
    device = environment.device()
    model  = build_seg_model_from_cfg(cfg, device)
    state  = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    ds = SegCSV(csv_path, cfg, augment=build_val_augs(cfg), is_train=False)
    bs = int(cfg.get("train",{}).get("batch_size", 1))
    ld = DataLoader(ds, batch_size=bs, shuffle=False,
                    num_workers=int(cfg.get("train",{}).get("num_workers",2)),
                    pin_memory=not torch.backends.mps.is_available())

    num_classes = int(cfg["data"]["num_classes"])
    labels_map  = cfg["data"]["labels"]
    class_names = cfg["data"].get("class_names", [f"class_{i}" for i in range(num_classes)])

    rows: List[Dict] = []
    panels_dir = out_path.parent / (out_path.stem + "_previews")
    if save_panels:
        panels_dir.mkdir(parents=True, exist_ok=True)

    gidx = 0
    for batch in ld:
        # dataset might return (img, msk, path) or (img, msk)
        if len(batch) == 3:
            img, msk, _ = batch
        else:
            img, msk = batch
        img, msk = img.to(device), msk.to(device)

        logits    = model(img)                           # [B,C,H,W]
        preds_ids = torch.argmax(logits, dim=1).cpu()   # [B,H,W]

        B = img.shape[0]
        for j in range(B):
            # features
            feats = compute_region_features(
                img[j].detach().cpu().numpy(),
                preds_ids[j].numpy().astype(np.int64),
                labels_map=labels_map,
                class_names=class_names,
                ignore_background_id=0
            )

            # labels from GT (rule-based): keep under separate names
            y_rule, yname_rule = derive_labels_from_gt(
                msk[j].cpu().numpy(), labels_map, task=cfg.get("feature_clf", {}).get("task", "rd_binary")
            )
            feats["y_rule"] = int(y_rule)
            feats["yname_rule"] = yname_rule

            # always carry image_path for later label merge (normalized relative to work_dir)
            try:
                raw_ip = ds.df.iloc[gidx]["image_path"]
                # get absolute path using your resolver (handles already-relative inputs too)
                ip_abs = resolve_under_root_cfg(cfg, str(raw_ip))
                # store RELATIVE path like: images/PatientX/Subject 1.1.png
                feats["image_path"] = _rel_from_workdir(cfg, str(ip_abs))
            except Exception:
                pass
            rows.append(feats)

            if save_panels:
                try:
                    ip = resolve_under_root_cfg(cfg, str(ds.df.iloc[gidx]["image_path"]))
                    save_preview_panel(
                        panels_dir / f"panel_{gidx:06d}.png",
                        img[j].detach().cpu(),
                        msk[j].detach().cpu(),
                        logits[j].detach().cpu(),
                        cfg,
                        meta_json=None,
                        raw_img_path=str(ip)
                    )
                except Exception:
                    pass
            # if gidx == 0:
            #     print("[dbg] work_dir:", Path(cfg["data"]["work_dir"].format(**cfg)).resolve())
            #     print("[dbg] raw_ip:", raw_ip)
            #     print("[dbg] ip_abs:", ip_abs)
            #     print("[dbg] feats.image_path:", feats["image_path"])
            gidx += 1

    df = pd.DataFrame(rows)
    _write_df(df, out_path)
    print(f"[feature_clf] wrote {out_path} shape={df.shape}")
    return out_path

def _make_Xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    y = df["y"].astype(int).values
    X = df.drop(columns=[c for c in ["y", "y_name", "image_path"] if c in df.columns], errors="ignore")
    # keep numeric only
    X = X.select_dtypes(include=[np.number]).copy()
    cols = list(X.columns)
    return X.values.astype(np.float32), y, cols

def _build_models(which: str, include_knn: bool=False):
    pipe_lr = Pipeline([("sc", StandardScaler(with_mean=False)),  # some sparse-like cols
                        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    rf = RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced")
    models = [("lr", pipe_lr), ("rf", rf)]
    if include_knn:
        from sklearn.neighbors import KNeighborsClassifier
        models.append(("knn", Pipeline([("sc", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))])))
    # limit to top-two choices by default
    if which:
        keep = [w.strip() for w in which.split(",")]
        models = [m for m in models if m[0] in keep]
    return models

def _eval_classification(y_true, y_pred, labels_names=None):
    import numpy as np
    from sklearn.metrics import classification_report, f1_score

    uniq = np.unique(y_true)
    uniq_list = sorted(uniq.tolist())

    # Only use binary average when labels are exactly {0,1}
    if len(uniq_list) == 2 and set(uniq_list) == {0, 1}:
        avg = "binary"
    else:
        avg = "macro"  # safe for {0,2}, 3-class, etc.

    # Make reports stable across subsets: pass explicit labels
    rep = classification_report(
        y_true, y_pred,
        labels=uniq_list,
        target_names=(labels_names if labels_names and len(labels_names) == len(uniq_list) else None),
        digits=4, zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred,
        labels=uniq_list, average=avg, zero_division=0
    )
    return rep, f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["extract","train","eval","predict","all"],
                    help="Run part of the pipeline or everything.")
    # seg + extract
    ap.add_argument("--config", help="YAML config")
    ap.add_argument("--ckpt", help="segmentation checkpoint")
    ap.add_argument("--train_csv")
    ap.add_argument("--val_csv")
    ap.add_argument("--test_csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--save_panels", action="store_true")
    # clf options
    ap.add_argument("--task", default=None, choices=["rd_binary","rd_vh_normal"],
                    help="Overrides cfg.feature_clf.task if given.")
    ap.add_argument("--models", default="lr,rf", help="subset of lr,rf,knn")
    ap.add_argument("--include_knn", action="store_true")
    # direct files (when not doing extract)
    ap.add_argument("--train_feats")
    ap.add_argument("--val_feats")
    ap.add_argument("--test_feats")
    # predict mode
    ap.add_argument("--clf_ckpt")
    ap.add_argument('--labels_csv', default=None,
                    help="CSV with per-image diagnosis; defaults to <work_dir>/metadata/labels.csv")
    ap.add_argument('--label_col', default='diagnosis',
                    help="Column name in labels CSV (default: diagnosis)")
    args = ap.parse_args()
    print(f"[feature_clf] args: {args}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # if args.labels_csv is None:
    #     args.labels_csv = Path("work_dir") / "metadata" / "labels.csv"
    cfg_for_defaults = None
    if args.config and os.path.exists(args.config):
        cfg_for_defaults = yaml.safe_load(open(args.config, "r"))

    if args.labels_csv is None:
        if cfg_for_defaults and "data" in cfg_for_defaults and "work_dir" in cfg_for_defaults["data"]:
            args.labels_csv = str(Path(cfg_for_defaults["data"]["work_dir"].format(**cfg_for_defaults)) / "metadata" / "labels.csv")
        else:
            args.labels_csv = "metadata/labels.csv"  # reasonable fallback


    # ---- EXTRACT ----
    feats_paths = {}
    if args.mode in ("extract","all"):
        assert args.config and args.ckpt, "Need --config and --ckpt for extraction"
        cfg = yaml.safe_load(open(args.config, "r"))
        if args.task:  # runtime override
            cfg.setdefault("feature_clf", {})["task"] = args.task

        if args.train_csv:
            feats_paths["train"] = _extract_one_split(cfg, args.ckpt, args.train_csv, out_dir / "features_train.parquet", args.save_panels)
        if args.val_csv:
            feats_paths["val"]   = _extract_one_split(cfg, args.ckpt, args.val_csv,   out_dir / "features_val.parquet",   args.save_panels)
        if args.test_csv:
            feats_paths["test"]  = _extract_one_split(cfg, args.ckpt, args.test_csv,  out_dir / "features_test.parquet",  args.save_panels)

    # allow re-use
    train_feats = feats_paths.get("train") or (Path(args.train_feats) if args.train_feats else None)
    val_feats   = feats_paths.get("val")   or (Path(args.val_feats) if args.val_feats else None)
    test_feats  = feats_paths.get("test")  or (Path(args.test_feats) if args.test_feats else None)

    # ---- TRAIN ----
    if args.mode in ("train","all"):
        assert SK_OK, "scikit-learn missing"
        assert train_feats and val_feats, "Need --train_feats and --val_feats (or provide CSVs for extraction)."

        df_tr = _read_df(str(train_feats))
        df_vl = _read_df(str(val_feats))

        # Optional supervised override from labels.csv for 3-class task
        if (args.task == "rd_vh_normal") and (args.labels_csv is not None):
            if args.config:
                cfg_for_wd = yaml.safe_load(open(args.config, "r"))
                work_dir = Path(cfg_for_wd["data"]["work_dir"]).resolve()
            else:
                work_dir = Path(".").resolve()
            df_lab = _load_labels_table(args.labels_csv, work_dir, args.label_col)
            df_tr = _merge_labels_into_feats(df_tr, df_lab, work_dir, "train")
            df_vl = _merge_labels_into_feats(df_vl, df_lab, work_dir, "val")

        X_tr, y_tr, feat_cols = _make_Xy(df_tr)
        X_vl = _read_df(str(val_feats)).reindex(columns=feat_cols, fill_value=0.0).select_dtypes(include=[np.number]).values.astype(np.float32)
        y_vl = df_vl["y"].astype(int).values

        # sanity: require ≥2 classes in train
        if np.unique(y_tr).size < 2:
            counts = dict(zip(*np.unique(y_tr, return_counts=True)))
            raise ValueError(f"Train split has <2 classes after labeling. Class counts: {counts}")

        # label names for pretty reports if available
        names = None
        if "y_name" in df_tr.columns:
            # keep index order based on observed classes (0..K-1 expected)
            uniq = sorted(np.unique(np.concatenate([y_tr, y_vl]).astype(int)))
            lut = {i: f for i, f in enumerate(uniq)}
            # can't guarantee dense mapping of y_name, so let sklearn format default if mismatch
            names = None

        best = None
        results = []
        for name, model in _build_models(args.models, include_knn=args.include_knn):
            print(f"[model] name: {name}")
            model.fit(X_tr, y_tr)
            yhat = model.predict(X_vl)
            print("[debug] val classes:", sorted(np.unique(y_vl).tolist()))
            rep, f1 = _eval_classification(y_vl, yhat, labels_names=names)
            print(f"\n=== {name.upper()} (val) ===\n{rep}\nmacro-F1={f1:.4f}")
            results.append({"model": name, "val_macro_f1": float(f1)})

            if best is None or f1 > best[0]:
                best = (f1, name, model)

        # persist best
        assert best is not None
        job = {"model": best[2], "features": feat_cols, "task": args.task or "auto"}
        job_path = out_dir / "model.joblib"
        joblib.dump(job, job_path)
        (out_dir / "clf_results.json").write_text(json.dumps(results, indent=2))
        print(f"\n[feature_clf] chose '{best[1]}' (val macro-F1={best[0]:.4f}) → {job_path}")

    # ---- EVAL ----
    if args.mode in ("eval","all") and test_feats is not None:
        assert SK_OK, "scikit-learn missing"
        job = joblib.load(out_dir / "model.joblib")
        df_te = _read_df(str(test_feats))
        if (args.task == "rd_vh_normal") and (args.labels_csv is not None):
            if args.config:
                cfg_for_wd = yaml.safe_load(open(args.config, "r"))
                work_dir = Path(cfg_for_wd["data"]["work_dir"]).resolve()
            else:
                work_dir = Path(".").resolve()
            df_lab = _load_labels_table(args.labels_csv, work_dir, args.label_col)
            df_te = _merge_labels_into_feats(df_te, df_lab, work_dir, "test")

        X_te = df_te.drop(columns=[c for c in ["y","y_name","image_path"] if c in df_te.columns], errors="ignore")
        X_te = X_te.reindex(columns=job["features"], fill_value=0.0).select_dtypes(include=[np.number]).values.astype(np.float32)
        y_te = df_te["y"].astype(int).values

        yhat = job["model"].predict(X_te)
        print("[debug] test classes:", sorted(np.unique(y_te).tolist()))
        rep, f1 = _eval_classification(y_te, yhat)
        print(f"\n=== TEST ===\n{rep}\nmacro-F1={f1:.4f}")
        (out_dir / "test_report.txt").write_text(rep)

    # ---- PREDICT (features → labels) ----
    if args.mode == "predict":
        assert SK_OK, "scikit-learn missing"
        assert args.clf_ckpt and test_feats, "Need --clf_ckpt and --test_feats"
        job = joblib.load(args.clf_ckpt)
        df = _read_df(str(test_feats))
        X = df.drop(columns=[c for c in ["y","y_name","image_path"] if c in df.columns], errors="ignore")
        X = X.reindex(columns=job["features"], fill_value=0.0).select_dtypes(include=[np.number]).values.astype(np.float32)
        yhat = job["model"].predict(X)
        print("Predictions:", yhat.tolist())

if __name__ == "__main__":
    main()
