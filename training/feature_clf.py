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
from training.metrics import pixel_accuracy
from utils import environment
from utils.region_features import compute_region_features, derive_labels_from_gt
from utils.paths import resolve_under_root_cfg
from utils.vis_usg import save_preview_panel

# --- optional sklearn
try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, f1_score
    import joblib
    SK_OK = True
except Exception:
    SK_OK = False

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

            # labels from GT
            y, y_name = derive_labels_from_gt(msk[j].cpu().numpy(), labels_map, task=cfg.get("feature_clf", {}).get("task", "rd_binary"))
            feats["y"] = int(y)
            feats["y_name"] = y_name

            # add some meta if present
            if "image_path" in ds.df.columns:
                feats["image_path"] = ds.df.iloc[gidx]["image_path"]

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
    avg = "macro" if len(np.unique(y_true)) > 2 else "binary"
    rep = classification_report(y_true, y_pred, digits=4, zero_division=0, target_names=labels_names)
    f1  = f1_score(y_true, y_pred, average=avg, zero_division=0)
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
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

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

        X_tr, y_tr, feat_cols = _make_Xy(df_tr)
        X_vl = _read_df(str(val_feats)).reindex(columns=feat_cols, fill_value=0.0).select_dtypes(include=[np.number]).values.astype(np.float32)
        y_vl = df_vl["y"].astype(int).values

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
            model.fit(X_tr, y_tr)
            yhat = model.predict(X_vl)
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
        X_te = df_te.drop(columns=[c for c in ["y","y_name","image_path"] if c in df_te.columns], errors="ignore")
        X_te = X_te.reindex(columns=job["features"], fill_value=0.0).select_dtypes(include=[np.number]).values.astype(np.float32)
        y_te = df_te["y"].astype(int).values

        yhat = job["model"].predict(X_te)
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
