# # classify/train_cls.py
# import argparse
# import joblib
# import numpy as np
# import os
# import pandas as pd
#
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
#
# """
# Train a lightweight RD-presence classifier on extracted region features.
#
# Reads train/val feature tables (Parquet or CSV) produced by extract_features.py,
# splits into X (features) and y (has_rd_gt label), and fits either:
#
#     • Logistic Regression (default, class_weight='balanced')
#     • Random Forest (n_estimators=300)
#
# Prints precision/recall/F1 and saves model.joblib with feature schema.
#
# Reference:
#     • Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 2011.
# """
#
# NON_FEATURE_COLS_DEFAULT = {"image_path", "mask_path"}
#
# def _read_table(path: str) -> pd.DataFrame:
#     try:
#         return pd.read_parquet(path)
#     except Exception:
#         return pd.read_csv(path)
#
# def _ensure_label(df: pd.DataFrame, args) -> tuple[pd.DataFrame, str]:
#     """Return (df_with_label, label_col). Prefer explicit label; else derive from rd_area_frac."""
#     # Explicit label?
#     if args.label and args.label in df.columns:
#         return df, args.label
#     # Common default label
#     if "has_rd_gt" in df.columns:
#         return df, "has_rd_gt"
#     # Derive from rd_area_frac if available
#     if "rd_area_frac" in df.columns:
#         thr = float(args.rd_thresh)
#         out = df.copy()
#         out["has_rd_gt"] = (out["rd_area_frac"].astype(float) >= thr).astype(int)
#         return out, "has_rd_gt"
#     raise ValueError("No label column found (looked for --label, 'has_rd_gt', or 'rd_area_frac').")
#
# def _make_feature_matrix(df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, list[str]]:
#     drop_cols = set([label_col]).union(NON_FEATURE_COLS_DEFAULT)
#     # Keep only numeric feature columns
#     X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
#     X = X.select_dtypes(include=[np.number]).copy()
#     if X.empty:
#         raise ValueError("No numeric feature columns found after dropping non-feature columns.")
#     cols = list(X.columns)
#     return X, cols
#
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument('--train', required=True)
#     ap.add_argument('--val', required=True)
#     ap.add_argument('--out', required=True)
#     ap.add_argument('--clf', default='logreg', choices=['logreg','rf'])
#
#     # Optional robustness:
#     ap.add_argument('--label', default=None, help="Name of label column (default: auto: has_rd_gt or rd_area_frac>=rd_thresh)")
#     ap.add_argument('--rd_thresh', type=float, default=0.001, help="Threshold on rd_area_frac to derive label if needed")
#     ap.add_argument('--min_per_class', type=int, default=2, help="Require at least this many samples per class in train")
#     args = ap.parse_args()
#
#     os.makedirs(args.out, exist_ok=True)
#
#     # Load
#     df_tr = _read_table(args.train)
#     df_vl = _read_table(args.val)
#
#     # Ensure labels
#     df_tr, label_col = _ensure_label(df_tr, args)
#     df_vl, _         = _ensure_label(df_vl, argparse.Namespace(label=label_col, rd_thresh=args.rd_thresh))
#
#     # Report label distribution
#     y_tr = df_tr[label_col].astype(int)
#     y_vl = df_vl[label_col].astype(int)
#     print("Train label counts:\n", y_tr.value_counts(dropna=False).to_string())
#     print("Val   label counts:\n", y_vl.value_counts(dropna=False).to_string())
#
#     # Guardrails
#     vc_tr = y_tr.value_counts()
#     if vc_tr.shape[0] < 2:
#         raise ValueError(
#             f"Training set has only one class:\n{vc_tr}\n"
#             "Fix: ensure both RD and non-RD are present in TRAIN, then re-extract features."
#         )
#     if (vc_tr.min() < args.min_per_class) or (len(y_tr) < 2):
#         raise ValueError(
#             f"Training set too small per class (min required per class: {args.min_per_class}).\n"
#             f"Counts:\n{vc_tr}"
#         )
#
#     # Features (numeric only), remember column order for inference
#     X_tr, feat_cols = _make_feature_matrix(df_tr, label_col)
#     X_vl = df_vl.reindex(columns=feat_cols, fill_value=0.0)  # align columns
#
#     # Choose classifier
#     if args.clf == 'rf':
#         clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
#     else:
#         clf = LogisticRegression(max_iter=1000, class_weight="balanced")
#
#     # Fit / Eval
#     clf.fit(X_tr.values, y_tr.values)
#     y_pred = clf.predict(X_vl.values)
#
#     print(classification_report(y_vl, y_pred, digits=4, zero_division=0))
#     print('F1:', f1_score(y_vl, y_pred, zero_division=0))
#     print('Precision:', precision_score(y_vl, y_pred, zero_division=0))
#     print('Recall:', recall_score(y_vl, y_pred, zero_division=0))
#
#     # Persist model + feature schema
#     out_path = os.path.join(args.out, 'model.joblib')
#     joblib.dump({"model": clf, "features": feat_cols, "label": label_col}, out_path)
#     print('Saved:', out_path)
#
# if __name__ == '__main__':
#     main()