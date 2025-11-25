# # tools/make_patient_splits.py
# import argparse, re
# from pathlib import Path
# import pandas as pd
#
# """
# propose a patient-wise split
# prior to this run  python -m tools.scan_sizes ....
#
# python -m tools.make_patient_splits  --sizes_csv work_dir/sizes.csv  --out_dir work_dir/metadata
# """
# def get_patient_id(p: str) -> str:
#     m = re.search(r"(Patient\d+)", p)
#     if not m:
#         raise ValueError(f"Cannot parse patient id from path: {p}")
#     return m.group(1)
#
# def summarize(df: pd.DataFrame, name: str):
#     tot = len(df)
#     def pct(k):
#         v = int(df[k].sum()) if k in df.columns else 0
#         return v, (100.0 * v / max(1, tot))
#     v_vh, p_vh = pct("has_vitreous_humor")
#     v_on, p_on = pct("has_optic_nerve")
#     v_rt, p_rt = pct("has_retina")
#     v_ch, p_ch = pct("has_choroid")
#     print(f"\n[{name}] N={tot}")
#     print(f"  VH present       : {v_vh:>3d} ({p_vh:4.1f}%)")
#     print(f"  Optic nerve pres.: {v_on:>3d} ({p_on:4.1f}%)")
#     print(f"  Retina present   : {v_rt:>3d} ({p_rt:4.1f}%)")
#     print(f"  Choroid present  : {v_ch:>3d} ({p_ch:4.1f}%)")
#
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--sizes_csv", required=True)
#     # ap.add_argument("--out_dir", required=True)
#     ap.add_argument("--test_patients", nargs="*", default=[])
#     ap.add_argument("--val_patients", nargs="*", default=[])
#     args = ap.parse_args()
#
#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     df = pd.read_csv(args.sizes_csv)
#     # add patient_id column (once)
#     if "patient_id" not in df.columns:
#         df["patient_id"] = df["image_path"].apply(get_patient_id)
#
#     # if user didn’t pass lists, use recommended default for your dataset
#     test_pat = args.test_patients or ["Patient3", "Patient8"]
#     val_pat  = args.val_patients  or ["Patient5"]
#
#     # everything else goes to train
#     pats_all = set(df["patient_id"].unique())
#     pats_held = set(test_pat) | set(val_pat)
#     train_pat = sorted(pats_all - pats_held)
#
#     # guardrails
#     assert not (set(test_pat) & set(val_pat)), "Overlap between test and val patients."
#     assert set(test_pat).issubset(pats_all), "Unknown patient in test list."
#     assert set(val_pat).issubset(pats_all),  "Unknown patient in val list."
#
#     # build splits
#     df_train = df[df["patient_id"].isin(train_pat)].copy()
#     df_val   = df[df["patient_id"].isin(val_pat)].copy()
#     df_test  = df[df["patient_id"].isin(test_pat)].copy()
#
#     # write minimal CSVs used by your pipeline (image_path, mask_path)
#     meta_dir = out_dir
#     meta_dir.mkdir(parents=True, exist_ok=True)
#     df_train[["image_path","mask_path"]].to_csv(meta_dir / "train.csv", index=False)
#     df_val  [["image_path","mask_path"]].to_csv(meta_dir / "val.csv",   index=False)
#     df_test [["image_path","mask_path"]].to_csv(meta_dir / "test.csv",  index=False)
#
#     # quick console summaries
#     summarize(df_train, "TRAIN (" + ", ".join(train_pat) + ")")
#     summarize(df_val,   "VAL   (" + ", ".join(val_pat)  + ")")
#     summarize(df_test,  "TEST  (" + ", ".join(test_pat) + ")")
#
#     # also save a manifest of split membership
#     pd.DataFrame({
#         "patient": sorted(pats_all),
#         "split": [ "test" if p in test_pat else ("val" if p in val_pat else "train")
#                    for p in sorted(pats_all) ]
#     }).to_csv(out_dir / "patient_split_manifest.csv", index=False)
#
# if __name__ == "__main__":
#     main()