# classify/predict_cls.py

import argparse
import joblib
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--feats', required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.feats)
    X = df.drop(columns=[c for c in ['has_rd_gt'] if c in df.columns]).values
    clf = joblib.load(args.ckpt)
    y = clf.predict(X)
    print('Predictions:', y.tolist())

if __name__ == '__main__':
    main()
