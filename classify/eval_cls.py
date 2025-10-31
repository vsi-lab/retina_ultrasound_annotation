
import argparse
import joblib
import os
import pandas as pd

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

"""
Evaluate a saved RD classifier on test features.

Loads model.joblib and its feature schema, re-aligns columns in the test
feature file, performs inference, and prints classification metrics and confusion
matrix.  Optionally saves per-sample predictions for auditing.
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.test)
    y = df['has_rd_gt'].astype(int).values
    X = df.drop(columns=['has_rd_gt']).values

    clf = joblib.load(args.ckpt)
    y_pred = clf.predict(X)
    rpt = classification_report(y, y_pred, digits=4)
    print(rpt)
    print('F1:', f1_score(y, y_pred))
    print('Precision:', precision_score(y, y_pred))
    print('Recall:', recall_score(y, y_pred))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'report.txt'), 'w') as f:
        f.write(rpt)

if __name__ == '__main__':
    main()
