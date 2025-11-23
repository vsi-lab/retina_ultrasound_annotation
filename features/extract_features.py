# features/extract_features.py
import argparse
import numpy as np
import os
import pandas as pd
import torch
import yaml

from torch.utils.data import DataLoader

from features.region_features import rd_region_features
from models.transunet import TransUNet
from models.unet import UNet
from training.augments import build_val_augs
from training.dataset import SegCSV
from utils import environment

"""
Feature extraction from segmentation predictions for RD classification.

This module runs a trained pixel-wise segmentation model (TransUNet/UNet) over a
set of images listed in a CSV, converts the predicted mask into a compact set of
region-level features (area fraction, connected components, perimeter, etc.),
and writes a Parquet/CSV table that can be used by a downstream classifier
(e.g., Logistic Regression or Random Forest) to detect the presence of retinal
detachment (RD).

End-to-end context:
    1) Train a segmentation model with `training/train_seg.py`.
    2) Evaluate to pick a checkpoint (e.g., `best.ckpt`).
    3) Run this script on train/val/test splits to produce feature tables:
       - work_dir/features/train_feats.parquet
       - work_dir/features/val_feats.parquet
       - work_dir/features/test_feats.parquet
    4) Train a light classifier on these features with `classify/train_cls.py`.

Inputs:
    --config: Path to YAML config (model/data/aug/eval settings).
    --ckpt:   Path to trained segmentation checkpoint to load.
    --csv:    CSV with columns ['image_path','mask_path'] for the split of interest.
    --out:    Output path for features table (e.g., .parquet).

Outputs:
    A table with columns like:
        ['rd_area_frac', 'rd_num_cc', 'rd_max_cc_frac',
         'rd_perimeter_norm', 'rd_bbox_aspect', 'rd_center_y',
         'has_rd_gt']
    where 'has_rd_gt' (0/1) reflects RD presence in the *ground-truth* mask.
    
    
    
"""

def build_model(cfg, device):
    name = cfg.get('model', {}).get('name','transunet').lower()
    if name == 'unet':
        return UNet(in_ch=1, num_classes=cfg['data']['num_classes'], base=cfg['model'].get('base',32)).to(device)
    else:
        m = cfg['model']
        return TransUNet(in_ch=1,
                         num_classes=cfg['data']['num_classes'],
                         base=m.get('base',32),
                         embed_dim=m.get('embed_dim',256),
                         depth=m.get('depth',4),
                         heads=m.get('heads',8)).to(device)

@torch.no_grad()
def main():
    """
    CLI entry point.

    Steps:
        1) Parse args, load YAML config, detect device.
        2) Build model and load checkpoint weights.
        3) Construct SegCSV dataset with validation-time transforms (build_val_augs)
           so inference uses the same resize/normalize as evaluation.
        4) Iterate images in DataLoader:
             - Run model forward pass, take argmax over channels -> predicted class map.
             - Compute RD region features via `rd_region_features(pred, rd_label)`.
             - Derive `has_rd_gt` = 1 if any GT pixel equals rd_label else 0.
             - Append row to an in-memory list.
        5) Write the feature table to `--out` (Parquet preferred; CSV fallback is possible).

    Notes:
        - `rd_label` is taken from cfg['data']['labels']['retinal_detachment'] (default=2).
        - Argmax is used for stable, discrete region computations; if we want
          probability-aware features, then can add thresholds/soft features here.
        - Keep the feature schema stable across train/val/test; the classifier
          will reindex validation/test columns to match the training schema.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config,'r'))
    device = environment.device()

    model = build_model(cfg, device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    ds = SegCSV(args.csv, cfg, augment=build_val_augs(cfg), is_train=False)
    ld = DataLoader(ds, batch_size=1, shuffle=False)
    rd_label = cfg['data']['labels']['retinal_detachment']
    rows = []
    for i, (img, msk) in enumerate(ld):
        img = img.to(device)
        logits = model(img)
        pred = logits.argmax(dim=1).cpu().numpy()[0].astype(np.uint8)
        feats = rd_region_features(pred, rd_label)
        gt_has_rd = int((msk.numpy()[0] == rd_label).any())
        feats.update({"has_rd_gt": gt_has_rd})
        rows.append(feats)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Saved: {args.out} ({len(df)} rows)")

if __name__ == '__main__':
    main()
