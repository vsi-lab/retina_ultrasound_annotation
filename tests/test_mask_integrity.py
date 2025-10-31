# tests/test_mask_integrity.py
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from training.augments import build_train_augs
from training.dataset import SegCSV

CFG_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'config_usg.yaml'

def test_mask_label_integrity(tmp_path):
    cfg = yaml.safe_load(open(CFG_PATH, 'r'))
    lbls = cfg['data']['labels']
    classes = sorted(lbls.values())

    H,W = 128,128
    img = np.full((H,W), 120, np.uint8)
    msk = np.zeros((H,W), np.uint8)
    # three bands: background, retina_sclera, RD
    msk[:, 10:40]  = lbls['retina_sclera']
    msk[:, 60:100] = lbls['retinal_detachment']

    ip = tmp_path/'img.png'; mp = tmp_path/'msk.png'
    cv2.imwrite(str(ip), img); cv2.imwrite(str(mp), msk)
    csvp = tmp_path/'toy.csv'
    pd.DataFrame([{'image_path': str(ip), 'mask_path': str(mp)}]).to_csv(csvp, index=False)

    ds = SegCSV(csvp, cfg, augment=build_train_augs(cfg), is_train=True)
    x, y = ds[0]
    uniq = torch.unique(y).cpu().numpy().tolist()
    # all mask values should be a subset of known classes (no 255/3/17 leaks)
    assert set(uniq).issubset(set(classes)), f"Unexpected labels in mask: {uniq}"