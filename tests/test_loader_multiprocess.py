# tests/test_loader_multiprocess.py
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from torch.utils.data import DataLoader

from training.augments import build_train_augs
from training.dataset import SegCSV

CFG_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'config_usg.yaml'

def test_multiprocess_workers(tmp_path):
    cfg = yaml.safe_load(open(CFG_PATH, 'r'))
    H,W = 64,64
    ip = tmp_path/'img.png'; mp = tmp_path/'msk.png'
    cv2.imwrite(str(ip), np.zeros((H,W), np.uint8))
    cv2.imwrite(str(mp), np.zeros((H,W), np.uint8))
    csvp = tmp_path/'toy.csv'; pd.DataFrame([{'image_path': str(ip), 'mask_path': str(mp)}]).to_csv(csvp, index=False)

    ds = SegCSV(csvp, cfg, augment=build_train_augs(cfg), is_train=True)
    # if aug pipeline contains lambda/locals, this will fail to pickle on some OSes
    ld = DataLoader(ds, batch_size=2, shuffle=False, num_workers=2)
    it = iter(ld)
    x, y = next(it)
    assert x.shape[0] <= 2 and y.shape[0] <= 2