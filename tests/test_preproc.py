# tests/test_preproc.py
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from training.dataset import SegCSV

CFG_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'config_usg.yaml'

def test_resize_and_normalize(tmp_path):
    cfg = yaml.safe_load(open(CFG_PATH, 'r'))
    H,W = cfg['data']['resize']
    img = np.full((200,300), 180, np.uint8)
    msk = np.zeros((200,300), np.uint8)
    ip = tmp_path/'img.png'; mp = tmp_path/'msk.png'
    cv2.imwrite(str(ip), img); cv2.imwrite(str(mp), msk)
    csvp = tmp_path/'toy.csv'
    pd.DataFrame([{'image_path': str(ip), 'mask_path': str(mp)}]).to_csv(csvp, index=False)

    ds = SegCSV(csvp, cfg, augment=None, is_train=False)
    x, y = ds[0]
    assert tuple(x.shape[-2:]) == (H, W)
    assert tuple(y.shape[-2:]) == (H, W)
    # zscore normalization should roughly center constant image near 0
    if cfg['data'].get('normalize','zscore') == 'zscore':
        m = float(x.mean()); s = float(x.std() + 1e-6)
        assert abs(m) < 1e-3, f"mean not ~0: {m}"
        assert s < 1e-2, "std too large for constant image"
