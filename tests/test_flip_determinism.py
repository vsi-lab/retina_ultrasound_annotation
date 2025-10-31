# tests/test_flip_determinism.py
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from training.augments import build_train_augs
from training.dataset import SegCSV

CFG_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'config_usg.yaml'

def test_hflip_always(tmp_path):
    cfg = yaml.safe_load(open(CFG_PATH, 'r'))
    cfg['aug']['hflip'] = 1.0
    H,W = 64,64
    img = np.zeros((H,W), np.uint8); img[:, :10] = 255  # bright strip on the left
    msk = np.zeros((H,W), np.uint8); msk[:, :10] = 1
    ip = tmp_path/'img.png'; mp = tmp_path/'msk.png'
    cv2.imwrite(str(ip), img); cv2.imwrite(str(mp), msk)
    csvp = tmp_path/'toy.csv'; pd.DataFrame([{'image_path': str(ip), 'mask_path': str(mp)}]).to_csv(csvp, index=False)

    ds = SegCSV(csvp, cfg, augment=build_train_augs(cfg), is_train=True)
    x, y = ds[0]
    # bright strip should now be on the right
    left_mean  = float(x[0, :, :W//4].mean())
    right_mean = float(x[0, :, -W//4:].mean())
    assert right_mean > left_mean, "hflip=1.0 did not flip image"

def test_hflip_never(tmp_path):
    cfg = yaml.safe_load(open(CFG_PATH, 'r'))
    cfg['aug']['hflip'] = 0.0
    H,W = 64,64
    img = np.zeros((H,W), np.uint8); img[:, :10] = 255
    msk = np.zeros((H,W), np.uint8); msk[:, :10] = 1
    ip = tmp_path/'img.png'; mp = tmp_path/'msk.png'
    cv2.imwrite(str(ip), img); cv2.imwrite(str(mp), msk)
    csvp = tmp_path/'toy.csv'; pd.DataFrame([{'image_path': str(ip), 'mask_path': str(mp)}]).to_csv(csvp, index=False)

    from training.augments import build_train_augs
    ds = SegCSV(csvp, cfg, augment=build_train_augs(cfg), is_train=True)
    x, y = ds[0]
    left_mean  = float(x[0, :, :W//4].mean())
    right_mean = float(x[0, :, -W//4:].mean())
    assert left_mean > right_mean, "hflip=0.0 unexpectedly flipped image"
