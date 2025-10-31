from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from training.augments import build_train_augs
from training.dataset import SegCSV

CFG_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'config_usg.yaml'

def test_dataset_roundtrip(tmp_path):
    cfg = yaml.safe_load(open(CFG_PATH, 'r'))
    rd_id = cfg['data']['labels']['retinal_detachment']

    H, W = 128, 128
    img = np.zeros((H, W), np.uint8)
    cv2.circle(img, (W // 2, H // 2), 24, 200, -1)

    msk = np.zeros((H, W), np.uint8)
    cv2.ellipse(msk, (W // 2, H // 2), (30, 20), 0, 0, 360, rd_id, -1)

    imgp = tmp_path / 'img.png'
    mskp = tmp_path / 'msk.png'
    cv2.imwrite(str(imgp), img)
    cv2.imwrite(str(mskp), msk)

    df = pd.DataFrame([{'image_path': str(imgp), 'mask_path': str(mskp)}])
    csvp = tmp_path / 'toy.csv'
    df.to_csv(csvp, index=False)

    ds = SegCSV(csvp, cfg, augment=build_train_augs(cfg))
    x, y = ds[0]

    assert x.ndim == 3 and x.shape[0] == 1
    assert y.ndim == 2
    # Optional sanity check:
    assert (y.numpy() == rd_id).any()

