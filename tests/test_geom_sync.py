# tests/test_geom_sync.py
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from training.augments import build_train_augs
from training.dataset import SegCSV

CFG_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'config_usg.yaml'

def test_image_mask_alignment(tmp_path):
    cfg = yaml.safe_load(open(CFG_PATH, 'r'))
    # make flips/affines always apply for determinism
    cfg['aug']['hflip'] = 1.0
    cfg['aug']['rotate_deg'] = [10,10]
    cfg['aug']['translate'] = [0.1,0.1]
    cfg['aug']['scale'] = [1.0,1.0]
    cfg['aug']['shear_deg'] = [0.0,0.0]

    H,W = 128,128
    img = np.zeros((H,W), np.uint8)
    cv2.rectangle(img, (20,40), (60,80), 200, -1)
    msk = np.zeros((H,W), np.uint8); msk[40:80,20:60] = cfg['data']['labels']['retina_sclera']

    ip = tmp_path/'img.png'; mp = tmp_path/'msk.png'
    cv2.imwrite(str(ip), img); cv2.imwrite(str(mp), msk)
    csvp = tmp_path/'toy.csv'
    pd.DataFrame([{'image_path': str(ip), 'mask_path': str(mp)}]).to_csv(csvp, index=False)

    ds = SegCSV(csvp, cfg, augment=build_train_augs(cfg), is_train=True)
    x, y = ds[0]  # tensors
    # bounding-box of foreground in mask should match bounding-box of bright patch in image
    yy = (y.numpy() > 0).astype(np.uint8)
    xi = (x.numpy()[0] > x.numpy()[0].mean()).astype(np.uint8)
    # compare centroids (loose tolerance)
    def centroid(a):
        ys, xs = np.where(a>0)
        return np.array([ys.mean(), xs.mean()]) if len(xs)>0 else np.array([np.nan, np.nan])
    c_img, c_msk = centroid(xi), centroid(yy)
    assert np.allclose(c_img, c_msk, atol=3), f"Out of sync: img {c_img} vs msk {c_msk}"