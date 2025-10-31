import argparse, cv2, numpy as np, os
from pathlib import Path
import yaml

"""
Migrate Supervisely Color annotations to greyscale

python -m utils.color_to_ids --config configs/config_usg.yaml --masks_color work_dir/masks_color --out work_dir/masks_ids
  
  
"""

def build_bgr_to_id(cfg):
    labels = cfg['data']['labels']
    pal = cfg['data']['color_palette']
    return {tuple(map(int, pal[name])): int(cid) for name, cid in labels.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--masks_color', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config,'r'))
    bgr2id = build_bgr_to_id(cfg)
    os.makedirs(args.out, exist_ok=True)

    for p in Path(args.masks_color).glob('*.*'):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        H,W,_ = img.shape
        ids = np.zeros((H,W), np.uint8)
        for bgr, cid in bgr2id.items():
            m = (img[:,:,0]==bgr[0]) & (img[:,:,1]==bgr[1]) & (img[:,:,2]==bgr[2])
            ids[m] = cid
        cv2.imwrite(str(Path(args.out)/p.name), ids)

if __name__ == "__main__":
    main()