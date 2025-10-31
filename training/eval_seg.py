
import argparse
import os
import pandas as pd
import torch
import yaml

from torch.utils.data import DataLoader
from tqdm import tqdm

from models.transunet import TransUNet
from models.unet import UNet
from training.augments import build_val_augs
from training.dataset import SegCSV
from training.metrics import dice_per_class_from_logits, pixel_accuracy
from utils import environment

"""
Evaluate a trained segmentation checkpoint on a validation or test split.

Loads the model specified in --config and --ckpt, performs inference on
the given CSV split, and reports class-wise and mean Dice coefficients,
foreground accuracy, and optional visual previews.

Outputs summary tables and optional prediction masks to the run directory.
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--eval_csv', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config,'r'))
    os.makedirs(args.out, exist_ok=True)
    device = environment.device()

    model = build_model(cfg, device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    ds = SegCSV(args.eval_csv, cfg, augment=build_val_augs(cfg), is_train=False)
    ld = DataLoader(ds, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=2)

    rows = []
    for img, msk in tqdm(ld, desc='eval', leave=False):
        img, msk = img.to(device), msk.to(device)
        logits = model(img)
        dice_c = dice_per_class_from_logits(logits, msk, cfg['data']['num_classes'])
        acc = pixel_accuracy(logits, msk)
        rows.append({
            'dice_bg': float(dice_c[0]),
            'dice_retina': float(dice_c[1]),
            'dice_rd': float(dice_c[2]),
            'acc': float(acc),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, 'metrics.csv'), index=False)
    print(df.describe())

if __name__ == '__main__':
    main()
