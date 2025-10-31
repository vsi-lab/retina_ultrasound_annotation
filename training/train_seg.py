
import argparse
import os
import torch
import warnings
import yaml

from torch.utils.data import DataLoader
from tqdm import tqdm

from models.transunet import TransUNet
from models.unet import UNet
from training.augments import build_train_augs, build_val_augs
from training.dataset import SegCSV
from training.losses import composite_loss
from training.metrics import dice_per_class_from_logits, pixel_accuracy
from utils import environment

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
warnings.filterwarnings("ignore", message="enable_nested_tensor")


"""
Train a segmentation model (TransUNet or UNet) for retinal-ultrasound images.

This script launches supervised training using configuration parameters
from configs/config_usg.yaml.  It builds the chosen model, applies dynamic
Albumentations-based augmentations to grayscale ultrasound images and masks,
and optimizes a composite Dice + Focal loss to segment:
    (0) background
    (1) retina/sclera
    (2) retinal detachment (RD)

Training uses stochastic online augmentation: each image is randomly
transformed (flip, affine, brightness/contrast, noise) on-the-fly every epoch.

References:
    • Chen et al., *TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation*, arXiv:2102.04306
    • Buslaev et al., *Albumentations: Fast and Flexible Image Augmentations*, Information 2020, https://albumentations.ai
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
                         heads=m.get('heads',8),
                         ).to(device)

def train_one_epoch(model, loader, optimizer, cfg, device):
    model.train()
    tot = 0.0
    for img, msk in tqdm(loader, desc='train', leave=False):
        img, msk = img.to(device), msk.to(device)
        logits = model(img)
        loss = composite_loss(logits, msk, cfg['data']['num_classes'],
                              dice_w=cfg['loss']['dice_weight'],
                              focal_w=cfg['loss']['focal_weight'],
                              focal_gamma=cfg['loss']['focal_gamma'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot += float(loss.detach().cpu())
    return tot/len(loader)

@torch.no_grad()
def validate(model, loader, cfg, device):
    model.eval()
    import torch as T
    dices, accs = [], []
    for img, msk in tqdm(loader, desc='val', leave=False):
        img, msk = img.to(device), msk.to(device)
        logits = model(img)
        d = dice_per_class_from_logits(logits, msk, cfg['data']['num_classes'])
        a = pixel_accuracy(logits, msk)
        dices.append(d.cpu())
        accs.append(a.cpu())
    dice_mean = T.stack(dices).mean(0)  # [C]
    acc_mean = T.stack(accs).mean().item()
    return dice_mean, acc_mean

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--train_csv', required=False)
    ap.add_argument('--val_csv', required=False)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config,'r'))
    os.makedirs(args.out, exist_ok=True)
    device = environment.device()
    train_csv = args.train_csv or cfg['data']['train_csv']
    val_csv = args.val_csv or cfg['data']['val_csv']

    model = build_model(cfg, device)
    train_ds = SegCSV(train_csv, cfg, augment=build_train_augs(cfg), is_train=True)
    val_ds   = SegCSV(val_csv, cfg, augment=build_val_augs(cfg), is_train=False)
    num_workers = getattr(cfg.get('train', {}), 'num_workers', _num_workers_default())
    pin = not torch.backends.mps.is_available()
    train_ld = DataLoader(train_ds, batch_size=cfg['train']['batch_size'],
                          shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_ld = DataLoader(val_ds, batch_size=cfg['train']['batch_size'],
                        shuffle=False, num_workers=num_workers, pin_memory=pin)

    optim = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    best = -1.0
    for ep in range(cfg['train']['epochs']):
        tl = train_one_epoch(model, train_ld, optim, cfg, device)
        dice_c, acc = validate(model, val_ld, cfg, device)
        dice_fg = float(dice_c[1:].mean())
        print(f'Epoch {ep+1}/{cfg["train"]["epochs"]}: train_loss={tl:.4f} val_dice_fg={dice_fg:.4f} (per-class {dice_c.tolist()}) acc={acc:.4f}')
        torch.save(model.state_dict(), os.path.join(args.out, 'last.ckpt'))
        if dice_fg > best:
            best = dice_fg
            torch.save(model.state_dict(), os.path.join(args.out, 'best.ckpt'))
    print(f'Best val dice (foreground avg): {best:.4f}')


def _num_workers_default():
    # if platform.system() == "Darwin":
    #     return 0  # safer default on macOS to avoid pickling issues
    return max(2, (os.cpu_count() or 2) // 2)


if __name__ == '__main__':
    main()
