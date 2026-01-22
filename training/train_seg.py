# training/train_seg.py
import argparse
import os
import shutil

import torch
import warnings
import yaml
import json
import random
from pathlib import Path

from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.paths import clean_or_make
from models.model_factory import build_seg_model_from_cfg
from training.augments import build_train_augs, build_val_augs
from training.dataset import SegCSV
from training.losses import composite_loss
from training.metrics import per_class_dice_from_logits
from utils import environment
from utils.vis_usg import save_preview_panel
from utils.paths import resolve_under_root_cfg

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
warnings.filterwarnings("ignore", message="enable_nested_tensor")

import numpy as np

def _bincount_np(t: torch.Tensor, K: int) -> np.ndarray:
    """t: [B,H,W] or [H,W] int tensor -> np counts of length K."""
    return torch.bincount(t.view(-1), minlength=K).detach().cpu().numpy()

def _hist_str(counts: np.ndarray) -> str:
    tot = int(counts.sum())
    parts = [f"{i}:{int(c)}" for i, c in enumerate(counts) if c > 0]
    return " ".join(parts) + f" | tot:{tot}"

def _softmax_mean_str(logits: torch.Tensor) -> str:
    """Mean class probability over the batch to detect collapse."""
    probs = torch.softmax(logits, dim=1).mean(dim=(0,2,3)).detach().cpu().numpy()
    return " ".join([f"{i}:{p:.3f}" for i, p in enumerate(probs)])

def _read_json(p: Path):
    if not p.exists(): return None
    with open(p, "r") as f: return json.load(f)


def train_one_epoch(model, loader, optimizer, cfg, device, out_dir: str,
                    epoch: int, preview_every: int = 0):
    """
    Adds rich diagnostics:
      - Batch GT and Pred (argmax) histograms: counts per class
      - Mean softmax per class (to catch collapse)
      - Optional per-sample panel (existing)
      - Tries to print the file path if the dataset returns it as a 3rd item
    Control frequency with cfg.train.debug_every_steps (default: same as preview_every or 50)
    """
    model.train()
    tot = 0.0
    step = 0
    K = int(cfg['data']['num_classes'])
    meta_json = _read_json(Path(cfg["work_root"]) / "meta.json")

    debug_every = int(cfg.get('train', {}).get('debug_every_steps', preview_every or 50))

    for batch in tqdm(loader, desc='train', leave=False):
        # Accept (img, msk) OR (img, msk, path)
        img, msk, paths = batch
            # paths can be a list of strings; we'll show the first when dumping
        path0 = (paths[0] if isinstance(paths, (list, tuple)) and len(paths) > 0 else None)

        img, msk = img.to(device), msk.to(device)
        logits = model(img)

        loss = composite_loss(
            logits, msk,
            cfg['data']['num_classes'],
            dice_w=cfg['loss']['dice_weight'],
            focal_w=cfg['loss']['focal_weight'],
            focal_gamma=cfg['loss']['focal_gamma']
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot += float(loss.detach().cpu())

        # --- Optional step-wise preview panel (as you had) ---
        if preview_every and (step % preview_every == 0):
            # If we have a path, pass it through so left column shows original
            save_preview_panel(
                Path(out_dir) / "previews" / f"preview_train_e{epoch:03d}_s{step:05d}.png",
                img[0].detach().cpu(),
                msk[0].detach().cpu(),
                logits[0].detach().cpu(),
                cfg,
                meta_json=meta_json,
                raw_img_path=(path0 if isinstance(path0, str) else None)
            )

        step += 1

    return tot/len(loader)

@torch.no_grad()
def validate(model, loader, cfg, device, out_dir: str, epoch: int,
             previews_per_epoch: int = 2):
    """
    Returns:
      overall_mean (dict cid->dice), present_mean (dict cid->dice or None)
    Also drops exactly `previews_per_epoch` panels from the first val batch,
    with filenames printed on the panels.
    """
    model.eval()
    K = int(cfg['data']['num_classes'])
    meta_json = _read_json(Path(cfg["work_root"]) / "meta.json")

    # accumulators
    sum_overall = {c: 0.0 for c in range(K)}
    cnt_overall = {c: 0   for c in range(K)}
    sum_present = {c: 0.0 for c in range(K)}
    cnt_present = {c: 0   for c in range(K)}

    ds = loader.dataset  # SegCSV; has .df with file paths

    for b_idx, (img, msk, _) in enumerate(tqdm(loader, desc='val', leave=False)):
        img, msk = img.to(device), msk.to(device)
        logits = model(img)

        overall_d, present_d, _ = per_class_dice_from_logits(
            logits, msk, num_classes=K, ignore_index=0
        )
        for c, d in overall_d.items():
            sum_overall[c] += float(d); cnt_overall[c] += 1
        for c, d in present_d.items():
            if d is not None:
                sum_present[c] += float(d); cnt_present[c] += 1

        # Exactly 2 previews/epoch from first batch (or fewer if batch < 2)
        if b_idx == 0 and previews_per_epoch > 0:
            n = min(previews_per_epoch, img.shape[0])
            base_index = 0  # because val loader is shuffle=False
            # compute where this batch starts: b_idx * batch_size (safe even if last batch is smaller)
            start_idx = b_idx * loader.batch_size
            for j in range(n):
                global_idx = start_idx + j
                # resolve image path for printing
                ip = resolve_under_root_cfg(cfg, ds.df.iloc[global_idx].image_path).as_posix()
                save_preview_panel(
                    Path(out_dir) / "previews" / f"preview_val_e{epoch:03d}_{j:02d}.png",
                    img[j].detach().cpu(),
                    msk[j].detach().cpu(),
                    logits[j].detach().cpu(),
                    cfg,
                    meta_json=meta_json,
                    raw_img_path=ip
                )

    overall_mean = {c: (sum_overall[c] / max(cnt_overall[c], 1)) for c in range(K)}
    present_mean = {c: (sum_present[c] / cnt_present[c] if cnt_present[c] > 0 else None) for c in range(K)}
    return overall_mean, present_mean

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--train_csv', required=False)
    ap.add_argument('--val_csv', required=False)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config,'r'))

    clean_or_make(args.out)

    device = environment.device()
    train_csv = args.train_csv or (cfg['data']['train_csv']).format(**cfg)
    val_csv   = args.val_csv   or (cfg['data']['val_csv']).format(**cfg)

    model = build_seg_model_from_cfg(cfg, device)

    train_ds = SegCSV(train_csv, cfg, augment=build_train_augs(cfg), is_train=True)
    val_ds   = SegCSV(val_csv,   cfg, augment=build_val_augs(cfg), is_train=False)

    num_workers = int(cfg.get('train', {}).get('num_workers', max(2, (os.cpu_count() or 2) // 2)))
    pin = not torch.backends.mps.is_available()

    train_ld = DataLoader(train_ds, batch_size=cfg['train']['batch_size'],
                          shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_ld = DataLoader(val_ds, batch_size=cfg['train']['batch_size'],
                        shuffle=False, num_workers=num_workers, pin_memory=pin)
    if "unetpp" in  cfg.get("model")['name']:
        base_lr = 3e-4  # decoder/head
        enc_lr = base_lr * 0.1

        optim = AdamW([
            {"params": model.encoder.parameters(), "lr": enc_lr},
            {"params": model.decoder.parameters(), "lr": base_lr},
            {"params": model.segmentation_head.parameters(), "lr": base_lr},
        ], lr=base_lr, weight_decay=cfg['train']['weight_decay'])
    elif "transunet" in cfg.get("model")['name']:
        base_lr = 3e-4
        enc_lr = base_lr * 0.1

        # TransUNet encoder is the ViT/Hybrid block
        enc_params = []
        if hasattr(model, "transformer"):
            enc_params += list(model.transformer.parameters())
        elif hasattr(model, "encoder"):
            enc_params += list(model.encoder.parameters())

        enc_ids = {id(p) for p in enc_params}
        dec_params = [p for p in model.parameters() if id(p) not in enc_ids]

        optim = AdamW([
            {"params": enc_params, "lr": enc_lr},
            {"params": dec_params, "lr": base_lr},
        ], lr=base_lr, weight_decay=cfg['train']['weight_decay'])
    else:
        optim     = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'],
                             weight_decay=cfg['train']['weight_decay'])

    # knobs
    previews_per_epoch = int(cfg.get('train', {}).get('previews_per_epoch', 2))
    patience           = int(cfg.get('train', {}).get('early_stop_patience', 5))
    min_delta          = float(cfg.get('train', {}).get('early_stop_min_delta', 1e-3))

    best = -1.0
    no_improve = 0
    class_names = cfg["data"].get("class_names", [f"class{i}" for i in range(cfg["data"]["num_classes"])])
    K = cfg['data']['num_classes']
    fg_ids = [c for c in range(K) if c != 0]

    for ep in range(cfg['train']['epochs']):
        tl = train_one_epoch(model, train_ld, optim, cfg, device, args.out, ep, previews_per_epoch * 5)
        overall_mean, present_mean = validate(model, val_ld, cfg, device, args.out, ep, previews_per_epoch)

        dice_fg = sum(overall_mean[c] for c in fg_ids) / max(len(fg_ids), 1)

        # pretty log (omit background, include P-Dice)
        parts = []
        for c in fg_ids:
            po = present_mean[c]
            po_s = f"{po:.3f}" if po is not None else "N/A"
            parts.append(f"{class_names[c]}={overall_mean[c]:.3f} (P:{po_s})")
        msg = ", ".join(parts)

        print(f'Epoch {ep+1}/{cfg["train"]["epochs"]}: '
              f'train_loss={tl:.4f}  val_dice_fg={dice_fg:.4f}  [{msg}]')

        # save last & best
        torch.save(model.state_dict(), os.path.join(args.out, 'last.ckpt'))
        if dice_fg > best + min_delta:
            best = dice_fg
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.out, 'best.ckpt'))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping: no improvement ≥ {patience} epoch(s). Best val_dice_fg={best:.4f}")
                break

    print(f'Best val dice (foreground avg): {best:.4f}')

if __name__ == '__main__':
    main()