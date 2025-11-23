# training/train_seg.py
import argparse
import os

import cv2
import torch
import warnings
import yaml
import json
import random
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

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

# def train_one_epoch_GOOD(model, loader, optimizer, cfg, device, out_dir: str,
#                     epoch: int, preview_every: int = 0):
#     model.train()
#     tot = 0.0
#     step = 0
#     meta_json = _read_json(Path(cfg["work_root"]) / "meta.json")
#
#     for img, msk, paths  in tqdm(loader, desc='train', leave=False):
#         img, msk = img.to(device), msk.to(device)
#         logits = model(img)
#
#         loss = composite_loss(
#             logits, msk, cfg['data']['num_classes'],
#             dice_w=cfg['loss']['dice_weight'],
#             focal_w=cfg['loss']['focal_weight'],
#             focal_gamma=cfg['loss']['focal_gamma']
#         )
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         tot += float(loss.detach().cpu())
#
#         # Optional step-wise preview (kept for parity; filenames not printed here)
#         if preview_every and (step % preview_every == 0):
#             raw_path = paths[0] if isinstance(paths, (list, tuple)) else str(paths)
#             save_preview_panel(
#                 Path(out_dir) / f"preview_train_e{epoch:03d}_s{step:05d}.png",
#                 img[0].detach().cpu(),
#                 msk[0].detach().cpu(),
#                 logits[0].detach().cpu(),
#                 cfg,
#                 meta_json=meta_json,
#                 raw_img_path=raw_path
#             )
#         step += 1
#
#     return tot/len(loader)

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

    debug_every = int(cfg.get('train', {}).get('debug_every_steps', preview_every or 50))

    for batch in tqdm(loader, desc='train', leave=False):
        # Accept (img, msk) OR (img, msk, path)
        img, msk, paths = batch
            # paths can be a list of strings; we'll show the first when dumping
        path0 = (paths[0] if isinstance(paths, (list, tuple)) and len(paths) > 0 else None)

        img, msk = img.to(device), msk.to(device)
        logits = model(img)

        # --------------------# --------------------# --------------------# --------------------# --------------------# --------------------# --------------------
        # === DEBUG: per-class counts + softmax mean (1st item) ===
        if step % max(1, preview_every or 50) == 0:
            with torch.no_grad():
                # counts at model input resolution
                gt0  = msk[0].detach().cpu().numpy()
                pr0  = torch.argmax(logits[0], dim=0).detach().cpu().numpy()
                H, W = gt0.shape
                tot  = H * W

                def counts(arr):
                    uniq, cnt = np.unique(arr, return_counts=True)
                    return {int(u): int(c) for u, c in zip(uniq, cnt)}

                gt_cnt  = counts(gt0)
                pr_cnt  = counts(pr0)
                sm_mean = torch.softmax(logits[0], dim=0).mean(dim=(1,2)).cpu().numpy()

                print(f"[gt  ] {path0} | " + " ".join([f"{k}:{v}" for k,v in sorted(gt_cnt.items())]) + f" | tot:{tot}")
                print(f"[pred] {path0} | " + " ".join([f"{k}:{v}" for k,v in sorted(pr_cnt.items())]) + f" | tot:{tot}")
                print("[softmax_mean] " + " ".join([f"{i}:{sm_mean[i]:0.3f}" for i in range(len(sm_mean))]))

                # optional: dump class-3 (optic nerve) masks for quick eyeballing
                dbg_dir = Path(out_dir) / "debug_train"
                dbg_dir.mkdir(parents=True, exist_ok=True)
                on_gt  = (gt0 == 3).astype(np.uint8) * 255
                on_pr  = (pr0 == 3).astype(np.uint8) * 255
                stem   = Path(path0).stem.replace("/", "_")
                cv2.imwrite(str(dbg_dir / f"{stem}_e{epoch:03d}_s{step:05d}_gt_on.png"), on_gt)
                cv2.imwrite(str(dbg_dir / f"{stem}_e{epoch:03d}_s{step:05d}_pr_on.png"), on_pr)


        # --- DEBUG: per-sample histograms + optic nerve dumps + color pred ---
        # import numpy as np, cv2
        # from pathlib import Path
        # from utils.vis_usg import build_id2color_from_meta, colorize_ids
        #
        # # handle (img, msk, path) or (img, msk)
        # paths = None
        # if isinstance(loader.dataset[0], (tuple, list)) and len(loader.dataset[0]) == 3:
        #     paths = loader.dataset.df.image_path  # we’ll use df index below
        #
        meta_json = _read_json(Path(cfg["work_root"]) / "meta.json")
        # id2color = build_id2color_from_meta(meta_json, cfg["data"]["labels"]) if meta_json else {}
        #
        # B = img.shape[0]
        # for j in range(min(B, 1)):  # dump first item of the batch
        #     # logits[j]: [C,H,W], msk[j]: [H,W]
        #     lo = logits[j].detach().cpu()
        #     gt = msk[j].detach().cpu()
        #     pr = torch.argmax(lo, dim=0).numpy().astype(np.int64)
        #
        #     # counts
        #     def hist_str(arr):
        #         u, c = np.unique(arr, return_counts=True)
        #         return " ".join([f"{int(ui)}:{int(ci)}" for ui, ci in zip(u, c)])
        #
        #     H, W = pr.shape
        #     tot = H * W
        #     # best-effort file name
        #     fname = "<na>"
        #     if paths is not None:
        #         # approximate global index for display
        #         fname = str(loader.dataset.df.iloc[step % len(loader.dataset)].image_path)
        #
        #     print(f"[gt  ] {fname} | {hist_str(gt.numpy())} | tot:{tot}")
        #     print(f"[pred] {fname} | {hist_str(pr)} | tot:{tot}")
        #
        #     # class-wise mean softmax (helps spot a collapsed class)
        #     sm = torch.softmax(lo, dim=0).mean(dim=(1, 2)).numpy()
        #     print("[softmax_mean]", " ".join([f"{i}:{sm[i]:.3f}" for i in range(len(sm))]))
        #
        #     # optic nerve (cid=3) binary dumps
        #     cid = 3
        #     on_gt = (gt.numpy() == cid).astype(np.uint8) * 255
        #     on_pr = (pr == cid).astype(np.uint8) * 255
        #     cv2.imwrite(str(Path(out_dir) / f"debug_gt_optic_e{epoch:03d}_s{step:05d}.png"), on_gt)
        #     cv2.imwrite(str(Path(out_dir) / f"debug_pr_optic_e{epoch:03d}_s{step:05d}.png"), on_pr)
        #
        #     # full colorized predicted mask (BGR) using meta.json colors
        #     pred_color = colorize_ids(pr, id2color)
        #     cv2.imwrite(str(Path(out_dir) / f"pred_color_e{epoch:03d}_s{step:05d}.png"), pred_color)
        # # --- end DEBUG ---

        # --------------------# --------------------# --------------------# --------------------# --------------------# --------------------# --------------------



        loss = composite_loss(
            logits, msk, cfg['data']['num_classes'],
            dice_w=cfg['loss']['dice_weight'],
            focal_w=cfg['loss']['focal_weight'],
            focal_gamma=cfg['loss']['focal_gamma']
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot += float(loss.detach().cpu())

        # --- Diagnostics every debug_every steps ---
        if debug_every and (step % debug_every == 0):
            # Argmax prediction
            pred = torch.argmax(logits, dim=1)  # [B,H,W]

            # Batch histograms
            gt_counts   = _bincount_np(msk,  K)
            pred_counts = _bincount_np(pred, K)

            # Mean softmax per class
            smx_str = _softmax_mean_str(logits)

            # Optional short path hint
            pstr = ""
            if path0 is not None:
                try:
                    root = Path(cfg.get("work_root", ".")).resolve()
                    pstr = " " + Path(path0).resolve().relative_to(root).as_posix()
                except Exception:
                    pstr = " " + str(path0)

            # print(f"[gt  ]{pstr} | {_hist_str(gt_counts)}")
            # print(f"[pred]{pstr} | {_hist_str(pred_counts)}")
            # print(f"[softmax_mean] {smx_str}")
            Hm, Wm = logits.shape[-2:]
            print(f"[debug] model_input={Hm}x{Wm}")

        # --- Optional step-wise preview panel (as you had) ---
        if preview_every and (step % preview_every == 0):
            # If we have a path, pass it through so left column shows original
            save_preview_panel(
                Path(out_dir) / f"preview_train_e{epoch:03d}_s{step:05d}.png",
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
            # global dataset index of sample j in this batch:
            start_idx = 0
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
    ap.add_argument('--preview_every', type=int, default=0)  # keep step-previews optional
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config,'r'))
    os.makedirs(args.out, exist_ok=True)
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

    optim = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'],
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
        tl = train_one_epoch(model, train_ld, optim, cfg, device, args.out, ep, args.preview_every)
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