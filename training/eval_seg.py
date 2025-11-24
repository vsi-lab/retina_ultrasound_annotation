# training/eval_seg.py  — DROP-IN
import argparse
import os
from pathlib import Path
import cv2
from utils.vis_usg import colorize_ids, build_id2color_from_meta
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model_factory import build_seg_model_from_cfg
from training.augments import build_val_augs
from training.dataset import SegCSV
from training.metrics import per_class_dice_from_logits, pixel_accuracy
from utils import environment
from utils.paths import resolve_under_root_cfg, clean_or_make
from utils.vis_usg import save_preview_panel
from typing import Dict, List


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    # ap.add_argument('--save_panels', action='store_true', help='save a preview for every sample')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r'))

    eval_csv = str(resolve_under_root_cfg(cfg, cfg['data']['test_csv']))

    out_dir = Path(args.out)
    clean_or_make(args.out)

    panels_dir = out_dir / "previews"
    panels_dir.mkdir(parents=True, exist_ok=True)

    device = environment.device()

    # model
    model = build_seg_model_from_cfg(cfg, device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # data
    ds = SegCSV(eval_csv, cfg, augment=build_val_augs(cfg), is_train=False)
    bs = int(cfg['train']['batch_size'])
    ld = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=int(cfg['train']['num_workers']), pin_memory=not torch.backends.mps.is_available())

    num_classes = int(cfg["data"]["num_classes"])
    labels_map = cfg["data"]["labels"]

    # ID -> name mapping from config to avoid ordering mismatches
    id2name = {int(v): k for k, v in labels_map.items()}
    class_names = [id2name.get(i, f"class_{i}") for i in range(num_classes)]

    fg_ids = [c for c in range(num_classes) if c != 0]

    # Track how many eval samples actually contain each class in GT
    gt_presence_counts = {c: 0 for c in range(num_classes)}


    # per-sample table we’ll write out
    rows: List[Dict] = []

    # accumulate for summary
    per_class_overall: Dict[int, List[float]] = {c: [] for c in range(num_classes)}
    per_class_present: Dict[int, List[float]] = {c: [] for c in range(num_classes)}  # NaN if not present
    pixel_accs: List[float] = []

    meta_json = None
    meta_path = Path(cfg.get("work_root", ".")) / "meta.json"
    if meta_path.exists():
        import json
        meta_json = json.loads(meta_path.read_text())
    id2color = build_id2color_from_meta(meta_json, labels_map)
    global_idx = 0
    for b_idx, (img, msk, _) in enumerate(tqdm(ld, desc="eval", leave=False)):
        img, msk = img.to(device), msk.to(device)
        logits = model(img)
        Hm, Wm = logits.shape[-2:]
        print(f"[debug] model_input={Hm}x{Wm}")
        B = img.shape[0]
        for j in range(B):
            lo  = logits[j:j+1]  # [1,C,H,W]
            gt  = msk[j:j+1]     # [1,H,W]

            def _hist_str(arr):
                uniq, cnt = np.unique(arr, return_counts=True)
                return " ".join(f"{int(u)}:{int(c)}" for u, c in zip(uniq, cnt))

            # print(f"[pred] {Path(ds.df.iloc[global_idx].image_path).name} | {_hist_str(lo_np)}")


            # Compute per-class dice (overall)
            overall_d, _, _ = per_class_dice_from_logits(
                lo, gt, num_classes=num_classes, ignore_index=0
            )

            # Determine GT presence directly from the mask
            gt_np = gt[0].detach().cpu().numpy()
            for c in range(num_classes):
                if (gt_np == c).any():
                    gt_presence_counts[c] += 1

            # Rebuild present-only dice: use overall dice value ONLY if class is present in GT
            present_d = {}
            for c in range(num_classes):
                present_d[c] = overall_d.get(c, 0.0) if (gt_np == c).any() else None

            # Pixel accuracy (unchanged)
            acc = float(pixel_accuracy(lo, gt).cpu())
            pixel_accs.append(acc)

            # Per-sample row (use the id2name-based class_names for stable labeling)
            row = {"pixel_acc": acc}
            for c in range(1, num_classes):
                name = class_names[c]
                row[f"dice_{name}"] = overall_d.get(c, 0.0)
                row[f"diceP_{name}"] = present_d.get(c, None)
            rows.append(row)

            # Accumulate for summary
            for c in range(num_classes):
                per_class_overall[c].append(float(overall_d.get(c, 0.0)))
                v = present_d.get(c, None)
                per_class_present[c].append(np.nan if v is None else float(v))

            # per-sample panel (optional)

            # resolve raw path for left column
            ip = resolve_under_root_cfg(cfg, str(ds.df.iloc[global_idx].image_path))

            #
            # --- Predicted id mask for this sample ---
            pred_ids = torch.argmax(lo, dim=1)[0].detach().cpu().numpy().astype(np.uint8)

            # (A) Sanity check: class histogram of the prediction
            #     Comment out after a quick run if too chatty.
            uniq, cnt = np.unique(pred_ids, return_counts=True)
            print(f"[gt ] {Path(ds.df.iloc[global_idx].image_path).name} | {_hist_str(gt_np)}")
            print(f"[pred] {Path(ip).name} | " + " ".join(f"{int(u)}:{int(c)}" for u, c in zip(uniq, cnt)))

            # (B) Dump predictions as images (ids + colorized)
            #     File name references the original image path.
            try:
                rel = Path(ip).relative_to(Path(cfg.get("work_root", ".")))
                safe_name = str(rel).replace("/", "__")
            except Exception:
                safe_name = Path(ip).name

            pred_ids_dir = out_dir / "pred_ids"
            pred_viz_dir = out_dir / "pred_viz"
            pred_ids_dir.mkdir(parents=True, exist_ok=True)
            pred_viz_dir.mkdir(parents=True, exist_ok=True)

            # Save raw id map (grayscale PNG)
            cv2.imwrite(str(pred_ids_dir / f"{safe_name}_pred_ids.png"), pred_ids)

            # Save colorized visualization (uses your vis_usg.colorize_ids)
            pred_color = colorize_ids(pred_ids, id2color)  # HxWx3 (BGR) from your helper
            cv2.imwrite(str(pred_viz_dir / f"{safe_name}_pred_viz.png"), pred_color)
            #

            save_preview_panel(
                panels_dir / f"panel_{global_idx:06d}.png",
                img[j].detach().cpu(),
                msk[j].detach().cpu(),
                logits[j].detach().cpu(),
                cfg,
                meta_json=meta_json,
                raw_img_path=str(ip)
            )
            global_idx += 1
            #  DEBUG #######
            # === Also dump the raw predicted id map and per-class binaries ===
            pred_ids = torch.argmax(lo[0], dim=0).cpu().numpy().astype(np.uint8)  # [H,W]
            stem = Path(ip).stem

            # save compact PNG with raw ids (palette-free; just integers 0..C-1)
            raw_pred_dir = out_dir / "pred_ids"
            raw_pred_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(raw_pred_dir / f"{stem}_pred_ids.png"), pred_ids)

            # optional: per-class binary masks (e.g., to inspect class 3 “blue” FPs)
            bin_dir = out_dir / "pred_bins"
            bin_dir.mkdir(parents=True, exist_ok=True)
            for cid in range(num_classes):
                bin_img = (pred_ids == cid).astype(np.uint8) * 255
                cv2.imwrite(str(bin_dir / f"{stem}_pred_c{cid}.png"), bin_img)

            #  DEBUG #######





    # write per-sample CSV
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics.csv", index=False)

    # ---- Console summary ----
    # pixel acc
    pix_m, pix_s = np.mean(pixel_accs), np.std(pixel_accs)
    print("\n=== Test Summary ===")
    print(f"Pixel Accuracy          : {pix_m:.4f} ± {pix_s:.4f}")

    # FG Dice (overall and present-only)
    fg_overall = np.array([np.mean(per_class_overall[c]) for c in fg_ids])
    fg_present = np.array([np.nanmean(per_class_present[c]) for c in fg_ids])

    print(f"FG Dice (overall)       : {np.mean(fg_overall):.4f} ± {np.std(fg_overall):.4f}")
    print(f"FG Dice (present-only)  : {np.nanmean(fg_present):.4f} ± {np.nanstd(fg_present):.4f}")

    # Per-class (overall)
    print("Per class Overall Dice")
    for c in fg_ids:
        name = class_names[c] if c < len(class_names) else f"class_{c}"
        vals = np.array(per_class_overall[c])
        print(f"{'dice_' + name:>18}: {vals.mean():.4f} ± {vals.std():.4f}")

    # Per-class (present-only) with counts
    print("\n-- Per-class (present-only) Dice --")
    N = len(rows)
    for c in fg_ids:
        name = class_names[c]
        vals = np.array(per_class_present[c])  # contains NaNs when class absent in GT
        # Use GT presence count to report n and %
        n_pres = gt_presence_counts.get(c, int(np.sum(~np.isnan(vals))))
        pct = (100.0 * n_pres / max(1, N))
        print(f"{'diceP_' + name:>18}: {np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f}   (n={n_pres}, {pct:.1f}% present)")


if __name__ == '__main__':
    main()