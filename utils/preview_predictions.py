# utils/preview_predictions.py
import argparse
import random
from pathlib import Path
from typing import Dict, Optional
from utils.vis_usg import save_preview_panel  # add at top

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

# NOTE: do NOT import TransUNet here anymore.
# We'll import it lazily only if cfg requests it.
# from models.unet import UNet  # keep handcrafted UNet as fallback
from training.dataset import SegCSV
from training.metrics import per_class_dice_from_logits
from utils.paths import resolve_under_root_cfg
from utils.vis_usg import build_id2color_from_meta, colorize_mask


def _read_json(p: Path):
    if not p.exists():
        return None
    import json
    with open(p, "r") as f:
        return json.load(f)


# --------- robust path resolve (works even if resolve_under_root_cfg is buggy) ---------
def _safe_resolve(cfg: Dict, p: str) -> Path:
    try:
        rp = resolve_under_root_cfg(cfg, p)
        if rp is None:
            raise ValueError("resolve_under_root_cfg returned None")
        return Path(rp)
    except Exception:
        root = Path(cfg.get("work_root", cfg.get("work_dir", ".")))
        pp = Path(p)
        return pp if pp.is_absolute() else (root / pp)


# --------- SMP builder (local, no factory) ---------
def _build_smp_model(name: str, cfg: Dict, out_channels: int, in_channels: int = 1):
    try:
        import segmentation_models_pytorch as smp
    except Exception as e:
        raise RuntimeError(
            "segmentation_models_pytorch not installed, but config requests an SMP model "
            f"({name}). Install it or switch model.name to 'transunet' or 'unet'."
        ) from e

    enc_name = cfg["model"].get("encoder_name", "resnet34")
    enc_wts  = cfg["model"].get("encoder_weights", "imagenet")

    if name == "unetpp":
        return smp.UnetPlusPlus(
            encoder_name=enc_name,
            encoder_weights=enc_wts,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,
        )
    if name == "unet":
        return smp.Unet(
            encoder_name=enc_name,
            encoder_weights=enc_wts,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,
        )
    raise ValueError(f"Unsupported SMP model: {name}")


def load_model(cfg: Dict, ckpt_path: str, device):
    name = str(cfg["model"]["name"]).lower()
    num_classes = int(cfg["data"]["num_classes"])
    base = int(cfg["model"].get("base", 32))

    # SMP pretrained paths
    if name in ("unetpp", "unet"):
        model = _build_smp_model(name, cfg, out_channels=num_classes, in_channels=1)

    # Handcrafted TransUNet (lazy import)
    elif "trans" in name:
        from models.transunet import TransUNet  # <- only imported if needed
        model = TransUNet(
            in_ch=1, num_classes=num_classes, base=base,
            embed_dim=int(cfg["model"].get("embed_dim", 256)),
            depth=int(cfg["model"].get("depth", 4)),
            heads=int(cfg["model"].get("heads", 8)),
        )

    # Handcrafted UNet fallback
    else:
        model = UNet(in_ch=1, num_classes=num_classes, base=base)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)
    return model


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--eval_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_samples", type=int, default=6)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    df = pd.read_csv(args.eval_csv)
    ds = SegCSV(args.eval_csv, cfg, augment=None, is_train=False)

    meta_json = _read_json(Path(cfg["work_root"]) / "meta.json")
    labels_map = cfg["data"]["labels"]
    id2color = build_id2color_from_meta(meta_json, labels_map) if meta_json else None

    num_classes = int(cfg["data"]["num_classes"])
    class_names = cfg["data"].get("class_names", [f"class_{i}" for i in range(num_classes)])

    model = load_model(cfg, args.ckpt, device)

    idxs = random.sample(range(len(df)), min(args.num_samples, len(df)))
    for k, idx in enumerate(idxs):
        row = df.iloc[idx]
        ip = _safe_resolve(cfg, str(row.image_path))
        mp = _safe_resolve(cfg, str(row.mask_path))

        img_raw = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        gt_raw  = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)

        if img_raw is None or gt_raw is None:
            print(f"[WARN] Skipping unreadable pair: {ip}, {mp}")
            continue

        img_t, gt_t, _ = ds[idx]
        logits = model(img_t.unsqueeze(0).to(device))[0].cpu()

        save_preview_panel(
            out_dir / f"pred_panel_{k:02d}.png",
            img_t,  # [1,Hm,Wm]
            gt_t,  # [Hm,Wm]
            logits,  # [C,Hm,Wm]
            cfg,
            meta_json=meta_json,
            raw_img_path=str(ip)
        )
        print("Saved:", out_dir / f"pred_panel_{k:02d}.png")

        pred = torch.argmax(logits, dim=0).numpy().astype(np.int64)
        gt_ids = gt_t.numpy().astype(np.int64)

        # GT / pred colorization
        if id2color:
            gt_rgb   = colorize_mask(gt_ids, id2color)
            pred_rgb = colorize_mask(pred, id2color)
        else:
            # fallback: show raw GT and a simple pred viz
            gt_rgb = gt_raw
            if gt_rgb.ndim == 2:
                gt_rgb = cv2.cvtColor(gt_rgb, cv2.COLOR_GRAY2RGB)
            pred_rgb = (pred[..., None] * (255 // max(1, num_classes-1))).astype(np.uint8)
            pred_rgb = np.repeat(pred_rgb, 3, axis=2)

        panel = np.concatenate([
            cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB),
            gt_rgb,
            pred_rgb
        ], axis=1)

        overall, present, _ = per_class_dice_from_logits(
            logits.unsqueeze(0), gt_t.unsqueeze(0),
            num_classes=num_classes, ignore_index=0
        )

        y0 = 18
        for c in range(1, num_classes):
            nm = class_names[c] if c < len(class_names) else f"class_{c}"
            d  = overall.get(c, 0.0)
            po = present.get(c, None)
            po_str = f"{po:.3f}" if po is not None else "N/A"
            cv2.putText(panel, f"{nm}: {d:.3f} (P:{po_str})",
                        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255,255,255), 1, cv2.LINE_AA)
            y0 += 16

        out_path = out_dir / f"pred_panel_{k:02d}.png"
        cv2.imwrite(str(out_path), panel[:, :, ::-1])  # RGB->BGR for cv2
        print("Saved:", out_path)


if __name__ == "__main__":
    main()

