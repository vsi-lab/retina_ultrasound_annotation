# models/model_factory.py
from typing import Dict, Optional

import numpy as np
import torch

import segmentation_models_pytorch as smp
# models/model_factory.py (add near top)
from models.segformer_wrap import SegFormerWrap

def check_pretrained(model, encoder_weights: Optional[str]):
    print(f"[enc] requested encoder_weights={encoder_weights}")
    pcfg = getattr(model, "encoder", None)
    pcfg = getattr(pcfg, "pretrained_cfg", None)
    print("[enc] pretrained_cfg:", None if pcfg is None else pcfg.get("tag", pcfg.get("hf_hub_id", "unknown")))

    # Only do the conv1 delta sanity if the encoder has a conv stem (CNN encoders).
    enc = getattr(model, "encoder", None)
    conv1 = getattr(enc, "conv1", None)
    if conv1 is None or not hasattr(conv1, "weight"):
        print("[enc] conv1 not found (likely a ViT/transformer encoder); skipping Kaiming delta check.")
        return

    with torch.no_grad():
        w = conv1.weight.detach().cpu()
        torch.manual_seed(123)
        w_rand = torch.empty_like(w)
        torch.nn.init.kaiming_normal_(w_rand, nonlinearity="relu")
        delta = torch.norm(w - w_rand).item()
        print(f"[enc] L2 delta vs fresh Kaiming: {delta:.4f}")

def _maybe_load_google_vit_npz(transunet_model, npz_path: Optional[str]):
    """
    If provided, load Google ViT '.npz' weights into the TransUNet ViT backbone.
    This mirrors the official TransUNet repo behavior where the model exposes
    model.load_from(npz) to copy weights into its ViT encoder.
    """
    if not npz_path:
        print("[transunet] No vit_pretrained_npz provided; training ViT from scratch.")
        return
    import os
    if not os.path.isfile(npz_path):
        print(f"[transunet][WARN] vit_pretrained_npz not found: {npz_path}. Skipping ViT init.")
        return
    try:
        import numpy as np
        if hasattr(transunet_model, "load_from"):
            npz = np.load(npz_path, allow_pickle=True)
            transunet_model.load_from(npz)
            print(f"[transunet] Loaded ViT weights from: {npz_path}")
        else:
            print("[transunet][WARN] Model has no load_from(npz). Add the official loader to your TransUNet class.")
    except Exception as e:
        print(f"[transunet][WARN] Failed to load ViT npz weights: {e}")

def build_seg_model_from_cfg(cfg: Dict, device) -> torch.nn.Module:
    """
    Supported cfg.model.name:
      - "unetpp" : SMP UnetPlusPlus w/ pretrained CNN encoder
      - "transunet_pretrained" : SMP Unet w/ pretrained ViT/Swin encoder (TransUNet-like)
      - "unet" : SMP Unet w/ pretrained CNN encoder (backup baseline)

    Grayscale input => in_channels=1
    """
    if smp is None:
        raise RuntimeError("segmentation_models_pytorch not installed. pip install segmentation-models-pytorch timm")

    mcfg = cfg.get("model", {})
    name = mcfg.get("name", "unetpp").lower()
    num_classes = int(cfg["data"]["num_classes"])

    encoder_name = mcfg.get("encoder_name", "resnet34")
    encoder_weights = mcfg.get("encoder_weights", "imagenet")  # None disables weights

    if name in ("segformer_b0", "segformer_b1", "segformer_b2", "segformer_b3", "segformer_b4", "segformer_b5"):
        ignore_idx = 0
        variant = name.split("_")[1]  # "b0", "b2", etc.
        model = SegFormerWrap(variant=variant, num_labels=num_classes,
                              ignore_index=ignore_idx, pretrained=True)
        print(f"[model] SegFormer {variant} (HF) with pretrained backbone")

        n_params = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[model] params: total={n_params / 1e6:.1f}M, trainable={n_train / 1e6:.1f}M")
        raise Exception("segformer not supported")
        return model.to(device)

    if name == "unetpp":
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=1,
            classes=num_classes,
            activation=None,
        )
        print(f"[model] UnetPlusPlus encoder {encoder_name} weights={encoder_weights} ")
        check_pretrained(model, encoder_weights)
        return model.to(device)

    if name in ("transunet_npz", "transunet"):
        # https://github.com/Beckschen/TransUNet
        # refer to README.md for instructions

        # cloned it to : third_party/transunet
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "transunet"))

        # Official modules
        from third_party.transunet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
        from third_party.transunet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
        from third_party.transunet.networks.vit_seg_configs import get_r50_b16_config

        # Choose config and matching .npz
        # encoder_name: one of {"ViT-B_16", "R50+ViT-B_16" etc.
        enc_name = mcfg.get("encoder_name", "R50-ViT-B_16")

        if enc_name not in CONFIGS_ViT_seg:
            raise ValueError(f"Unknown TransUNet encoder_name '{enc_name}'. "
                             f"Available: {list(CONFIGS_ViT_seg.keys())[:6]}...")

        config = get_r50_b16_config()
        with config.unlocked():
            # Use 3ch if using R50 ImageNet npz
            use_r50_npz = ("pretrained_npz" in mcfg) and ("R50" in mcfg.get("encoder_name", ""))
            config.n_channels = 1
            config.n_classes = num_classes
            config.n_skip = int(mcfg.get("n_skip"))

            img_size = int(cfg["data"]["resize"][0])  # assumes square
            config.img_size = img_size
            config.patches.size = (16, 16)
            gh = gw = img_size // 16
            config.patches.grid = (gh, gw)

            # R50 tuple + width
            if not isinstance(getattr(config.resnet, "num_layers", ()), tuple):
                config.resnet.num_layers = (3, 4, 6, 3)
            config.resnet.width_factor = int(mcfg.get("resnet_width", 1))

        npz_path = mcfg.get("pretrained_npz", "").format(**cfg)

        img_sz = config.img_size if isinstance(config.img_size, int) else max(config.img_size)
        model = ViT_seg(config, img_size=img_sz, num_classes=num_classes)

        print(f"[model] ViT config.img_size={config.img_size}")
        if npz_path and Path(npz_path).exists():
            print(f"[TransUNet] loading ViT weights from: {npz_path}")
            try:
                w = np.load(npz_path, allow_pickle=True)
                model.load_from(weights=w)  # <-- this fork expects a dict-like, not a path
                del w
                print("[TransUNet] pretrained ViT weights loaded.")
            except Exception as e:
                print(f"[TransUNet][WARN] failed to load npz ({e}). Continuing with random init.")
        else:
            print("[TransUNet] no valid pretrained_npz provided; using random init.")

        return model.to(device)

    if name == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=1,
            classes=num_classes,
            activation=None,
        )
        check_pretrained(model, encoder_weights)
        return model.to(device)

    raise ValueError(f"Unknown cfg.model.name={name}")