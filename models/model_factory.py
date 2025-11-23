# models/model_factory.py
from typing import Dict, Optional
import torch

import segmentation_models_pytorch as smp

# def check_pretrained(model, encoder_weights: Optional[str]):
#     print(f"[enc] requested encoder_weights={encoder_weights}")
#     # 2) If using timm under the hood, this exposes the pretrained recipe
#     pcfg = getattr(model.encoder, "pretrained_cfg", None)
#     print("[enc] pretrained_cfg:", None if pcfg is None else pcfg.get("tag", pcfg.get("hf_hub_id", "unknown")))
#
#     # Kaiming distance sanity (same as ERM)
#     with torch.no_grad():
#         w = model.encoder.conv1.weight.detach().cpu()
#         torch.manual_seed(123)
#         w_rand = torch.empty_like(w)
#         torch.nn.init.kaiming_normal_(w_rand, nonlinearity="relu")
#         delta = torch.norm(w - w_rand).item()
#         print(f"[enc] L2 delta vs fresh Kaiming: {delta:.4f}")

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

    if name == "unetpp":
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=1,
            classes=num_classes,
            activation=None,
        )
        print(f"[model] UnetPlusPlus encoder weights={encoder_weights}")
        check_pretrained(model, encoder_weights)
        return model.to(device)

    if name in ("transunet_pretrained", "transunet"):
        # ---- Official-style TransUNet with ViT backbone and optional Google .npz init ----
        try:
            # Your file should expose a class compatible with the authors' repo
            # (e.g., models/transunet.py having TransUNet(..., embed_dim, depth, heads, patch_size, img_size))
            from models.transunet import TransUNet  # <- authors' style or faithful port

            # Common defaults for ViT-B/16 @ 512 (multiple of 16). Adjust via YAML if needed.
            mcfg = cfg.get("model", {})
            embed_dim = int(mcfg.get("embed_dim", 768))  # ViT-B
            depth = int(mcfg.get("depth", 12))
            heads = int(mcfg.get("heads", 12))
            patch_size = int(mcfg.get("patch_size", 16))
            img_h, img_w = cfg["data"].get("resize", [512, 512])
            img_size = int(mcfg.get("img_size", max(img_h, img_w)))  # ensure multiple of patch_size in your preprocess
            base = int(mcfg.get("base", 32))  # decoder base channels (TransUNet dec side)
            in_ch = int(mcfg.get("in_channels", 1))
            num_classes = int(cfg["data"]["num_classes"])

            model = TransUNet(
                in_ch=in_ch,
                num_classes=num_classes,
                base=base,
                embed_dim=embed_dim,
                depth=depth,
                heads=heads,
                patch_size=patch_size,
                img_size=img_size,
            )

            # Optional: Google ViT .npz weights
            vit_npz = mcfg.get("vit_pretrained_npz", None)
            _maybe_load_google_vit_npz(model, vit_npz)

            print(f"[model] TransUNet (ViT) embed_dim={embed_dim} depth={depth} heads={heads} "
                  f"patch={patch_size} img_size={img_size} | in_ch={in_ch} classes={num_classes}")
            # No check_pretrained() here because ViT has no conv1
            return model.to(device)

        except Exception as e:
            print(f"[WARN] TransUNet import/init failed: {e}")
            print("[WARN] Falling back to SMP UnetPlusPlus(resnet50, imagenet).")
            model = smp.UnetPlusPlus(
                encoder_name="resnet50",
                encoder_weights="imagenet",
                in_channels=1,
                classes=int(cfg['data']['num_classes']),
                activation=None,
            )
            check_pretrained(model, "imagenet")
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