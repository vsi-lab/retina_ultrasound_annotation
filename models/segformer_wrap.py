# models/segformer_wrap.py
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
except Exception as e:
    raise RuntimeError(
        "Please `pip install transformers>=4.40 timm` to use SegFormer."
    ) from e

_VARIANTS = {"b0":"nvidia/mit-b0", "b1":"nvidia/mit-b1", "b2":"nvidia/mit-b2",
             "b3":"nvidia/mit-b3", "b4":"nvidia/mit-b4", "b5":"nvidia/mit-b5"}

class SegFormerWrap(nn.Module):
    """
    Thin wrapper that:
      - accepts [B,1,H,W] or [B,3,H,W] (replicates gray->3ch)
      - upsamples logits back to input HxW
    """
    def __init__(self, variant: str = "b0", num_labels: int = 5,
                 ignore_index: int = 0, pretrained: bool = True):
        super().__init__()
        variant = variant.lower()
        if variant not in _VARIANTS:
            raise ValueError(f"Unknown SegFormer variant: {variant}")

        base_id = _VARIANTS[variant]
        if pretrained:
            cfg = SegformerConfig.from_pretrained(
                base_id,
                num_labels=num_labels,
                semantic_loss_ignore_index=ignore_index
            )
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                base_id, config=cfg, ignore_mismatched_sizes=True
            )
        else:
            cfg = SegformerConfig(
                num_labels=num_labels,
                semantic_loss_ignore_index=ignore_index
            )
            self.net = SegformerForSemanticSegmentation(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,H,W] or [B,3,H,W]
        B, C, H, W = x.shape
        if C == 1:
            x = x.repeat(1, 3, 1, 1)  # gray -> 3ch for pretrained backbone
        out = self.net(pixel_values=x)
        logits = out.logits  # [B, num_labels, H/4, W/4] (SegFormer head)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits