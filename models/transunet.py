import torch
import torch.nn as nn
import torch.nn.functional as F

"""
TransUNet: Transformer-enhanced U-Net encoder–decoder model.

Combines a ViT encoder with CNN decoding blocks to capture both
global contextual information and local texture structure in ultrasound images.

Config parameters (from YAML):
    • embed_dim, depth, heads, patch_size
    • num_classes, in_channels

Reference:
    • Chen et al., *TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation*, arXiv 2102.04306.
"""


class DoubleConv(nn.Module):
    """(Conv-BN-ReLU) x2"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class EncoderStage(nn.Module):
    """DoubleConv + optional downsample"""
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = True):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2) if downsample else nn.Identity()
    def forward(self, x):
        y = self.conv(x)
        y_down = self.pool(y)
        return y, y_down


class DecoderBlock(nn.Module):
    """Upsample (outside) -> concat(skip) -> conv(in+skip->out) -> conv(out->out)"""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
    def forward(self, x_up: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        if x_up.shape[-2:] != x_skip.shape[-2:]:
            x_up = F.interpolate(x_up, size=x_skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x_up, x_skip], dim=1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


# ----------------------------
# Lightweight "TransUNet"
# ----------------------------

class MHSAEncoder(nn.Module):
    """
    Transformer encoder applied on the deepest encoder feature map.
    We project e4 (C=c4) -> embed_dim, flatten to (HW, B, C), run N layers,
    then project back to c4 so the UNet decoder can consume it.
    """
    def __init__(self, c_in: int, embed_dim: int = 256, depth: int = 4, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.proj_in  = nn.Conv2d(c_in, embed_dim, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim*4,
            batch_first=False, dropout=dropout, activation='gelu', norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.proj_out = nn.Conv2d(embed_dim, c_in, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        b, _, h, w = x.shape
        z = self.proj_in(x)                  # [B, E, H, W]
        z = z.flatten(2).permute(2, 0, 1)    # [HW, B, E] for Transformer
        z = self.encoder(z)                  # [HW, B, E]
        z = z.permute(1, 2, 0).reshape(b, -1, h, w)  # [B, E, H, W]
        z = self.proj_out(z)                 # [B, C, H, W]
        return z


class TransUNet(nn.Module):
    """
    UNet backbone with a Transformer encoder on the deepest level.
    - Robust channel math for decoder (concat of upsampled + skip).
    - Works for 1-channel grayscale by default.
    """
    def __init__(self, in_ch: int = 1, num_classes: int = 3, base: int = 32,
                 embed_dim: int = 256, depth: int = 4, heads: int = 8):
        super().__init__()

        # Encoder pyramid
        c1, c2, c3, c4 = base, base*2, base*4, base*8
        self.enc1 = EncoderStage(in_ch, c1, downsample=True)
        self.enc2 = EncoderStage(c1, c2, downsample=True)
        self.enc3 = EncoderStage(c2, c3, downsample=True)
        self.enc4 = EncoderStage(c3, c4, downsample=False)  # deepest stage, no further pooling

        # Transformer on deepest features
        self.trans_enc = MHSAEncoder(c_in=c4, embed_dim=embed_dim, depth=depth, heads=heads)

        # Decoder (explicit channels: in_ch + skip_ch)
        self.up4  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # to e3 size
        self.dec4 = DecoderBlock(in_ch=c4, skip_ch=c3, out_ch=c3)

        self.up3  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # to e2 size
        self.dec3 = DecoderBlock(in_ch=c3, skip_ch=c2, out_ch=c2)

        self.up2  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # to e1 size
        self.dec2 = DecoderBlock(in_ch=c2, skip_ch=c1, out_ch=c1)

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder with skips
        e1, x1 = self.enc1(x)     # c1
        e2, x2 = self.enc2(x1)    # c2
        e3, x3 = self.enc3(x2)    # c3
        e4, _  = self.enc4(x3)    # c4

        # Transformer at deepest level
        t = self.trans_enc(e4)    # still c4, same spatial as e4

        # Decoder
        d4 = self.up4(t);  d4 = self.dec4(d4, e3)  # -> c3
        d3 = self.up3(d4); d3 = self.dec3(d3, e2)  # -> c2
        d2 = self.up2(d3); d2 = self.dec2(d2, e1)  # -> c1

        logits = self.head(d2)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return logits


# Smoke test
if __name__ == "__main__":
    with torch.no_grad():
        m = TransUNet(in_ch=1, num_classes=3, base=32, embed_dim=256, depth=2, heads=4)
        x = torch.randn(2, 1, 512, 512)
        y = m(x)
        print("OK:", y.shape)
