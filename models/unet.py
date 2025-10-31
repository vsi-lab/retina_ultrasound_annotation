import torch
import torch.nn as nn
import torch.nn.functional as F

"""
UNet backbone for medical image segmentation.

Implements a classical encoder–decoder convolutional network with skip
connections.  Serves as a baseline alternative to TransUNet for grayscale
ultrasound segmentation tasks.

Reference:
    • Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015.
"""

class DoubleConv(nn.Module):
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

class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))

class UpBlock(nn.Module):
    """Upsample -> concat skip -> conv(in+skip->out) -> conv(out->out)"""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up    = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, num_classes: int = 3, base: int = 32):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base*2, base*4, base*8, base*16

        # Encoder
        self.enc1 = DoubleConv(in_ch, c1)
        self.enc2 = Down(c1, c2)
        self.enc3 = Down(c2, c3)
        self.enc4 = Down(c3, c4)

        # Bottleneck
        self.bottleneck = DoubleConv(c4, c5)

        # Decoder (use explicit in+skip channels)
        self.up4 = UpBlock(in_ch=c5, skip_ch=c4, out_ch=c4)
        self.up3 = UpBlock(in_ch=c4, skip_ch=c3, out_ch=c3)
        self.up2 = UpBlock(in_ch=c3, skip_ch=c2, out_ch=c2)
        self.up1 = UpBlock(in_ch=c2, skip_ch=c1, out_ch=c1)

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        # Skips
        s1 = self.enc1(x)   # c1
        s2 = self.enc2(s1)  # c2
        s3 = self.enc3(s2)  # c3
        s4 = self.enc4(s3)  # c4

        b  = self.bottleneck(s4)  # c5

        d4 = self.up4(b,  s4)     # c4
        d3 = self.up3(d4, s3)     # c3
        d2 = self.up2(d3, s2)     # c2
        d1 = self.up1(d2, s1)     # c1

        logits = self.head(d1)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return logits

if __name__ == "__main__":
    with torch.no_grad():
        m = UNet(in_ch=1, num_classes=3, base=32)
        x = torch.randn(2,1,512,512)
        y = m(x)
        print("OK:", y.shape)