"""
models: Deep Neural Architectures for Ultrasound Segmentation
=============================================================

Defines modular neural architectures for medical image segmentation,
including CNN-based and Transformer-enhanced variants.

Available Models:
-----------------
- UNet       : Classic encoder–decoder CNN with skip connections.
- TransUNet  : Transformer-augmented UNet (ViT encoder + CNN decoder).

Highlights:
------------
- Supports grayscale single-channel ultrasound input.
- Configurable depth, embedding dimension, patch size, and attention heads.
- Unified API for training and inference.

References:
------------
1. Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015.
2. Chen et al., *TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation*, arXiv:2102.04306.
"""

