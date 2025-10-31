"""
usg_segmentation: Retinal Ultrasound Segmentation and RD Classification
=======================================================================

A modular deep learning package for automated detection of retinal detachment (RD)
in B-mode ocular ultrasound images.

Main Modules:
--------------
• training      – Model training, augmentation, and evaluation
• models        – UNet and TransUNet architectures
• features      – Region-level feature extraction from segmentation masks
• classify      – Lightweight RD presence classifier (LogisticRegression / RandomForest)
• utils         – Visualization utilities for augmentation and predictions
• tests         – Unit tests ensuring geometric and preprocessing consistency

Highlights:
-----------
- Dynamic augmentations using Albumentations
- TransUNet-based segmentation with Dice+Focal loss
- Region descriptors for explainable RD classification
- Clinician-oriented visual preview tools

References:
-----------
1. Chen et al., *TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation*, arXiv:2102.04306
2. Buslaev et al., *Albumentations: Fast and Flexible Image Augmentations*, Information 2020, https://albumentations.ai
3. TVST 2025, *Automated Detection of Retinal Detachment using Deep Learning-based Segmentation on Ocular Ultrasonography*, PMC11875030

Usage:
------
To train and evaluate:
    $ python -m training.train_seg --config configs/config_usg.yaml
    $ python -m training.eval_seg --config configs/config_usg.yaml --ckpt work_dir/runs/seg_transunet/best.ckpt

To extract features and classify RD presence:
    $ python -m features.extract_features ...
    $ python -m classify.train_cls ...

This package can be imported or run as standalone modules for research and reproducibility.
"""
import os
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
