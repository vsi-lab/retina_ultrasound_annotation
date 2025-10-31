"""
training: Model Training and Evaluation Routines
================================================

Implements dataset loading, Albumentations-based data augmentation,
and supervised training for segmentation models (UNet / TransUNet).

Modules:
---------
- dataset.py       : SegCSV dataset for reading image–mask pairs and preprocessing.
- augments.py      : Online augmentations (flip, affine, brightness/contrast, noise).
- train_seg.py     : Main segmentation training script (Dice + Focal loss).
- eval_seg.py      : Validation and test evaluation routines.

Key Features:
--------------
- On-the-fly stochastic augmentations each epoch.
- Config-driven architecture for reproducibility.
- Multi-class segmentation (background / retina_sclera / RD).

References:
------------
1. Chen et al., *TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation*, arXiv:2102.04306
2. Buslaev et al., *Albumentations: Fast and Flexible Image Augmentations*, Information 2020
"""