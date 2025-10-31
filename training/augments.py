# training/augments.py
import albumentations as A
import numpy as np

from utils.usg_transforms import augment_pair

"""
Albumentations augmentation policies for ultrasound segmentation.

Defines build_train_augs(cfg) and build_val_augs(cfg):

Train augmentations (applied dynamically each epoch):
    • Horizontal flip (p=cfg.aug.hflip)
    • Affine: rotation, scale, translation, shear
    • Brightness / contrast jitter
    • Gaussian noise injection
    • Optional despeckle filtering

Validation augmentations:
    • Deterministic resize + normalization only

References:
    • Buslaev et al., *Albumentations: Fast and Flexible Image Augmentations*, 2020.
"""

def build_train_augs(cfg):
    """
    Build stochastic, on-the-fly training augs.

    Semantics:
      - With probability `p_all` (cfg.aug.p), apply the whole augmentation block.
      - Otherwise, return the original image/mask unchanged.
      - Inside the block, individual transforms have their own probabilities.

    This yields: over many epochs, the model sees both originals and a wide
    variety of perturbed samples. It does NOT replicate the dataset 5x;
    it applies stochastic transforms at sample-fetch time.
    """
    aug = cfg.get('aug', {})
    p_all = float(aug.get('p', 0.9))

    # Geometric ranges
    rot_min, rot_max = aug.get('rotate_deg', [-5, 5])
    sx_min, sx_max   = aug.get('scale', [0.98, 1.02])
    tx_min, tx_max   = aug.get('translate', [-0.02, 0.02])
    sh_min, sh_max   = aug.get('shear_deg', [-2, 2])

    # Photometric ranges
    b_min, b_max = aug.get('brightness', [0.95, 1.05])
    c_min, c_max = aug.get('contrast',   [0.95, 1.05])
    noise_std    = float(aug.get('gaussian_noise_std', 0.0))

    # Compose with top-level probability p_all
    transforms = [
        # If the dice probability is too less, then try to CROP masks. Or ad a pre proessing step to crop mask,image pairs.
        # A.CropNonEmptyMaskIfExists(height=256, width=256, p=0.5)
        A.HorizontalFlip(p=float(aug.get('hflip', 0.0))),
        A.Affine(
            rotate=(rot_min, rot_max),
            scale=(sx_min, sx_max),
            translate_percent=(tx_min, tx_max),
            shear=(sh_min, sh_max),
            interpolation=1,  # bilinear for image
            mask_interpolation=0,  # nearest for mask
            fit_output=False, p=0.7
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(b_min - 1.0, b_max - 1.0),
            contrast_limit=(c_min - 1.0, c_max - 1.0),
            p=0.3
        ),

    ]
    if noise_std > 0:
        # Albumentations uses GaussNoise(std_range=...), not var_limit
        transforms.append(A.GaussNoise(std_range=(0.0, noise_std), p=0.2))  # type: ignore[arg-type]

    return A.Compose(
        transforms,
        p=p_all,  #  THIS gates the entire block (original shows up with 1 - p_all)
        additional_targets={'mask': 'mask'}
    )

def build_val_augs(cfg):
    """
    Validation-time transforms: only deterministic resize/normalize happen
    elsewhere in the Dataset; no randomness here. Keep as identity.
    """
    return A.Compose([], p=1.0, additional_targets={'mask': 'mask'})
