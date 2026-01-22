# tests/conftest.py
# Ensure project root is importable as a module during pytest runs




# tests/conftest_usg.py
import os
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

# ROOT = Path(__file__).resolve().parents[1]
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))


def _seed_usg_synthetic_images(img_dir: Path, mask_dir: Path, n=4, size=(128, 128)):
    """
    Create tiny synthetic ultrasound-like B-mode images + 2-class masks:
    - background (0)
    - retina (1): mid strip
    - choroid (2): deeper strip
    """
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    h, w = size

    for i in range(n):
        # Ultrasound-ish speckle background
        img = np.random.normal(loc=80, scale=30, size=(h, w)).astype(np.float32)
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Simple layered mask: retina + choroid
        mask = np.zeros((h, w), np.uint8)
        # retina band
        cv2.rectangle(mask, (0, h // 3), (w, h // 2), 1, thickness=-1)
        # choroid band below
        cv2.rectangle(mask, (0, h // 2), (w, 2 * h // 3), 2, thickness=-1)

        img_path = img_dir / f"usg_{i:03d}.png"
        mask_path = mask_dir / f"usg_{i:03d}.png"

        cv2.imwrite(str(img_path), img)
        cv2.imwrite(str(mask_path), mask)


def _write_usg_metadata_csv(meta_dir: Path, img_dir: Path, mask_dir: Path):
    """
    Create tiny train / val / test CSVs.
    Column names are a best guess for SegCSV.
    Adjust them if your dataset class expects different headers.
    """
    meta_dir.mkdir(parents=True, exist_ok=True)

    # list all synthetic images
    imgs = sorted(img_dir.glob("*.png"))
    masks = sorted(mask_dir.glob("*.png"))
    assert len(imgs) == len(masks) and len(imgs) >= 3, "Need >=3 synthetic samples"

    # simple split: 2 train, 1 val, 1 test (reusing last one if needed)
    pairs = list(zip(imgs, masks))
    train_pairs = pairs[:2]
    val_pairs = pairs[2:3]
    test_pairs = pairs[3:4] or pairs[2:3]

    def _write_csv(path: Path, rows):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            # HEADER: adjust if SegCSV expects different names
            writer.writerow(["image_path", "mask_path", "patient_id"])
            for i, (img_p, mask_p) in enumerate(rows):
                # single dummy patient id
                writer.writerow([str(img_p), str(mask_p), f"Patient{i+1}"])

    _write_csv(meta_dir / "train.csv", train_pairs)
    _write_csv(meta_dir / "val.csv", val_pairs)
    _write_csv(meta_dir / "test.csv", test_pairs)


def _write_usg_test_config(work_root: Path) -> Path:
    """
    Minimal config for USG segmentation tests.
    Mirrors configs/config_usg.yaml but with:
      - tiny epochs
      - paths pointing inside work_root
    """
    cfg = {
        "seed": 42,
        "work_root": str(work_root),

        "data": {
            "num_classes": 3,
            "class_names": ["background", "retina", "choroid"],
            "labels": {
                "background": 0,
                "retina": 1,
                "choroid": 2,
            },
            # used by SegCSV via .format(**cfg)
            "work_dir": "{work_root}",
            "out_dir": "data",
            "reader": "auto",
            "train_csv": "{work_root}/metadata/train.csv",
            "val_csv": "{work_root}/metadata/val.csv",
            "test_csv": "{work_root}/metadata/test.csv",
            "resize": [512, 512],
            "normalize": "zscore",
            "despeckle": "median3",
            "fan_mask": "none",
            "return_path": True,
            "mask_mode": "grayscale",
        },

        "model": {
            "name": "unetpp",
            "encoder_name": "resnet50",
            "encoder_weights": "imagenet",
            "n_channels": 3,
            "n_skip": 3,
        },

        "aug": {
            "hflip": 0.0,
            "rotate_deg": [-5, 5],
            "scale": [0.98, 1.02],
            "translate": [-0.02, 0.02],
            "shear_deg": [-2, 2],
            "brightness": [0.95, 1.05],
            "contrast": [0.95, 1.05],
            "gaussian_noise_std": 0.01,
            "p": 0.5,
        },

        "train": {
            "batch_size": 2,
            "epochs": 1,           # tiny for CI / smoke test
            "lr": 1.0e-3,
            "weight_decay": 1.0e-4,
            "previews_per_epoch": 0,
            "early_stop_patience": 5,
            "early_stop_min_delta": 0.001,
            "num_workers": 0,
        },

        "loss": {
            "dice_weight": 0.7,
            "focal_weight": 0.3,
            "focal_gamma": 2.0,
        },

        "eval": {
            "threshold": 0.5,
        },
    }

    cfg_path = work_root / "config_usg_test.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return cfg_path


@pytest.fixture
def tmp_usg_repo(tmp_path):
    """
    Self-contained USG test repo with:
      - synthetic images + masks
      - train/val/test CSVs
      - minimal USG config YAML
    Returns (config_path, work_root)
    """
    root = tmp_path
    work_root = root / "work_dir_usg_test"

    img_dir = work_root / "images"
    mask_dir = work_root / "masks"
    meta_dir = work_root / "metadata"

    _seed_usg_synthetic_images(img_dir, mask_dir, n=4, size=(128, 128))
    _write_usg_metadata_csv(meta_dir, img_dir, mask_dir)
    cfg_path = _write_usg_test_config(work_root)

    # cd into tmp repo so relative imports (if any) behave
    os.chdir(root)

    return cfg_path, work_root