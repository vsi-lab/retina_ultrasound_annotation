### TESTS : [![ci](https://github.com/vsi-lab/retina_ultrasound_annotation/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/vsi-lab/retina_ultrasound_annotation/actions/workflows/ci.yml)

# Retinal Ultrasound Segmentation

This repository implements a segmentation pipeline for **retinal ultrasound (USG)** imaging.  
It focuses on delineating *retinal detachment* and related anatomical layers using modern architectures like **TransUNet** and **U-Net**.  
The project emphasizes:
- High-quality preprocessing and augmentation of clinical ultrasound data,
- Consistent data/model configuration,
- Clinician-readable augmentation previews for validation.

Note : This project closely follows the pipeline described in the TVST 2025 study on Automated Detection of Retinal Detachment using Deep Learning-based Segmentation on Ocular Ultrasonography (PMCID: PMC11875030)
https://pmc.ncbi.nlm.nih.gov/articles/PMC11875030/ 

---

## Note on Models to train
- Unet++ : Gold standard baseline for segmentation
- TransUnet : A very strong option here as noted in TVST 2025 study.
- MedSAM / 2 : 

Time permitting, these ones too
- SegFormer / Mask2Former: These are newer, pure-transformer architectures. They are replacing U-Net in many benchmarks and available on huggingface
- DeepLabV3+ : Available on Huggungface


### How to get Pretrained TransUnet
```
# as a subfolder
mkdir -p third_party && cd third_party

[//]: # (git clone https://github.com/Beckschen/TransUNet.git transunet)
Or better 
# TransUNet (pinned to a known-good commit)
pip install "git+https://github.com/Beckschen/TransUNet@192e441f2e2979ea289b4f521dd6d07a2d0f1f5f#egg=transunet"

cd ../work_dir
mkdir "vit_checkpoint/imagenet21k"  && cd "vit_checkpoint/imagenet21k"
cd vit_checkpoint/imagenet21k

# Hybrid ResNet50 + ViT-B/16 (TransUNet paper’s common choice)
curl -L  'https://storage.googleapis.com/vit_models/imagenet21k/R50%2BViT-B_16.npz' -o R50+ViT-B_16.npz
```
---

## Quick End-to-End Summary

| **Stage** | **Input** | **Output** | **Goal / Description**                                                                                                                |
|------------|------------|------------|---------------------------------------------------------------------------------------------------------------------------------------|
|  **Segmentation** | Raw ultrasound images (`work_dir/images/`) | Retinal layer and RD masks (`work_dir/runs/seg/`) | Perform pixel-level segmentation using TransUNet / U-Net to delineate retinal detachment regions.                                     |
|  **Feature Extraction** | Segmented masks + corresponding images (`work_dir/images/`, `work_dir/runs/seg/`) | Feature vectors (`work_dir/features/*.parquet`) | Converts the predicted mask into a compact set of region-level features (area fraction, connected components, perimeter, etc.).       |
|  **Classification** | Extracted feature vectors (`train/val/test_feats.parquet`) | Disease label: *RD* or *Normal* (`work_dir/runs/cls_rd/`) | Train a lightweight classifier (e.g., RandomForest, XGBoost) to determine presence of retinal detachment based on extracted features. |


---

####  Pipeline Overview

```text
Ultrasound Image
      │
      ▼
[Segmentation Model: TransUNet]
      │
      ▼
Segmentation Mask
      │
      ▼
[Feature Extraction Module]
      │
      ▼
Feature Vectors (.parquet)
      │
      ▼
[Classification Model]
      │
      ▼
Disease Prediction → RD / Normal
```

---



## Project Structure
```
usg_segmentation/
├── .venv/                         # Virtual environment (excluded from version control)
│
├── classify/                      # Classification stage (RD vs Normal)
│   ├── eval_cls.py                # Evaluate trained classifier on test features
│   ├── predict_cls.py             # Predict RD/Normal on new feature data
│   └── train_cls.py               # Train classifier on extracted features
│
├── configs/
│   └── config_usg.yaml            # Central experiment configuration (data, model, aug, train)
│
├── docs/                          # Documentation and visual outputs
│   └── images/                    # e.g., preview_panel.png, diagrams, etc.
│
├── features/                      # Feature extraction modules
│   ├── extract_features.py        # End-to-end feature pipeline from masks
│   └── region_features.py         # Individual region-based feature computations
│
├── models/                        # Segmentation model definitions
│   ├── transunet.py               # Transformer-based segmentation model
│   └── unet.py                    # Baseline U-Net segmentation model
│
├── tests/                         # Unit and integration tests
│   ├── test_dataset.py            # Tests for dataset loading, CSV scanning, etc.
│   ├─  <other tests>
│
├── training/                      # Segmentation training components
│   ├── augments.py                # Augmentations : Albumentations transforms for training
│   ├── dataset.py                 # Dataset and DataLoader utilities
│   ├── eval_seg.py                # Evaluation metrics and visualization
│   ├── losses.py                  # Dice, focal, and hybrid loss functions
│   ├── metrics.py                 # IoU, Dice coefficient, and confusion matrix metrics
│   └── train_seg.py               # Core segmentation training loop
│
├── utils/                         # Helper scripts for visualization and data prep
│   ├── preview_augs.py            # Augmentation preview panel generator
│   ├── scan_to_csv.py             # Scan image/mask dirs to auto-generate CSV splits
│   ├── usg_transforms.py          # Custom preprocessing & ultrasound-specific transforms
│   └── vis.py                     # Visualization helpers for masks, contours, overlays
│
├── work_dir/                      # Working directory for outputs
│    ├──                           # Refer next section.
│
└── README.md                      # Project documentation and usage guide

```
---
### Data layout

Keep each experiment self‑contained in one folder:

```
work_dir/
 ├── aug_previews/  # Augmentation preview panels (e.g., img1__preview_panel.png)
 ├── preview_predictions/        # Preview Original, ground truth, prediction as image panel 
 │
 ├── images/
 │    ├── Patient1/
 │    │     ├── Subject 1.1.png
 │    │     ├── Subject 1.2.png
 │    │     └── ...
 │    ├── Patient2/
 │    └── ...
 ├── masks/
 │    └── (mirrors the images/ structure; same filenames)
 ├── metadata/
 │    ├── labels.csv                      # image_path, mask_path, patient_id, scan_id, diagnosis
 │    ├── train.csv                       # patient-level splits
 │    ├── val.csv
 │    ├── test.csv
 │    └── stats.csv                       # basic dataset stats
 ├── meta.json                            # Supervisely class definitions
 └── obj_class_to_machine_color.json      # Supervisely class to greyscale color mapping
```


### Create CSVs automatically
```bash
python -m utils.build_ultrasound_csvs --work_dir work_dir --config configs/config_usg.yaml
Oe 
python -m utils.build_ultrasound_csvs --work_dir work_dir --config configs/config_usg.yaml --labels_csv work_dir/metadata/disease_labels.csv
```
This pairs files by **matching filename stems** and writes:
```
work_dir/metadata/train.csv
work_dir/metadata/val.csv
work_dir/metadata/test.csv
work_dir/metadata/labels.csv
work_dir/metadata/stats.csv
```

### Point the config to your `work_dir`
Edit `configs/config_usg.yaml` so that:
```yaml
data:
  work_dir: <root>/work_dir
  train_csv: <root>/work_dir/metadata/train.csv
  val_csv:   <root>/work_dir/metadata/val.csv
  test_csv:  <root>/work_dir/metadata/test.csv
```

---

# Quickstart

```bash
# 1) Install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2) Build CSVs / Stats from work_dir
python -m utils.build_ultrasound_csvs --work_dir work_dir --config configs/config_usg.yaml --labels_csv work_dir/metadata/disease_labels.csv

# 3) train seg-only
python -m training.train_seg --config configs/config_usg.yaml --out work_dir/runs/seg_transunet

# 3a) Eval step 3)  
python -m training.eval_seg --config configs/config_usg.yaml --ckpt work_dir/runs/seg_transunet/best.ckpt --out work_dir/runs/seg_transunet/eval

# 3b) Preview panels (optional)  
python -m utils.preview_predictions --num_samples 6 --config configs/config_usg.yaml --ckpt work_dir/runs/seg_transunet/best.ckpt --eval_csv work_dir/metadata/test.csv --out_dir work_dir/preview_predictions 
 
---- OLD 

# 5) Extract region features from predicted masks  → converts predictions to 6–7 tabular features + has_rd_gt label.
python -m features.extract_features --config configs/config_usg.yaml  --ckpt work_dir/runs/seg_transunet/best.ckpt  --csv  work_dir/data/train.csv --out work_dir/features/train_feats.parquet

python -m features.extract_features --config configs/config_usg.yaml  --ckpt work_dir/runs/seg_transunet/best.ckpt  --csv  work_dir/data/val.csv   --out work_dir/features/val_feats.parquet

python -m features.extract_features --config configs/config_usg.yaml  --ckpt work_dir/runs/seg_transunet/best.ckpt  --csv  work_dir/data/test.csv  --out work_dir/features/test_feats.parquet

# 6) Train & evaluate RD detector  → learns RD presence using those features.
python -m classify.train_cls --train work_dir/features/train_feats.parquet  --val   work_dir/features/val_feats.parquet --out   work_dir/runs/cls_rd

python -m classify.eval_cls  --ckpt  work_dir/runs/cls_rd/model.joblib    --test  work_dir/features/test_feats.parquet  --out   work_dir/runs/cls_rd/eval
```

### Preview augmentations
```bash
python -m utils.preview_augs  --config configs/config_usg.yaml --n 12
```

### Preview predictions
```bash
  python -m utils.preview_predictions \
      --config   configs/config_usg.yaml \
      --ckpt     work_dir/runs/seg_transunet/best.ckpt \
      --eval_csv work_dir/metadata/test.csv \
      --out_dir  work_dir/preview_predictions \
      --num_samples 6 \
      --pred_mode prob|binary|argmax \
      --th 0.50 \
      --prob_contrast none|auto \
      --gt_mode raw|colorized
      
  python -m utils.preview_predictions \
      --config   configs/config_usg.yaml \
      --ckpt     work_dir/runs/seg_transunet/best.ckpt \
      --eval_csv work_dir/metadata/test.csv \
      --out_dir  work_dir/preview_predictions \
      --num_samples 6 \
      --pred_mode prob \
      --th 0.50 \
      --prob_contrast auto \
      --gt_mode raw      
```
**Panel output**
![Prediction Preview sample only]<img src="docs/images/prediction_panel.png" width="850">
---


# Augmentation & Preprocessing

This project uses a custom augmentation pipeline tailored for ocular ultrasound (B-scan). The same core functions are used for both training and preview, so what we see in the preview panels is what the model is trained on.

### Preprocessing (applied before augmentations)
- Despeckle (optional): data.despeckle
    -	median3 → 3×3 median filter on the image (mask unchanged).
    -	none (default) → no despeckling.
- Denomalization/resize (training only): handled inside the dataset after augs:
  - 	data.normalize: zscore|minmax|log
  - data.resize: e.g., [512, 512]

### Augmentations


###  Augmentation Summary Table

| **Augmentation** | **Affects Image** | **Affects Mask** | **Purpose / Notes** |
|------------------|---------|-------------|----------------------|
| **Despeckle** | ✔️ | ❌ | Reduces speckle noise using a 3×3 median filter; improves visual clarity of retinal layers. |
| **Rotate** | ✔️ | ✔️ | Rotates the ultrasound cone slightly to simulate probe tilt or eye rotation. |
| **Shear** | ✔️ | ✔️ | Applies lateral shear to mimic tissue deformation or probe pressure. |
| **Horizontal Flip (HFlip)** | ✔️ | ✔️ | Mirrors the scan left–right; simulates OD vs OS orientation differences. |
| **Scale** | ✔️ | ✔️ | Slightly zooms in/out to mimic changes in probe distance or magnification. |
| **Translate** | ✔️ | ✔️ | Shifts the image by a few pixels to simulate small probe or patient movements. |
| **Speckle Noise** | ✔️ | ❌ | Adds multiplicative Gaussian noise to reproduce ultrasound speckle patterns. |
| **Gamma Adjustment** | ✔️ | ❌ | Alters image brightness/contrast to simulate gain or dynamic range variations. |

---

### Commands

Generate preview panels (saved to work_dir/aug_previews/<stem>__preview_panel.png):
```bash
python -m utils.preview_augs --config configs/config_usg.yaml --n 4
 
``` 

###  Augmentation Preview Guide

 **Augmentation Preview Panel:**  
![Augmentation Preview]<img src="docs/images/preview_panel.png" width="950">

Each row in the augmentation panel displays **the image on the left** and its **corresponding segmentation mask on the right**.  
All transformations are deterministic examples (not random) to allow clinicians to visually verify the effects of augmentation on retinal structures.

---

### 1. Original (grayscale)
- The unaltered ultrasound frame exactly as read from disk.
- Mask shows anatomical regions in color, aligned pixel-for-pixel with the image.

---

### 2. Despeckle (median3)
- Applies a 3×3 median filter to reduce speckle noise and smooth high-frequency artifacts.
- Used to simulate light post-processing or denoising.
- Mask remains unchanged.

---

### 3. Rotate: ±X°
- Rotates both image and mask by a small angle (± as per config).
- Simulates probe tilt or eye rotation during examination.
- Preserves geometric alignment between mask and image.

---

### 4. Shear: ±Y°
- Applies an x-axis shear to mimic probe pressure or tissue deformation.
- Useful for model robustness against non-rigid distortions.
- Affects both image and mask identically.

---

### 5. HFlip (Horizontal Flip, p=Z)
- Mirrors the frame horizontally with probability `p`.
- Simulates left/right eye or reversed probe orientation.
- Mask flipped accordingly to preserve anatomical correspondence.

---

### 6. Scale ×S
- Zooms in or out by a small factor near 1.0 (e.g., ×0.98–×1.02).
- Represents probe distance variation or optical zoom changes.
- Cropped/padded to preserve fixed resolution.

---

### 7. Translate +ΔW, +ΔH
- Shifts both image and mask horizontally and vertically by small fractions.
- Simulates minor probe movements or patient motion.
- Ensures output remains same-sized via padding.

---

### 8. Speckle σ=...
- Adds multiplicative Gaussian noise to simulate ultrasound speckle variability.
- Only the image is affected (mask unchanged).
- Helps the model generalize across devices and acquisition settings.

---

### 9. Gamma (range [a,b])
- Nonlinear brightness adjustment to mimic gain or contrast variations.
- Midpoint of configured range is shown in the preview for determinism.
- Mask remains unaffected.

---

###  Notes
- All previewed transformations are drawn from the same config (`configs/config_usg.yaml`).
- Training uses the **exact same augmentation definitions** via shared transform modules.
- Clinicians can use the preview panel to confirm that **anatomical structures are preserved** while the model gains exposure to realistic ultrasound variability.


---
# Model Training and Evaluation
The model can be trained using TransUNet or U-Net.
Both share the same data pipeline and loss configuration.

### Metrics reported include:
- Dice coefficient
-  IoU (Jaccard Index)
-  F1-score
-  Pixel accuracy


### Loss

- Dice Loss : It is based on the Dice coefficient, a metric that measures the overlap between the predicted segmentation and the ground truth. 
- Focal loss : It is a modification of the standard cross-entropy loss that adds a focusing parameter \((\gamma )\) to down-weight the loss from easy-to-classify examples.

By combining Dice Loss and Focal Loss, the combined loss function addresses two types of imbalances: foreground-background imbalance (via Dice) and the imbalance between easy and hard examples (via Focal). 

Read up : https://www.sciencedirect.com/science/article/pii/S0895611121001750

**Read Losses outputs**
```
Training Loop
Epoch 2/2: train_loss=0.4590 val_dice_fg=0.3083 (per-class [0.455, 0.0081, 0.608]) acc=0.6922                                                                     
Best val dice (foreground avg): 0.3083

--> Average loss (Dice × 0.7 + Focal × 0.3) over the training set.
--> Forward pass on batches → compute dice + focal loss → back-prop → update model weights.
--> Higher is better (1 = perfect overlap). value 0.31 ≈ 31 % overlap shows early-stage learning.
--> per-class [Dice_bg, Dice_retina, Dice_RD]


Evaluation on test 
        dice_bg  dice_retina   dice_rd       acc                                                                                                                                                           
mean   0.455316     0.008196  0.608444  0.692165
......

--> focus on mean 
--> accuracy of 69% may be dominated by large background area
--> dice_retina is very poor, network hasn’t yet learned retina contours
--> dice_rd : ~61 %, solid retinal detach overlap detection signal 




```


**Train segmentation** (TransUNet/U-Net as per config):
```bash
python -m training.train_seg \
  --config configs/config_usg.yaml \
  --train_csv work_dir/data/train.csv \
  --val_csv   work_dir/data/val.csv \
  --out       work_dir/runs/seg_transunet
```

---

# Segmentation, Feature Extraction, and Classification Pipeline

After segmentation, the pipeline extends to **feature extraction** and **disease classification** to determine the presence of *retinal detachment (RD)* from ultrasound images.

---

### a) Segmentation using TransUNet

The first stage performs **pixel-level segmentation** of the retinal layers and detachment regions.

- **Model:** TransUNet or U-Net (configurable)
- **Input:** Grayscale ocular ultrasound image (resized to 512×512)
- **Output:** Semantic mask with classes — `background`, `retina_sclera`, and `retinal_detachment`

Training:
```bash
python -m training.train_seg --config configs/config_usg.yaml
```

Evaluation:
```bash
python -m training.evaluate --config configs/config_usg.yaml
```

### b) Feature Extraction

After segmentation, each predicted RD mask is analyzed to extract **region-level features** summarizing the geometry and spatial distribution of the detachment.

Current extracted features include:
- **rd_area_frac** – Fraction of the image area labeled as RD.  
- **rd_num_cc** – Number of connected components (distinct RD regions).  
- **rd_max_cc_frac** – Fraction of pixels in the largest connected RD region.  
- **rd_perimeter_norm** – Normalized perimeter length of the largest RD contour.  
- **rd_bbox_aspect** – Aspect ratio of the bounding box enclosing the RD region.  
- **rd_center_y** – Normalized vertical position of the RD centroid.  
- **has_rd_gt** – Ground-truth binary label for RD presence (used in downstream classification).

These features are stored as `.parquet` tables (`train_feats.parquet`, `val_feats.parquet`, etc.)
for downstream disease classification.
Features stored 
```
work_dir/features/
 ├── train_feats.parquet
 ├── val_feats.parquet
 └── test_feats.parquet
 ```


### c)  Classification — Retinal Detachment (RD) vs Normal
The final stage trains a lightweight classifier (e.g., RandomForest, XGBoost, or Logistic Regression) on the extracted features to determine whether an eye image contains retinal detachment or not.

Training
```
python -m classify.train_cls \
  --train work_dir/features/train_feats.parquet \
  --val   work_dir/features/val_feats.parquet \
  --out   work_dir/runs/cls_rd

```
Evaluation
```
python -m classify.eval_cls \
  --ckpt work_dir/runs/cls_rd/model.joblib \
  --test work_dir/features/test_feats.parquet
```

This step outputs key metrics:
- Accuracy, Precision, Recall, F1-score
- ROC-AUC for RD vs Normal discrimination

Results logged --> work_dir/runs/cls_rd/


---



#  Configuration Overview

Below is a simplified view of the main configuration file  
[`configs/config_usg.yaml`](configs/config_usg.yaml).  
Each section defines a clear part of the Retinal USG segmentation pipeline.

---

###  Data Section
```yaml
data:
  num_classes: 3                           # Number of segmentation classes (excluding background)
  class_names: ["background", "retina_sclera", "retinal_detachment"]
  labels:                                  # Mapping from class name to integer ID
    background: 0
    retina_sclera: 1
    retinal_detachment: 2
    # optic_nerve: 3                       # Uncomment if optic nerve sheath is included

  work_dir: work_dir                       # Base directory for current experiment
  images_sub: images                       # Folder containing input images
  masks_sub: masks                         # Folder containing ground-truth masks
  out_dir: data                            # Folder for generated CSVs (train/val/test)

  train_frac: 0.70                         # Split ratios
  val_frac: 0.15
  test_frac: 0.15
  stratify_by_rd: true                     # Ensure RD cases are balanced across splits

  input_mode: grayscale                    # "grayscale" for USG, RGB if needed later
  reader: auto                             # Automatically selects cv2 or PIL for reading
  train_csv: work_dir/data/train.csv
  val_csv:   work_dir/data/val.csv
  test_csv:  work_dir/data/test.csv

  resize: [512, 512]                       # Target spatial resolution (HxW)
  normalize: zscore                        # Normalization type: zscore|minmax|log
  despeckle: median3                       # Apply median filter for speckle reduction
  fan_mask: none                           # For optional sector cropping (none|left|right|center)



### Relevant config keys (configs/config_usg.yaml)
```yaml
data:
  resize: [512, 512]        # training only; preview does not resize
  normalize: zscore         # zscore|minmax|log (training only)
  despeckle: median3        # none|median3 (applied pre-augment in training; preview shows both)

aug:
  hflip: 0.2                # probability
  rotate_deg: [-5, 5]
  shear_deg: [-2, 2]
  scale: [0.98, 1.02]
  translate: [-0.02, 0.02]
  gaussian_noise_std: 0.01  # speckle σ (multiplicative)
  brightness: [0.95, 1.05]  # gamma min/max (around 1.0)
 ```

--- 
# References

- **TVST 2025 Study:** *Automated Detection of Retinal Detachment using Deep Learning-based Segmentation on Ocular Ultrasonography*  
  [PMCID: PMC11875030](https://pmc.ncbi.nlm.nih.gov/articles/PMC11875030/)

- **TransUNet:** *Transformers Make Strong Encoders for Medical Image Segmentation* (Chen et al., 2021)  
  [https://arxiv.org/abs/2102.04306](https://arxiv.org/abs/2102.04306)

- **Albumentations:** *Fast and Flexible Image Augmentations* (Buslaev et al., 2020)  
  [https://albumentations.ai](https://albumentations.ai)


# Appendix

### Inspect class distribution in your feature files

```bash
python - << 'PY'
import pandas as pd
for p in ["work_dir/features/train_feats.parquet","work_dir/features/val_feats.parquet"]:
    try:
        df = pd.read_parquet(p)
    except Exception:
        df = pd.read_csv(p)
    print("==", p)
    print("columns:", list(df.columns))
    # common label candidates:
    for col in ["label","rd_present","has_rd","y","target"]:
        if col in df.columns:
            print(col, "-> counts\n", df[col].value_counts(dropna=False), "\n")
PY
```

### If rd_present is missing, check if we wrote RD stats

```bash
python - << 'PY'
import pandas as pd
df = pd.read_parquet("work_dir/features/train_feats.parquet")
print(df.filter(regex="rd|RD|label|target|frac|pix", axis=1).head())
PY

```
