"""
utils: Visualization and Diagnostic Utilities
=============================================

Provides utility scripts for visual sanity checks, previews, and diagnostics.

Modules:
---------
- preview_augs.py         : Visualize Albumentations transformations.
- preview_predictions.py  : Compare model predictions vs ground truth.

Usage:
-------
- For clinician validation of augmentation realism.
- For inspecting segmentation results and model performance visually.

Output:
--------
Creates grid panels with grayscale images, color-coded masks, and predictions
in docs/images or work_dir/preview_*/ folders.
"""
