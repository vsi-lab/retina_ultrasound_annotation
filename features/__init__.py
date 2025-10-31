"""
features: Region-Level Feature Extraction from Segmentation Outputs
===================================================================

Transforms predicted segmentation masks into interpretable numeric
descriptors capturing retinal detachment (RD) characteristics.

Modules:
---------
- extract_features.py : Runs a trained model and generates RD region descriptors.
- region_features.py  : Computes geometric and morphological features.

Extracted Features:
-------------------
- rd_area_frac       : Fraction of image occupied by RD region.
- rd_num_cc          : Number of connected components.
- rd_max_cc_frac     : Largest component area ratio.
- rd_perimeter_norm  : Normalized perimeter.
- rd_bbox_aspect     : Aspect ratio of RD bounding box.
- rd_center_y        : RD centroid vertical position.
- has_rd_gt          : Binary RD presence (for classifier training).

References:
------------
1. TVST 2025, *Automated Detection of Retinal Detachment using Deep Learning-based Segmentation on Ocular Ultrasonography*, PMC11875030.
"""
