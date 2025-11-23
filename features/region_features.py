# features/region_features.py

import cv2
import numpy as np

"""
Region-level features computed from a predicted segmentation mask.

Given an integer-labeled mask and an RD class id (`rd_label`), this module
extracts compact, geometry-based descriptors of the RD region(s). These
features are intentionally simple, fast, and robust, and they are suitable
as inputs to a downstream tabular classifier (e.g., Logistic Regression).

Features:
    - rd_area_frac      : RD area / image area
    - rd_num_cc         : number of connected RD components
    - rd_max_cc_frac    : largest RD component area / image area
    - rd_perimeter_norm : total RD contour perimeter normalized by (2*(H+W))
    - rd_bbox_aspect    : aspect ratio (width/height) of the bounding box of RD
    - rd_center_y       : vertical position of the RD bbox center in [0,1]
    
    
Compute geometric and morphological features from a predicted mask.

Given a 2-D class-index mask and the integer RD label id:
    • Computes RD area fraction and connected-component statistics.
    • Measures normalized perimeter, aspect ratio, and vertical position.

Returns a dict of region descriptors robust to scale and resolution.    
"""

def rd_region_features(mask, rd_label):
    """
    Compute RD region descriptors from a predicted class-index mask.

    Args:
        mask (np.ndarray): 2D integer array (HxW) where each pixel is a class id
                           produced by argmax over model logits.
        rd_label (int)   : Integer id for retinal detachment in the mask.

    Returns:
        dict: {
            "rd_area_frac": float,
            "rd_num_cc": int,
            "rd_max_cc_frac": float,
            "rd_perimeter_norm": float,
            "rd_bbox_aspect": float,
            "rd_center_y": float,
        }

    Implementation details:
        - Converts to a binary mask: (mask == rd_label).
        - Finds contours (cv2.findContours) to count components and perimeter.
        - Largest-component area is the max of contour areas (or CC areas).
        - Normalizes perimeter by 2*(H+W) to be scale-invariant.
        - Uses cv2.boundingRect on the binary mask to compute aspect ratio and
          the bbox center y, normalized by image height.

    Notes:
        - If there is no RD pixel, features return area=0, num_cc=0, etc.
        - For very small blobs, we may want to ignore components below a
          pixel threshold prior to computing features; add it here if needed.
    """
    m = (mask == rd_label).astype(np.uint8)
    h, w = m.shape
    area = float(m.sum()) / (h*w + 1e-6)
    if area <= 0.0:
        return {
            "rd_area_frac": 0.0, "rd_num_cc": 0, "rd_max_cc_frac": 0.0,
            "rd_perimeter_norm": 0.0, "rd_bbox_aspect": 0.0, "rd_center_y": 0.0,
        }
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_cc = len(cnts)
    areas = [cv2.contourArea(c) for c in cnts] or [0.0]
    max_cc = max(areas)
    max_cc_frac = float(max_cc) / (h*w + 1e-6)
    perim = sum(cv2.arcLength(c, True) for c in cnts)
    perim_norm = float(perim) / (2.0*(h+w) + 1e-6)
    x,y,bw,bh = cv2.boundingRect(m)
    bbox_aspect = float(bw)/(bh+1e-6)
    cy = (y + bh*0.5) / (h + 1e-6)
    return {
        "rd_area_frac": area,
        "rd_num_cc": num_cc,
        "rd_max_cc_frac": max_cc_frac,
        "rd_perimeter_norm": perim_norm,
        "rd_bbox_aspect": bbox_aspect,
        "rd_center_y": cy,
    }
