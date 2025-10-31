"""
classify: Study-Level RD Presence Classification
================================================

Performs binary classification (RD / No-RD) based on region-level
features extracted from predicted segmentation masks.

Modules:
---------
- train_cls.py : Trains Logistic Regression or Random Forest classifier.
- eval_cls.py  : Evaluates saved classifier on test features.

Pipeline:
----------
1. Segmentation predicts RD masks.
2. Feature extraction computes geometric descriptors.
3. Classifier learns decision boundary for RD presence.

Metrics:
---------
Precision, Recall, F1, and confusion matrix.

References:
------------
1. Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 2011.
"""