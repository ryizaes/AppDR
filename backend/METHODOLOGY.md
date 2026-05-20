# Thesis Methodology: Handcrafted Feature ML for DR Grading

## Methodology Explanation

This system grades diabetic retinopathy using a hybrid classical computer vision
and supervised machine learning pipeline. Each retinal fundus image is first
normalized to a consistent analysis size, then processed through green-channel
enhancement, CLAHE contrast normalization, denoising, and illumination
correction. Classical image processing is used to segment diagnostically
meaningful structures: microaneurysm candidates, bright exudate regions,
retinal vessels, and gray-level co-occurrence texture patterns.

The deterministic rule-based staging logic is replaced by a supervised
classifier trained on six handcrafted retinal features:

- microaneurysm count
- exudate area
- vessel density
- GLCM contrast
- GLCM homogeneity
- GLCM energy

The features are saved into `features.csv` and used to train Random Forest and
Support Vector Machine classifiers. Model selection uses stratified train/test
splitting and 5-fold StratifiedKFold cross-validation with macro F1-score,
which is appropriate for multi-class DR grading where minority stages matter.

## Feature-Based Supervised Learning

Feature-based supervised learning separates image understanding from
classification. The image processing stage converts each fundus image into a
compact numerical description of lesion load, vascular structure, and retinal
texture. The machine learning stage then learns the relationship between those
measurements and the labeled DR stage. Unlike hand-coded thresholds, the
classifier estimates decision boundaries from labeled data, allowing the model
to adapt to the dataset while remaining interpretable.

## Why Random Forest Was Selected

Random Forest is suitable for this thesis because it performs well on small to
medium tabular datasets, handles nonlinear feature interactions, is robust to
outliers, and provides feature importance scores. In DR grading, lesion count,
exudate burden, vessel density, and texture statistics may interact in nonlinear
ways. Random Forest can model those relationships without requiring deep
learning. Its feature importance output also supports thesis defense by showing
which retinal measurements contributed most to the final predictions.

## CNNs vs. Handcrafted Feature ML

A CNN learns image features directly from pixels through many trainable
convolutional layers. This usually requires large datasets, heavy computation,
GPU resources, and deep learning frameworks such as TensorFlow or PyTorch.

This project does not use CNNs or deep learning. Instead, it uses classical
computer vision to explicitly extract medically motivated features, then trains
traditional machine learning classifiers on those features. The approach is more
transparent: each input variable has a direct retinal interpretation, such as
microaneurysm count or exudate area. This makes the method easier to explain in
an undergraduate thesis and more aligned with feature-engineering-based medical
image analysis.

## Defense-Ready Wording

The proposed system is a non-deep-learning diabetic retinopathy grading
pipeline. It uses classical computer vision to extract handcrafted retinal
features and supervised machine learning to classify disease severity. The
method improves over deterministic threshold rules by learning stage boundaries
from labeled samples while preserving interpretability through explicit lesion,
vessel, and texture features. To avoid data leakage, preprocessing statistics
used by the classifier are fit only within the training folds through a
scikit-learn Pipeline. Because DR datasets are commonly imbalanced, the
classifiers use balanced class weights and are evaluated with macro F1-score,
balanced accuracy, Cohen's kappa, sensitivity, and specificity in addition to
overall accuracy.
