# Thesis Methodology: Handcrafted Feature ML for DR Screening Support

## Methodology Explanation

This system provides clinician-reviewed diabetic retinopathy screening support
using a hybrid classical computer vision and supervised machine learning
pipeline. Each retinal fundus image is first normalized to a consistent analysis
size, then processed through green-channel enhancement, CLAHE contrast
normalization, denoising, and illumination correction. Classical image
processing is used to segment screening-relevant structures: microaneurysm
candidates, bright exudate regions, retinal vessels, and gray-level
co-occurrence texture patterns.

The deterministic rule-based staging logic is replaced by a shallow supervised
tabular decision engine trained on an expanded handcrafted retinal feature
vector. The current implementation keeps the original six core measurements and
adds vessel morphology, color-space, and direction-aware texture descriptors:

- microaneurysm count, area, density, and mean area
- exudate count, area, density, and mean area
- vessel density, skeleton length, endpoints, branchpoints, and tortuosity
- L*a*b* b-channel statistics for yellow exudate isolation
- GLCM contrast, homogeneity, energy, and multi-angle contrast/energy values

The features are saved into `features.csv` and used to train Random Forest,
Support Vector Machine, and HistGradientBoosting classifiers. Model selection
uses stratified train/test splitting and 5-fold StratifiedKFold cross-validation
with macro F1-score, which is appropriate for multi-class DR grading where
minority stages matter.

## Feature-Based Supervised Learning

Feature-based supervised learning separates image understanding from stage
estimation. The image processing stage converts each fundus image into a compact
numerical description of lesion load, vascular structure, and retinal texture.
The machine learning stage then learns the relationship between those
measurements and the labeled DR stage. Unlike hand-coded thresholds, the shallow
tabular model estimates decision boundaries from labeled data, allowing the
system to adapt to the dataset while remaining interpretable.

## Why HistGradientBoosting Was Selected

HistGradientBoosting produced the best cross-validated macro F1 score during
the expanded-feature training run. It is still traditional tabular machine
learning, not deep learning: it operates only on handcrafted lesion, vessel,
color, and texture measurements. Random Forest is retained as a comparison model
and for feature-importance reporting because it shows which retinal
measurements are most influential.

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

The proposed system is a non-deep-learning diabetic retinopathy screening
support pipeline. It uses classical computer vision to extract handcrafted
retinal features and traditional supervised machine learning to estimate a
screening stage for clinician review. The method improves over deterministic
threshold rules by learning stage boundaries from labeled samples while
preserving interpretability through explicit lesion, vessel, and texture
features. To avoid data leakage, preprocessing statistics used by the classifier
are fit only within the training folds through a scikit-learn Pipeline. Because
DR datasets are commonly imbalanced, Random Forest and SVC use balanced class
weights, while HistGradientBoosting uses in-fold SMOTE inside the
cross-validation pipeline. The models are evaluated with macro F1-score,
balanced accuracy, Cohen's kappa, sensitivity, and specificity in addition to
overall accuracy. The system is not an autonomous diagnostic device; severe and
proliferative recall remains limited, so a specialist review and manual override
step is required before saving a clinical report.
