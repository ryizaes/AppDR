# AppDR Evaluation Report

This app is a screening support tool only and does not provide a final medical diagnosis. Please consult an ophthalmologist for confirmation.

## MULTICLASS DR GRADING RESULTS

Overall accuracy: 67.98%

### Per-stage results

#### Class 0 — No apparent diabetic retinopathy

Test images: 1593
Correct predictions: 1298
Precision: 85.11%
Recall: 81.48%
F1-score: 83.26%
Most common misclassification: predicted as class 2 — Moderate non-proliferative diabetic retinopathy (202 images).

#### Class 1 — Mild non-proliferative diabetic retinopathy

Test images: 194
Correct predictions: 64
Precision: 25.91%
Recall: 32.99%
F1-score: 29.02%
Most common misclassification: predicted as class 2 — Moderate non-proliferative diabetic retinopathy (71 images).

#### Class 2 — Moderate non-proliferative diabetic retinopathy

Test images: 1082
Correct predictions: 635
Precision: 64.08%
Recall: 58.69%
F1-score: 61.26%
Most common misclassification: predicted as class 0 — No apparent diabetic retinopathy (167 images).

#### Class 3 — Severe non-proliferative diabetic retinopathy

Test images: 84
Correct predictions: 26
Precision: 27.08%
Recall: 30.95%
F1-score: 28.89%
Most common misclassification: predicted as class 4 — Proliferative diabetic retinopathy (30 images).

#### Class 4 — Proliferative diabetic retinopathy

Test images: 239
Correct predictions: 147
Precision: 44.14%
Recall: 61.51%
F1-score: 51.40%
Most common misclassification: predicted as class 2 — Moderate non-proliferative diabetic retinopathy (63 images).

### Summary

Balanced accuracy: 53.12%
Macro precision: 49.27%
Macro recall: 53.12%
Macro F1-score: 50.77%
Weighted F1-score: 68.69%

### Confusion matrix

Rows are true labels. Columns are predicted labels.

| True \ Predicted | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 0 — No apparent diabetic retinopathy | 1298 | 62 | 202 | 5 | 26 |
| 1 — Mild non-proliferative diabetic retinopathy | 48 | 64 | 71 | 1 | 10 |
| 2 — Moderate non-proliferative diabetic retinopathy | 167 | 112 | 635 | 48 | 120 |
| 3 — Severe non-proliferative diabetic retinopathy | 3 | 5 | 20 | 26 | 30 |
| 4 — Proliferative diabetic retinopathy | 9 | 4 | 63 | 16 | 147 |

## BINARY REFERABLE DR SCREENING RESULTS

Overall accuracy: 79.32%
Balanced accuracy: 80.87%
Precision: 69.70%
Recall: 93.73%
F1-score: 79.95%
Referable recall: 93.73%

### Non-referable DR

Test images: 1788
Correct predictions: 1216
Precision: 93.25%
Recall: 68.01%
F1-score: 78.65%

### Referable DR

Test images: 1404
Correct predictions: 1316
Precision: 69.70%
Recall: 93.73%
F1-score: 79.95%

### Confusion matrix

Rows are true labels. Columns are predicted labels.

| True \ Predicted | 0 | 1 |
|---|---|---|
| 0 — Non-referable DR | 1216 | 572 |
| 1 — Referable DR | 88 | 1316 |

## Interpretation

The binary referable model is stronger for screening referable disease (referable recall 93.73%) than the multiclass model is for exact five-class grading.
Class 3 / Severe non-proliferative diabetic retinopathy remains the weakest exact-grade class: 26 of 84 test images were correctly predicted (30.95% recall).
These results should be used as screening-support performance numbers, not as proof that the app can make final medical diagnoses.
