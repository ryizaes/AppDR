# OPTIMEYE Dual-Model Result Update

## Active Model Mode

`MODEL_MODE=dual_model_screening_hybrid_severity`

OPTIMEYE uses task-specific model selection rather than asking one model to serve
both screening and exact severity grading.

## Model Routing

- Main referable screening: the existing AppDR binary SVM with the validated 0.20 threshold.
- Supporting severity assessment: the validation-selected full-training hybrid 5-class XGBoost model.
- Severity input: AppDR's 203 handcrafted retinal features plus prediction features from the validation-selected EfficientNet-B3 checkpoint.
- Medical severity mapping:
  - No apparent diabetic retinopathy
  - Mild non-proliferative diabetic retinopathy
  - Moderate non-proliferative diabetic retinopathy
  - Severe non-proliferative diabetic retinopathy
  - Proliferative diabetic retinopathy

Numeric model classes remain optional developer/debug data in the API. They are
not displayed on the normal result screen or the specialist review controls.

## Consistency Rule

The main screening and supporting severity outputs are checked before a result is
shown:

- A non-referable binary result paired with moderate NPDR, severe NPDR, or PDR is escalated to `Referable / Needs ophthalmologist review`.
- A referable binary result paired with no apparent DR or mild NPDR is shown as `Referable / Needs ophthalmologist review`.
- Disagreement includes the note: `The screening and severity outputs require clinical confirmation.`
- Aligned outputs are shown as `Referable DR` or `Non-referable DR`.

This prevents a moderate-or-worse medical severity assessment from appearing next
to a final non-referable message.

## User-Facing Result

The normal result screen prioritizes:

1. Screening result
2. Supporting medical severity assessment
3. Image quality
4. Clinical note
5. Limitations and supported findings

Technical model names and numeric class labels remain available only in backend
status/debug fields and are not shown in the normal UI.

## Clinical Scope

OPTIMEYE remains a screening-support system, not a final diagnosis. Image quality
and field of view can affect results. Venous beading, IRMA, neovascularization,
and vitreous or preretinal hemorrhage are not directly assessed unless separately
validated. Every result requires ophthalmologist confirmation.

## Local Artifact Paths

- Binary screening: `backend/results/binary/best_model.pkl`
- Binary threshold: `backend/results/binary/optimal_threshold.json`
- Severity model: `backend/results/full_hybrid_cnn_appdr_full_training/hybrid_5class_best_model.pkl`
- Image feature checkpoint: `backend/results/full_hybrid_cnn_appdr_full_training/cnn_sources/efficientnet_b3_full_training/best_model.pt`

Model artifacts are local deployment dependencies and are not added to Git.
