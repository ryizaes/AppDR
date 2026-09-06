# Optimeye

React Native mobile app plus FastAPI backend for a production diabetic
retinopathy screening-support workflow.

The deployed backend uses a task-specific dual-model flow: an AppDR binary SVM
for referable DR screening and a full-training hybrid 5-class XGBoost model for
severity support. The live feature payload currently reports `203` expanded
features; not `207`.

The current trained artifacts use labeled images from the local Downloads
datasets:

- `C:\Users\User\Downloads\DR_grading.csv` with images in `C:\Users\User\Downloads\DR_grading`
- `C:\Users\User\Downloads\train.csv` with images in `C:\Users\User\Downloads\train_images`
- `C:\Users\User\Downloads\test.csv` with images in `C:\Users\User\Downloads\test_images` for unlabeled smoke prediction only

The encoded classes remain `0` through `4`, mapped in the UI to:

- `0`: No apparent diabetic retinopathy
- `1`: Mild non-proliferative diabetic retinopathy
- `2`: Moderate non-proliferative diabetic retinopathy
- `3`: Severe non-proliferative diabetic retinopathy
- `4`: Proliferative diabetic retinopathy

The app is a screening/assistive tool only and is not a final diagnosis.

## Main Pieces

- `App.tsx` contains the React Native screens, camera capture flow, gallery save
  flow, backend connection checks, and result display.
- `backend/` contains the FastAPI analysis service.
- `android/app/src/main/java/com/appdr/` contains Android native code used by
  React Native, currently the gallery saver module.
- `__tests__/` contains the React Native smoke test and native module mocks.

## Production Deployment

- Google Cloud project: `project-7ae532d9-6b7d-4f6c-8db`
- Region: `asia-southeast1`
- Backend Cloud Run service: `optimeye-api`
- Backend URL: `https://optimeye-api-jmogcbpd7a-as.a.run.app`
- Website Cloud Run service: `optimeye-site`
- Website URL: `https://optimeye-site-335900035513.asia-southeast1.run.app`
- Final APK URL: `https://optimeye-site-335900035513.asia-southeast1.run.app/downloads/OPTIMEYE-v1.0.1-release.apk`
- Compatibility APK URLs: `https://optimeye-site-335900035513.asia-southeast1.run.app/downloads/optimeye.apk`, `https://optimeye-site-335900035513.asia-southeast1.run.app/downloads/drapp.apk`
- Release APK: `release/OPTIMEYE-v1.0.1-release.apk`
- Release APK SHA256: `A7AA7C85023DF753009B631DCC89A3675E9C023B249C7810DB283FF9ACDB8D6A`

Verified production `/health` reports:

- `model_mode`: `dual_model_screening_hybrid_severity`
- `dual_model_ready`: `true`
- `demo_hybrid_ready`: `true`
- `multiclass_model`: `XGBoost`
- `binary_model`: `SVM RBF`
- `binary_threshold`: `0.2`

The deployed model summary reports:

- Severity model: Full-training hybrid 5-class XGBoost
- Severity metrics: accuracy `83.85%`, balanced accuracy `72.35%`, macro F1 `74.52%`
- Screening model: AppDR binary SVM
- Screening metrics: referable recall `95.70%`, false negatives `51`, false positives `398`, F1 `83.50%`
- Metrics note: validation/research metrics, not clinical deployment validation

Verified production inference on
`backend/results/heldout_demo_images/class2_moderate_npdr_01.jpg` returned:

- `screening_result`: `referable`
- `predicted_class`: `2`
- `medical_label`: `Moderate non-proliferative diabetic retinopathy`
- `grade_confidence`: `0.9126101732254028`
- `referable_probability`: `0.8570645676759512`
- `feature_count`: `203`

## Run The Backend

From the project root:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```text
http://127.0.0.1:8000/health
```

## Run Android

From the project root:

```powershell
npm.cmd run android
```

For a physical Android phone over USB, keep the phone plugged in and run:

```powershell
& "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" reverse tcp:8000 tcp:8000
```

The app tries `http://127.0.0.1:8000` first on Android so USB reverse works
without relying on Wi-Fi routing.

## Training And Model Files

The current training pipeline validates the feature CSV, removes exact duplicate
rows and duplicate feature vectors before the train/test split, uses a
stratified split, evaluates scaler/feature-selection choices, runs Optuna
optimization, and saves all deployment artifacts.

Prepare the combined OIA-DDR/DDR + APTOS feature table:

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\prepare_combined_dataset.py --downloads-dir C:\Users\User\Downloads --output-csv features_combined.csv --workers 4 --resume
```

Train both the multiclass DR-grade model and binary referable screening model:

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\train_all_models.py --features-csv features_combined.csv --trials 50
```

If a long run is interrupted and you want to reuse completed Optuna CSVs, or if
the slow SHAP/permutation artifact step is not needed:

```powershell
.\.venv\Scripts\python.exe scripts\train_all_models.py --features-csv features_combined.csv --trials 50 --resume --skip-interpretability
```

Important artifacts:

- `backend/results/best_model.pkl`: multiclass DR-grade model
- `backend/results/binary/best_model.pkl`: binary referable screening model
- `backend/results/best_model_metadata.json`: feature order and label metadata
- `backend/results/metrics.json`: accuracy, precision, recall, F1, confusion matrix, and classification report
- `backend/results/feature_order.json`: the exact 203-feature input order

Live prediction uses the same `feature_extraction.extract_feature_payload`
feature order saved in metadata, then sends that vector through the saved
scikit-learn pipeline.

### Weak-Stage Extra Dataset Experiment

Three additional Downloads datasets were inspected without extracting them into
the project folder:

- `C:\Users\User\Downloads\archive.zip`: Diabetic Retinopathy Dataset by Sachin Kumar
- `C:\Users\User\Downloads\Imagenes.zip` plus `C:\Users\User\Downloads\idrid_labels.csv`: IDRiD grading
- `C:\Users\User\Downloads\content.zip`: Diabetic_Retinopathy_Balanced

They are safely extracted, if needed, under:

```powershell
C:\Users\User\Downloads\AppDR_extra_datasets\
```

Run the inspection and count-only report:

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\prepare_extra_weak_stage_data.py --inspect-only
```

Build the selected weak-stage feature table:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_extra_weak_stage_data.py
```

This reuses `backend/features_combined.csv`, selects only clear-label weak-stage
images, extracts the same 203 handcrafted features only for the new images, and
writes:

- `backend/features_combined_balanced.csv`
- `backend/results/selected_extra_dataset_manifest.csv`
- `backend/results/selected_extra_clean.csv`
- `backend/results/selected_extra_rejected.csv`
- `backend/results/duplicate_report.csv`
- `backend/results/extra_image_quality_report.csv`

The balanced selection used no extra class 0 or class 2 images. It added:

- Class 1: `529`
- Class 3: `582`
- Class 4: `308`

New combined counts:

- Class 0: `7966`
- Class 1: `1500`
- Class 2: `5411`
- Class 3: `1000`
- Class 4: `1500`

Medium experimental training was run with classical ML only:

```powershell
.\.venv\Scripts\python.exe train.py --features-csv features_combined_balanced.csv --results-dir results\experimental_balanced_medium --trials 10 --smoke --skip-interpretability --skip-ensembles
.\.venv\Scripts\python.exe train.py --features-csv features_combined_balanced.csv --results-dir results\experimental_balanced_medium\binary --trials 10 --binary-referable --smoke --skip-interpretability --skip-ensembles
.\.venv\Scripts\python.exe scripts\generate_balanced_reports.py
```

Balanced experiment reports are saved as:

- `backend/results/evaluation_report_balanced.md`
- `backend/results/evaluation_report_balanced.json`
- `backend/results/evaluation_report_balanced.csv`
- `backend/results/confusion_matrix_balanced.csv`
- `backend/results/screening_report_balanced.md`

Balanced multiclass result: HistGradientBoosting, overall accuracy `54.57%`,
balanced accuracy `57.21%`, macro F1 `47.90%`, and class 3 recall `68.00%`.
Because macro F1 dropped below the production baseline `50.77%`, the production
multiclass model was not replaced.

Balanced binary result: XGBoost Calibrated Isotonic, referable recall `96.46%`,
F1 `77.62%`, and false negatives `56`. This is promising for screening safety
but over-refers more non-referable cases, so it is kept experimental unless that
tradeoff is intentionally accepted later.

### Study-Based Feature Selection Experiment

Exact 5-class grading remained unstable after dataset balancing, so a separate
study-style feature audit was added instead of adding more images blindly. It
keeps the existing 203 handcrafted features and compares classical ML models
only:

- Logistic Regression baseline
- Random Forest
- XGBoost
- SVM RBF

Run it with:

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\study_feature_selection_experiments.py --skip-shap
```

Outputs are saved under:

```text
backend/results/study_feature_selection/
```

Important files:

- `feature_audit.json`
- `feature_audit_removed_features.csv`
- `feature_importance_study.csv`
- `ranked_feature_sets.json`
- `model_comparison_study.csv`
- `binary_threshold_sweep.csv`
- `study_feature_selection_report.md`
- `study_feature_selection_report.json`

The study backs up current production artifacts before it trains:

```text
backend/results/study_feature_selection/production_backup/
```

Current study result:

- Best exact-grade candidate: SVM RBF with top `100` selected features
- Accuracy: `66.02%`
- Balanced accuracy: `60.83%`
- Macro F1: `57.18%`
- Class 1 recall: `45.33%`
- Class 3 recall: `64.00%`
- Class 4 recall: `62.33%`

The best threshold sweep candidate was Random Forest with top `100` features at
threshold `0.30`, with referable recall `94.82%` and `82` false negatives. The
3-vs-4 sub-classifier did not reduce Severe NPDR vs Proliferative DR confusion.

The production app was not automatically replaced because the best deployed
screening model must remain strong and the selected-feature exact-grade model
needs a production export/integration pass before replacing the 203-feature
artifact. The app should continue to show binary referable screening as the main
result and 5-class grading as supporting information.

### Current Training Snapshot

- Raw OIA-DDR CSV counts: class `0=6266`, `1=630`, `2=4477`, `3=236`, `4=913`.
- Raw APTOS train CSV counts: class `0=1805`, `1=370`, `2=999`, `3=193`, `4=295`.
- Clean combined feature table: `backend/features_combined.csv`, `15,958` rows and exactly `203` feature columns.
- Clean OIA-DDR rows: `0=6168`, `1=630`, `2=4477`, `3=236`, `4=913`.
- Clean APTOS rows: `0=1798`, `1=341`, `2=934`, `3=182`, `4=279`.
- Clean combined counts: `0=7966`, `1=971`, `2=5411`, `3=418`, `4=1192`.
- Multiclass model: XGBoost, test macro F1 `0.5077`, balanced accuracy `0.5312`.
- Binary referable model: SVM RBF, test F1 `0.7995`, recall `0.9373`, balanced accuracy `0.8087`.
- Severe NPDR class `3` remains weak: multiclass test recall `0.3095`, so clinician review and the binary referable tier remain important.

## Recent Fixes

- Added clean API response fields for `predicted_class`, `medical_label`,
  `confidence`, `explanation`, `recommendation`, `image_quality_status`, and
  `detected_features`.
- Added screening-first API fields: `screening_result`, `screening_label`,
  `screening_confidence`, `screening_confidence_level`,
  `referable_probability`, `non_referable_probability`, `grade_confidence`, and
  `disclaimer`.
- Updated the mobile result screen so binary referable screening is the main
  result and the 5-class DR grade is shown as supporting information.
- Replaced short stage labels in the user interface with medical terminology.
- Prevented duplicate feature vectors from leaking across train/test splits.
- Added train/validation/test split reporting and feature-value sanitization for
  NaN, infinity, and invalid numeric values.
- Preserved feature names through model fit/predict for estimators such as
  LightGBM and Extra Trees.
- Disabled stale Optuna trial reuse unless `--resume` is explicitly provided.
- Added safer user-facing error messages for unreadable images and failed
  analysis.

## Analysis Crop Behavior

The square camera viewport is the analysis target. Android creates a cropped
analysis copy from that centered square and uploads the cropped copy to FastAPI.
The camera still saves the full image to the gallery. The processed `ROI` view
in the result screen shows the exact region that was analyzed.

## Verification

Useful checks from the project root:

```powershell
npm.cmd run lint
npx.cmd tsc --noEmit
npm.cmd test -- --runInBand
```

Backend smoke test against bundled APTOS-style samples:

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\smoke_aptos_samples.py --samples-per-label 2
```
