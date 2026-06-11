# AppDR

React Native mobile app plus FastAPI backend for a classical image-processing
diabetic retinopathy screening workflow.

This project does not use CNNs or deep-learning image embeddings. The backend
uses deterministic OpenCV classical image processing to extract 203 handcrafted
retinal measurements, then loads traditional ML models from `backend/results/`
for diabetic-retinopathy screening support.

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
