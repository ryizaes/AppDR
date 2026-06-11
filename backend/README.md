# DR Screening Backend

FastAPI service for the AppDR mobile app.

The service uses deterministic classical image processing only. It does not load
a CNN or deep-learning model. A traditional scikit-learn classifier may be loaded
as an internal tabular decision engine over handcrafted retinal measurements.
All returned classifications are decision-support estimates for clinician
review, not autonomous diagnoses.

The current trained artifacts use local Downloads datasets: OIA-DDR/DDR from
`C:\Users\User\Downloads\DR_grading.csv` and
`C:\Users\User\Downloads\DR_grading`, plus APTOS train from
`C:\Users\User\Downloads\train.csv` and
`C:\Users\User\Downloads\train_images`. APTOS `test.csv` has no diagnosis
column in this workspace, so it is used only for unlabeled smoke prediction.
Encoded labels are mapped to: `0` no apparent diabetic retinopathy, `1` mild
non-proliferative diabetic retinopathy, `2` moderate non-proliferative diabetic
retinopathy, `3` severe non-proliferative diabetic retinopathy, and `4`
proliferative diabetic retinopathy.

## Setup

From the project root:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Endpoints

- `GET /health` returns `{"status":"ok"}` for app connectivity checks.
- `POST /analyze` accepts an uploaded image and returns a `task_id`.
- `GET /status/{task_id}` returns the background task state and final result.
- `POST /analyze-sync` runs the same analysis directly for debugging.

## Processing Flow

1. Decode the uploaded image.
2. Crop the image to the centered square that matches the mobile capture guide.
3. Build the FOV mask and mask the optic disc as a critical exudate fail-safe.
4. Assess blur, brightness, contrast, retinal field size, retinal field shape,
   and whether the crop looks like a retinal image.
5. Extract the green channel, apply CLAHE, and denoise with a median filter.
6. Apply Frangi-style vesselness, adaptive thresholding, and morphology.
7. Detect microaneurysms with vessel-suppressed black-hat plus circular Hough
   validation, and detect exudates with L*a*b* lightness, Otsu thresholding,
   local bright-lesion gating, and optic-disc exclusion.
8. Extract the configured 203 handcrafted measurements covering lesion
   counts/areas, vessel density and tortuosity, L*a*b* statistics, multi-angle
   GLCM texture, color, frequency, wavelet, quadrant, and quality-adjusted
   features.
9. Reject unsuitable captures before estimating the decision-support DR grade.
10. Map grades 0-1 to Non-Referable and grades 2-4 to Referable.
11. Return processed images, lesion overlay masks, scalar features, and lesion
   coordinates for clinical visual verification.

## Important Scope

This backend provides screening support only. It does not provide a medical
diagnosis or treatment recommendation. Current evaluation shows weaker recall
for severe and proliferative disease than for no-DR/moderate cases, so every
result requires qualified eye-care review and the frontend keeps a manual
override step.

## Handcrafted-Feature Training

Build the combined OIA-DDR/DDR + APTOS feature table:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_combined_dataset.py --downloads-dir C:\Users\User\Downloads --output-csv features_combined.csv --workers 4 --resume
```

Train traditional classifiers:

```powershell
.\.venv\Scripts\python.exe scripts\train_all_models.py --features-csv features_combined.csv --trials 50
```

The trainer saves the multiclass model under `results/best_model.pkl` and the
binary referable model under `results/binary/best_model.pkl`. It also saves
metadata, selected features, scaler/imputer artifacts, metrics, confusion
matrices, feature importance, and misclassified-case analysis. Use `--resume`
only when continuing an interrupted run; otherwise retraining starts fresh and
does not reuse stale Optuna results.

Use this faster finalization command when completed Optuna trials exist and the
slow SHAP/permutation interpretability step should be skipped:

```powershell
.\.venv\Scripts\python.exe train.py --features-csv features_combined.csv --results-dir results\binary --trials 50 --binary-referable --resume --skip-interpretability
```

## Current Dataset And Metrics

- Raw OIA-DDR counts: `0=6266`, `1=630`, `2=4477`, `3=236`, `4=913`.
- Raw APTOS train counts: `0=1805`, `1=370`, `2=999`, `3=193`, `4=295`.
- Clean combined table: `features_combined.csv`, `15,958` labeled rows, `203`
  feature columns.
- Clean combined counts: `0=7966`, `1=971`, `2=5411`, `3=418`, `4=1192`.
- Multiclass artifact: XGBoost, test macro F1 `0.5077`, balanced accuracy
  `0.5312`.
- Binary referable artifact: SVM RBF, test F1 `0.7995`, recall `0.9373`,
  balanced accuracy `0.8087`.
- Severe NPDR/class `3` remains the weakest class with multiclass test recall
  `0.3095`; the app presents this as screening support and recommends
  ophthalmologist confirmation.

Run the API:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Run a Celery worker when Redis is available:

```powershell
$env:APPDR_USE_CELERY="1"
celery -A app.tasks worker --loglevel=info --pool=solo
```
