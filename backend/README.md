# DR Screening Backend

FastAPI service for the AppDR mobile app.

The service uses deterministic classical image processing only. It does not load
a CNN or deep-learning model. A traditional scikit-learn classifier may be loaded
as an internal tabular decision engine over handcrafted retinal measurements.
All returned stages are decision-support estimates for clinician review, not
autonomous diagnoses.

The practical dataset basis in this workspace is APTOS 2019 Blindness Detection
under `images/aptos2019/`. Its labels are `0` no DR, `1` mild, `2` moderate,
`3` severe, and `4` proliferative DR.

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
8. Extract 30 handcrafted measurements covering lesion counts/areas, vessel
   density and tortuosity, L*a*b* b-channel statistics, and multi-angle GLCM
   texture values.
9. Reject unsuitable captures before estimating the decision-support stage.
10. Map stages 0-1 to Non-Referable and stages 2-4 to Referable.
11. Return processed images, lesion overlay masks, scalar features, and lesion
   coordinates for clinical visual verification.

## Important Scope

This backend provides screening support only. It does not provide a medical
diagnosis or treatment recommendation. Current evaluation shows weaker recall
for severe and proliferative stages than for no-DR/moderate cases, so every
result requires qualified eye-care review and the frontend keeps a manual
override step.

## Handcrafted-Feature Training

APTOS-style `id_code,diagnosis` CSVs are supported. Build features from the
Downloads dataset:

```powershell
.\.venv\Scripts\python.exe dataset_builder.py --csv C:\Users\User\Downloads\train.csv --images-dir C:\Users\User\Downloads\train_images --workers 4
```

Train traditional classifiers:

```powershell
.\.venv\Scripts\python.exe train.py
```

Run the API:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Run a Celery worker when Redis is available:

```powershell
$env:APPDR_USE_CELERY="1"
celery -A app.tasks worker --loglevel=info --pool=solo
```
