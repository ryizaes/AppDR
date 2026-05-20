# DR Screening Backend

FastAPI service for the AppDR mobile app.

The service uses deterministic classical image processing only. It does not load
a CNN, deep-learning model, or saved machine-learning model.

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
- `POST /analyze` accepts an uploaded image and returns quality checks,
  extracted features, a screening-support result, and processed preview images.

## Processing Flow

1. Decode the uploaded image.
2. Crop the image to the centered square that matches the mobile capture guide.
3. Build the FOV mask and mask the optic disc as a critical exudate fail-safe.
4. Assess blur, brightness, contrast, retinal field size, retinal field shape,
   and whether the crop looks like a retinal image.
5. Extract the green channel, apply CLAHE, and denoise with a median filter.
6. Apply Scikit-Image Frangi vesselness, adaptive thresholding, and morphology.
7. Detect microaneurysms with vessel-suppressed black-hat plus circular Hough
   validation, and detect exudates with L*a*b* lightness, Otsu thresholding,
   local bright-lesion gating, and optic-disc exclusion.
8. Extract quadrant spread, PAI percentage, vessel density, and GLCM contrast,
   homogeneity, and energy.
9. Reject unsuitable captures before assigning the strict rule-based DR stage.
10. Return processed images as base64 PNG data URLs.

## Important Scope

This backend provides screening support only. It does not provide a medical
diagnosis or treatment recommendation.

## Classical Pipeline Evaluation

No model is trained or loaded. To measure the deterministic pipeline against a
labeled dataset, create a CSV with `image_path,label` columns, where `label` is
`0` for healthy/no DR and `1` to `4` for DR severity. APTOS-style
`id_code,diagnosis` CSVs are also supported.

Evaluate:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_classical_pipeline.py --csv images\aptos2019\labels.csv --workers 8
```

Quick smoke test with a few bundled samples from each APTOS label:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_aptos_samples.py --samples-per-label 2
```
