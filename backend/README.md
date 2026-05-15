# DR Screening Backend

FastAPI service for the AppDR mobile app.

The service uses deterministic classical image processing only. It does not use
CNNs, deep learning, machine learning, or AI classification.

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
3. Build a fundus mask.
4. Assess blur, brightness, contrast, retinal field size, retinal field shape,
   and whether the crop looks like a retinal image.
5. Enhance the green channel with illumination correction and CLAHE.
6. Segment vessels and lesion candidates with classical image processing.
7. Extract global and regional retina feature measurements.
8. Reject unsuitable captures before assigning a referable/non-referable DR
   screening-support label.
9. Return processed images as base64 PNG data URLs.

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
