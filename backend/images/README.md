# AppDR Dataset Folder

Put your retinal dataset here before training.

## Recommended Structure

```text
backend/images/
  labels.csv
  healthy/
    healthy_001.jpg
    healthy_002.jpg
  dr/
    dr_001.jpg
    dr_002.jpg
```

## CSV Format

Use `labels.csv` with two columns:

```csv
image_path,label
healthy/healthy_001.jpg,0
dr/dr_001.jpg,1
```

Labels:

- `0` = healthy / no diabetic retinopathy
- `1` = mild DR
- `2` = moderate DR
- `3` = severe DR
- `4` = proliferative DR

The current deterministic pipeline returns a DR-detected flag and a stage
estimate on the same `0` to `4` scale. No trainer or learned model is used.

## Smoke Test

From `backend/`:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_aptos_samples.py --csv images\aptos2019\labels.csv --samples-per-label 2
```
