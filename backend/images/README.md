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

The current trainer converts labels `1` to `4` into one DR-positive class.

## Train Command

From `backend/`:

```powershell
.\.venv\Scripts\python.exe scripts\train_feature_model.py --csv images\labels.csv
```

Restart the backend after training.
