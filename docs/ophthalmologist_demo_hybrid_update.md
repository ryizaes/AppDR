# Ophthalmologist Demo Hybrid Update

The backend default model mode is now `ophthalmologist_demo_hybrid` for the ophthalmologist demo. In this mode, `/analyze` and `/analyze-sync` use the hybrid pipeline instead of the previous production-only handcrafted model path when the local demo artifacts are present.

Production artifacts were backed up locally before the demo update:

- `backend/results/backup_before_ophthalmologist_demo/`

The large model artifacts remain local under ignored `backend/results/` paths and are not committed to Git:

- `backend/results/demo_ophthalmologist_update/models/cnn_best_model.pt`
- `backend/results/demo_ophthalmologist_update/models/embedding_pca.pkl`
- `backend/results/demo_ophthalmologist_update/models/hybrid_5class_best_model.pkl`
- `backend/results/demo_ophthalmologist_update/models/hybrid_binary_best_model.pkl`

## Demo Hybrid Pipeline

Image input -> quality check -> AppDR preprocessing -> 203 handcrafted feature extraction -> EfficientNet-B3 CNN inference -> CNN prediction and PCA embedding features -> hybrid 5-class grading model -> hybrid binary referable screening model -> result JSON.

## Training Pause State

- CNN source: EfficientNet-B3, 384px, ImageNet pretrained, GeM pooling
- Last completed epoch: 14
- Best validation checkpoint: epoch 13
- Best validation macro F1: 72.69%
- Resume instructions: `backend/results/full_hybrid_cnn_appdr_full_training/RESUME_TRAINING.md`

## Rebuilt Hybrid Metrics

5-class validation-selected model:

- Model: Logistic Regression
- Feature version: `v4_appdr203_best_cnn_predictions_embedding_pca64`
- Accuracy: 82.74%
- Balanced accuracy: 73.20%
- Macro precision: 74.88%
- Macro recall: 73.20%
- Macro F1: 73.93%
- Class 1 recall: 56.00%
- Class 3 recall: 62.00%
- Class 4 recall: 72.89%

Binary validation-selected model:

- Model: LightGBM
- Feature version: `v3_appdr203_best_cnn_embedding_pca128`
- Threshold: 0.09
- Referable recall: 95.96%
- False negatives: 48
- False positives: 251
- F1: 88.40%
- Balanced accuracy: 89.14%

These are validation/research metrics on the fixed full split, not final clinical deployment validation.

## Run Backend

```powershell
cd C:\Users\User\AppDR\backend
..\backend\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`MODEL_MODE` defaults to `ophthalmologist_demo_hybrid`. To roll back behavior without changing files:

```powershell
$env:MODEL_MODE = "production"
..\backend\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Run Mobile App

```powershell
cd C:\Users\User\AppDR
npm start
npm run android
```

The app result screen now shows the model update summary, medical DR labels, screening-support wording, and limitations for findings not directly assessed.
