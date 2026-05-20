# AppDR

React Native mobile app plus FastAPI backend for a classical image-processing
diabetic retinopathy screening workflow.

This project does not load a CNN, deep-learning model, or saved machine-learning
model. The backend is based on a deterministic OpenCV classical
image-processing pipeline: FOV masking, optic-disc masking, green-channel
CLAHE, Scikit-Image Frangi vesselness, black-hat/Hough microaneurysm detection,
L*a*b*/Otsu exudate extraction, quadrant mapping, PAI, GLCM texture features,
and rule-based stage grading.

The dataset basis in this repo is APTOS 2019 Blindness Detection under
`backend/images/aptos2019/`, which uses the severity labels `0` no DR, `1`
mild, `2` moderate, `3` severe, and `4` proliferative DR.

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
