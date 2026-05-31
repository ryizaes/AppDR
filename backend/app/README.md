# Backend App Code

This folder contains the FastAPI application and the classical image-processing
pipeline. It returns screening-support estimates for clinician review, not
autonomous diagnoses.

## Files

- `main.py` defines the FastAPI app, CORS policy, `/health`, and `/analyze`.
- `pipeline.py` performs the full image-processing workflow.
- `schemas.py` defines the Pydantic response models shared by the API and
  pipeline.
- `__init__.py` marks the folder as a Python package.

## Pipeline Notes

`pipeline.py` starts by decoding the upload and normalizing it to a centered
square. On Android, the frontend already uploads a cropped square analysis copy;
the backend square step is a safety fallback for other clients.

The pipeline then follows the thesis six-stage flow: FOV and optic-disc masking,
green-channel CLAHE preprocessing, Frangi vessel segmentation, black-hat/Hough
microaneurysm detection, L*a*b*/Otsu exudate detection, quadrant and PAI/GLCM
feature extraction, and shallow tabular stage estimation with clinician review.

Unsuitable captures are stopped before DR stage estimation. A photo can be marked
unsuitable when it is blurry, too dark, too bright, too low contrast, not shaped
like a retinal field, fills the crop like a background surface, or has no visible
retinal vessel pattern.

## Developer Tips

- Keep API response shape changes in sync with `App.tsx`.
- Prefer adding new response fields through `schemas.py` instead of returning
  loose dictionaries from `main.py`.
- If the camera capture target changes in the mobile UI, update the Android
  cropper and `CENTER_CROP_SCALE` in `pipeline.py` together.
