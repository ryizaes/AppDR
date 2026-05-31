from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.pipeline import analyze_image, build_analyze_response, normalize_calibration
from app.schemas import AnalyzeResponse, AnalyzeTaskResponse, AnalyzeTaskStatusResponse
from app.task_queue import get_task_status, submit_analysis

app = FastAPI(
    title="DR Screening Handcrafted Feature ML API",
    version="0.1.0",
    description=(
        "Classical image processing plus supervised handcrafted-feature machine "
        "learning for diabetic retinopathy screening support. No CNN or deep learning."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, object]:
    return {
        "name": "AppDR clinical decision-support API",
        "status": "ok",
        "scope": (
            "Semi-automated retinal screening support using handcrafted classical "
            "image-processing features and shallow tabular ML. This service does "
            "not provide an autonomous diagnosis."
        ),
        "clinical_review_required": True,
        "limitations": [
            "Stage estimates must be reviewed by a qualified eye-care professional.",
            "Severe and proliferative stages may be under-called on some images.",
            "Use /docs for API testing and /health for connectivity checks.",
        ],
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeTaskResponse)
async def analyze(
    file: UploadFile = File(...),
    clahe_clip_limit: float = Form(2.0),
    exudate_percentile: float = Form(97.5),
    exudate_local_percentile: float = Form(98.0),
) -> AnalyzeTaskResponse:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    calibration = normalize_calibration(
        {
            "clahe_clip_limit": clahe_clip_limit,
            "exudate_percentile": exudate_percentile,
            "exudate_local_percentile": exudate_local_percentile,
        },
    )
    task_id = submit_analysis(file.filename or "uploaded-image", image_bytes, calibration)

    return AnalyzeTaskResponse(
        task_id=task_id,
        status_url=f"/status/{task_id}",
        message="Image accepted for semi-automated screening support processing.",
    )


@app.get("/status/{task_id}", response_model=AnalyzeTaskStatusResponse)
def task_status(task_id: str) -> AnalyzeTaskStatusResponse:
    status = get_task_status(task_id)

    if status is None:
        raise HTTPException(status_code=404, detail="Task ID was not found.")

    return AnalyzeTaskStatusResponse(**status)


@app.post("/analyze-sync", response_model=AnalyzeResponse)
async def analyze_sync(
    file: UploadFile = File(...),
    clahe_clip_limit: float = Form(2.0),
    exudate_percentile: float = Form(97.5),
    exudate_local_percentile: float = Form(98.0),
) -> AnalyzeResponse:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    calibration = normalize_calibration(
        {
            "clahe_clip_limit": clahe_clip_limit,
            "exudate_percentile": exudate_percentile,
            "exudate_local_percentile": exudate_local_percentile,
        },
    )

    try:
        output = analyze_image(image_bytes, calibration=calibration)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Image processing failed: {error}",
        ) from error

    return build_analyze_response(file.filename or "uploaded-image", output)
