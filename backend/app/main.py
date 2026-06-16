import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.pipeline import (
    analyze_image,
    analyze_session_images,
    build_analyze_response,
    get_supervised_model_status,
)
from app.schemas import (
    AnalyzeResponse,
    AnalyzeSessionResponse,
    AnalyzeTaskResponse,
    AnalyzeTaskStatusResponse,
    SessionImageMetadata,
    UsabilityTrialFeedback,
    UsabilityTrialFeedbackResponse,
)
from app.task_queue import get_task_status, submit_analysis, user_safe_analysis_error


BACKEND_DIR = Path(__file__).resolve().parents[1]
TRIAL_FEEDBACK_PATH = BACKEND_DIR / "results" / "ml_full_study_upgrade" / "trial_feedback.jsonl"

app = FastAPI(
    title="DR Screening Classical Processing API",
    version="0.1.0",
    description=(
        "Automated classical retinal image processing for diabetic retinopathy "
        "screening support with required clinician review."
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
            "Automated retinal screening support using handcrafted classical "
            "image-processing features. This service does "
            "not provide an autonomous diagnosis."
        ),
        "clinical_review_required": True,
        "limitations": [
            "Screening classifications must be reviewed by a qualified eye-care professional.",
            "Severe and proliferative stages may be under-called on some images.",
            "Use /docs for API testing and /health for connectivity checks.",
        ],
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "models": get_supervised_model_status(),
    }


@app.post("/analyze", response_model=AnalyzeTaskResponse)
async def analyze(
    file: UploadFile = File(...),
) -> AnalyzeTaskResponse:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    task_id = submit_analysis(file.filename or "uploaded-image", image_bytes)

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
) -> AnalyzeResponse:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        output = analyze_image(image_bytes)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=user_safe_analysis_error(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=user_safe_analysis_error(error),
        ) from error

    return build_analyze_response(file.filename or "uploaded-image", output)


@app.post("/analyze-session", response_model=AnalyzeSessionResponse)
async def analyze_session(
    files: list[UploadFile] = File(...),
    session_id: str | None = Form(default=None),
    eyes: list[str] | None = Form(default=None),
    fields: list[str] | None = Form(default=None),
    image_sources: list[str] | None = Form(default=None),
) -> AnalyzeSessionResponse:
    if not 1 <= len(files) <= 9:
        raise HTTPException(
            status_code=400,
            detail="Please upload 1 to 9 fundus images for a session.",
        )

    session_files: list[tuple[str, bytes, SessionImageMetadata]] = []
    for index, file in enumerate(files):
        if file.content_type is None or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Please upload image files only.")
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail=f"{file.filename or 'Image'} is empty.")
        session_files.append(
            (
                file.filename or f"session-image-{index + 1}",
                image_bytes,
                SessionImageMetadata(
                    eye=form_value_at(eyes, index, "unknown"),
                    field=form_value_at(fields, index, "unknown"),
                    image_source=form_value_at(image_sources, index, "unknown"),
                ),
            )
        )

    try:
        return analyze_session_images(session_files, session_id=session_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=user_safe_analysis_error(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=user_safe_analysis_error(error)) from error


@app.post("/trial-feedback", response_model=UsabilityTrialFeedbackResponse)
def save_trial_feedback(
    feedback: UsabilityTrialFeedback = Body(...),
) -> UsabilityTrialFeedbackResponse:
    TRIAL_FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = feedback.model_dump()
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    with TRIAL_FEEDBACK_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return UsabilityTrialFeedbackResponse(
        status="ok",
        saved=True,
        path=str(TRIAL_FEEDBACK_PATH),
    )


def form_value_at(values: list[str] | None, index: int, default: str) -> str:
    if not values or index >= len(values):
        return default
    value = str(values[index]).strip()
    return value or default
