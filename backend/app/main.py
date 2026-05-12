from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.pipeline import analyze_image
from app.schemas import AnalyzeResponse

app = FastAPI(
    title="DR Screening Classical Image Processing API",
    version="0.1.0",
    description=(
        "Classical image processing service for referable diabetic retinopathy "
        "screening support. No CNN, deep learning, machine learning, or AI."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        output = analyze_image(image_bytes)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Image processing failed: {error}",
        ) from error

    return AnalyzeResponse(
        filename=file.filename or "uploaded-image",
        quality=output.quality,
        features=output.features,
        result=output.result,
        processed_images=output.processed_images,
    )
