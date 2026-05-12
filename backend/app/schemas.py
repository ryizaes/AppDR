from pydantic import BaseModel, Field


class QualityReport(BaseModel):
    is_acceptable: bool
    blur_score: float
    brightness_mean: float
    contrast_std: float
    fundus_area_ratio: float
    warnings: list[str] = Field(default_factory=list)


class FeatureReport(BaseModel):
    vessel_density: float
    vessel_area: int
    bright_lesion_area: int
    dark_lesion_area: int
    microaneurysm_count: int
    hemorrhage_candidate_count: int
    optic_disc_area: int
    mean_intensity: float
    intensity_std: float
    texture_contrast: float


class ScreeningResult(BaseModel):
    classification: str
    referable: bool
    reason: str
    disclaimer: str


class AnalyzeResponse(BaseModel):
    filename: str
    quality: QualityReport
    features: FeatureReport
    result: ScreeningResult
    processed_images: dict[str, str]
