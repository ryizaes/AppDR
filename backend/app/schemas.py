from pydantic import BaseModel, Field


class QualityReport(BaseModel):
    is_acceptable: bool
    blur_score: float
    brightness_mean: float
    contrast_std: float
    fundus_area_ratio: float
    warnings: list[str] = Field(default_factory=list)


class FeatureReport(BaseModel):
    fundus_area: int
    vessel_density: float
    vessel_area: int
    bright_lesion_area: int
    dark_lesion_area: int
    bright_lesion_count: int = 0
    dark_lesion_count: int = 0
    microaneurysm_count: int
    microaneurysm_area: int = 0
    exudate_count: int = 0
    exudate_area: int = 0
    exudate_quadrants: list[str] = Field(default_factory=list)
    exudate_quadrant_count: int = 0
    pathology_area_index: float = 0.0
    hemorrhage_candidate_count: int
    optic_disc_area: int
    optic_disc_detected: bool = False
    mean_intensity: float
    intensity_std: float
    texture_contrast: float
    glcm_contrast: float = 0.0
    glcm_homogeneity: float = 0.0
    glcm_energy: float = 0.0
    spatial_features: list[float] = Field(default_factory=list)


class ScreeningResult(BaseModel):
    classification: str
    referable: bool
    dr_probability: float
    stage: int | None = None
    stage_label: str
    reason: str
    disclaimer: str
    model_type: str = "rule_based"
    confidence: float | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)


class AnalyzeResponse(BaseModel):
    filename: str
    quality: QualityReport
    features: FeatureReport
    result: ScreeningResult
    processed_images: dict[str, str]
