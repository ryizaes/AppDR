from pydantic import BaseModel, Field


class QualityReport(BaseModel):
    is_acceptable: bool
    blur_score: float
    sharpness: float = 0.0
    brightness_mean: float
    contrast_std: float
    signal_to_noise_ratio: float = 0.0
    quality_score: int = 0
    quality_label: str = "Poor"
    fundus_area_ratio: float
    warnings: list[str] = Field(default_factory=list)
    retake_recommendations: list[str] = Field(default_factory=list)


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
    expanded_features: dict[str, float] = Field(default_factory=dict)


class LesionBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class LesionPoint(BaseModel):
    x: float
    y: float


class LesionRegion(BaseModel):
    bbox: LesionBox
    centroid: LesionPoint
    area: float
    contour: list[LesionPoint] = Field(default_factory=list)


class ScreeningTier(BaseModel):
    status: str
    referable: bool
    rule: str
    recommendation: str


class DetectionFinding(BaseModel):
    label: str
    detected: bool


class AnalysisHistoryEntry(BaseModel):
    image_id: str
    date_analyzed: str
    dr_stage: int | None = None
    confidence_level: str
    screening_recommendation: str


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
    confidence_label: str = "Low Confidence"
    probabilities: dict[str, float] = Field(default_factory=dict)
    screening: ScreeningTier | None = None
    screening_recommendation: str = ""


class AnalyzeResponse(BaseModel):
    filename: str
    quality: QualityReport
    features: FeatureReport
    result: ScreeningResult
    processed_images: dict[str, str]
    detected_findings: list[DetectionFinding] = Field(default_factory=list)
    history_entry: AnalysisHistoryEntry | None = None
    lesion_regions: dict[str, list[LesionRegion]] = Field(default_factory=dict)
    image_shape: dict[str, int] = Field(default_factory=dict)


class AnalyzeTaskResponse(BaseModel):
    task_id: str
    status_url: str
    message: str


class AnalyzeTaskStatusResponse(BaseModel):
    task_id: str
    state: str
    message: str
    result: AnalyzeResponse | None = None
    error: str | None = None
