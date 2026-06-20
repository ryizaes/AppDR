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


class ClinicalBasisItem(BaseModel):
    grade: int | None = None
    medical_label: str
    clinical_reference: str
    app_mapping: str
    directly_assessed: bool = True


class SessionImageMetadata(BaseModel):
    eye: str = "unknown"
    field: str = "unknown"
    image_source: str = "unknown"


class SessionImageResult(BaseModel):
    filename: str
    metadata: SessionImageMetadata = Field(default_factory=SessionImageMetadata)
    analysis: "AnalyzeResponse"


class SessionSummary(BaseModel):
    session_id: str
    screening_result: str
    screening_label: str
    recommendation: str
    image_count: int
    poor_quality_count: int
    strongest_referable_probability: float | None = None
    strongest_image_filename: str | None = None
    max_predicted_class: int | None = None
    max_medical_label: str = ""
    average_probabilities: dict[str, float] = Field(default_factory=dict)
    patient_level_training_used: bool = False
    note: str = (
        "Session aggregation combines image-level screening outputs only. "
        "Patient-level model training is future work until true patient/session "
        "labels and image-angle metadata are available."
    )


class AnalyzeSessionResponse(BaseModel):
    session: SessionSummary
    images: list[SessionImageResult]
    disclaimer: str = (
        "This app is a screening support tool only and does not provide a final "
        "medical diagnosis. Please consult an ophthalmologist for confirmation."
    )


class UsabilityTrialFeedback(BaseModel):
    session_id: str | None = None
    image_count: int = 0
    retake_count: int = 0
    time_to_finish_seconds: float | None = None
    image_quality_warnings: list[str] = Field(default_factory=list)
    result_shown: str = ""
    ease_of_use_rating: int | None = Field(default=None, ge=1, le=5)
    result_understanding_rating: int | None = Field(default=None, ge=1, le=5)
    recommendation_clarity_rating: int | None = Field(default=None, ge=1, le=5)
    confusion_notes: str = ""
    free_text_feedback: str = ""


class UsabilityTrialFeedbackResponse(BaseModel):
    status: str
    saved: bool
    path: str | None = None


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
    medical_label: str = ""
    explanation: str = ""
    recommendation: str = ""
    reason: str
    disclaimer: str
    model_type: str = "rule_based"
    confidence: float | None = None
    confidence_label: str = "Low Confidence"
    probabilities: dict[str, float] = Field(default_factory=dict)
    screening: ScreeningTier | None = None
    screening_recommendation: str = ""
    consistency_status: str = "aligned"
    raw_binary_prediction: int | None = None
    raw_severity_prediction: int | None = None
    binary_model_source: str = ""
    severity_model_source: str = ""


class AnalyzeResponse(BaseModel):
    filename: str
    screening_result: str = "uncertain"
    screening_label: str = "Uncertain screening result"
    referable_result: str = "Uncertain screening result"
    screening_confidence: float | None = None
    screening_confidence_level: str = "low"
    referable_probability: float | None = None
    non_referable_probability: float | None = None
    predicted_class: int | None = None
    severity_grade: int | None = None
    medical_label: str = ""
    severity_label_medical: str = ""
    grade_confidence: float | None = None
    confidence: float | None = None
    explanation: str = ""
    recommendation: str = ""
    model_type: str = ""
    model_version: str = "production_handcrafted_203"
    model_mode: str = ""
    binary_model_source: str = ""
    severity_model_source: str = ""
    consistency_status: str = "aligned"
    raw_binary_prediction: int | None = None
    raw_severity_prediction: int | None = None
    clinical_basis: list[ClinicalBasisItem] = Field(default_factory=list)
    detected_supported_findings: list[str] = Field(default_factory=list)
    not_directly_assessed_findings: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "This app is a screening support tool only and does not provide a final "
        "medical diagnosis. Please consult an ophthalmologist for confirmation."
    )
    image_quality_status: dict[str, object] = Field(default_factory=dict)
    image_quality: dict[str, object] = Field(default_factory=dict)
    detected_features: dict[str, object] = Field(default_factory=dict)
    detected_feature_summary: dict[str, object] = Field(default_factory=dict)
    clinical_note: str = (
        "This result is an automated screening support output and is not a final "
        "diagnosis. Please confirm with an ophthalmologist."
    )
    limitations: list[str] = Field(default_factory=list)
    model_update_summary: dict[str, object] = Field(default_factory=dict)
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
