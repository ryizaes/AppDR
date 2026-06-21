import base64
import hashlib
import json
import pickle
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config as ml_config
from app.schemas import (
    AnalysisHistoryEntry,
    AnalyzeSessionResponse,
    AnalyzeResponse,
    ClinicalBasisItem,
    DetectionFinding,
    FeatureReport,
    QualityReport,
    ScreeningResult,
    SessionImageMetadata,
    SessionImageResult,
    SessionSummary,
)
from app.demo_hybrid import (
    BINARY_MODEL_PATH as DUAL_BINARY_MODEL_PATH,
    CNN_CHECKPOINT_PATH as DUAL_CNN_CHECKPOINT_PATH,
    SCREENING_MODEL_SOURCE,
    SEVERITY_MODEL_PATH as DUAL_SEVERITY_MODEL_PATH,
    SEVERITY_MODEL_SOURCE,
    classify_demo_hybrid,
)
from feature_extraction import (
    FeatureExtractionPayload,
    extract_feature_payload,
    mask_to_regions as feature_mask_to_regions,
)


@dataclass
class PipelineOutput:
    quality: QualityReport
    features: FeatureReport
    result: ScreeningResult
    processed_images: dict[str, str]
    detected_findings: list[DetectionFinding]
    lesion_regions: dict[str, list[dict[str, Any]]]
    image_shape: dict[str, int]
    image_id: str


@dataclass
class Stage0Masks:
    gray: np.ndarray
    fov_mask: np.ndarray
    optic_disc_mask: np.ndarray
    optic_disc_contour: np.ndarray | None


@dataclass
class PreprocessedImage:
    green: np.ndarray
    enhanced: np.ndarray
    denoised: np.ndarray


@dataclass
class VesselSegmentation:
    vesselness: np.ndarray
    vessels: np.ndarray


@dataclass
class LesionMasks:
    microaneurysms: np.ndarray
    exudates: np.ndarray
    microaneurysm_candidates: np.ndarray
    exudate_candidates: np.ndarray


CENTER_CROP_SCALE = 1.0
MAX_ANALYSIS_SIZE = 900
FOV_THRESHOLD = 10
FUNDUS_CROP_THRESHOLD = 12
FUNDUS_CROP_MARGIN_RATIO = 0.03
OD_BRIGHT_PERCENTILE = 95.0
OD_MIN_AREA_RATIO = 0.0015
OD_MAX_AREA_RATIO = 0.16
MA_BLACKHAT_RADIUS = 6
STAGE_SCORE_BY_STAGE = {
    0: 0.0,
    1: 25.0,
    2: 55.0,
    3: 80.0,
    4: 95.0,
}
FEATURE_VECTOR_NAMES = [
    "fundus_area",
    "vessel_density",
    "vessel_area",
    "exudate_area",
    "microaneurysm_area",
    "exudate_count",
    "microaneurysm_count",
    "exudate_quadrant_count",
    "pathology_area_index",
    "optic_disc_area",
    "mean_intensity",
    "intensity_std",
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
]
ML_MODEL_PATH = Path(__file__).resolve().parents[1] / "results" / "best_model.pkl"
ML_METADATA_PATH = Path(__file__).resolve().parents[1] / "results" / "best_model_metadata.json"
BINARY_SCREENING_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "results" / "binary" / "best_model.pkl"
)
BINARY_SCREENING_METADATA_PATH = (
    Path(__file__).resolve().parents[1] / "results" / "binary" / "best_model_metadata.json"
)
BINARY_REFERABLE_THRESHOLD_PATH = (
    Path(__file__).resolve().parents[1] / "results" / "binary" / "optimal_threshold.json"
)
DEFAULT_BINARY_REFERABLE_THRESHOLD = 0.20
ML_FEATURE_NAMES = ml_config.FEATURE_NAMES
ML_STAGE_LABELS = dict(ml_config.CLASS_NAMES)
SPATIAL_GRID_SIZE = 4
SPATIAL_FEATURE_KINDS = (
    "mask_coverage",
    "enhanced_mean",
    "enhanced_std",
    "vessel_ratio",
    "exudate_ratio",
    "microaneurysm_ratio",
)
_SUPERVISED_MODEL: Any | None = None
_SUPERVISED_MODEL_LOAD_ATTEMPTED = False
_SUPERVISED_MODEL_FEATURE_NAMES: list[str] | None = None
_SUPERVISED_MODEL_METADATA: dict[str, Any] = {}
_BINARY_SCREENING_MODEL: Any | None = None
_BINARY_SCREENING_MODEL_LOAD_ATTEMPTED = False
_BINARY_SCREENING_MODEL_METADATA: dict[str, Any] = {}
_BINARY_REFERABLE_THRESHOLD: float | None = None


INTERNAL_PROCESSING_PARAMETERS = {
    "clahe_clip_limit": 2.0,
    "exudate_percentile": 97.5,
    "exudate_local_percentile": 98.0,
}
MIN_ANALYSIS_QUALITY_SCORE = 50


def analyze_image(
    image_bytes: bytes,
    include_processed_images: bool = True,
) -> PipelineOutput:
    processing_parameters = fixed_processing_parameters()
    image_id = hashlib.sha256(image_bytes).hexdigest()[:16]
    image = prepare_analysis_image(decode_image(image_bytes))
    stage0 = stage0_fov_and_optic_disc_masking(image)
    quality = assess_quality(image, image[:, :, 1], stage0.fov_mask)
    if not quality.is_acceptable:
        return build_quality_blocked_output(
            image=image,
            stage0=stage0,
            quality=quality,
            image_id=image_id,
            include_processed_images=include_processed_images,
        )

    preprocessed = stage1_preprocess_green_channel(
        image,
        stage0.fov_mask,
        clahe_clip_limit=processing_parameters["clahe_clip_limit"],
    )
    vessels = stage2_segment_vessels(preprocessed.denoised, stage0.fov_mask)
    lesions = stage3_extract_lesions(
        image=image,
        preprocessed=preprocessed.denoised,
        fov_mask=stage0.fov_mask,
        optic_disc_mask=stage0.optic_disc_mask,
        vessels=vessels.vessels,
        processing_parameters=processing_parameters,
    )
    # Use the same resized/cropped frame as the dashboard overlays so lesion
    # coordinates align with processed_images. Raw upload bytes can differ in size.
    expanded_payload = extract_payload_from_image(image)
    features = stage4_extract_features(
        preprocessed=preprocessed.denoised,
        vessels=vessels.vessels,
        microaneurysms=lesions.microaneurysms,
        exudates=lesions.exudates,
        optic_disc_mask=stage0.optic_disc_mask,
        fov_mask=stage0.fov_mask,
        optic_disc_detected=stage0.optic_disc_contour is not None,
    )
    features.expanded_features = expanded_payload.features
    quality = add_feature_quality_warnings(quality, features)
    result = stage5_classify(features, quality, image=image)
    detected_findings = build_detection_findings(features)
    processed_images: dict[str, str] = {}

    if include_processed_images:
        overlay = create_overlay(
            image=image,
            vessels=vessels.vessels,
            microaneurysms=lesions.microaneurysms,
            exudates=lesions.exudates,
            optic_disc_mask=stage0.optic_disc_mask,
            fov_mask=stage0.fov_mask,
        )
        processed_images = {
            "original": encode_png(image),
            "fov_mask": encode_png(stage0.fov_mask),
            "optic_disc_mask": encode_png(stage0.optic_disc_mask),
            "green_channel": encode_png(preprocessed.green),
            "enhanced": encode_png(preprocessed.denoised),
            "vesselness": encode_png(vessels.vesselness),
            "vessels": encode_png(vessels.vessels),
            "microaneurysms": encode_png(lesions.microaneurysms),
            "exudates": encode_png(lesions.exudates),
            "lesion_overlay": encode_png(overlay),
        }

    return PipelineOutput(
        quality=quality,
        features=features,
        result=result,
        processed_images=processed_images,
        detected_findings=detected_findings,
        lesion_regions=lesion_regions_from_masks(
            vessels=vessels.vessels,
            microaneurysms=lesions.microaneurysms,
            exudates=lesions.exudates,
        ),
        image_shape={
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
        },
        image_id=image_id,
    )


def process_image_path(image_path: str | Path) -> dict[str, Any]:
    path = Path(image_path)
    output = analyze_image(path.read_bytes())

    return build_analyze_response(path.name, output).model_dump()


def analyze_session_images(
    files: list[tuple[str, bytes, SessionImageMetadata]],
    session_id: str | None = None,
) -> AnalyzeSessionResponse:
    if not 1 <= len(files) <= 9:
        raise ValueError("A patient session must contain 1 to 9 retinal images.")

    image_results: list[SessionImageResult] = []
    for filename, image_bytes, metadata in files:
        output = analyze_image(image_bytes, include_processed_images=False)
        analysis = build_analyze_response(filename, output)
        image_results.append(
            SessionImageResult(
                filename=filename,
                metadata=metadata,
                analysis=analysis,
            )
        )

    return aggregate_session_response(
        image_results,
        session_id=session_id or datetime.now(timezone.utc).strftime("session-%Y%m%d%H%M%S"),
    )


def build_analyze_response(filename: str, output: PipelineOutput) -> AnalyzeResponse:
    detected_finding_labels = [
        finding.label for finding in output.detected_findings if finding.detected
    ]
    detected_features = {
        "findings": detected_finding_labels,
        "feature_count": len(ML_FEATURE_NAMES),
        "expanded_feature_count": len(output.features.expanded_features or {}),
        "summary": {
            "microaneurysm_count": output.features.microaneurysm_count,
            "microaneurysm_red_lesion_indicators": output.features.microaneurysm_count,
            "hemorrhage_dark_lesion_indicators": output.features.dark_lesion_count,
            "exudate_count": output.features.exudate_count,
            "exudate_bright_lesion_indicators": output.features.exudate_count,
            "exudate_area": output.features.exudate_area,
            "vessel_density": output.features.vessel_density,
            "vessel_texture_indicators": {
                "vessel_density": output.features.vessel_density,
                "glcm_contrast": output.features.glcm_contrast,
                "glcm_homogeneity": output.features.glcm_homogeneity,
                "glcm_energy": output.features.glcm_energy,
            },
            "pathology_area_index": output.features.pathology_area_index,
            "quality_indicators": {
                "blur_score": output.quality.blur_score,
                "brightness_mean": output.quality.brightness_mean,
                "contrast_std": output.quality.contrast_std,
                "quality_score": output.quality.quality_score,
            },
        },
    }
    history_entry = AnalysisHistoryEntry(
        image_id=output.image_id,
        date_analyzed=datetime.now(timezone.utc).isoformat(),
        dr_stage=output.result.stage,
        confidence_level=output.result.confidence_label,
        screening_recommendation=output.result.screening_recommendation,
    )
    screening_fields = build_screening_response_fields(output.result, output.quality)

    return AnalyzeResponse(
        filename=filename or "uploaded-image",
        screening_result=screening_fields["screening_result"],
        screening_label=screening_fields["screening_label"],
        referable_result=str(screening_fields["screening_label"]),
        screening_confidence=screening_fields["screening_confidence"],
        screening_confidence_level=screening_fields["screening_confidence_level"],
        referable_probability=screening_fields["referable_probability"],
        non_referable_probability=screening_fields["non_referable_probability"],
        predicted_class=output.result.stage,
        severity_grade=output.result.stage,
        medical_label=output.result.medical_label or output.result.stage_label,
        severity_label_medical=output.result.medical_label or output.result.stage_label,
        grade_confidence=output.result.confidence,
        confidence=output.result.confidence,
        explanation=screening_fields["explanation"],
        recommendation=screening_fields["recommendation"],
        model_type=output.result.model_type,
        model_version=model_version_label(),
        model_mode=ml_config.MODEL_MODE,
        binary_model_source=output.result.binary_model_source,
        severity_model_source=output.result.severity_model_source,
        consistency_status=output.result.consistency_status,
        raw_binary_prediction=output.result.raw_binary_prediction,
        raw_severity_prediction=output.result.raw_severity_prediction,
        clinical_basis=clinical_basis_for_stage(output.result.stage),
        detected_supported_findings=detected_finding_labels,
        not_directly_assessed_findings=not_directly_assessed_findings(),
        disclaimer=screening_fields["disclaimer"],
        image_quality_status=build_image_quality_status(output.quality),
        image_quality=build_image_quality_status(output.quality),
        detected_features=detected_features,
        detected_feature_summary=detected_features["summary"],
        clinical_note=(
            "This result is for screening support only and is not a "
            "final diagnosis. Please confirm with an ophthalmologist."
        ),
        limitations=not_directly_assessed_findings(),
        model_update_summary=model_update_summary(),
        quality=output.quality,
        features=output.features,
        result=output.result,
        processed_images=output.processed_images,
        detected_findings=output.detected_findings,
        history_entry=history_entry,
        lesion_regions=output.lesion_regions,
        image_shape=output.image_shape,
    )


def aggregate_session_response(
    image_results: list[SessionImageResult],
    session_id: str,
) -> AnalyzeSessionResponse:
    image_count = len(image_results)
    poor_quality_count = sum(
        1
        for item in image_results
        if item.analysis.image_quality_status.get("overall") != "acceptable"
    )
    usable = [item for item in image_results if item.analysis.quality.is_acceptable]

    referable_probabilities = [
        (
            float(item.analysis.referable_probability or 0.0),
            item,
        )
        for item in image_results
    ]
    strongest_probability, strongest_item = max(
        referable_probabilities,
        key=lambda pair: pair[0],
    )
    max_stage_items = [
        item for item in image_results if item.analysis.predicted_class is not None
    ]
    max_stage_item = (
        max(max_stage_items, key=lambda item: int(item.analysis.predicted_class or 0))
        if max_stage_items
        else None
    )
    average_probabilities = average_session_probabilities(image_results)

    if poor_quality_count > image_count / 2:
        screening_result = "uncertain"
        screening_label = "Uncertain session result"
        recommendation = (
            "Most session images have limited quality. Retake images before relying "
            "on the screening-support output, or consult an ophthalmologist."
        )
    elif any(item.analysis.screening_result == "referable" for item in usable):
        screening_result = "referable"
        screening_label = "Referable diabetic retinopathy suspected in session"
        recommendation = (
            "At least one image suggests referable diabetic retinopathy. "
            "Ophthalmology evaluation is recommended."
        )
    elif usable and all(item.analysis.screening_result == "non_referable" for item in usable):
        screening_result = "non_referable"
        screening_label = "No referable diabetic retinopathy detected in session"
        recommendation = (
            "No image produced a referable screening result, but confirm with an "
            "ophthalmologist as part of routine eye care."
        )
    else:
        screening_result = "uncertain"
        screening_label = "Uncertain session result"
        recommendation = (
            "The session has mixed or low-confidence image-level outputs. "
            "Review with an ophthalmologist or capture additional angles."
        )

    summary = SessionSummary(
        session_id=session_id,
        screening_result=screening_result,
        screening_label=screening_label,
        recommendation=recommendation,
        image_count=image_count,
        poor_quality_count=poor_quality_count,
        strongest_referable_probability=strongest_probability,
        strongest_image_filename=strongest_item.filename,
        max_predicted_class=(
            int(max_stage_item.analysis.predicted_class)
            if max_stage_item and max_stage_item.analysis.predicted_class is not None
            else None
        ),
        max_medical_label=max_stage_item.analysis.medical_label if max_stage_item else "",
        average_probabilities=average_probabilities,
    )
    return AnalyzeSessionResponse(session=summary, images=image_results)


def average_session_probabilities(image_results: list[SessionImageResult]) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for item in image_results:
        for key, value in item.analysis.result.probabilities.items():
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {
        key: round(totals[key] / max(counts.get(key, 1), 1), 6)
        for key in sorted(totals)
    }


def model_version_label() -> str:
    if ml_config.uses_dual_model_mode():
        return "dual_model_appdr_binary_svm_full_training_hybrid_severity"
    status = get_supervised_model_status()
    if status.get("dual_tier_ready"):
        return "production_xgboost_grading_svm_rbf_screening_203_features"
    if status.get("binary_loaded"):
        return "production_svm_rbf_screening_203_features"
    if status.get("multiclass_loaded"):
        return "production_xgboost_grading_203_features"
    return "rule_based_classical_processing"


def clinical_basis_for_stage(stage: int | None) -> list[ClinicalBasisItem]:
    references = {
        0: (
            "No apparent diabetic retinopathy.",
            "No apparent DR is represented by low lesion evidence in the image-level feature vector.",
        ),
        1: (
            "Mild NPDR is commonly associated with microaneurysms only.",
            "The app supports this with microaneurysm/red-lesion candidate features, but they need clinician review.",
        ),
        2: (
            "Moderate NPDR may include microaneurysms or hemorrhages with hard exudates, cotton wool spots, or venous beading.",
            "The app directly supports microaneurysm, hemorrhage-candidate, hard-exudate, cotton-wool-candidate, and vessel-proxy features.",
        ),
        3: (
            "Severe NPDR is clinically associated with the 4-2-1 rule: extensive hemorrhages, venous beading, or IRMA, without PDR signs.",
            "The app has quadrant/spread and vessel-abnormality proxies, but does not directly assess venous beading or IRMA.",
        ),
        4: (
            "Proliferative DR is associated with neovascularization or vitreous/preretinal hemorrhage.",
            "The current production model has severe-lesion and vessel proxies only; it does not directly detect neovascularization or vitreous/preretinal hemorrhage.",
        ),
    }
    label = ML_STAGE_LABELS.get(int(stage), "Unstageable") if stage is not None else "Unstageable"
    clinical_reference, app_mapping = references.get(
        int(stage) if stage is not None else -1,
        (
            "The image could not be assigned a reliable DR severity support grade.",
            "The app recommends retake or ophthalmologist review.",
        ),
    )
    return [
        ClinicalBasisItem(
            grade=stage,
            medical_label=label,
            clinical_reference=clinical_reference,
            app_mapping=app_mapping,
            directly_assessed=stage in {0, 1, 2},
        )
    ]


def not_directly_assessed_findings() -> list[str]:
    return [
        "Venous beading is not directly assessed unless validated.",
        "IRMA is not directly assessed unless validated.",
        "Neovascularization is not directly assessed unless validated.",
        "Vitreous or preretinal hemorrhage is not directly assessed unless validated.",
        "Result depends on fundus image quality and field of view.",
        "Patient-level progression is not assessed without true patient/session metadata.",
    ]


def model_update_summary() -> dict[str, object]:
    if not ml_config.uses_dual_model_mode():
        return {
            "current_model_mode": "Production",
            "production_unchanged": True,
        }
    return {
        "current_model_mode": "Task-specific dual model",
        "dataset_used": "17,377 readable labeled images",
        "split": "12,163 train / 2,607 validation / 2,607 test",
        "cnn_source": "EfficientNet-B3, 384px, ImageNet pretrained, GeM pooling",
        "best_current_validation_macro_f1": "75.98% at epoch 16",
        "severity_model": "Full-training hybrid 5-class XGBoost",
        "severity_model_metrics": "Accuracy 83.85%, balanced accuracy 72.35%, macro F1 74.52%",
        "screening_model": "AppDR binary SVM",
        "screening_model_metrics": "Referable recall 95.70%, FN 51, FP 398, F1 83.50%",
        "metrics_note": "Validation/research metrics, not clinical deployment validation.",
        "production_unchanged": True,
        "rollback_available": True,
    }


def build_screening_response_fields(
    result: ScreeningResult,
    quality: QualityReport,
) -> dict[str, object]:
    """Create the mobile-facing safety-first screening summary.

    The binary referable model is the primary screening signal. The 5-class DR
    grade remains a supporting estimate, not a final diagnosis.
    """
    disclaimer = (
        "This app is a screening support tool only and does not provide a final "
        "medical diagnosis. Please consult an ophthalmologist for confirmation."
    )

    if not quality.is_acceptable:
        return {
            "screening_result": "uncertain",
            "screening_label": "Uncertain screening result",
            "screening_confidence": None,
            "screening_confidence_level": "low",
            "referable_probability": None,
            "non_referable_probability": None,
            "explanation": (
                "The result is uncertain because the retinal image quality is low. "
                "Please retake the image before relying on screening output."
            ),
            "recommendation": (
                "Uncertain result - retake the image in better focus and lighting, "
                "or consult an ophthalmologist."
            ),
            "disclaimer": disclaimer,
        }

    if result.model_type == "dual_model_unavailable":
        return {
            "screening_result": "uncertain",
            "screening_label": "Analysis models unavailable",
            "screening_confidence": None,
            "screening_confidence_level": "low",
            "referable_probability": None,
            "non_referable_probability": None,
            "explanation": result.explanation,
            "recommendation": result.recommendation,
            "disclaimer": disclaimer,
        }

    referable_probability = result.probabilities.get(
        "Referable",
        float(result.dr_probability) / 100.0,
    )
    referable_probability = float(np.clip(referable_probability, 0.0, 1.0))
    non_referable_probability = result.probabilities.get(
        "Non-Referable",
        1.0 - referable_probability,
    )
    non_referable_probability = float(np.clip(non_referable_probability, 0.0, 1.0))
    screening_confidence = max(referable_probability, non_referable_probability)
    confidence_level = screening_confidence_level(screening_confidence)
    if result.consistency_status != "aligned":
        return {
            "screening_result": "referable_review",
            "screening_label": "Referable / Needs ophthalmologist review",
            "screening_confidence": screening_confidence,
            "screening_confidence_level": confidence_level,
            "referable_probability": referable_probability,
            "non_referable_probability": non_referable_probability,
            "explanation": (
                "The screening and supporting severity outputs require clinical "
                "confirmation."
            ),
            "recommendation": (
                "Please arrange ophthalmologist review before relying on this "
                "screening-support result."
            ),
            "disclaimer": disclaimer,
        }

    if screening_confidence < 0.60:
        return {
            "screening_result": "uncertain",
            "screening_label": "Uncertain screening result",
            "screening_confidence": screening_confidence,
            "screening_confidence_level": confidence_level,
            "referable_probability": referable_probability,
            "non_referable_probability": non_referable_probability,
            "explanation": (
                "The screening model does not have enough confidence to make a "
                "clear referable/non-referable call."
            ),
            "recommendation": (
                "Uncertain result - retake the image or consult an ophthalmologist."
            ),
            "disclaimer": disclaimer,
        }

    if result.referable:
        return {
            "screening_result": "referable",
            "screening_label": "Referable DR",
            "screening_confidence": screening_confidence,
            "screening_confidence_level": confidence_level,
            "referable_probability": referable_probability,
            "non_referable_probability": non_referable_probability,
            "explanation": (
                "The screening model suggests signs that may require ophthalmology "
                "referral. The medical severity assessment is supporting information only."
            ),
            "recommendation": (
                "Referable DR suspected - ophthalmology evaluation recommended."
            ),
            "disclaimer": disclaimer,
        }

    return {
        "screening_result": "non_referable",
        "screening_label": "Non-referable DR",
        "screening_confidence": screening_confidence,
        "screening_confidence_level": confidence_level,
        "referable_probability": referable_probability,
        "non_referable_probability": non_referable_probability,
        "explanation": (
            "The screening model did not detect signs that strongly suggest "
            "referable diabetic retinopathy in this image."
        ),
        "recommendation": (
            "No urgent referral indicated by the screening model, but confirm with "
            "an ophthalmologist as part of routine eye care."
        ),
        "disclaimer": disclaimer,
    }


def screening_confidence_level(confidence: float) -> str:
    if confidence >= 0.80:
        return "high"
    if confidence >= 0.60:
        return "medium"
    return "low"


def build_image_quality_status(quality: QualityReport) -> dict[str, object]:
    """Compact mobile-facing quality summary without exposing raw internals."""
    warnings_text = " ".join(quality.warnings).lower()
    blur_status = "needs_retake" if "blur" in warnings_text else "acceptable"
    brightness_status = (
        "too_dark"
        if "dark" in warnings_text or "underexposed" in warnings_text
        else "too_bright"
        if "bright" in warnings_text or "overexposed" in warnings_text
        else "acceptable"
    )
    overall = (
        "acceptable"
        if quality.is_acceptable
        else "poor"
    )
    contrast_status = "too_low" if "contrast" in warnings_text else "acceptable"
    return {
        "overall": overall,
        "blur": blur_status,
        "brightness": brightness_status,
        "contrast": contrast_status,
        "warnings": quality.warnings,
        "retake_recommendations": quality.retake_recommendations,
        "quality_score": quality.quality_score,
        "quality_label": quality.quality_label,
    }


def fixed_processing_parameters() -> dict[str, float]:
    return dict(INTERNAL_PROCESSING_PARAMETERS)


def build_quality_blocked_output(
    image: np.ndarray,
    stage0: Stage0Masks,
    quality: QualityReport,
    image_id: str,
    include_processed_images: bool,
) -> PipelineOutput:
    result = stage5_classify(empty_feature_report(), quality)
    processed_images: dict[str, str] = {}

    if include_processed_images:
        processed_images = {
            "original": encode_png(image),
            "fov_mask": encode_png(stage0.fov_mask),
            "optic_disc_mask": encode_png(stage0.optic_disc_mask),
        }

    return PipelineOutput(
        quality=quality,
        features=empty_feature_report(),
        result=result,
        processed_images=processed_images,
        detected_findings=build_detection_findings(empty_feature_report()),
        lesion_regions={"microaneurysms": [], "exudates": [], "vessels": []},
        image_shape={
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
        },
        image_id=image_id,
    )


def empty_feature_report() -> FeatureReport:
    return FeatureReport(
        fundus_area=0,
        vessel_density=0.0,
        vessel_area=0,
        bright_lesion_area=0,
        dark_lesion_area=0,
        microaneurysm_count=0,
        microaneurysm_area=0,
        exudate_count=0,
        exudate_area=0,
        hemorrhage_candidate_count=0,
        optic_disc_area=0,
        mean_intensity=0.0,
        intensity_std=0.0,
        texture_contrast=0.0,
    )


def extract_payload_from_image(
    image: np.ndarray,
) -> FeatureExtractionPayload:
    success, buffer = cv2.imencode(".png", image)

    if not success:
        raise ValueError("Failed to encode prepared image for feature extraction.")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_file.write(buffer.tobytes())
        temp_path = Path(temp_file.name)

    try:
        processing_parameters = fixed_processing_parameters()
        return extract_feature_payload(
            temp_path,
            clahe_clip_limit=processing_parameters["clahe_clip_limit"],
            exudate_percentile=processing_parameters["exudate_percentile"],
            exudate_local_percentile=processing_parameters["exudate_local_percentile"],
        )
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def lesion_regions_from_masks(
    vessels: np.ndarray,
    microaneurysms: np.ndarray,
    exudates: np.ndarray,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "microaneurysms": feature_mask_to_regions(microaneurysms, min_area=1),
        "exudates": feature_mask_to_regions(exudates, min_area=ml_config.EXUDATE_MIN_AREA),
        "vessels": feature_mask_to_regions(vessels, min_area=20),
    }


def decode_image(image_bytes: bytes) -> np.ndarray:
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Uploaded file is not a readable image.")

    return image


def crop_center_square(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    crop_size = max(1, int(min(height, width) * CENTER_CROP_SCALE))
    y_start = max(0, (height - crop_size) // 2)
    x_start = max(0, (width - crop_size) // 2)

    return image[y_start : y_start + crop_size, x_start : x_start + crop_size]


def prepare_analysis_image(image: np.ndarray) -> np.ndarray:
    cropped = crop_to_fundus_bounds(image)
    cropped = crop_center_square(cropped)
    height, width = cropped.shape[:2]
    longest_side = max(height, width)

    if longest_side <= MAX_ANALYSIS_SIZE:
        return cropped

    scale = MAX_ANALYSIS_SIZE / longest_side
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))

    return cv2.resize(
        cropped,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )


def crop_to_fundus_bounds(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, mask = cv2.threshold(blurred, FUNDUS_CROP_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = disk_kernel(10)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contour = get_largest_contour(mask)

    if contour is None:
        return image

    height, width = image.shape[:2]
    x, y, box_width, box_height = cv2.boundingRect(contour)
    margin = int(round(min(height, width) * FUNDUS_CROP_MARGIN_RATIO))
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(width, x + box_width + margin)
    y1 = min(height, y + box_height + margin)

    if x1 <= x0 or y1 <= y0:
        return image

    return image[y0:y1, x0:x1]


def stage0_fov_and_optic_disc_masking(image: np.ndarray) -> Stage0Masks:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, fov_mask = cv2.threshold(gray, FOV_THRESHOLD, 255, cv2.THRESH_BINARY)
    fov_mask = cv2.morphologyEx(fov_mask, cv2.MORPH_CLOSE, disk_kernel(10))
    fov_mask = cv2.morphologyEx(fov_mask, cv2.MORPH_OPEN, disk_kernel(2))
    fov_mask = keep_largest_component(fov_mask)
    optic_disc_mask, optic_disc_contour = detect_optic_disc_mask(gray, fov_mask)

    return Stage0Masks(
        gray=gray,
        fov_mask=fov_mask,
        optic_disc_mask=optic_disc_mask,
        optic_disc_contour=optic_disc_contour,
    )


def detect_optic_disc_mask(
    gray: np.ndarray,
    fov_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    optic_disc_mask = np.zeros_like(gray)
    fov_pixels = gray[fov_mask > 0]

    if fov_pixels.size == 0:
        return optic_disc_mask, None

    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=17, sigmaY=17)
    threshold = float(np.percentile(blurred[fov_mask > 0], OD_BRIGHT_PERCENTILE))
    bright = np.zeros_like(gray)
    bright[(blurred >= threshold) & (fov_mask > 0)] = 255
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, disk_kernel(8))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, disk_kernel(3))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fov_area = max(float(np.count_nonzero(fov_mask)), 1.0)
    best_contour: np.ndarray | None = None
    best_score = 0.0

    for contour in contours:
        area = float(cv2.contourArea(contour))
        circularity, aspect_ratio, solidity = contour_shape_metrics(contour)

        if not OD_MIN_AREA_RATIO * fov_area <= area <= OD_MAX_AREA_RATIO * fov_area:
            continue
        if circularity < 0.18 or aspect_ratio > 3.0 or solidity < 0.35:
            continue

        candidate_mask = np.zeros_like(gray)
        cv2.drawContours(candidate_mask, [contour], -1, 255, thickness=cv2.FILLED)
        mean_brightness = float(np.mean(gray[candidate_mask > 0]))
        score = area * circularity * solidity * max(mean_brightness, 1.0)

        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return optic_disc_mask, None

    cv2.drawContours(optic_disc_mask, [best_contour], -1, 255, thickness=cv2.FILLED)
    optic_disc_mask = cv2.dilate(optic_disc_mask, disk_kernel(10), iterations=1)
    optic_disc_mask[fov_mask == 0] = 0

    return optic_disc_mask, best_contour


def stage1_preprocess_green_channel(
    image: np.ndarray,
    fov_mask: np.ndarray,
    clahe_clip_limit: float = 2.0,
) -> PreprocessedImage:
    green = image[:, :, 1]
    green_for_clahe = fill_outside_mask(green, fov_mask)
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip_limit), tileGridSize=(8, 8))
    enhanced = clahe.apply(green_for_clahe)
    denoised = cv2.medianBlur(enhanced, 3)
    enhanced[fov_mask == 0] = 0
    denoised[fov_mask == 0] = 0

    return PreprocessedImage(green=green, enhanced=enhanced, denoised=denoised)


def stage2_segment_vessels(preprocessed: np.ndarray, fov_mask: np.ndarray) -> VesselSegmentation:
    normalized = preprocessed.astype(np.float32) / 255.0
    vesselness_float = frangi_vesselness_opencv(
        normalized,
        sigmas=(1, 2, 4),
        beta=0.5,
        gamma=15.0,
    )
    vesselness = normalize_uint8(np.nan_to_num(vesselness_float, nan=0.0))
    vesselness[fov_mask == 0] = 0
    vessels = cv2.adaptiveThreshold(
        vesselness,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10,
    )
    masked_values = vesselness[fov_mask > 0]
    high_vessel_threshold = max(8.0, float(np.percentile(masked_values, 97)))
    vessels[vesselness < high_vessel_threshold] = 0
    vessels[fov_mask == 0] = 0
    vessels = cv2.morphologyEx(vessels, cv2.MORPH_CLOSE, disk_kernel(2))
    vessels = cv2.morphologyEx(vessels, cv2.MORPH_OPEN, disk_kernel(1))
    vessels = remove_small_components(vessels, min_area=8, max_area=50000)

    return VesselSegmentation(vesselness=vesselness, vessels=vessels)


def stage3_extract_lesions(
    image: np.ndarray,
    preprocessed: np.ndarray,
    fov_mask: np.ndarray,
    optic_disc_mask: np.ndarray,
    vessels: np.ndarray,
    processing_parameters: dict[str, float] | None = None,
) -> LesionMasks:
    values = processing_parameters or fixed_processing_parameters()
    microaneurysms, microaneurysm_candidates = detect_microaneurysms(
        image=image,
        preprocessed=preprocessed,
        fov_mask=fov_mask,
        optic_disc_mask=optic_disc_mask,
        vessels=vessels,
    )
    exudates, exudate_candidates = detect_exudates_lab_otsu(
        image=image,
        fov_mask=fov_mask,
        optic_disc_mask=optic_disc_mask,
        vessels=vessels,
        exudate_percentile=values["exudate_percentile"],
        exudate_local_percentile=values["exudate_local_percentile"],
    )

    return LesionMasks(
        microaneurysms=microaneurysms,
        exudates=exudates,
        microaneurysm_candidates=microaneurysm_candidates,
        exudate_candidates=exudate_candidates,
    )


def detect_microaneurysms(
    image: np.ndarray,
    preprocessed: np.ndarray,
    fov_mask: np.ndarray,
    optic_disc_mask: np.ndarray,
    vessels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vessel_exclusion = cv2.dilate(vessels, disk_kernel(2), iterations=1)
    valid_mask = build_valid_lesion_mask(fov_mask, optic_disc_mask, vessel_exclusion)
    vessel_removed = fill_masked_pixels(preprocessed, vessel_exclusion, valid_mask)
    blackhat = cv2.morphologyEx(vessel_removed, cv2.MORPH_BLACKHAT, disk_kernel(MA_BLACKHAT_RADIUS))
    blackhat[valid_mask == 0] = 0
    valid_values = blackhat[valid_mask > 0]

    if valid_values.size == 0:
        return np.zeros_like(preprocessed), np.zeros_like(preprocessed)

    threshold = max(22.0, float(np.percentile(valid_values, 99.7)))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue = image[:, :, 0].astype(np.int16)
    green = image[:, :, 1].astype(np.int16)
    red = image[:, :, 2].astype(np.int16)
    median_green = float(np.median(green[fov_mask > 0])) if np.any(fov_mask > 0) else 0.0
    dark_red_candidate = (
        (green <= median_green + 6.0)
        & (red >= blue + 3)
        & (hsv[:, :, 1] >= 18)
    )
    candidates = np.zeros_like(preprocessed)
    candidates[(blackhat >= threshold) & dark_red_candidate & (valid_mask > 0)] = 255
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, disk_kernel(1))
    hough_mask = detect_microaneurysm_circles(blackhat, candidates, valid_mask)
    strict_components = filter_microaneurysm_components(candidates, hough_mask)
    microaneurysms = strict_components
    microaneurysms[valid_mask == 0] = 0
    microaneurysms = remove_small_components(microaneurysms, min_area=8, max_area=95)

    return microaneurysms, candidates


def detect_microaneurysm_circles(
    blackhat: np.ndarray,
    candidates: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    hough_mask = np.zeros_like(blackhat)
    hough_input = cv2.GaussianBlur(blackhat, (3, 3), 0)
    circles = cv2.HoughCircles(
        hough_input,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=8,
        param1=60,
        param2=12,
        minRadius=1,
        maxRadius=8,
    )

    if circles is None:
        return hough_mask

    height, width = blackhat.shape[:2]

    for x_float, y_float, radius_float in np.round(circles[0]).astype(int):
        x = int(np.clip(x_float, 0, width - 1))
        y = int(np.clip(y_float, 0, height - 1))
        radius = int(np.clip(radius_float, 1, 8))

        if valid_mask[y, x] == 0:
            continue

        circle_mask = np.zeros_like(blackhat)
        cv2.circle(circle_mask, (x, y), radius, 255, thickness=cv2.FILLED)
        overlap = np.count_nonzero((circle_mask > 0) & (candidates > 0))
        circle_area = max(int(np.count_nonzero(circle_mask)), 1)

        if overlap / circle_area >= 0.25:
            cv2.circle(hough_mask, (x, y), radius, 255, thickness=cv2.FILLED)

    return hough_mask


def filter_microaneurysm_components(candidates: np.ndarray, hough_mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = np.zeros_like(candidates)
    has_hough = np.any(hough_mask > 0)

    for contour in contours:
        area = cv2.contourArea(contour)

        if not 8 <= area <= 95:
            continue

        circularity, aspect_ratio, solidity = contour_shape_metrics(contour)
        component_mask = np.zeros_like(candidates)
        cv2.drawContours(component_mask, [contour], -1, 255, thickness=cv2.FILLED)
        hough_overlap = np.count_nonzero((component_mask > 0) & (hough_mask > 0))

        if (
            circularity >= 0.60
            and aspect_ratio <= 1.6
            and solidity >= 0.55
            and (hough_overlap > 0 or not has_hough)
        ):
            cv2.drawContours(output, [contour], -1, 255, thickness=cv2.FILLED)

    return output


def detect_exudates_lab_otsu(
    image: np.ndarray,
    fov_mask: np.ndarray,
    optic_disc_mask: np.ndarray,
    vessels: np.ndarray,
    exudate_percentile: float = 97.5,
    exudate_local_percentile: float = 98.0,
) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    b_channel = lab[:, :, 2]
    local_bright = cv2.morphologyEx(lightness, cv2.MORPH_TOPHAT, disk_kernel(8))
    vessel_exclusion = cv2.dilate(vessels, disk_kernel(1), iterations=1)
    od_exclusion = cv2.dilate(optic_disc_mask, disk_kernel(12), iterations=1)
    interior_mask = create_interior_fundus_mask(fov_mask, margin_ratio=0.035)
    valid_mask = build_valid_lesion_mask(interior_mask, od_exclusion, vessel_exclusion)
    valid_values = lightness[valid_mask > 0]

    if valid_values.size == 0:
        return np.zeros_like(lightness), np.zeros_like(lightness)

    threshold = max(
        masked_otsu_threshold(valid_values),
        float(np.percentile(valid_values, exudate_percentile)),
    )
    local_values = local_bright[valid_mask > 0]
    local_threshold = max(14.0, float(np.percentile(local_values, exudate_local_percentile)))
    b_values = b_channel[valid_mask > 0]
    b_threshold = max(128.0, float(np.percentile(b_values, ml_config.EXUDATE_B_PERCENTILE)))
    candidates = np.zeros_like(lightness)
    candidates[
        (lightness >= threshold)
        & (local_bright >= local_threshold)
        & (b_channel >= b_threshold)
        & (valid_mask > 0)
    ] = 255
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, disk_kernel(2))
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, disk_kernel(1))
    candidates = apply_exudate_color_gate(image, candidates)
    exudates = filter_exudate_components(candidates)
    exudates[valid_mask == 0] = 0

    return exudates, candidates


def apply_exudate_color_gate(image: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue = image[:, :, 0].astype(np.int16)
    green = image[:, :, 1].astype(np.int16)
    red = image[:, :, 2].astype(np.int16)
    yellow_white = (
        (red >= 80)
        & (green >= 65)
        & (red >= blue + 8)
        & (green >= blue + 4)
        & (hsv[:, :, 1] >= 20)
    )
    gated = np.zeros_like(candidates)
    gated[(candidates > 0) & yellow_white] = 255

    return gated


def filter_exudate_components(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = np.zeros_like(mask)

    for contour in contours:
        area = cv2.contourArea(contour)

        if not 12 <= area <= 5000:
            continue

        _, aspect_ratio, solidity = contour_shape_metrics(contour)

        if aspect_ratio <= 5.5 and solidity >= 0.28:
            cv2.drawContours(output, [contour], -1, 255, thickness=cv2.FILLED)

    return output


def stage4_extract_features(
    preprocessed: np.ndarray,
    vessels: np.ndarray,
    microaneurysms: np.ndarray,
    exudates: np.ndarray,
    optic_disc_mask: np.ndarray,
    fov_mask: np.ndarray,
    optic_disc_detected: bool,
) -> FeatureReport:
    fundus_area = max(int(np.count_nonzero(fov_mask)), 1)
    vessel_area = int(np.count_nonzero(vessels))
    ma_area = int(np.count_nonzero(microaneurysms))
    exudate_area = int(np.count_nonzero(exudates))
    ma_count = count_components(microaneurysms)
    exudate_count = count_components(exudates)
    exudate_quadrants = find_exudate_quadrants(exudates, fov_mask)
    pai = float(((ma_area + exudate_area) / fundus_area) * 100.0)
    pixels = preprocessed[fov_mask > 0]
    glcm_contrast, glcm_homogeneity, glcm_energy = extract_glcm_features(
        preprocessed,
        fov_mask,
    )

    return FeatureReport(
        fundus_area=fundus_area,
        vessel_density=float(vessel_area / fundus_area),
        vessel_area=vessel_area,
        bright_lesion_area=exudate_area,
        dark_lesion_area=ma_area,
        bright_lesion_count=exudate_count,
        dark_lesion_count=ma_count,
        microaneurysm_count=ma_count,
        microaneurysm_area=ma_area,
        exudate_count=exudate_count,
        exudate_area=exudate_area,
        exudate_quadrants=exudate_quadrants,
        exudate_quadrant_count=len(exudate_quadrants),
        pathology_area_index=round(pai, 4),
        hemorrhage_candidate_count=0,
        optic_disc_area=int(np.count_nonzero(optic_disc_mask)),
        optic_disc_detected=optic_disc_detected,
        mean_intensity=float(np.mean(pixels)) if pixels.size else 0.0,
        intensity_std=float(np.std(pixels)) if pixels.size else 0.0,
        texture_contrast=glcm_contrast,
        glcm_contrast=glcm_contrast,
        glcm_homogeneity=glcm_homogeneity,
        glcm_energy=glcm_energy,
        spatial_features=extract_spatial_features(
            preprocessed,
            vessels,
            exudates,
            microaneurysms,
            fov_mask,
        ),
    )


def find_exudate_quadrants(exudates: np.ndarray, fov_mask: np.ndarray) -> list[str]:
    centroid = find_mask_centroid(fov_mask)

    if centroid is None or not np.any(exudates > 0):
        return []

    cx, cy = centroid
    y_indices, x_indices = np.where(exudates > 0)
    quadrant_masks = {
        "Top-Left": (x_indices < cx) & (y_indices < cy),
        "Top-Right": (x_indices >= cx) & (y_indices < cy),
        "Bottom-Left": (x_indices < cx) & (y_indices >= cy),
        "Bottom-Right": (x_indices >= cx) & (y_indices >= cy),
    }

    min_quadrant_pixels = max(50, int(np.count_nonzero(fov_mask) * 0.0003))

    return [
        name
        for name, quadrant_mask in quadrant_masks.items()
        if int(np.count_nonzero(quadrant_mask)) >= min_quadrant_pixels
    ]


def extract_glcm_features(gray: np.ndarray, fov_mask: np.ndarray) -> tuple[float, float, float]:
    contour = get_largest_contour(fov_mask)

    if contour is None:
        return 0.0, 0.0, 0.0

    x, y, width, height = cv2.boundingRect(contour)
    prepared = fill_outside_mask(gray, fov_mask)
    roi = prepared[y : y + height, x : x + width].astype(np.uint8)

    if roi.size == 0:
        return 0.0, 0.0, 0.0

    contrast, homogeneity, energy = glcm_texture_props(roi)

    return round(contrast, 4), round(homogeneity, 4), round(energy, 4)


def extract_spatial_features(
    enhanced: np.ndarray,
    vessels: np.ndarray,
    exudates: np.ndarray,
    microaneurysms: np.ndarray,
    fov_mask: np.ndarray,
) -> list[float]:
    height, width = enhanced.shape[:2]
    values: list[float] = []

    for row in range(SPATIAL_GRID_SIZE):
        y0 = int(round((row / SPATIAL_GRID_SIZE) * height))
        y1 = int(round(((row + 1) / SPATIAL_GRID_SIZE) * height))

        for col in range(SPATIAL_GRID_SIZE):
            x0 = int(round((col / SPATIAL_GRID_SIZE) * width))
            x1 = int(round(((col + 1) / SPATIAL_GRID_SIZE) * width))
            tile_mask = fov_mask[y0:y1, x0:x1] > 0
            tile_area = max(int(np.count_nonzero(tile_mask)), 1)
            tile_pixels = enhanced[y0:y1, x0:x1][tile_mask]

            values.extend(
                [
                    float(tile_area / max(tile_mask.size, 1)),
                    float(np.mean(tile_pixels) / 255.0) if tile_pixels.size else 0.0,
                    float(np.std(tile_pixels) / 128.0) if tile_pixels.size else 0.0,
                    float(np.count_nonzero(vessels[y0:y1, x0:x1][tile_mask]) / tile_area),
                    float(np.count_nonzero(exudates[y0:y1, x0:x1][tile_mask]) / tile_area),
                    float(
                        np.count_nonzero(microaneurysms[y0:y1, x0:x1][tile_mask])
                        / tile_area,
                    ),
                ],
            )

    return values


def stage5_classify(
    features: FeatureReport,
    quality: QualityReport,
    image: np.ndarray | None = None,
) -> ScreeningResult:
    if not quality.is_acceptable:
        screening = screening_tier_for_stage(None)
        return ScreeningResult(
            classification="Image not suitable for DR screening",
            referable=False,
            dr_probability=0.0,
            stage=None,
            stage_label="Unstageable",
            medical_label="Image quality insufficient for screening",
            explanation=(
                "The retinal image quality was not good enough for reliable "
                "automated screening."
            ),
            recommendation="Please retake the image in better focus and lighting.",
            reason=", ".join(quality.retake_recommendations or quality.warnings),
            disclaimer=(
                "Screening support only. Retake with a clear retinal image before "
                "reviewing diabetic retinopathy features."
            ),
            confidence_label="Low Confidence",
            screening=screening,
            screening_recommendation=str(screening["recommendation"]),
        )

    if ml_config.uses_dual_model_mode() and image is not None:
        demo_result = classify_demo_hybrid(image, features)
        if demo_result is not None:
            return demo_result
        screening = screening_tier_for_stage(None)
        return ScreeningResult(
            classification="Analysis models unavailable",
            referable=False,
            dr_probability=0.0,
            stage=None,
            stage_label="Medical severity assessment unavailable",
            medical_label="Medical severity assessment unavailable",
            explanation=(
                "The backend is configured for task-specific screening and severity "
                "models, but the required artifacts were not loaded."
            ),
            recommendation=(
                "Confirm that the configured screening, severity, and image-feature "
                "model artifacts are available."
            ),
            reason="Dual-model mode does not silently fall back to another model route.",
            disclaimer=(
                "This app is a screening support tool only and does not provide "
                "a final medical diagnosis."
            ),
            model_type="dual_model_unavailable",
            confidence_label="Low Confidence",
            probabilities={"Non-Referable": 1.0, "Referable": 0.0},
            screening=screening,
            screening_recommendation=str(screening["recommendation"]),
        )

    supervised_result = classify_by_supervised_feature_model(features)
    if supervised_result is not None:
        return supervised_result

    stage, label, reason = classify_by_strict_stage_rules(features)
    screening = screening_tier_for_stage(stage)
    referable = bool(screening["referable"])
    score = fallback_referable_score(stage, features.pathology_area_index)

    return ScreeningResult(
        classification=label,
        referable=referable,
        dr_probability=score,
        stage=stage,
        stage_label=label,
        medical_label=label,
        explanation=medical_explanation_for_stage(stage),
        recommendation=str(screening["recommendation"]),
        reason=reason,
        disclaimer=(
            "Rule-based classical screening support only. This result is not a "
            "medical diagnosis and must be reviewed by a qualified eye-care professional."
        ),
        model_type="rule_based",
        confidence_label=confidence_label(None),
        screening=screening,
        screening_recommendation=str(screening["recommendation"]),
    )


def classify_by_supervised_feature_model(features: FeatureReport) -> ScreeningResult | None:
    stage_model = load_supervised_model()
    binary_model = load_binary_screening_model()

    if stage_model is None and binary_model is None:
        return None

    feature_names = supervised_model_feature_names()
    feature_vector = np.array(
        [supervised_feature_vector_from_report(features, feature_names)],
        dtype=np.float64,
    )
    if len(feature_names) != len(ML_FEATURE_NAMES):
        raise ValueError(
            "Loaded model metadata does not match the 203-feature AppDR extractor."
        )
    if feature_vector.shape[1] != len(ML_FEATURE_NAMES):
        raise ValueError(
            f"Feature extraction produced {feature_vector.shape[1]} features; "
            f"expected {len(ML_FEATURE_NAMES)}."
        )

    stage: int | None = None
    stage_label = "Unstageable"
    stage_probabilities: dict[int, float] = {}
    stage_confidence: float | None = None

    if stage_model is not None:
        stage = int(stage_model.predict(feature_vector)[0])
        stage_probabilities = supervised_probabilities(stage_model, feature_vector)
        stage_confidence = (
            max(stage_probabilities.values()) if stage_probabilities else None
        )
        stage_label = ML_STAGE_LABELS.get(stage, "Medical severity label unavailable")

    referable = bool(screening_tier_for_stage(stage)["referable"])
    dr_probability = supervised_referable_probability(stage, stage_probabilities)
    screening = screening_tier_for_stage(stage)
    binary_probabilities: dict[int, float] = {}
    binary_confidence: float | None = None
    binary_threshold: float | None = None

    if binary_model is not None:
        binary_probabilities = supervised_probabilities(binary_model, feature_vector)
        binary_threshold = load_binary_referable_threshold()
        referable_probability = float(binary_probabilities.get(1, 0.0))
        referable = bool(referable_probability >= binary_threshold)
        dr_probability = round(float(np.clip(referable_probability * 100.0, 0.0, 100.0)), 1)
        binary_confidence = (
            max(binary_probabilities.values()) if binary_probabilities else None
        )
        screening = screening_tier_for_binary_referable(referable, binary_threshold)
        if stage_model is None:
            stage_label = (
                "Referable diabetic retinopathy"
                if referable
                else "No referable diabetic retinopathy detected"
            )

    feature_summary = (
        f"MA={features.microaneurysm_count}, "
        f"exudate area={features.exudate_area}, "
        f"vessel density={features.vessel_density:.4f}, "
        f"GLCM contrast={features.glcm_contrast:.2f}"
    )
    reason_parts: list[str] = []

    if stage_model is not None and stage is not None:
        reason_parts.append(
            f"{supervised_model_display_name()} classified the image as "
            f"{stage_label} from {len(feature_names)} handcrafted retinal "
            f"measurements ({feature_summary})."
        )
        if stage_confidence is not None:
            reason_parts.append(f"Stage confidence is {stage_confidence:.2%}.")

    if binary_model is not None:
        reason_parts.append(
            f"{binary_screening_model_display_name()} screened the case as "
            f"{'Referable' if referable else 'Non-Referable'} "
            f"using a recall-oriented probability threshold of {binary_threshold:.2f}."
        )
        if binary_confidence is not None:
            reason_parts.append(f"Screening confidence is {binary_confidence:.2%}.")

    model_type = dual_tier_model_type(stage_model is not None, binary_model is not None)
    confidence = stage_confidence if stage_confidence is not None else binary_confidence
    response_probabilities = build_dual_tier_probabilities(
        stage_probabilities,
        binary_probabilities,
    )

    explanation = (
        medical_explanation_for_stage(stage)
        if stage_model is not None
        else str(screening["recommendation"])
    )
    recommendation = str(screening["recommendation"])

    return ScreeningResult(
        classification=f"{screening['status']}: {stage_label}",
        referable=referable,
        dr_probability=dr_probability,
        stage=stage,
        stage_label=stage_label,
        medical_label=stage_label,
        explanation=explanation,
        recommendation=recommendation,
        reason=" ".join(reason_parts),
        disclaimer=(
            "Supervised handcrafted-feature screening support only. This result "
            "is not a medical diagnosis and must be reviewed by a qualified "
            "eye-care professional. Severe and proliferative disease may be "
            "under-called on some images, so clinician review and manual override "
            "are required."
        ),
        model_type=model_type,
        confidence=confidence,
        confidence_label=confidence_label(confidence),
        probabilities=response_probabilities,
        screening=screening,
        screening_recommendation=str(screening["recommendation"]),
    )


def confidence_label(confidence: float | None) -> str:
    if confidence is None:
        return "Medium Confidence"
    if confidence >= 0.75:
        return "High Confidence"
    if confidence >= 0.45:
        return "Medium Confidence"
    return "Low Confidence"


def build_detection_findings(features: FeatureReport) -> list[DetectionFinding]:
    expanded = features.expanded_features or {}
    vessel_abnormality = float(expanded.get("vessel_abnormality_score", 0.0))
    cotton_wool_count = int(round(float(expanded.get("cotton_wool_count", 0.0))))

    return [
        DetectionFinding(
            label="Microaneurysms",
            detected=features.microaneurysm_count > 0,
        ),
        DetectionFinding(
            label="Hard Exudates",
            detected=features.exudate_count > 0,
        ),
        DetectionFinding(
            label="Vessel Abnormalities",
            detected=features.vessel_density > 0.08 or vessel_abnormality > 0.15,
        ),
        DetectionFinding(
            label="Cotton Wool Spots",
            detected=cotton_wool_count > 0,
        ),
    ]


def load_supervised_model() -> Any | None:
    global _SUPERVISED_MODEL, _SUPERVISED_MODEL_FEATURE_NAMES
    global _SUPERVISED_MODEL_LOAD_ATTEMPTED, _SUPERVISED_MODEL_METADATA

    if _SUPERVISED_MODEL_LOAD_ATTEMPTED:
        return _SUPERVISED_MODEL

    _SUPERVISED_MODEL_LOAD_ATTEMPTED = True
    _SUPERVISED_MODEL_METADATA = load_supervised_metadata(ML_METADATA_PATH)
    _SUPERVISED_MODEL_FEATURE_NAMES = metadata_feature_names(_SUPERVISED_MODEL_METADATA)

    if not ML_MODEL_PATH.exists():
        return None

    try:
        with ML_MODEL_PATH.open("rb") as file:
            _SUPERVISED_MODEL = pickle.load(file)
    except Exception:
        _SUPERVISED_MODEL = None

    return _SUPERVISED_MODEL


def load_binary_screening_model() -> Any | None:
    global _BINARY_SCREENING_MODEL, _BINARY_SCREENING_MODEL_LOAD_ATTEMPTED
    global _BINARY_SCREENING_MODEL_METADATA

    if _BINARY_SCREENING_MODEL_LOAD_ATTEMPTED:
        return _BINARY_SCREENING_MODEL

    _BINARY_SCREENING_MODEL_LOAD_ATTEMPTED = True
    _BINARY_SCREENING_MODEL_METADATA = load_supervised_metadata(BINARY_SCREENING_METADATA_PATH)

    if not BINARY_SCREENING_MODEL_PATH.exists():
        return None

    try:
        with BINARY_SCREENING_MODEL_PATH.open("rb") as file:
            _BINARY_SCREENING_MODEL = pickle.load(file)
    except Exception:
        _BINARY_SCREENING_MODEL = None

    return _BINARY_SCREENING_MODEL


def load_binary_referable_threshold() -> float:
    global _BINARY_REFERABLE_THRESHOLD

    if _BINARY_REFERABLE_THRESHOLD is not None:
        return _BINARY_REFERABLE_THRESHOLD

    threshold = DEFAULT_BINARY_REFERABLE_THRESHOLD
    if BINARY_REFERABLE_THRESHOLD_PATH.exists():
        try:
            payload = json.loads(BINARY_REFERABLE_THRESHOLD_PATH.read_text(encoding="utf-8"))
            value = payload.get("threshold") if isinstance(payload, dict) else None
            if value is not None:
                threshold = float(value)
        except Exception:
            threshold = DEFAULT_BINARY_REFERABLE_THRESHOLD

    _BINARY_REFERABLE_THRESHOLD = threshold
    return threshold


def load_supervised_metadata(metadata_path: Path = ML_METADATA_PATH) -> dict[str, Any]:
    if not metadata_path.exists():
        return {}

    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except Exception:
        return {}

    return metadata if isinstance(metadata, dict) else {}


def metadata_feature_names(metadata: dict[str, Any]) -> list[str]:
    names = metadata.get("feature_names")

    if isinstance(names, list) and all(isinstance(name, str) for name in names):
        return list(names)

    return list(ML_FEATURE_NAMES)


def supervised_model_feature_names() -> list[str]:
    if _SUPERVISED_MODEL_FEATURE_NAMES:
        return _SUPERVISED_MODEL_FEATURE_NAMES
    binary_names = metadata_feature_names(_BINARY_SCREENING_MODEL_METADATA)
    if binary_names:
        return binary_names

    return list(ML_FEATURE_NAMES)


def supervised_model_display_name() -> str:
    name = _SUPERVISED_MODEL_METADATA.get("best_model_name", "ScikitLearnClassifier")
    return str(name)


def binary_screening_model_display_name() -> str:
    name = _BINARY_SCREENING_MODEL_METADATA.get("best_model_name", "BinaryScreeningClassifier")
    return str(name)


def dual_tier_model_type(has_stage_model: bool, has_binary_model: bool) -> str:
    if has_stage_model and has_binary_model:
        return "dual_tier_handcrafted_features"
    if has_binary_model:
        return f"{binary_screening_model_display_name()}_binary_screening"
    return f"{supervised_model_display_name()}_handcrafted_features"


def get_supervised_model_status() -> dict[str, object]:
    stage_model = load_supervised_model()
    binary_model = load_binary_screening_model()
    dual_model_ready = all(
        path.exists()
        for path in [
            DUAL_BINARY_MODEL_PATH,
            DUAL_SEVERITY_MODEL_PATH,
            DUAL_CNN_CHECKPOINT_PATH,
        ]
    )
    return {
        "model_mode": ml_config.MODEL_MODE,
        "dual_model_ready": dual_model_ready,
        "demo_hybrid_ready": dual_model_ready,
        "binary_model_source": SCREENING_MODEL_SOURCE,
        "severity_model_source": SEVERITY_MODEL_SOURCE,
        "severity_model_path": str(DUAL_SEVERITY_MODEL_PATH),
        "rollback_dir": str(ml_config.RESULTS_DIR / "backup_before_ophthalmologist_demo"),
        "multiclass_loaded": stage_model is not None,
        "multiclass_model": (
            supervised_model_display_name() if stage_model is not None else None
        ),
        "binary_loaded": binary_model is not None,
        "binary_model": (
            binary_screening_model_display_name() if binary_model is not None else None
        ),
        "dual_tier_ready": stage_model is not None and binary_model is not None,
        "binary_threshold": load_binary_referable_threshold(),
    }


def build_dual_tier_probabilities(
    stage_probabilities: dict[int, float],
    binary_probabilities: dict[int, float],
) -> dict[str, float]:
    probabilities: dict[str, float] = {}

    for label, value in stage_probabilities.items():
        probabilities[ML_STAGE_LABELS.get(int(label), str(label))] = float(value)

    if binary_probabilities:
        probabilities["Non-Referable"] = float(binary_probabilities.get(0, 0.0))
        probabilities["Referable"] = float(binary_probabilities.get(1, 0.0))

    return probabilities


def supervised_feature_vector_from_report(
    features: FeatureReport,
    feature_names: list[str],
) -> list[float]:
    expanded = dict(features.expanded_features or {})
    fundus_area = max(float(features.fundus_area), 1.0)

    fallback_values = {
        "ma_count": float(features.microaneurysm_count),
        "ma_area": float(features.microaneurysm_area),
        "ma_density": float(features.microaneurysm_area) / fundus_area,
        "ma_mean_area": float(features.microaneurysm_area)
        / max(float(features.microaneurysm_count), 1.0),
        "exudate_count": float(features.exudate_count),
        "exudate_area": float(features.exudate_area),
        "exudate_density": float(features.exudate_area) / fundus_area,
        "exudate_mean_area": float(features.exudate_area)
        / max(float(features.exudate_count), 1.0),
        "vessel_density": float(features.vessel_density),
        "glcm_contrast": float(features.glcm_contrast),
        "glcm_homogeneity": float(features.glcm_homogeneity),
        "glcm_energy": float(features.glcm_energy),
    }

    return [
        float(expanded.get(name, fallback_values.get(name, 0.0)))
        for name in feature_names
    ]


def supervised_probabilities(model: Any, feature_vector: np.ndarray) -> dict[int, float]:
    if not hasattr(model, "predict_proba"):
        return {}

    values = model.predict_proba(feature_vector)[0]
    classes = model_classes(model)
    return {
        int(label): float(probability)
        for label, probability in zip(classes, values)
    }


def model_classes(model: Any) -> list[int]:
    classes = getattr(model, "classes_", None)

    if classes is None and hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier")
        classes = getattr(classifier, "classes_", None)

    if classes is None:
        return list(ML_STAGE_LABELS.keys())

    return [int(label) for label in classes]


def supervised_referable_probability(
    stage: int | None,
    probabilities: dict[int, float],
) -> float:
    if probabilities:
        referable_probability = sum(
            probability
            for label, probability in probabilities.items()
            if int(label) >= 2
        )
        return round(float(np.clip(referable_probability * 100.0, 0.0, 100.0)), 1)

    if stage is None:
        return 0.0

    return fallback_referable_score(stage, 0.0)


def fallback_referable_score(stage: int, pathology_area_index: float) -> float:
    if stage < 2:
        return 0.0

    return deterministic_stage_score(stage, pathology_area_index)


def screening_tier_for_stage(stage: int | None) -> dict[str, Any]:
    if stage is None:
        return {
            "status": "Unstageable",
            "referable": False,
            "rule": "Quality gate failed",
            "recommendation": "Retake the retinal image before clinician review.",
        }

    if stage >= 2:
        return {
            "status": "Referable",
            "referable": True,
            "rule": "DR grades 2-4 are mapped to referable screening-support review.",
            "recommendation": (
                "Referable diabetic retinopathy detected. Specialist evaluation "
                "with an ophthalmologist is recommended."
            ),
        }

    return {
        "status": "Non-Referable",
        "referable": False,
        "rule": "DR grades 0-1 are mapped to non-referable screening-support review.",
        "recommendation": (
            "No significant referable diabetic retinopathy findings detected. "
            "Routine ophthalmology follow-up recommended after clinician review."
        ),
    }


def screening_tier_for_binary_referable(
    referable: bool,
    threshold: float | None = None,
) -> dict[str, Any]:
    threshold_text = (
        f"{threshold:.2f}"
        if threshold is not None
        else f"{DEFAULT_BINARY_REFERABLE_THRESHOLD:.2f}"
    )

    if referable:
        return {
            "status": "Referable",
            "referable": True,
            "rule": (
                "Binary screening model classified the case as referable using "
                f"optimized probability threshold {threshold_text}."
            ),
            "recommendation": (
                "Referable diabetic retinopathy detected. Specialist evaluation "
                "recommended."
            ),
        }

    return {
        "status": "Non-Referable",
        "referable": False,
        "rule": (
            "Binary screening model classified the case as non-referable using "
            f"optimized probability threshold {threshold_text}."
        ),
        "recommendation": (
            "No significant referable diabetic retinopathy findings detected. "
            "Routine ophthalmology follow-up recommended after clinician review."
        ),
    }


def classify_by_strict_stage_rules(features: FeatureReport) -> tuple[int, str, str]:
    ma_count = features.microaneurysm_count
    exudate_count = features.exudate_count
    exudate_quadrants = features.exudate_quadrant_count

    if features.vessel_density > 0.12 and features.glcm_contrast > 500.0:
        return (
            4,
            ML_STAGE_LABELS[4],
            "vessel density and GLCM contrast exceeded the proliferative override",
        )
    if ma_count == 0 and exudate_count == 0:
        return 0, ML_STAGE_LABELS[0], "no microaneurysms or exudates detected"
    if ma_count > 15 or exudate_quadrants >= 2:
        return (
            3,
            ML_STAGE_LABELS[3],
            "more than 15 microaneurysms or exudates across at least two quadrants",
        )
    if 1 <= ma_count <= 5 and exudate_count == 0:
        return 1, ML_STAGE_LABELS[1], "1 to 5 microaneurysms and no exudates detected"
    if 6 <= ma_count <= 15 or (exudate_count > 0 and exudate_quadrants == 1):
        return (
            2,
            ML_STAGE_LABELS[2],
            "6 to 15 microaneurysms or exudates localized to one quadrant",
        )

    return 1, ML_STAGE_LABELS[1], "minimal non-zero lesion evidence detected"


def medical_explanation_for_stage(stage: int | None) -> str:
    if stage is None:
        return (
            "The image could not be reliably classified. Retake the image and "
            "ask an eye-care professional to review it."
        )
    return ml_config.CLASS_EXPLANATIONS.get(
        int(stage),
        "The model produced a diabetic-retinopathy screening classification.",
    )


def deterministic_stage_score(stage: int, pathology_area_index: float) -> float:
    base_score = STAGE_SCORE_BY_STAGE.get(stage, 0.0)

    if stage == 0:
        return 0.0

    return round(float(np.clip(base_score + min(pathology_area_index * 2.0, 4.0), 0.0, 99.0)), 1)


def assess_quality(image: np.ndarray, green: np.ndarray, mask: np.ndarray) -> QualityReport:
    pixels = green[mask > 0]
    warnings: list[str] = []
    blocking_warnings: list[str] = []

    if pixels.size == 0:
        return QualityReport(
            is_acceptable=False,
            blur_score=0.0,
            sharpness=0.0,
            brightness_mean=0.0,
            contrast_std=0.0,
            signal_to_noise_ratio=0.0,
            quality_score=0,
            quality_label="Poor",
            fundus_area_ratio=0.0,
            warnings=["Retinal field could not be detected."],
            retake_recommendations=["Retina not fully visible. Please retake the image."],
        )

    blur_score = float(cv2.Laplacian(green, cv2.CV_64F).var())
    sharpness = float(np.sqrt(max(blur_score, 0.0)))
    brightness_mean = float(np.mean(pixels))
    contrast_std = float(np.std(pixels))
    signal_to_noise_ratio = float(brightness_mean / max(contrast_std, 1.0))
    fundus_area_ratio = float(np.count_nonzero(mask) / mask.size)
    vessel_hint = estimate_vessel_hint(green, mask)
    retinal_warnings, retinal_blockers = assess_retinal_field(
        image,
        green,
        mask,
        fundus_area_ratio,
        vessel_hint,
    )
    warnings.extend(retinal_warnings)
    blocking_warnings.extend(retinal_blockers)

    if blur_score < 6.0:
        blocking_warnings.append("Image is too blurry for screening support.")
    elif blur_score < 45.0:
        warnings.append("Image may be blurry.")
    if brightness_mean < 20.0:
        blocking_warnings.append("Image is too dark for screening support.")
    elif brightness_mean < 35.0:
        warnings.append("Image may be underexposed.")
    if brightness_mean > 240.0:
        blocking_warnings.append("Image is too bright for screening support.")
    elif brightness_mean > 220.0:
        warnings.append("Image may be overexposed.")
    if contrast_std < 5.0:
        blocking_warnings.append("Image contrast is too low for screening support.")
    elif contrast_std < 12.0:
        warnings.append("Image contrast may be too low.")
    if fundus_area_ratio < 0.025:
        blocking_warnings.append("Detected retinal field is too small.")
    elif fundus_area_ratio < 0.08:
        warnings.append("Detected retinal field is too small.")

    warnings = unique_warnings([*blocking_warnings, *warnings])
    quality_score = calculate_quality_score(
        blur_score=blur_score,
        sharpness=sharpness,
        brightness_mean=brightness_mean,
        contrast_std=contrast_std,
        signal_to_noise_ratio=signal_to_noise_ratio,
        fundus_area_ratio=fundus_area_ratio,
        blocking_warning_count=len(blocking_warnings),
    )
    retake_recommendations = retake_recommendations_from_warnings(warnings)
    is_acceptable = len(blocking_warnings) == 0 and quality_score >= MIN_ANALYSIS_QUALITY_SCORE

    return QualityReport(
        is_acceptable=is_acceptable,
        blur_score=blur_score,
        sharpness=sharpness,
        brightness_mean=brightness_mean,
        contrast_std=contrast_std,
        signal_to_noise_ratio=signal_to_noise_ratio,
        quality_score=quality_score,
        quality_label=quality_label(quality_score),
        fundus_area_ratio=fundus_area_ratio,
        warnings=warnings,
        retake_recommendations=retake_recommendations,
    )


def calculate_quality_score(
    blur_score: float,
    sharpness: float,
    brightness_mean: float,
    contrast_std: float,
    signal_to_noise_ratio: float,
    fundus_area_ratio: float,
    blocking_warning_count: int,
) -> int:
    blur_component = np.clip(blur_score / 90.0, 0.0, 1.0)
    sharpness_component = np.clip(sharpness / 14.0, 0.0, 1.0)
    brightness_component = 1.0 - np.clip(abs(brightness_mean - 115.0) / 115.0, 0.0, 1.0)
    contrast_component = np.clip(contrast_std / 45.0, 0.0, 1.0)
    snr_component = 1.0 - np.clip(abs(signal_to_noise_ratio - 4.0) / 8.0, 0.0, 1.0)
    coverage_component = np.clip(fundus_area_ratio / 0.55, 0.0, 1.0)
    raw_score = (
        blur_component * 22.0
        + sharpness_component * 14.0
        + brightness_component * 20.0
        + contrast_component * 18.0
        + snr_component * 10.0
        + coverage_component * 16.0
    )
    penalty = blocking_warning_count * 18.0
    return int(round(float(np.clip(raw_score - penalty, 0.0, 100.0))))


def quality_label(score: int) -> str:
    if score >= 80:
        return "Good"
    if score >= MIN_ANALYSIS_QUALITY_SCORE:
        return "Acceptable"
    return "Poor"


def retake_recommendations_from_warnings(warnings: list[str]) -> list[str]:
    recommendations: list[str] = []
    joined = " ".join(warnings).lower()

    if "blurry" in joined or "blur" in joined:
        recommendations.append("Image too blurry. Please retake the image.")
    if "retinal field" in joined or "round enough" in joined:
        recommendations.append("Retina not fully visible.")
    if "dark" in joined or "bright" in joined or "exposed" in joined or "contrast" in joined:
        recommendations.append("Poor lighting detected.")
    if "too small" in joined or "boundary" in joined:
        recommendations.append("Fundus coverage insufficient.")

    return unique_warnings(recommendations)


def assess_retinal_field(
    image: np.ndarray,
    green: np.ndarray,
    mask: np.ndarray,
    fundus_area_ratio: float,
    vessel_hint: float,
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    blocking_warnings: list[str] = []
    contour = get_largest_contour(mask)

    if contour is None:
        return [], ["Retinal field could not be detected."]

    height, width = mask.shape[:2]
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, box_width, box_height = cv2.boundingRect(contour)
    circularity = 0.0
    extent = 0.0

    if perimeter > 0:
        circularity = float((4.0 * np.pi * area) / (perimeter * perimeter))
    if box_width > 0 and box_height > 0:
        extent = float(area / (box_width * box_height))

    touches_border = (
        x <= 2
        or y <= 2
        or x + box_width >= width - 2
        or y + box_height >= height - 2
    )
    full_frame_retina_candidate = fundus_area_ratio > 0.80 and extent > 0.88

    if fundus_area_ratio > 0.94:
        warnings.append("Retinal field fills most of the image.")
    elif touches_border and fundus_area_ratio > 0.72:
        warnings.append("Retinal field boundary could not be separated from background.")
    if circularity < 0.20 and not full_frame_retina_candidate:
        blocking_warnings.append("Detected retinal field is not round enough.")
    elif circularity < 0.38 and not full_frame_retina_candidate:
        warnings.append("Detected retinal field is not round enough.")

    masked_pixels = image[mask > 0]
    if masked_pixels.size:
        blue_mean = float(np.mean(masked_pixels[:, 0]))
        green_mean = float(np.mean(masked_pixels[:, 1]))
        red_mean = float(np.mean(masked_pixels[:, 2]))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation_mean = float(np.mean(hsv[:, :, 1][mask > 0]))
        red_orange_score = red_mean - max(green_mean, blue_mean)
        has_retinal_color = saturation_mean >= 8.0 and red_orange_score >= -25.0
        has_retinal_texture = float(np.std(green[mask > 0])) >= 8.0
        has_vessel_signal = vessel_hint >= 0.001

        if not has_retinal_color and not has_vessel_signal and not has_retinal_texture:
            blocking_warnings.append("Image does not match expected retinal color.")
        elif saturation_mean < 18.0 or red_orange_score < -5.0:
            warnings.append("Image does not match expected retinal color.")

    if vessel_hint < 0.0004:
        warnings.append("Retinal vessel pattern is weak.")

    return warnings, blocking_warnings


def add_feature_quality_warnings(
    quality: QualityReport,
    features: FeatureReport,
) -> QualityReport:
    warnings = list(quality.warnings)

    if features.vessel_density < 0.001:
        warnings.append("Retinal vessel pattern is not visible.")
    if not features.optic_disc_detected:
        warnings.append("Optic disc was not confidently localized.")

    return QualityReport(
        is_acceptable=quality.is_acceptable,
        blur_score=quality.blur_score,
        sharpness=quality.sharpness,
        brightness_mean=quality.brightness_mean,
        contrast_std=quality.contrast_std,
        signal_to_noise_ratio=quality.signal_to_noise_ratio,
        quality_score=quality.quality_score,
        quality_label=quality.quality_label,
        fundus_area_ratio=quality.fundus_area_ratio,
        warnings=unique_warnings(warnings),
        retake_recommendations=quality.retake_recommendations,
    )


def create_overlay(
    image: np.ndarray,
    vessels: np.ndarray,
    microaneurysms: np.ndarray,
    exudates: np.ndarray,
    optic_disc_mask: np.ndarray,
    fov_mask: np.ndarray,
) -> np.ndarray:
    overlay = image.copy()
    overlay[vessels > 0] = blend_color(overlay[vessels > 0], np.array([255, 120, 0]))
    draw_mask_contours(overlay, exudates, color=(0, 255, 255), min_area=8)
    draw_mask_contours(overlay, microaneurysms, color=(0, 0, 255), min_area=3)
    draw_mask_contours(overlay, optic_disc_mask, color=(0, 255, 0), min_area=20)
    draw_quadrant_axes(overlay, fov_mask)

    return overlay


def draw_mask_contours(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    min_area: int,
) -> None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        cv2.drawContours(image, [contour], -1, color, thickness=2)
        cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness=1)


def draw_quadrant_axes(image: np.ndarray, fov_mask: np.ndarray) -> None:
    centroid = find_mask_centroid(fov_mask)

    if centroid is None:
        return

    cx, cy = centroid
    contour = get_largest_contour(fov_mask)

    if contour is None:
        return

    x, y, width, height = cv2.boundingRect(contour)
    cv2.line(image, (cx, y), (cx, y + height), (255, 255, 255), 1)
    cv2.line(image, (x, cy), (x + width, cy), (255, 255, 255), 1)


def blend_color(pixels: np.ndarray, color: np.ndarray) -> np.ndarray:
    return np.clip((pixels.astype(np.float32) * 0.45) + (color * 0.55), 0, 255).astype(
        np.uint8,
    )


def build_valid_lesion_mask(
    fov_mask: np.ndarray,
    optic_disc_mask: np.ndarray,
    vessel_exclusion: np.ndarray,
) -> np.ndarray:
    valid = np.zeros_like(fov_mask)
    valid[(fov_mask > 0) & (optic_disc_mask == 0) & (vessel_exclusion == 0)] = 255

    return valid


def create_interior_fundus_mask(mask: np.ndarray, margin_ratio: float) -> np.ndarray:
    if not np.any(mask > 0):
        return np.zeros_like(mask)

    margin = max(8.0, min(mask.shape[:2]) * margin_ratio)
    distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    interior = np.zeros_like(mask)
    interior[distance >= margin] = 255

    return interior


def fill_outside_mask(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = gray.copy()
    pixels = gray[mask > 0]
    fill_value = int(np.median(pixels)) if pixels.size else 0
    output[mask == 0] = fill_value

    return output


def fill_masked_pixels(gray: np.ndarray, blocked_mask: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    output = gray.copy()
    valid_pixels = gray[valid_mask > 0]
    fill_value = int(np.median(valid_pixels)) if valid_pixels.size else 255
    output[blocked_mask > 0] = fill_value

    return output


def find_mask_centroid(mask: np.ndarray) -> tuple[int, int] | None:
    moments = cv2.moments(mask)

    if moments["m00"] == 0:
        return None

    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])


def count_components(mask: np.ndarray) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    component_count = 0

    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] > 0:
            component_count += 1

    return component_count


def estimate_vessel_hint(green: np.ndarray, mask: np.ndarray) -> float:
    candidate_pixels = green[mask > 0]

    if candidate_pixels.size == 0:
        return 0.0

    blackhat = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, disk_kernel(8))
    threshold = max(8.0, float(np.percentile(blackhat[mask > 0], 92)))
    candidate_count = int(np.count_nonzero((blackhat >= threshold) & (mask > 0)))

    return float(candidate_count / candidate_pixels.size)


def contour_shape_metrics(contour: np.ndarray) -> tuple[float, float, float]:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, width, height = cv2.boundingRect(contour)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))

    circularity = 0.0
    if perimeter > 0:
        circularity = float((4.0 * np.pi * area) / (perimeter * perimeter))

    aspect_ratio = 999.0
    if width > 0 and height > 0:
        aspect_ratio = float(max(width, height) / max(min(width, height), 1))

    solidity = 0.0
    if hull_area > 0:
        solidity = float(area / hull_area)

    return circularity, aspect_ratio, solidity


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if count <= 1:
        return mask

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    output = np.zeros_like(mask)
    output[labels == largest_label] = 255

    return output


def get_largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    return max(contours, key=cv2.contourArea)


def unique_warnings(warnings: list[str]) -> list[str]:
    unique: list[str] = []

    for warning in warnings:
        if warning not in unique:
            unique.append(warning)

    return unique


def remove_small_components(mask: np.ndarray, min_area: int, max_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    output = np.zeros_like(mask)

    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]

        if min_area <= area <= max_area:
            output[labels == label] = 255

    return output


def disk_kernel(radius: int) -> np.ndarray:
    size = (radius * 2) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def masked_otsu_threshold(values: np.ndarray) -> float:
    if values.size == 0:
        return 255.0

    value_image = values.astype(np.uint8).reshape(-1, 1)
    threshold, _ = cv2.threshold(
        value_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return float(threshold)


def frangi_vesselness_opencv(
    image: np.ndarray,
    sigmas: tuple[float, ...],
    beta: float,
    gamma: float,
) -> np.ndarray:
    responses: list[np.ndarray] = []
    epsilon = 1e-8
    gamma_value = gamma / 255.0 if gamma > 1.0 else gamma

    for sigma in sigmas:
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
        dxx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3) * (sigma**2)
        dxy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3) * (sigma**2)
        dyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3) * (sigma**2)
        trace = dxx + dyy
        determinant_term = np.sqrt(np.maximum((dxx - dyy) ** 2 + 4.0 * (dxy**2), 0.0))
        lambda_1 = 0.5 * (trace - determinant_term)
        lambda_2 = 0.5 * (trace + determinant_term)
        swap = np.abs(lambda_1) > np.abs(lambda_2)
        small_lambda = np.where(swap, lambda_2, lambda_1)
        large_lambda = np.where(swap, lambda_1, lambda_2)
        blobness = (small_lambda / (large_lambda + epsilon)) ** 2
        structureness = small_lambda**2 + large_lambda**2
        response = np.exp(-blobness / (2.0 * (beta**2)))
        response *= 1.0 - np.exp(-structureness / (2.0 * (gamma_value**2)))
        response[large_lambda <= 0] = 0.0
        responses.append(response.astype(np.float32))

    if not responses:
        return np.zeros_like(image, dtype=np.float32)

    return np.maximum.reduce(responses)


def glcm_texture_props(gray: np.ndarray, levels: int = 64) -> tuple[float, float, float]:
    if gray.size == 0:
        return 0.0, 0.0, 0.0

    quantized = np.clip(gray.astype(np.uint16) * levels // 256, 0, levels - 1).astype(
        np.int32,
    )
    offsets = ((0, 1), (1, 1), (1, 0), (1, -1))
    contrasts: list[float] = []
    homogeneities: list[float] = []
    energies: list[float] = []
    row_indices = np.arange(levels, dtype=np.float32)
    diff = row_indices[:, None] - row_indices[None, :]
    contrast_weights = diff**2
    homogeneity_weights = 1.0 / (1.0 + np.abs(diff))
    contrast_scale = (256.0 / levels) ** 2

    for dy, dx in offsets:
        y0 = max(0, dy)
        y1 = gray.shape[0] + min(0, dy)
        x0 = max(0, dx)
        x1 = gray.shape[1] + min(0, dx)
        source = quantized[y0:y1, x0:x1]
        target = quantized[y0 - dy : y1 - dy, x0 - dx : x1 - dx]

        if source.size == 0 or target.size == 0:
            continue

        matrix = np.zeros((levels, levels), dtype=np.float64)
        np.add.at(matrix, (source.ravel(), target.ravel()), 1)
        matrix += matrix.T
        matrix_sum = float(matrix.sum())

        if matrix_sum <= 0:
            continue

        probabilities = matrix / matrix_sum
        contrasts.append(float(np.sum(probabilities * contrast_weights) * contrast_scale))
        homogeneities.append(float(np.sum(probabilities * homogeneity_weights)))
        energies.append(float(np.sqrt(np.sum(probabilities**2))))

    if not contrasts:
        return 0.0, 0.0, 0.0

    return float(np.mean(contrasts)), float(np.mean(homogeneities)), float(np.mean(energies))


def feature_vector_from_report(features: FeatureReport) -> list[float]:
    return [
        float(features.fundus_area),
        float(features.vessel_density),
        float(features.vessel_area),
        float(features.exudate_area),
        float(features.microaneurysm_area),
        float(features.exudate_count),
        float(features.microaneurysm_count),
        float(features.exudate_quadrant_count),
        float(features.pathology_area_index),
        float(features.optic_disc_area),
        float(features.mean_intensity),
        float(features.intensity_std),
        float(features.glcm_contrast),
        float(features.glcm_homogeneity),
        float(features.glcm_energy),
    ]


def encode_png(image: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", image)

    if not success:
        raise ValueError("Failed to encode processed image.")

    encoded = base64.b64encode(buffer).decode("ascii")
    return f"data:image/png;base64,{encoded}"
