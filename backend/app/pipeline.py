import base64
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.schemas import FeatureReport, QualityReport, ScreeningResult


@dataclass
class PipelineOutput:
    quality: QualityReport
    features: FeatureReport
    result: ScreeningResult
    processed_images: dict[str, str]


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
ML_FEATURE_NAMES = [
    "ma_count",
    "exudate_area",
    "vessel_density",
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
]
ML_STAGE_LABELS = {
    0: "Stage 0: No DR",
    1: "Stage 1: Mild NPDR",
    2: "Stage 2: Moderate NPDR",
    3: "Stage 3: Severe NPDR",
    4: "Stage 4: Proliferative DR",
}
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


def analyze_image(image_bytes: bytes, include_processed_images: bool = True) -> PipelineOutput:
    image = prepare_analysis_image(decode_image(image_bytes))
    stage0 = stage0_fov_and_optic_disc_masking(image)
    quality = assess_quality(image, image[:, :, 1], stage0.fov_mask)
    preprocessed = stage1_preprocess_green_channel(image, stage0.fov_mask)
    vessels = stage2_segment_vessels(preprocessed.denoised, stage0.fov_mask)
    lesions = stage3_extract_lesions(
        image=image,
        preprocessed=preprocessed.denoised,
        fov_mask=stage0.fov_mask,
        optic_disc_mask=stage0.optic_disc_mask,
        vessels=vessels.vessels,
    )
    features = stage4_extract_features(
        preprocessed=preprocessed.denoised,
        vessels=vessels.vessels,
        microaneurysms=lesions.microaneurysms,
        exudates=lesions.exudates,
        optic_disc_mask=stage0.optic_disc_mask,
        fov_mask=stage0.fov_mask,
        optic_disc_detected=stage0.optic_disc_contour is not None,
    )
    quality = add_feature_quality_warnings(quality, features)
    result = stage5_classify(features, quality)
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
    )


def process_image_path(image_path: str | Path) -> dict[str, Any]:
    path = Path(image_path)
    output = analyze_image(path.read_bytes())

    return {
        "filename": path.name,
        "quality": output.quality.model_dump(),
        "features": output.features.model_dump(),
        "result": output.result.model_dump(),
        "processed_images": output.processed_images,
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


def stage1_preprocess_green_channel(image: np.ndarray, fov_mask: np.ndarray) -> PreprocessedImage:
    green = image[:, :, 1]
    green_for_clahe = fill_outside_mask(green, fov_mask)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
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
) -> LesionMasks:
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
) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
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
        float(np.percentile(valid_values, 97.5)),
    )
    local_values = local_bright[valid_mask > 0]
    local_threshold = max(14.0, float(np.percentile(local_values, 98.0)))
    candidates = np.zeros_like(lightness)
    candidates[
        (lightness >= threshold)
        & (local_bright >= local_threshold)
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


def stage5_classify(features: FeatureReport, quality: QualityReport) -> ScreeningResult:
    if not quality.is_acceptable:
        return ScreeningResult(
            classification="Image not suitable for DR screening",
            referable=False,
            dr_probability=0.0,
            stage=None,
            stage_label="Unstageable",
            reason=", ".join(quality.warnings),
            disclaimer=(
                "Screening support only. Retake with a clear retinal image before "
                "reviewing diabetic retinopathy features."
            ),
        )

    supervised_result = classify_by_supervised_feature_model(features)
    if supervised_result is not None:
        return supervised_result

    stage, label, reason = classify_by_strict_stage_rules(features)
    referable = stage > 0
    score = deterministic_stage_score(stage, features.pathology_area_index)

    return ScreeningResult(
        classification=label,
        referable=referable,
        dr_probability=score,
        stage=stage,
        stage_label=label,
        reason=reason,
        disclaimer=(
            "Rule-based classical screening support only. This result is not a "
            "medical diagnosis and must be reviewed by a qualified eye-care professional."
        ),
        model_type="rule_based",
    )


def classify_by_supervised_feature_model(features: FeatureReport) -> ScreeningResult | None:
    model = load_supervised_model()

    if model is None:
        return None

    feature_vector = np.array([supervised_feature_vector_from_report(features)], dtype=np.float64)
    stage = int(model.predict(feature_vector)[0])
    probabilities = supervised_probabilities(model, feature_vector)
    confidence = max(probabilities.values()) if probabilities else None
    dr_probability = supervised_dr_probability(stage, probabilities)
    label = ML_STAGE_LABELS.get(stage, f"Stage {stage}: DR")

    feature_summary = (
        f"MA={features.microaneurysm_count}, "
        f"exudate area={features.exudate_area}, "
        f"vessel density={features.vessel_density:.4f}, "
        f"GLCM contrast={features.glcm_contrast:.2f}"
    )
    reason = (
        "RandomForestClassifier predicted the stage from the six handcrafted "
        f"retinal features ({feature_summary})."
    )

    if confidence is not None:
        reason += f" Model confidence for the selected stage is {confidence:.2%}."

    return ScreeningResult(
        classification=label,
        referable=stage > 0,
        dr_probability=dr_probability,
        stage=stage,
        stage_label=label,
        reason=reason,
        disclaimer=(
            "Supervised handcrafted-feature screening support only. This result "
            "is not a medical diagnosis and must be reviewed by a qualified "
            "eye-care professional."
        ),
        model_type="random_forest_handcrafted_features",
        confidence=confidence,
        probabilities={str(label): value for label, value in probabilities.items()},
    )


def load_supervised_model() -> Any | None:
    global _SUPERVISED_MODEL, _SUPERVISED_MODEL_LOAD_ATTEMPTED

    if _SUPERVISED_MODEL_LOAD_ATTEMPTED:
        return _SUPERVISED_MODEL

    _SUPERVISED_MODEL_LOAD_ATTEMPTED = True

    if not ML_MODEL_PATH.exists():
        return None

    try:
        with ML_MODEL_PATH.open("rb") as file:
            _SUPERVISED_MODEL = pickle.load(file)
    except Exception:
        _SUPERVISED_MODEL = None

    return _SUPERVISED_MODEL


def supervised_feature_vector_from_report(features: FeatureReport) -> list[float]:
    return [
        float(features.microaneurysm_count),
        float(features.exudate_area),
        float(features.vessel_density),
        float(features.glcm_contrast),
        float(features.glcm_homogeneity),
        float(features.glcm_energy),
    ]


def supervised_probabilities(model: Any, feature_vector: np.ndarray) -> dict[int, float]:
    if not hasattr(model, "predict_proba"):
        return {}

    values = model.predict_proba(feature_vector)[0]
    classes = getattr(model, "classes_", list(ML_STAGE_LABELS.keys()))
    return {
        int(label): float(probability)
        for label, probability in zip(classes, values)
    }


def supervised_dr_probability(stage: int, probabilities: dict[int, float]) -> float:
    if probabilities:
        non_dr_probability = probabilities.get(0, 0.0)
        return round(float(np.clip((1.0 - non_dr_probability) * 100.0, 0.0, 100.0)), 1)

    return deterministic_stage_score(stage, 0.0)


def classify_by_strict_stage_rules(features: FeatureReport) -> tuple[int, str, str]:
    ma_count = features.microaneurysm_count
    exudate_count = features.exudate_count
    exudate_quadrants = features.exudate_quadrant_count

    if features.vessel_density > 0.12 and features.glcm_contrast > 500.0:
        return (
            4,
            "Stage 4: Proliferative DR (Referable)",
            "vessel density and GLCM contrast exceeded the proliferative override",
        )
    if ma_count == 0 and exudate_count == 0:
        return 0, "Stage 0: No DR", "no microaneurysms or exudates detected"
    if ma_count > 15 or exudate_quadrants >= 2:
        return (
            3,
            "Stage 3: Severe NPDR",
            "more than 15 microaneurysms or exudates across at least two quadrants",
        )
    if 1 <= ma_count <= 5 and exudate_count == 0:
        return 1, "Stage 1: Mild NPDR", "1 to 5 microaneurysms and no exudates detected"
    if 6 <= ma_count <= 15 or (exudate_count > 0 and exudate_quadrants == 1):
        return (
            2,
            "Stage 2: Moderate NPDR",
            "6 to 15 microaneurysms or exudates localized to one quadrant",
        )

    return 1, "Stage 1: Mild NPDR", "minimal non-zero lesion evidence detected"


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
            brightness_mean=0.0,
            contrast_std=0.0,
            fundus_area_ratio=0.0,
            warnings=["Retinal field could not be detected."],
        )

    blur_score = float(cv2.Laplacian(green, cv2.CV_64F).var())
    brightness_mean = float(np.mean(pixels))
    contrast_std = float(np.std(pixels))
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

    return QualityReport(
        is_acceptable=len(blocking_warnings) == 0,
        blur_score=blur_score,
        brightness_mean=brightness_mean,
        contrast_std=contrast_std,
        fundus_area_ratio=fundus_area_ratio,
        warnings=warnings,
    )


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
        brightness_mean=quality.brightness_mean,
        contrast_std=quality.contrast_std,
        fundus_area_ratio=quality.fundus_area_ratio,
        warnings=unique_warnings(warnings),
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
