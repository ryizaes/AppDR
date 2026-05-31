from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config
from preprocessing import (
    PreprocessingResult,
    contour_shape_metrics,
    disk_kernel,
    fill_outside_mask,
    preprocess_retinal_image,
)
from utils import ensure_dir

try:
    from skimage.morphology import skeletonize as skimage_skeletonize
except Exception:  # pragma: no cover - optional production dependency fallback.
    skimage_skeletonize = None

try:
    from skimage.feature import local_binary_pattern as skimage_local_binary_pattern
except Exception:  # pragma: no cover - optional production dependency fallback.
    skimage_local_binary_pattern = None


class FeatureExtractionError(RuntimeError):
    """Raised when a sample cannot be converted into numerical features."""


@dataclass(frozen=True)
class FeatureExtractionPayload:
    """Full classical-CV output for ML and frontend visual verification."""

    features: dict[str, float]
    masks: dict[str, np.ndarray]
    coordinates: dict[str, list[dict[str, Any]]]
    image_shape: tuple[int, int]


def extract_features(
    image_path: str | Path,
    debug: bool = False,
    debug_dir: str | Path | None = None,
    clahe_clip_limit: float | None = None,
    exudate_percentile: float | None = None,
    exudate_local_percentile: float | None = None,
) -> list[float]:
    """Return the expanded handcrafted feature vector in config.FEATURE_NAMES order."""
    feature_dict = extract_feature_dict(
        image_path,
        debug=debug,
        debug_dir=debug_dir,
        clahe_clip_limit=clahe_clip_limit,
        exudate_percentile=exudate_percentile,
        exudate_local_percentile=exudate_local_percentile,
    )
    return [float(feature_dict[name]) for name in config.FEATURE_NAMES]


def extract_feature_dict(
    image_path: str | Path,
    debug: bool = False,
    debug_dir: str | Path | None = None,
    clahe_clip_limit: float | None = None,
    exudate_percentile: float | None = None,
    exudate_local_percentile: float | None = None,
) -> dict[str, float]:
    return extract_feature_payload(
        image_path,
        debug=debug,
        debug_dir=debug_dir,
        clahe_clip_limit=clahe_clip_limit,
        exudate_percentile=exudate_percentile,
        exudate_local_percentile=exudate_local_percentile,
    ).features


def extract_feature_payload(
    image_path: str | Path,
    debug: bool = False,
    debug_dir: str | Path | None = None,
    clahe_clip_limit: float | None = None,
    exudate_percentile: float | None = None,
    exudate_local_percentile: float | None = None,
) -> FeatureExtractionPayload:
    """Extract scalar features plus masks and lesion coordinates.

    The scalar feature vector is intentionally handcrafted and interpretable.
    Traditional/shallow ML uses these values as a statistical decision engine;
    no pixel-level neural network or deep feature extractor is used.
    """
    try:
        preprocessed = preprocess_retinal_image(
            image_path,
            debug=debug,
            debug_dir=debug_dir,
            clahe_clip_limit=clahe_clip_limit,
        )
        vessels, vesselness = segment_vessels(preprocessed.denoised_green, preprocessed.fov_mask)
        microaneurysms, ma_candidates = detect_microaneurysms(preprocessed, vessels)
        hemorrhages, hemorrhage_candidates = detect_hemorrhages(preprocessed, vessels)
        exudates, exudate_candidates = detect_exudates(
            preprocessed,
            vessels,
            exudate_percentile=exudate_percentile,
            exudate_local_percentile=exudate_local_percentile,
        )
        soft_exudates, soft_exudate_candidates = detect_soft_exudates(preprocessed, vessels, exudates)
        cotton_wool_spots, cotton_wool_candidates = detect_cotton_wool_spots(
            preprocessed,
            vessels,
            exudates,
        )
        glcm_by_direction = extract_multidirectional_glcm_features(
            preprocessed.denoised_green,
            preprocessed.fov_mask,
        )
        glcm_contrast, glcm_homogeneity, glcm_energy = summarize_glcm_features(glcm_by_direction)
        fov_area = max(int(np.count_nonzero(preprocessed.fov_mask)), 1)
        ma_area = int(np.count_nonzero(microaneurysms))
        exudate_area = int(np.count_nonzero(exudates))
        ma_count = count_components(microaneurysms)
        exudate_count = count_components(exudates)
        hemorrhage_area = int(np.count_nonzero(hemorrhages))
        soft_exudate_area = int(np.count_nonzero(soft_exudates))
        cotton_wool_area = int(np.count_nonzero(cotton_wool_spots))
        vessel_metrics = extract_vessel_morphology_features(vessels, preprocessed.fov_mask)
        lab_metrics = extract_lab_b_features(preprocessed.normalized_bgr, preprocessed.fov_mask, exudates)
        optic_disc_center = mask_centroid(preprocessed.optic_disc_mask)
        fov_center = mask_centroid(preprocessed.fov_mask)
        macula_center = estimate_macula_center(preprocessed.fov_mask, optic_disc_center)
        all_lesions = combine_masks(microaneurysms, hemorrhages, exudates, soft_exudates, cotton_wool_spots)
        quadrant_masks = retinal_quadrant_masks(
            preprocessed.fov_mask,
            optic_disc_center=optic_disc_center,
            fov_center=fov_center,
        )

        features = {
            "ma_count": float(ma_count),
            "ma_area": float(ma_area),
            "ma_density": float(ma_area / fov_area),
            "ma_mean_area": float(ma_area / max(ma_count, 1)),
            "exudate_count": float(exudate_count),
            "exudate_area": float(exudate_area),
            "exudate_density": float(exudate_area / fov_area),
            "exudate_mean_area": float(exudate_area / max(exudate_count, 1)),
            "vessel_density": float(np.count_nonzero(vessels) / fov_area),
            "glcm_contrast": float(glcm_contrast),
            "glcm_homogeneity": float(glcm_homogeneity),
            "glcm_energy": float(glcm_energy),
        }
        features.update(vessel_metrics)
        features.update(lab_metrics)
        features.update(
            extract_hemorrhage_features(
                hemorrhages,
                fov_area=fov_area,
            ),
        )
        features.update(
            extract_microaneurysm_advanced_features(
                microaneurysms,
                fov_area=fov_area,
                quadrant_masks=quadrant_masks,
                optic_disc_center=optic_disc_center,
            ),
        )
        features.update(
            extract_exudate_advanced_features(
                image=preprocessed.normalized_bgr,
                gray=preprocessed.denoised_green,
                hard_exudates=exudates,
                soft_exudates=soft_exudates,
                fov_area=fov_area,
                optic_disc_center=optic_disc_center,
                macula_center=macula_center,
            ),
        )
        features.update(extract_cotton_wool_features(cotton_wool_spots, quadrant_masks))
        features.update(extract_texture_features(preprocessed.denoised_green, preprocessed.fov_mask, glcm_by_direction))
        features.update(extract_color_features(preprocessed.normalized_bgr, preprocessed.fov_mask))
        features.update(extract_frequency_features(preprocessed.denoised_green, preprocessed.fov_mask))
        features.update(extract_lesion_morphology_features(all_lesions))
        features.update(
            extract_quadrant_analysis_features(
                gray=preprocessed.denoised_green,
                vessels=vessels,
                lesions=all_lesions,
                quadrant_masks=quadrant_masks,
            ),
        )
        features.update(
            extract_quality_features(
                gray=preprocessed.green_channel,
                fov_mask=preprocessed.fov_mask,
            ),
        )
        features.update(
            extract_severity_features(
                features=features,
                fov_area=fov_area,
                ma_count=ma_count,
                exudate_count=exudate_count,
                hemorrhage_count=count_components(hemorrhages),
                cotton_wool_count=count_components(cotton_wool_spots),
                combined_lesion_area=ma_area + exudate_area + hemorrhage_area + soft_exudate_area + cotton_wool_area,
            ),
        )
        features.update(extract_engineered_features(features, fov_area=fov_area))
        features.update(flatten_directional_glcm_features(glcm_by_direction))

        if debug and debug_dir is not None:
            save_feature_debug(
                Path(debug_dir),
                Path(image_path).stem,
                preprocessed,
                vessels,
                vesselness,
                microaneurysms,
                ma_candidates,
                exudates,
                exudate_candidates,
            )

        masks = {
            "fov_mask": preprocessed.fov_mask,
            "optic_disc_mask": preprocessed.optic_disc_mask,
            "vessels": vessels,
            "vesselness": vesselness,
            "microaneurysms": microaneurysms,
            "microaneurysm_candidates": ma_candidates,
            "hemorrhages": hemorrhages,
            "hemorrhage_candidates": hemorrhage_candidates,
            "exudates": exudates,
            "exudate_candidates": exudate_candidates,
            "soft_exudates": soft_exudates,
            "soft_exudate_candidates": soft_exudate_candidates,
            "cotton_wool_spots": cotton_wool_spots,
            "cotton_wool_candidates": cotton_wool_candidates,
        }
        coordinates = {
            "microaneurysms": mask_to_regions(microaneurysms, min_area=1),
            "hemorrhages": mask_to_regions(hemorrhages, min_area=config.HEMORRHAGE_MIN_AREA),
            "exudates": mask_to_regions(exudates, min_area=config.EXUDATE_MIN_AREA),
            "soft_exudates": mask_to_regions(soft_exudates, min_area=config.SOFT_EXUDATE_MIN_AREA),
            "cotton_wool_spots": mask_to_regions(cotton_wool_spots, min_area=config.COTTON_WOOL_MIN_AREA),
            "vessels": mask_to_regions(vessels, min_area=20),
        }

        return FeatureExtractionPayload(
            features={name: float(features.get(name, 0.0)) for name in config.FEATURE_NAMES},
            masks=masks,
            coordinates=coordinates,
            image_shape=preprocessed.denoised_green.shape[:2],
        )
    except Exception as exc:
        raise FeatureExtractionError(f"Failed to extract features from {image_path}: {exc}") from exc


def segment_vessels(gray: np.ndarray, fov_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized = gray.astype(np.float32) / 255.0
    vesselness_float = frangi_vesselness_opencv(
        normalized,
        sigmas=config.FRANGI_SIGMAS,
        beta=config.FRANGI_BETA,
        gamma=config.FRANGI_GAMMA,
    )
    vesselness = normalize_uint8(np.nan_to_num(vesselness_float, nan=0.0))
    vesselness[fov_mask == 0] = 0
    valid_values = vesselness[fov_mask > 0]

    vessels = np.zeros_like(gray, dtype=np.uint8)
    if valid_values.size == 0:
        return vessels, vesselness

    threshold = max(4.0, float(np.percentile(valid_values, config.VESSEL_PERCENTILE)))
    vessels[(vesselness >= threshold) & (fov_mask > 0)] = 255
    vessels = cv2.morphologyEx(vessels, cv2.MORPH_CLOSE, disk_kernel(2))
    vessels = cv2.morphologyEx(vessels, cv2.MORPH_OPEN, disk_kernel(1))
    vessels = remove_components_by_area(vessels, min_area=8, max_area=50000)
    vessels[fov_mask == 0] = 0
    return vessels, vesselness


def detect_microaneurysms(
    preprocessed: PreprocessingResult,
    vessels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vessel_exclusion = cv2.dilate(vessels, disk_kernel(2), iterations=1)
    valid_mask = build_valid_lesion_mask(
        preprocessed.fov_mask,
        preprocessed.optic_disc_mask,
        vessel_exclusion,
    )
    vessel_removed = fill_masked_pixels(
        preprocessed.denoised_green,
        vessel_exclusion,
        valid_mask,
    )
    blackhat = cv2.morphologyEx(
        vessel_removed,
        cv2.MORPH_BLACKHAT,
        disk_kernel(config.MA_BLACKHAT_RADIUS),
    )
    blackhat[valid_mask == 0] = 0
    valid_values = blackhat[valid_mask > 0]

    if valid_values.size == 0:
        empty = np.zeros_like(preprocessed.denoised_green)
        return empty, empty

    threshold = max(10.0, float(np.percentile(valid_values, config.MA_PERCENTILE)))
    image = preprocessed.normalized_bgr
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue = image[:, :, 0].astype(np.int16)
    green = image[:, :, 1].astype(np.int16)
    red = image[:, :, 2].astype(np.int16)
    fov_green = green[preprocessed.fov_mask > 0]
    median_green = float(np.median(fov_green)) if fov_green.size else 0.0

    dark_red_candidate = (
        (green <= median_green + 6.0)
        & (red >= blue + 3)
        & (hsv[:, :, 1] >= 18)
    )

    candidates = np.zeros_like(preprocessed.denoised_green)
    candidates[(blackhat >= threshold) & dark_red_candidate & (valid_mask > 0)] = 255
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, disk_kernel(1))
    filtered = filter_round_components(
        candidates,
        min_area=config.MA_MIN_AREA,
        max_area=config.MA_MAX_AREA,
        min_circularity=0.55,
        max_aspect_ratio=1.8,
        min_solidity=0.45,
    )
    filtered[valid_mask == 0] = 0
    return filtered, candidates


def detect_hemorrhages(
    preprocessed: PreprocessingResult,
    vessels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    vessel_exclusion = cv2.dilate(vessels, disk_kernel(3), iterations=1)
    valid_mask = build_valid_lesion_mask(
        preprocessed.fov_mask,
        preprocessed.optic_disc_mask,
        vessel_exclusion,
    )
    green = preprocessed.denoised_green
    blackhat = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, disk_kernel(12))
    blackhat[valid_mask == 0] = 0
    valid_values = blackhat[valid_mask > 0]

    if valid_values.size == 0:
        empty = np.zeros_like(green)
        return empty, empty

    image = preprocessed.normalized_bgr
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue = image[:, :, 0].astype(np.int16)
    green_channel = image[:, :, 1].astype(np.int16)
    red = image[:, :, 2].astype(np.int16)
    fov_green = green_channel[preprocessed.fov_mask > 0]
    median_green = float(np.median(fov_green)) if fov_green.size else 0.0
    threshold = max(14.0, float(np.percentile(valid_values, 98.8)))
    dark_red = (
        (blackhat >= threshold)
        & (green_channel <= median_green + 12.0)
        & (red >= blue)
        & (hsv[:, :, 1] >= 18)
        & (valid_mask > 0)
    )

    candidates = np.zeros_like(green)
    candidates[dark_red] = 255
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, disk_kernel(3))
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, disk_kernel(1))
    hemorrhages = filter_components_by_shape(
        candidates,
        min_area=config.HEMORRHAGE_MIN_AREA,
        max_area=config.HEMORRHAGE_MAX_AREA,
        max_aspect_ratio=6.0,
        min_solidity=0.18,
    )
    hemorrhages[valid_mask == 0] = 0
    return hemorrhages, candidates


def detect_exudates(
    preprocessed: PreprocessingResult,
    vessels: np.ndarray,
    exudate_percentile: float | None = None,
    exudate_local_percentile: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    image = preprocessed.normalized_bgr
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    b_channel = lab[:, :, 2]
    local_bright = cv2.morphologyEx(lightness, cv2.MORPH_TOPHAT, disk_kernel(8))
    vessel_exclusion = cv2.dilate(vessels, disk_kernel(1), iterations=1)
    optic_disc_exclusion = cv2.dilate(preprocessed.optic_disc_mask, disk_kernel(12), iterations=1)
    interior_mask = create_interior_fundus_mask(preprocessed.fov_mask, margin_ratio=0.035)
    valid_mask = build_valid_lesion_mask(interior_mask, optic_disc_exclusion, vessel_exclusion)
    valid_values = lightness[valid_mask > 0]

    if valid_values.size == 0:
        empty = np.zeros_like(lightness)
        return empty, empty

    threshold = max(
        masked_otsu_threshold(valid_values),
        float(np.percentile(valid_values, exudate_percentile or config.EXUDATE_PERCENTILE)),
    )
    local_values = local_bright[valid_mask > 0]
    local_threshold = max(
        10.0,
        float(
            np.percentile(
                local_values,
                exudate_local_percentile or config.EXUDATE_LOCAL_PERCENTILE,
            ),
        ),
    )
    b_values = b_channel[valid_mask > 0]
    b_threshold = max(128.0, float(np.percentile(b_values, config.EXUDATE_B_PERCENTILE)))
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


def detect_soft_exudates(
    preprocessed: PreprocessingResult,
    vessels: np.ndarray,
    hard_exudates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    image = preprocessed.normalized_bgr
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lightness = lab[:, :, 0]
    b_channel = lab[:, :, 2]
    saturation = hsv[:, :, 1]
    vessel_exclusion = cv2.dilate(vessels, disk_kernel(2), iterations=1)
    optic_disc_exclusion = cv2.dilate(preprocessed.optic_disc_mask, disk_kernel(14), iterations=1)
    hard_exudate_exclusion = cv2.dilate(hard_exudates, disk_kernel(2), iterations=1)
    valid_mask = build_valid_lesion_mask(
        create_interior_fundus_mask(preprocessed.fov_mask, margin_ratio=0.035),
        optic_disc_exclusion,
        cv2.bitwise_or(vessel_exclusion, hard_exudate_exclusion),
    )
    valid_values = lightness[valid_mask > 0]
    if valid_values.size == 0:
        empty = np.zeros_like(lightness)
        return empty, empty

    bright_threshold = max(150.0, float(np.percentile(valid_values, 96.5)))
    b_values = b_channel[valid_mask > 0]
    b_upper = float(np.percentile(b_values, 90.0)) if b_values.size else 255.0
    candidates = np.zeros_like(lightness)
    candidates[
        (lightness >= bright_threshold)
        & (b_channel <= b_upper)
        & (saturation <= 120)
        & (valid_mask > 0)
    ] = 255
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, disk_kernel(4))
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, disk_kernel(2))
    soft_exudates = filter_components_by_shape(
        candidates,
        min_area=config.SOFT_EXUDATE_MIN_AREA,
        max_area=config.SOFT_EXUDATE_MAX_AREA,
        max_aspect_ratio=5.0,
        min_solidity=0.15,
    )
    soft_exudates[valid_mask == 0] = 0
    return soft_exudates, candidates


def detect_cotton_wool_spots(
    preprocessed: PreprocessingResult,
    vessels: np.ndarray,
    hard_exudates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    image = preprocessed.normalized_bgr
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lightness = lab[:, :, 0]
    saturation = hsv[:, :, 1]
    local_bright = cv2.morphologyEx(lightness, cv2.MORPH_TOPHAT, disk_kernel(14))
    blocked = cv2.bitwise_or(cv2.dilate(vessels, disk_kernel(2), iterations=1), cv2.dilate(hard_exudates, disk_kernel(3), iterations=1))
    valid_mask = build_valid_lesion_mask(
        create_interior_fundus_mask(preprocessed.fov_mask, margin_ratio=0.04),
        cv2.dilate(preprocessed.optic_disc_mask, disk_kernel(14), iterations=1),
        blocked,
    )
    valid_values = lightness[valid_mask > 0]
    if valid_values.size == 0:
        empty = np.zeros_like(lightness)
        return empty, empty

    light_threshold = max(145.0, float(np.percentile(valid_values, 95.5)))
    local_threshold = max(8.0, float(np.percentile(local_bright[valid_mask > 0], 94.0)))
    candidates = np.zeros_like(lightness)
    candidates[
        (lightness >= light_threshold)
        & (local_bright >= local_threshold)
        & (saturation <= 145)
        & (valid_mask > 0)
    ] = 255
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, disk_kernel(5))
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, disk_kernel(2))
    spots = filter_components_by_shape(
        candidates,
        min_area=config.COTTON_WOOL_MIN_AREA,
        max_area=config.COTTON_WOOL_MAX_AREA,
        max_aspect_ratio=4.5,
        min_solidity=0.12,
    )
    spots[valid_mask == 0] = 0
    return spots, candidates


def extract_glcm_features(gray: np.ndarray, fov_mask: np.ndarray) -> tuple[float, float, float]:
    return summarize_glcm_features(extract_multidirectional_glcm_features(gray, fov_mask))


def extract_multidirectional_glcm_features(
    gray: np.ndarray,
    fov_mask: np.ndarray,
) -> dict[str, dict[str, float]]:
    levels = config.GLCM_LEVELS
    mask = fov_mask > 0
    if gray.size == 0 or not np.any(mask):
        return {
            angle: {
                "contrast": 0.0,
                "homogeneity": 0.0,
                "energy": 0.0,
                "correlation": 0.0,
                "dissimilarity": 0.0,
                "entropy": 0.0,
            }
            for angle in config.GLCM_DIRECTION_OFFSETS
        }

    quantized = np.clip(gray.astype(np.uint16) * levels // 256, 0, levels - 1).astype(np.int32)
    indices = np.arange(levels, dtype=np.float64)
    diff = indices[:, None] - indices[None, :]
    contrast_weights = diff**2
    dissimilarity_weights = np.abs(diff)
    homogeneity_weights = 1.0 / (1.0 + np.abs(diff))
    contrast_scale = (256.0 / levels) ** 2
    dissimilarity_scale = 256.0 / levels

    output: dict[str, dict[str, float]] = {}

    for angle, (dy, dx) in config.GLCM_DIRECTION_OFFSETS.items():
        y0 = max(0, dy)
        y1 = gray.shape[0] + min(0, dy)
        x0 = max(0, dx)
        x1 = gray.shape[1] + min(0, dx)

        source = quantized[y0:y1, x0:x1]
        target = quantized[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
        pair_mask = mask[y0:y1, x0:x1] & mask[y0 - dy : y1 - dy, x0 - dx : x1 - dx]

        if source.size == 0 or not np.any(pair_mask):
            output[angle] = {
                "contrast": 0.0,
                "homogeneity": 0.0,
                "energy": 0.0,
                "correlation": 0.0,
                "dissimilarity": 0.0,
                "entropy": 0.0,
            }
            continue

        matrix = np.zeros((levels, levels), dtype=np.float64)
        np.add.at(matrix, (source[pair_mask].ravel(), target[pair_mask].ravel()), 1.0)
        matrix += matrix.T
        matrix_sum = float(matrix.sum())

        if matrix_sum <= 0:
            output[angle] = {
                "contrast": 0.0,
                "homogeneity": 0.0,
                "energy": 0.0,
                "correlation": 0.0,
                "dissimilarity": 0.0,
                "entropy": 0.0,
            }
            continue

        probabilities = matrix / matrix_sum
        px = probabilities.sum(axis=1)
        py = probabilities.sum(axis=0)
        mean_x = float(np.sum(indices * px))
        mean_y = float(np.sum(indices * py))
        std_x = float(np.sqrt(np.sum(((indices - mean_x) ** 2) * px)))
        std_y = float(np.sqrt(np.sum(((indices - mean_y) ** 2) * py)))
        correlation = 0.0
        if std_x > 1e-8 and std_y > 1e-8:
            centered_x = indices[:, None] - mean_x
            centered_y = indices[None, :] - mean_y
            correlation = float(np.sum(probabilities * centered_x * centered_y) / (std_x * std_y))
        nonzero_probabilities = probabilities[probabilities > 0]
        output[angle] = {
            "contrast": float(np.sum(probabilities * contrast_weights) * contrast_scale),
            "homogeneity": float(np.sum(probabilities * homogeneity_weights)),
            "energy": float(np.sqrt(np.sum(probabilities**2))),
            "correlation": correlation,
            "dissimilarity": float(np.sum(probabilities * dissimilarity_weights) * dissimilarity_scale),
            "entropy": float(-np.sum(nonzero_probabilities * np.log2(nonzero_probabilities))),
        }

    return output


def summarize_glcm_features(glcm_by_direction: dict[str, dict[str, float]]) -> tuple[float, float, float]:
    if not glcm_by_direction:
        return 0.0, 0.0, 0.0

    contrasts = [values["contrast"] for values in glcm_by_direction.values()]
    homogeneities = [values["homogeneity"] for values in glcm_by_direction.values()]
    energies = [values["energy"] for values in glcm_by_direction.values()]

    return (
        float(np.mean(contrasts)),
        float(np.mean(homogeneities)),
        float(np.mean(energies)),
    )


def flatten_directional_glcm_features(glcm_by_direction: dict[str, dict[str, float]]) -> dict[str, float]:
    features: dict[str, float] = {}

    for angle in ("0", "45", "90", "135"):
        values = glcm_by_direction.get(angle, {"contrast": 0.0, "energy": 0.0})
        features[f"glcm_contrast_{angle}"] = float(values.get("contrast", 0.0))
        features[f"glcm_energy_{angle}"] = float(values.get("energy", 0.0))

    return features


def extract_vessel_morphology_features(vessels: np.ndarray, fov_mask: np.ndarray) -> dict[str, float]:
    skeleton = skeletonize_binary_mask(vessels)
    skeleton_length = int(np.count_nonzero(skeleton))
    vessel_area = int(np.count_nonzero(vessels))
    fov_area = max(int(np.count_nonzero(fov_mask)), 1)
    neighbor_counts = count_skeleton_neighbors(skeleton)
    endpoints = int(np.count_nonzero((skeleton > 0) & (neighbor_counts == 1)))
    branchpoints = int(np.count_nonzero((skeleton > 0) & (neighbor_counts >= 3)))
    tortuosities = vessel_component_tortuosities(skeleton)
    component_count = max(count_components(skeleton), 1)
    width_values = vessel_width_values(vessels, skeleton)
    curvature_values = skeleton_curvature_values(skeleton)
    tortuosity_mean = float(np.mean(tortuosities)) if tortuosities else 0.0
    fragmentation = float(component_count / max(skeleton_length, 1))
    branch_density = float(branchpoints / max(skeleton_length, 1))
    curvature_mean = float(np.mean(curvature_values)) if curvature_values else 0.0

    return {
        "vessel_skeleton_length": float(skeleton_length),
        "vessel_endpoint_count": float(endpoints),
        "vessel_branchpoint_count": float(branchpoints),
        "vessel_tortuosity_mean": tortuosity_mean,
        "vessel_tortuosity_max": float(np.max(tortuosities)) if tortuosities else 0.0,
        "vessel_tortuosity_std": float(np.std(tortuosities)) if tortuosities else 0.0,
        "vessel_area_ratio": float(vessel_area / fov_area),
        "vessel_length": float(skeleton_length),
        "vessel_branching_count": float(branchpoints),
        "vessel_average_width": float(np.mean(width_values)) if width_values.size else 0.0,
        "vessel_fragmentation_index": fragmentation,
        "vessel_complexity_score": float((branch_density * 100.0) + tortuosity_mean + curvature_mean),
        "vessel_curvature_mean": curvature_mean,
        "vessel_curvature_std": float(np.std(curvature_values)) if curvature_values else 0.0,
        "vessel_curvature_max": float(np.max(curvature_values)) if curvature_values else 0.0,
    }


def skeletonize_binary_mask(mask: np.ndarray) -> np.ndarray:
    binary = mask > 0

    if skimage_skeletonize is not None:
        return (skimage_skeletonize(binary).astype(np.uint8) * 255)

    # OpenCV fallback for environments where scikit-image is unavailable.
    working = binary.astype(np.uint8) * 255
    skeleton = np.zeros_like(working)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        opened = cv2.morphologyEx(working, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(working, opened)
        eroded = cv2.erode(working, element)
        skeleton = cv2.bitwise_or(skeleton, temp)
        working = eroded.copy()

        if cv2.countNonZero(working) == 0:
            break

    return skeleton


def count_skeleton_neighbors(skeleton: np.ndarray) -> np.ndarray:
    binary = (skeleton > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_counts = cv2.filter2D(binary, cv2.CV_16S, kernel, borderType=cv2.BORDER_CONSTANT)
    return neighbor_counts - binary


def vessel_component_tortuosities(skeleton: np.ndarray) -> list[float]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (skeleton > 0).astype(np.uint8),
        connectivity=8,
    )
    tortuosities: list[float] = []

    for label in range(1, count):
        arc_length = float(stats[label, cv2.CC_STAT_AREA])
        if arc_length < 12:
            continue

        y_coords, x_coords = np.where(labels == label)
        if x_coords.size < 2:
            continue

        chord = component_chord_length(x_coords, y_coords)
        if chord <= 1.0:
            continue

        tortuosities.append(float(arc_length / chord))

    return tortuosities


def component_chord_length(x_coords: np.ndarray, y_coords: np.ndarray) -> float:
    points = np.column_stack([x_coords.astype(np.float64), y_coords.astype(np.float64)])

    if points.shape[0] <= 2:
        return float(np.linalg.norm(points[-1] - points[0]))

    # Approximate the largest endpoint separation without an expensive all-pairs distance matrix.
    centroid = np.mean(points, axis=0)
    first = points[int(np.argmax(np.sum((points - centroid) ** 2, axis=1)))]
    second = points[int(np.argmax(np.sum((points - first) ** 2, axis=1)))]
    return float(np.linalg.norm(second - first))


def vessel_width_values(vessels: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
    if not np.any(vessels > 0) or not np.any(skeleton > 0):
        return np.array([], dtype=np.float32)
    distance = cv2.distanceTransform((vessels > 0).astype(np.uint8), cv2.DIST_L2, 5)
    return (distance[skeleton > 0] * 2.0).astype(np.float32)


def skeleton_curvature_values(skeleton: np.ndarray) -> list[float]:
    binary = skeleton > 0
    y_coords, x_coords = np.where(binary)
    values: list[float] = []
    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    for y, x in zip(y_coords.tolist()[::2], x_coords.tolist()[::2]):
        neighbors: list[np.ndarray] = []
        for dy, dx in offsets:
            yy = y + dy
            xx = x + dx
            if 0 <= yy < binary.shape[0] and 0 <= xx < binary.shape[1] and binary[yy, xx]:
                neighbors.append(np.array([dx, dy], dtype=np.float64))
        if len(neighbors) != 2:
            continue
        a, b = neighbors
        denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-8)
        cosine = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
        angle = float(np.arccos(cosine))
        values.append(abs(np.pi - angle))

    return values


def extract_lab_b_features(
    image: np.ndarray,
    fov_mask: np.ndarray,
    exudates: np.ndarray,
) -> dict[str, float]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    b_channel = lab[:, :, 2].astype(np.float32)
    fov_values = b_channel[fov_mask > 0]
    exudate_values = b_channel[exudates > 0]

    return {
        "lab_b_mean": float(np.mean(fov_values)) if fov_values.size else 0.0,
        "lab_b_std": float(np.std(fov_values)) if fov_values.size else 0.0,
        "lab_b_exudate_mean": float(np.mean(exudate_values)) if exudate_values.size else 0.0,
        "lab_b_exudate_std": float(np.std(exudate_values)) if exudate_values.size else 0.0,
    }


def extract_hemorrhage_features(mask: np.ndarray, fov_area: int) -> dict[str, float]:
    stats = component_area_stats(mask)
    area = int(np.count_nonzero(mask))
    return {
        "hemorrhage_count": float(stats["count"]),
        "hemorrhage_area": float(area),
        "hemorrhage_largest_area": float(stats["max_area"]),
        "hemorrhage_mean_area": float(stats["mean_area"]),
        "hemorrhage_density": float(area / max(fov_area, 1)),
        "hemorrhage_retina_affected_pct": float((area / max(fov_area, 1)) * 100.0),
    }


def extract_microaneurysm_advanced_features(
    mask: np.ndarray,
    fov_area: int,
    quadrant_masks: dict[str, np.ndarray],
    optic_disc_center: tuple[float, float] | None,
) -> dict[str, float]:
    stats = component_area_stats(mask)
    distances = component_distances_to_point(mask, optic_disc_center)
    features = {
        "ma_max_area": float(stats["max_area"]),
        "ma_distance_to_optic_disc_mean": float(np.mean(distances)) if distances else 0.0,
        "ma_distance_to_optic_disc_min": float(np.min(distances)) if distances else 0.0,
        "ma_density_per_retinal_area": float(stats["count"] / max(fov_area, 1)),
    }
    for quadrant in ("superior", "inferior", "nasal", "temporal"):
        features[f"ma_{quadrant}_count"] = float(count_components(mask_in_region(mask, quadrant_masks[quadrant])))
    return features


def extract_exudate_advanced_features(
    image: np.ndarray,
    gray: np.ndarray,
    hard_exudates: np.ndarray,
    soft_exudates: np.ndarray,
    fov_area: int,
    optic_disc_center: tuple[float, float] | None,
    macula_center: tuple[float, float] | None,
) -> dict[str, float]:
    features: dict[str, float] = {}
    for prefix, mask in (("hard_exudate", hard_exudates), ("soft_exudate", soft_exudates)):
        stats = component_area_stats(mask)
        pixels = gray[mask > 0]
        texture = local_variance_image(gray)
        texture_pixels = texture[mask > 0]
        od_distances = component_distances_to_point(mask, optic_disc_center)
        macula_distances = component_distances_to_point(mask, macula_center)
        features.update(
            {
                f"{prefix}_count": float(stats["count"]),
                f"{prefix}_area": float(np.count_nonzero(mask)),
                f"{prefix}_mean_area": float(stats["mean_area"]),
                f"{prefix}_max_area": float(stats["max_area"]),
                f"{prefix}_brightness_mean": float(np.mean(pixels)) if pixels.size else 0.0,
                f"{prefix}_brightness_std": float(np.std(pixels)) if pixels.size else 0.0,
                f"{prefix}_texture_mean": float(np.mean(texture_pixels)) if texture_pixels.size else 0.0,
                f"{prefix}_texture_std": float(np.std(texture_pixels)) if texture_pixels.size else 0.0,
                f"{prefix}_coverage_pct": float((np.count_nonzero(mask) / max(fov_area, 1)) * 100.0),
                f"{prefix}_distance_to_macula_mean": float(np.mean(macula_distances)) if macula_distances else 0.0,
                f"{prefix}_distance_to_optic_disc_mean": float(np.mean(od_distances)) if od_distances else 0.0,
            },
        )
    return features


def extract_cotton_wool_features(mask: np.ndarray, quadrant_masks: dict[str, np.ndarray]) -> dict[str, float]:
    morphology = contour_morphology_stats(mask)
    quadrant_counts = np.array(
        [
            count_components(mask_in_region(mask, quadrant_masks[quadrant]))
            for quadrant in ("superior", "inferior", "nasal", "temporal")
        ],
        dtype=np.float64,
    )
    return {
        "cotton_wool_count": float(count_components(mask)),
        "cotton_wool_area": float(np.count_nonzero(mask)),
        "cotton_wool_mean_area": float(component_area_stats(mask)["mean_area"]),
        "cotton_wool_circularity_mean": morphology["circularity_mean"],
        "cotton_wool_solidity_mean": morphology["solidity_mean"],
        "cotton_wool_aspect_ratio_mean": morphology["aspect_ratio_mean"],
        "cotton_wool_distribution_entropy": normalized_entropy(quadrant_counts),
    }


def extract_texture_features(
    gray: np.ndarray,
    fov_mask: np.ndarray,
    glcm_by_direction: dict[str, dict[str, float]],
) -> dict[str, float]:
    mask = fov_mask > 0
    pixels = gray[mask].astype(np.float64)
    variance = local_variance_image(gray)
    variance_pixels = variance[mask]
    lbp_values = local_binary_pattern_values(gray, fov_mask)
    correlations = [values.get("correlation", 0.0) for values in glcm_by_direction.values()]
    dissimilarities = [values.get("dissimilarity", 0.0) for values in glcm_by_direction.values()]
    entropies = [values.get("entropy", 0.0) for values in glcm_by_direction.values()]
    return {
        "glcm_correlation": float(np.mean(correlations)) if correlations else 0.0,
        "glcm_dissimilarity": float(np.mean(dissimilarities)) if dissimilarities else 0.0,
        "glcm_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "lbp_uniform_ratio": lbp_uniform_ratio(lbp_values),
        "lbp_entropy": histogram_entropy(lbp_values, bins=32),
        "lbp_mean": float(np.mean(lbp_values)) if lbp_values.size else 0.0,
        "lbp_std": float(np.std(lbp_values)) if lbp_values.size else 0.0,
        "local_texture_variance_mean": float(np.mean(variance_pixels)) if variance_pixels.size else 0.0,
        "local_texture_variance_std": float(np.std(variance_pixels)) if variance_pixels.size else 0.0,
        "texture_mean": float(np.mean(pixels)) if pixels.size else 0.0,
        "texture_variance": float(np.var(pixels)) if pixels.size else 0.0,
        "texture_std": float(np.std(pixels)) if pixels.size else 0.0,
        "texture_skewness": skewness(pixels),
        "texture_kurtosis": kurtosis(pixels),
    }


def extract_color_features(image: np.ndarray, fov_mask: np.ndarray) -> dict[str, float]:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    spaces = {
        "rgb": (rgb, ("r", "g", "b")),
        "hsv": (hsv, ("h", "s", "v")),
        "lab_color": (lab, ("l", "a", "b")),
    }
    mask = fov_mask > 0
    features: dict[str, float] = {}
    for space_name, (space_image, channels) in spaces.items():
        for index, channel_name in enumerate(channels):
            values = space_image[:, :, index][mask].astype(np.float64)
            prefix = f"{space_name}_{channel_name}"
            features[f"{prefix}_mean"] = float(np.mean(values)) if values.size else 0.0
            features[f"{prefix}_std"] = float(np.std(values)) if values.size else 0.0
            features[f"{prefix}_min"] = float(np.min(values)) if values.size else 0.0
            features[f"{prefix}_max"] = float(np.max(values)) if values.size else 0.0
            features[f"{prefix}_entropy"] = histogram_entropy(values, bins=32)
    return features


def extract_frequency_features(gray: np.ndarray, fov_mask: np.ndarray) -> dict[str, float]:
    prepared = fill_outside_mask(gray, fov_mask)
    small = cv2.resize(prepared, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    small -= float(np.mean(small))
    fft = np.fft.fftshift(np.fft.fft2(small))
    magnitude = np.abs(fft)
    power = magnitude**2
    total = float(np.sum(power)) or 1.0
    yy, xx = np.indices(power.shape)
    center = (np.array(power.shape) - 1) / 2.0
    radius = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
    max_radius = float(np.max(radius)) or 1.0
    low = power[radius <= max_radius * 0.15]
    mid = power[(radius > max_radius * 0.15) & (radius <= max_radius * 0.45)]
    high = power[radius > max_radius * 0.45]
    threshold = np.percentile(power, 99.5)
    dominant_radius = radius[power >= threshold]
    approx, horizontal, vertical, diagonal = haar_wavelet_bands(small)
    return {
        "fft_energy": total,
        "fft_low_frequency_ratio": float(np.sum(low) / total),
        "fft_mid_frequency_ratio": float(np.sum(mid) / total),
        "fft_high_frequency_ratio": float(np.sum(high) / total),
        "fft_dominant_radius_mean": float(np.mean(dominant_radius)) if dominant_radius.size else 0.0,
        "fft_dominant_radius_std": float(np.std(dominant_radius)) if dominant_radius.size else 0.0,
        "wavelet_approx_energy": band_energy(approx),
        "wavelet_horizontal_energy": band_energy(horizontal),
        "wavelet_vertical_energy": band_energy(vertical),
        "wavelet_diagonal_energy": band_energy(diagonal),
        "wavelet_detail_energy": band_energy(horizontal) + band_energy(vertical) + band_energy(diagonal),
    }


def extract_lesion_morphology_features(mask: np.ndarray) -> dict[str, float]:
    morphology = contour_morphology_stats(mask)
    return {
        "all_lesion_area": float(np.count_nonzero(mask)),
        "all_lesion_perimeter_mean": morphology["perimeter_mean"],
        "all_lesion_circularity_mean": morphology["circularity_mean"],
        "all_lesion_solidity_mean": morphology["solidity_mean"],
        "all_lesion_eccentricity_mean": morphology["eccentricity_mean"],
        "all_lesion_compactness_mean": morphology["compactness_mean"],
        "all_lesion_aspect_ratio_mean": morphology["aspect_ratio_mean"],
        "all_lesion_convex_hull_area_mean": morphology["convex_hull_area_mean"],
    }


def extract_quadrant_analysis_features(
    gray: np.ndarray,
    vessels: np.ndarray,
    lesions: np.ndarray,
    quadrant_masks: dict[str, np.ndarray],
) -> dict[str, float]:
    features: dict[str, float] = {}
    for quadrant, region in quadrant_masks.items():
        region_area = max(int(np.count_nonzero(region)), 1)
        region_pixels = gray[region > 0]
        lesion_region = mask_in_region(lesions, region)
        features[f"{quadrant}_lesion_count"] = float(count_components(lesion_region))
        features[f"{quadrant}_lesion_density"] = float(np.count_nonzero(lesion_region) / region_area)
        features[f"{quadrant}_vessel_density"] = float(np.count_nonzero(mask_in_region(vessels, region)) / region_area)
        features[f"{quadrant}_texture_mean"] = float(np.mean(region_pixels)) if region_pixels.size else 0.0
        features[f"{quadrant}_texture_std"] = float(np.std(region_pixels)) if region_pixels.size else 0.0
    return features


def extract_quality_features(gray: np.ndarray, fov_mask: np.ndarray) -> dict[str, float]:
    pixels = gray[fov_mask > 0].astype(np.float64)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = float(np.mean(np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0))) + np.mean(np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1))))
    brightness = float(np.mean(pixels)) if pixels.size else 0.0
    contrast = float(np.std(pixels)) if pixels.size else 0.0
    noise = gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigmaX=2)
    noise_values = noise[fov_mask > 0]
    snr = brightness / max(float(np.std(noise_values)) if noise_values.size else 0.0, 1e-6)
    return {
        "quality_blur_score": blur,
        "quality_sharpness": sharpness,
        "quality_brightness": brightness,
        "quality_contrast": contrast,
        "quality_snr": snr,
    }


def extract_severity_features(
    features: dict[str, float],
    fov_area: int,
    ma_count: int,
    exudate_count: int,
    hemorrhage_count: int,
    cotton_wool_count: int,
    combined_lesion_area: int,
) -> dict[str, float]:
    total_lesions = ma_count + exudate_count + hemorrhage_count + cotton_wool_count
    lesion_density = combined_lesion_area / max(fov_area, 1)
    vessel_abnormality = (
        features.get("vessel_tortuosity_mean", 0.0)
        + features.get("vessel_fragmentation_index", 0.0) * 100.0
        + features.get("vessel_curvature_mean", 0.0)
    )
    exudate_burden = features.get("exudate_density", 0.0) + features.get("soft_exudate_coverage_pct", 0.0) / 100.0
    advanced_indicator = (
        hemorrhage_count * 0.5
        + cotton_wool_count * 0.7
        + exudate_burden * 100.0
        + vessel_abnormality
    )
    return {
        "total_lesion_count": float(total_lesions),
        "combined_lesion_burden": float(combined_lesion_area),
        "lesion_density_score": float(lesion_density),
        "hemorrhage_to_ma_ratio": safe_ratio(hemorrhage_count, ma_count),
        "vessel_abnormality_score": float(vessel_abnormality),
        "exudate_burden_score": float(exudate_burden),
        "advanced_dr_indicator_score": float(advanced_indicator),
        "neovascularization_likelihood_score": float(
            features.get("vessel_branchpoint_count", 0.0) * features.get("vessel_tortuosity_mean", 0.0) / max(fov_area / 10000.0, 1.0),
        ),
    }


def extract_engineered_features(features: dict[str, float], fov_area: int) -> dict[str, float]:
    lesion_density = features.get("lesion_density_score", 0.0)
    contrast = max(features.get("quality_contrast", 0.0), 1.0)
    return {
        "ma_to_exudate_ratio": safe_ratio(features.get("ma_count", 0.0), features.get("exudate_count", 0.0)),
        "hemorrhage_to_exudate_ratio": safe_ratio(features.get("hemorrhage_count", 0.0), features.get("exudate_count", 0.0)),
        "lesion_vessel_interaction": float(lesion_density * features.get("vessel_density", 0.0)),
        "texture_lesion_interaction": float(lesion_density * features.get("glcm_contrast", 0.0)),
        "quality_adjusted_lesion_density": float(lesion_density * (contrast / 50.0)),
        "area_adjusted_ma_count": float(features.get("ma_count", 0.0) / max(fov_area / 10000.0, 1.0)),
        "area_adjusted_exudate_count": float(features.get("exudate_count", 0.0) / max(fov_area / 10000.0, 1.0)),
        "area_adjusted_hemorrhage_count": float(features.get("hemorrhage_count", 0.0) / max(fov_area / 10000.0, 1.0)),
        "referable_lesion_score": float(features.get("exudate_burden_score", 0.0) + features.get("hemorrhage_to_ma_ratio", 0.0) + lesion_density),
        "stage_progression_score": float(features.get("advanced_dr_indicator_score", 0.0) + features.get("vessel_abnormality_score", 0.0)),
    }


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


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def combine_masks(*masks: np.ndarray) -> np.ndarray:
    if not masks:
        raise ValueError("At least one mask is required.")
    output = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        output = cv2.bitwise_or(output, (mask > 0).astype(np.uint8) * 255)
    return output


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    moments = cv2.moments((mask > 0).astype(np.uint8))
    if moments["m00"] == 0:
        return None
    return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])


def estimate_macula_center(
    fov_mask: np.ndarray,
    optic_disc_center: tuple[float, float] | None,
) -> tuple[float, float] | None:
    fov_center = mask_centroid(fov_mask)
    if fov_center is None:
        return None
    if optic_disc_center is None:
        return fov_center
    contour = largest_contour(fov_mask)
    if contour is None:
        return fov_center
    x, _, width, _ = cv2.boundingRect(contour)
    direction = -1.0 if optic_disc_center[0] >= fov_center[0] else 1.0
    macula_x = float(np.clip(optic_disc_center[0] + (direction * width * 0.28), x, x + width))
    return macula_x, fov_center[1]


def retinal_quadrant_masks(
    fov_mask: np.ndarray,
    optic_disc_center: tuple[float, float] | None,
    fov_center: tuple[float, float] | None,
) -> dict[str, np.ndarray]:
    if fov_center is None:
        fov_center = (fov_mask.shape[1] / 2.0, fov_mask.shape[0] / 2.0)
    cx, cy = fov_center
    y_indices, x_indices = np.indices(fov_mask.shape[:2])
    valid = fov_mask > 0
    if optic_disc_center is not None:
        nasal_is_right = optic_disc_center[0] >= cx
    else:
        nasal_is_right = True
    nasal_selector = x_indices >= cx if nasal_is_right else x_indices < cx
    temporal_selector = ~nasal_selector
    return {
        "superior": binary_mask(valid & (y_indices < cy)),
        "inferior": binary_mask(valid & (y_indices >= cy)),
        "nasal": binary_mask(valid & nasal_selector),
        "temporal": binary_mask(valid & temporal_selector),
    }


def binary_mask(condition: np.ndarray) -> np.ndarray:
    return condition.astype(np.uint8) * 255


def mask_in_region(mask: np.ndarray, region: np.ndarray) -> np.ndarray:
    output = np.zeros_like(mask, dtype=np.uint8)
    output[(mask > 0) & (region > 0)] = 255
    return output


def component_area_stats(mask: np.ndarray) -> dict[str, float]:
    count, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    areas = [float(stats[label, cv2.CC_STAT_AREA]) for label in range(1, count)]
    return {
        "count": float(len(areas)),
        "mean_area": float(np.mean(areas)) if areas else 0.0,
        "max_area": float(np.max(areas)) if areas else 0.0,
    }


def component_distances_to_point(mask: np.ndarray, point: tuple[float, float] | None) -> list[float]:
    if point is None:
        return []
    _, _, _, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    px, py = point
    distances: list[float] = []
    for centroid in centroids[1:]:
        distances.append(float(np.hypot(float(centroid[0]) - px, float(centroid[1]) - py)))
    return distances


def contour_morphology_stats(mask: np.ndarray) -> dict[str, float]:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    values: dict[str, list[float]] = {
        "perimeter": [],
        "circularity": [],
        "solidity": [],
        "eccentricity": [],
        "compactness": [],
        "aspect_ratio": [],
        "convex_hull_area": [],
    }
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        circularity, aspect_ratio, solidity = contour_shape_metrics(contour)
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        values["perimeter"].append(perimeter)
        values["circularity"].append(circularity)
        values["solidity"].append(solidity)
        values["eccentricity"].append(contour_eccentricity(contour))
        values["compactness"].append(float((perimeter * perimeter) / max(area, 1e-6)))
        values["aspect_ratio"].append(aspect_ratio)
        values["convex_hull_area"].append(hull_area)
    return {
        f"{name}_mean": float(np.mean(items)) if items else 0.0
        for name, items in values.items()
    }


def contour_eccentricity(contour: np.ndarray) -> float:
    if len(contour) >= 5:
        _, axes, _ = cv2.fitEllipse(contour)
        major = max(float(axes[0]), float(axes[1]))
        minor = min(float(axes[0]), float(axes[1]))
        if major > 0:
            return float(np.sqrt(max(0.0, 1.0 - ((minor * minor) / (major * major)))))
    return 0.0


def local_variance_image(gray: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    values = gray.astype(np.float32)
    mean = cv2.blur(values, (kernel_size, kernel_size))
    mean_sq = cv2.blur(values * values, (kernel_size, kernel_size))
    return np.maximum(mean_sq - (mean * mean), 0.0)


def local_binary_pattern_values(gray: np.ndarray, fov_mask: np.ndarray) -> np.ndarray:
    if skimage_local_binary_pattern is not None:
        lbp = skimage_local_binary_pattern(gray, P=8, R=1, method="uniform")
        return lbp[fov_mask > 0].astype(np.float64)

    center = gray[1:-1, 1:-1]
    codes = np.zeros_like(center, dtype=np.uint8)
    shifts = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
    ]
    for bit, (dy, dx) in enumerate(shifts):
        neighbor = gray[1 + dy : gray.shape[0] - 1 + dy, 1 + dx : gray.shape[1] - 1 + dx]
        codes |= ((neighbor >= center).astype(np.uint8) << bit)
    mask = fov_mask[1:-1, 1:-1] > 0
    return codes[mask].astype(np.float64)


def lbp_uniform_ratio(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    # scikit-image uniform LBP uses P + 1 for non-uniform bins when P=8.
    return float(np.count_nonzero(values <= 8) / values.size)


def histogram_entropy(values: np.ndarray, bins: int) -> float:
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=bins)
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    probabilities = hist.astype(np.float64) / total
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def normalized_entropy(values: np.ndarray) -> float:
    total = float(np.sum(values))
    if total <= 0:
        return 0.0
    probabilities = values.astype(np.float64) / total
    probabilities = probabilities[probabilities > 0]
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    return float(entropy / max(np.log2(values.size), 1e-8))


def skewness(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    std = float(np.std(values))
    if std <= 1e-8:
        return 0.0
    centered = values - float(np.mean(values))
    return float(np.mean((centered / std) ** 3))


def kurtosis(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    std = float(np.std(values))
    if std <= 1e-8:
        return 0.0
    centered = values - float(np.mean(values))
    return float(np.mean((centered / std) ** 4) - 3.0)


def haar_wavelet_bands(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if image.shape[0] % 2 == 1:
        image = image[:-1, :]
    if image.shape[1] % 2 == 1:
        image = image[:, :-1]
    even_rows = image[0::2, :]
    odd_rows = image[1::2, :]
    low_rows = (even_rows + odd_rows) / 2.0
    high_rows = (even_rows - odd_rows) / 2.0
    ll = (low_rows[:, 0::2] + low_rows[:, 1::2]) / 2.0
    lh = (low_rows[:, 0::2] - low_rows[:, 1::2]) / 2.0
    hl = (high_rows[:, 0::2] + high_rows[:, 1::2]) / 2.0
    hh = (high_rows[:, 0::2] - high_rows[:, 1::2]) / 2.0
    return ll, lh, hl, hh


def band_energy(band: np.ndarray) -> float:
    return float(np.sum(np.square(band.astype(np.float64))))


def build_valid_lesion_mask(
    fov_mask: np.ndarray,
    optic_disc_mask: np.ndarray,
    vessel_exclusion: np.ndarray,
) -> np.ndarray:
    valid = np.zeros_like(fov_mask)
    valid[(fov_mask > 0) & (optic_disc_mask == 0) & (vessel_exclusion == 0)] = 255
    return valid


def fill_masked_pixels(gray: np.ndarray, blocked_mask: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    output = gray.copy()
    valid_pixels = gray[valid_mask > 0]
    fill_value = int(np.median(valid_pixels)) if valid_pixels.size else 255
    output[blocked_mask > 0] = fill_value
    return output


def create_interior_fundus_mask(mask: np.ndarray, margin_ratio: float) -> np.ndarray:
    if not np.any(mask > 0):
        return np.zeros_like(mask)

    margin = max(8.0, min(mask.shape[:2]) * margin_ratio)
    distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    interior = np.zeros_like(mask)
    interior[distance >= margin] = 255
    return interior


def masked_otsu_threshold(values: np.ndarray) -> float:
    if values.size == 0:
        return 255.0
    value_image = values.astype(np.uint8).reshape(-1, 1)
    threshold, _ = cv2.threshold(value_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(threshold)


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


def filter_round_components(
    mask: np.ndarray,
    min_area: int,
    max_area: int,
    min_circularity: float,
    max_aspect_ratio: float,
    min_solidity: float,
) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = np.zeros_like(mask)

    for contour in contours:
        area = cv2.contourArea(contour)
        if not min_area <= area <= max_area:
            continue

        circularity, aspect_ratio, solidity = contour_shape_metrics(contour)
        if (
            circularity >= min_circularity
            and aspect_ratio <= max_aspect_ratio
            and solidity >= min_solidity
        ):
            cv2.drawContours(output, [contour], -1, 255, thickness=cv2.FILLED)

    return output


def filter_exudate_components(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = np.zeros_like(mask)

    for contour in contours:
        area = cv2.contourArea(contour)
        if not config.EXUDATE_MIN_AREA <= area <= config.EXUDATE_MAX_AREA:
            continue

        _, aspect_ratio, solidity = contour_shape_metrics(contour)
        if aspect_ratio <= 5.5 and solidity >= 0.28:
            cv2.drawContours(output, [contour], -1, 255, thickness=cv2.FILLED)

    return output


def filter_components_by_shape(
    mask: np.ndarray,
    min_area: int,
    max_area: int,
    max_aspect_ratio: float,
    min_solidity: float,
) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = np.zeros_like(mask)

    for contour in contours:
        area = cv2.contourArea(contour)
        if not min_area <= area <= max_area:
            continue

        _, aspect_ratio, solidity = contour_shape_metrics(contour)
        if aspect_ratio <= max_aspect_ratio and solidity >= min_solidity:
            cv2.drawContours(output, [contour], -1, 255, thickness=cv2.FILLED)

    return output


def remove_components_by_area(mask: np.ndarray, min_area: int, max_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    output = np.zeros_like(mask)

    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            output[labels == label] = 255

    return output


def count_components(mask: np.ndarray) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return int(sum(1 for label in range(1, count) if stats[label, cv2.CC_STAT_AREA] > 0))


def largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def mask_to_regions(mask: np.ndarray, min_area: int) -> list[dict[str, Any]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[dict[str, Any]] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
        else:
            cx = float(x + (width / 2.0))
            cy = float(y + (height / 2.0))

        simplified = cv2.approxPolyDP(contour, epsilon=1.5, closed=True)
        points = [
            {"x": int(point[0][0]), "y": int(point[0][1])}
            for point in simplified[:80]
        ]

        regions.append(
            {
                "bbox": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(width),
                    "height": int(height),
                },
                "centroid": {"x": cx, "y": cy},
                "area": area,
                "contour": points,
            },
        )

    return sorted(regions, key=lambda item: float(item["area"]), reverse=True)


def create_overlay(
    image: np.ndarray,
    vessels: np.ndarray,
    microaneurysms: np.ndarray,
    exudates: np.ndarray,
) -> np.ndarray:
    overlay = image.copy()
    overlay[vessels > 0] = blend_color(overlay[vessels > 0], np.array([255, 120, 0]))
    draw_mask_contours(overlay, exudates, color=(0, 255, 255), min_area=8)
    draw_mask_contours(overlay, microaneurysms, color=(0, 0, 255), min_area=3)
    return overlay


def blend_color(pixels: np.ndarray, color: np.ndarray) -> np.ndarray:
    return np.clip((pixels.astype(np.float32) * 0.45) + (color * 0.55), 0, 255).astype(np.uint8)


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
        cv2.drawContours(image, [contour], -1, color, thickness=2)


def save_feature_debug(
    debug_dir: Path,
    image_id: str,
    preprocessed: PreprocessingResult,
    vessels: np.ndarray,
    vesselness: np.ndarray,
    microaneurysms: np.ndarray,
    ma_candidates: np.ndarray,
    exudates: np.ndarray,
    exudate_candidates: np.ndarray,
) -> None:
    output_dir = ensure_dir(debug_dir / image_id)
    overlay = create_overlay(preprocessed.normalized_bgr, vessels, microaneurysms, exudates)
    debug_images: dict[str, Any] = {
        "08_vesselness.png": vesselness,
        "09_vessels.png": vessels,
        "10_ma_candidates.png": ma_candidates,
        "11_microaneurysms.png": microaneurysms,
        "12_exudate_candidates.png": exudate_candidates,
        "13_exudates.png": exudates,
        "14_overlay.png": overlay,
    }
    for filename, image in debug_images.items():
        cv2.imwrite(str(output_dir / filename), image)
