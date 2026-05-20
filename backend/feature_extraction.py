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


class FeatureExtractionError(RuntimeError):
    """Raised when a sample cannot be converted into numerical features."""


def extract_features(
    image_path: str | Path,
    debug: bool = False,
    debug_dir: str | Path | None = None,
) -> list[float]:
    """Return the six thesis features as a numerical vector.

    Output order:
    [ma_count, exudate_area, vessel_density, glcm_contrast,
     glcm_homogeneity, glcm_energy]
    """
    feature_dict = extract_feature_dict(image_path, debug=debug, debug_dir=debug_dir)
    return [float(feature_dict[name]) for name in config.FEATURE_NAMES]


def extract_feature_dict(
    image_path: str | Path,
    debug: bool = False,
    debug_dir: str | Path | None = None,
) -> dict[str, float]:
    try:
        preprocessed = preprocess_retinal_image(
            image_path,
            debug=debug,
            debug_dir=debug_dir,
        )
        vessels, vesselness = segment_vessels(preprocessed.denoised_green, preprocessed.fov_mask)
        microaneurysms, ma_candidates = detect_microaneurysms(preprocessed, vessels)
        exudates, exudate_candidates = detect_exudates(preprocessed, vessels)
        glcm_contrast, glcm_homogeneity, glcm_energy = extract_glcm_features(
            preprocessed.denoised_green,
            preprocessed.fov_mask,
        )
        fov_area = max(int(np.count_nonzero(preprocessed.fov_mask)), 1)

        features = {
            "ma_count": float(count_components(microaneurysms)),
            "exudate_area": float(np.count_nonzero(exudates)),
            "vessel_density": float(np.count_nonzero(vessels) / fov_area),
            "glcm_contrast": float(glcm_contrast),
            "glcm_homogeneity": float(glcm_homogeneity),
            "glcm_energy": float(glcm_energy),
        }

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

        return features
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


def detect_exudates(
    preprocessed: PreprocessingResult,
    vessels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    image = preprocessed.normalized_bgr
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
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
        float(np.percentile(valid_values, config.EXUDATE_PERCENTILE)),
    )
    local_values = local_bright[valid_mask > 0]
    local_threshold = max(
        10.0,
        float(np.percentile(local_values, config.EXUDATE_LOCAL_PERCENTILE)),
    )
    candidates = np.zeros_like(lightness)
    candidates[(lightness >= threshold) & (local_bright >= local_threshold) & (valid_mask > 0)] = 255
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, disk_kernel(2))
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, disk_kernel(1))
    candidates = apply_exudate_color_gate(image, candidates)
    exudates = filter_exudate_components(candidates)
    exudates[valid_mask == 0] = 0
    return exudates, candidates


def extract_glcm_features(gray: np.ndarray, fov_mask: np.ndarray) -> tuple[float, float, float]:
    levels = config.GLCM_LEVELS
    mask = fov_mask > 0
    if gray.size == 0 or not np.any(mask):
        return 0.0, 0.0, 0.0

    quantized = np.clip(gray.astype(np.uint16) * levels // 256, 0, levels - 1).astype(np.int32)
    indices = np.arange(levels, dtype=np.float64)
    diff = indices[:, None] - indices[None, :]
    contrast_weights = diff**2
    homogeneity_weights = 1.0 / (1.0 + np.abs(diff))
    contrast_scale = (256.0 / levels) ** 2

    contrasts: list[float] = []
    homogeneities: list[float] = []
    energies: list[float] = []

    for dy, dx in config.GLCM_OFFSETS:
        y0 = max(0, dy)
        y1 = gray.shape[0] + min(0, dy)
        x0 = max(0, dx)
        x1 = gray.shape[1] + min(0, dx)

        source = quantized[y0:y1, x0:x1]
        target = quantized[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
        pair_mask = mask[y0:y1, x0:x1] & mask[y0 - dy : y1 - dy, x0 - dx : x1 - dx]

        if source.size == 0 or not np.any(pair_mask):
            continue

        matrix = np.zeros((levels, levels), dtype=np.float64)
        np.add.at(matrix, (source[pair_mask].ravel(), target[pair_mask].ravel()), 1.0)
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

    return (
        float(np.mean(contrasts)),
        float(np.mean(homogeneities)),
        float(np.mean(energies)),
    )


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
