from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app import pipeline as classical
from experiments.config import ImageProcessingParams


@dataclass
class FeatureExtractionResult:
    features: dict[str, float | int | str]
    masks: dict[str, np.ndarray]
    processed: dict[str, np.ndarray]


def extract_handcrafted_features(
    image_path: Path,
    params: ImageProcessingParams,
    include_images: bool = False,
) -> FeatureExtractionResult:
    image = classical.prepare_analysis_image(classical.decode_image(image_path.read_bytes()))
    stage0 = classical.stage0_fov_and_optic_disc_masking(image)
    preprocessed = preprocess_green_channel(image, stage0.fov_mask, params)
    vessels, vesselness = segment_vessels(preprocessed, stage0.fov_mask, params)
    microaneurysms, ma_candidates = detect_microaneurysms(
        image,
        preprocessed,
        stage0.fov_mask,
        stage0.optic_disc_mask,
        vessels,
        params,
    )
    exudates, exudate_candidates = detect_exudates(
        image,
        stage0.fov_mask,
        stage0.optic_disc_mask,
        vessels,
        params,
    )
    features = summarize_features(
        preprocessed,
        vessels,
        microaneurysms,
        exudates,
        stage0.fov_mask,
        stage0.optic_disc_mask,
    )

    if not include_images:
        return FeatureExtractionResult(features=features, masks={}, processed={})

    overlay = classical.create_overlay(
        image=image,
        vessels=vessels,
        microaneurysms=microaneurysms,
        exudates=exudates,
        optic_disc_mask=stage0.optic_disc_mask,
        fov_mask=stage0.fov_mask,
    )

    return FeatureExtractionResult(
        features=features,
        masks={
            "fov_mask": stage0.fov_mask,
            "optic_disc_mask": stage0.optic_disc_mask,
            "vessels": vessels,
            "microaneurysms": microaneurysms,
            "exudates": exudates,
            "ma_candidates": ma_candidates,
            "exudate_candidates": exudate_candidates,
        },
        processed={
            "original": image,
            "preprocessed": preprocessed,
            "vesselness": vesselness,
            "overlay": overlay,
        },
    )


def preprocess_green_channel(
    image: np.ndarray,
    fov_mask: np.ndarray,
    params: ImageProcessingParams,
) -> np.ndarray:
    green = image[:, :, 1]
    working = classical.fill_outside_mask(green, fov_mask)

    if params.illumination_sigma > 0:
        background = cv2.GaussianBlur(
            working,
            (0, 0),
            sigmaX=params.illumination_sigma,
            sigmaY=params.illumination_sigma,
        )
        working = cv2.divide(working, background, scale=128)

    if params.gamma != 1.0:
        inv_gamma = 1.0 / max(params.gamma, 1e-6)
        table = np.array(
            [((value / 255.0) ** inv_gamma) * 255 for value in range(256)],
            dtype=np.uint8,
        )
        working = cv2.LUT(working, table)

    tile_size = max(2, int(params.clahe_tile_grid_size))
    clahe = cv2.createCLAHE(
        clipLimit=float(params.clahe_clip_limit),
        tileGridSize=(tile_size, tile_size),
    )
    working = clahe.apply(working)
    working = cv2.medianBlur(working, 3)

    if params.gaussian_sigma > 0:
        working = cv2.GaussianBlur(
            working,
            (0, 0),
            sigmaX=params.gaussian_sigma,
            sigmaY=params.gaussian_sigma,
        )

    working[fov_mask == 0] = 0

    return working


def segment_vessels(
    preprocessed: np.ndarray,
    fov_mask: np.ndarray,
    params: ImageProcessingParams,
) -> tuple[np.ndarray, np.ndarray]:
    vesselness_float = classical.frangi_vesselness_opencv(
        preprocessed.astype(np.float32) / 255.0,
        sigmas=tuple(float(value) for value in params.frangi_sigmas),
        beta=params.frangi_beta,
        gamma=params.frangi_gamma,
    )
    vesselness = classical.normalize_uint8(np.nan_to_num(vesselness_float, nan=0.0))
    vesselness[fov_mask == 0] = 0
    block_size = make_odd_block_size(params.adaptive_block_size)
    vessels = cv2.adaptiveThreshold(
        vesselness,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        int(params.adaptive_c),
    )
    masked_values = vesselness[fov_mask > 0]

    if masked_values.size:
        threshold = max(4.0, float(np.percentile(masked_values, params.vessel_percentile)))
        vessels[vesselness < threshold] = 0

    vessels[fov_mask == 0] = 0
    vessels = cv2.morphologyEx(
        vessels,
        cv2.MORPH_CLOSE,
        classical.disk_kernel(params.morph_close_radius),
    )
    vessels = cv2.morphologyEx(
        vessels,
        cv2.MORPH_OPEN,
        classical.disk_kernel(params.morph_open_radius),
    )
    vessels = classical.remove_small_components(vessels, min_area=8, max_area=50000)

    return vessels, vesselness


def detect_microaneurysms(
    image: np.ndarray,
    preprocessed: np.ndarray,
    fov_mask: np.ndarray,
    optic_disc_mask: np.ndarray,
    vessels: np.ndarray,
    params: ImageProcessingParams,
) -> tuple[np.ndarray, np.ndarray]:
    vessel_exclusion = cv2.dilate(vessels, classical.disk_kernel(2), iterations=1)
    valid_mask = classical.build_valid_lesion_mask(fov_mask, optic_disc_mask, vessel_exclusion)
    vessel_removed = classical.fill_masked_pixels(preprocessed, vessel_exclusion, valid_mask)
    blackhat = cv2.morphologyEx(
        vessel_removed,
        cv2.MORPH_BLACKHAT,
        classical.disk_kernel(params.ma_blackhat_radius),
    )
    blackhat[valid_mask == 0] = 0
    valid_values = blackhat[valid_mask > 0]

    if valid_values.size == 0:
        return np.zeros_like(preprocessed), np.zeros_like(preprocessed)

    threshold = max(10.0, float(np.percentile(valid_values, params.ma_percentile)))
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
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, classical.disk_kernel(1))
    hough_mask = classical.detect_microaneurysm_circles(blackhat, candidates, valid_mask)
    strict_components = classical.filter_microaneurysm_components(candidates, hough_mask)
    microaneurysms = classical.remove_small_components(
        strict_components,
        min_area=params.ma_min_area,
        max_area=params.ma_max_area,
    )

    return microaneurysms, candidates


def detect_exudates(
    image: np.ndarray,
    fov_mask: np.ndarray,
    optic_disc_mask: np.ndarray,
    vessels: np.ndarray,
    params: ImageProcessingParams,
) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    local_bright = cv2.morphologyEx(lightness, cv2.MORPH_TOPHAT, classical.disk_kernel(8))
    vessel_exclusion = cv2.dilate(vessels, classical.disk_kernel(1), iterations=1)
    od_exclusion = cv2.dilate(optic_disc_mask, classical.disk_kernel(12), iterations=1)
    interior_mask = classical.create_interior_fundus_mask(fov_mask, margin_ratio=0.035)
    valid_mask = classical.build_valid_lesion_mask(interior_mask, od_exclusion, vessel_exclusion)
    valid_values = lightness[valid_mask > 0]

    if valid_values.size == 0:
        return np.zeros_like(lightness), np.zeros_like(lightness)

    threshold = max(
        classical.masked_otsu_threshold(valid_values),
        float(np.percentile(valid_values, params.exudate_percentile)),
    )
    local_values = local_bright[valid_mask > 0]
    local_threshold = max(10.0, float(np.percentile(local_values, params.exudate_local_percentile)))
    candidates = np.zeros_like(lightness)
    candidates[
        (lightness >= threshold)
        & (local_bright >= local_threshold)
        & (valid_mask > 0)
    ] = 255
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, classical.disk_kernel(2))
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, classical.disk_kernel(1))
    candidates = classical.apply_exudate_color_gate(image, candidates)
    exudates = filter_exudate_components(candidates, params)
    exudates[valid_mask == 0] = 0

    return exudates, candidates


def filter_exudate_components(mask: np.ndarray, params: ImageProcessingParams) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = np.zeros_like(mask)

    for contour in contours:
        area = cv2.contourArea(contour)

        if not params.exudate_min_area <= area <= params.exudate_max_area:
            continue

        _, aspect_ratio, solidity = classical.contour_shape_metrics(contour)

        if aspect_ratio <= 5.5 and solidity >= 0.28:
            cv2.drawContours(output, [contour], -1, 255, thickness=cv2.FILLED)

    return output


def summarize_features(
    preprocessed: np.ndarray,
    vessels: np.ndarray,
    microaneurysms: np.ndarray,
    exudates: np.ndarray,
    fov_mask: np.ndarray,
    optic_disc_mask: np.ndarray,
) -> dict[str, float | int | str]:
    fundus_area = max(int(np.count_nonzero(fov_mask)), 1)
    ma_area = int(np.count_nonzero(microaneurysms))
    exudate_area = int(np.count_nonzero(exudates))
    exudate_quadrants = classical.find_exudate_quadrants(exudates, fov_mask)
    glcm_contrast, glcm_homogeneity, glcm_energy = classical.extract_glcm_features(
        preprocessed,
        fov_mask,
    )

    return {
        "fundus_area": fundus_area,
        "vessel_density": float(np.count_nonzero(vessels) / fundus_area),
        "vessel_area": int(np.count_nonzero(vessels)),
        "microaneurysm_count": classical.count_components(microaneurysms),
        "microaneurysm_area": ma_area,
        "exudate_count": classical.count_components(exudates),
        "exudate_area": exudate_area,
        "exudate_quadrant_count": len(exudate_quadrants),
        "pathology_area_index": float(((ma_area + exudate_area) / fundus_area) * 100.0),
        "optic_disc_area": int(np.count_nonzero(optic_disc_mask)),
        "mean_intensity": float(np.mean(preprocessed[fov_mask > 0])),
        "intensity_std": float(np.std(preprocessed[fov_mask > 0])),
        "glcm_contrast": glcm_contrast,
        "glcm_homogeneity": glcm_homogeneity,
        "glcm_energy": glcm_energy,
        "exudate_quadrants": "|".join(exudate_quadrants),
    }


def make_odd_block_size(value: int) -> int:
    block_size = max(3, int(value))

    if block_size % 2 == 0:
        block_size += 1

    return block_size


def feature_names() -> list[str]:
    return [
        "fundus_area",
        "vessel_density",
        "vessel_area",
        "microaneurysm_count",
        "microaneurysm_area",
        "exudate_count",
        "exudate_area",
        "exudate_quadrant_count",
        "pathology_area_index",
        "optic_disc_area",
        "mean_intensity",
        "intensity_std",
        "glcm_contrast",
        "glcm_homogeneity",
        "glcm_energy",
    ]


def numeric_feature_vector(features: dict[str, Any]) -> list[float]:
    return [float(features[name]) for name in feature_names()]
