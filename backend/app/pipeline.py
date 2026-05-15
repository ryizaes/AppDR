import base64
from dataclasses import dataclass

import cv2
import numpy as np

from app.schemas import FeatureReport, QualityReport, ScreeningResult


@dataclass
class PipelineOutput:
    quality: QualityReport
    features: FeatureReport
    result: ScreeningResult
    processed_images: dict[str, str]


CENTER_CROP_SCALE = 1.0
MAX_ANALYSIS_SIZE = 900
FEATURE_VECTOR_NAMES = [
    "fundus_area",
    "vessel_density",
    "vessel_area",
    "bright_lesion_area",
    "dark_lesion_area",
    "bright_lesion_count",
    "dark_lesion_count",
    "microaneurysm_count",
    "hemorrhage_candidate_count",
    "optic_disc_area",
    "mean_intensity",
    "intensity_std",
    "texture_contrast",
]
SPATIAL_GRID_SIZE = 4
SPATIAL_FEATURE_KINDS = (
    "mask_coverage",
    "enhanced_mean",
    "enhanced_std",
    "vessel_ratio",
    "bright_ratio",
    "dark_ratio",
)
CLASSICAL_REFERABLE_THRESHOLD = 50.0


def analyze_image(image_bytes: bytes, include_processed_images: bool = True) -> PipelineOutput:
    image = prepare_analysis_image(decode_image(image_bytes))
    fundus_mask = create_fundus_mask(image)
    green = image[:, :, 1]
    quality = assess_quality(image, green, fundus_mask)

    corrected = correct_illumination(green, fundus_mask)
    enhanced = apply_clahe(corrected)
    denoised = cv2.medianBlur(enhanced, 3)

    vesselness = frangi_vesselness(denoised, fundus_mask)
    vessels = segment_vessels(vesselness, fundus_mask)
    optic_disc = detect_optic_disc(image, fundus_mask)
    bright_lesions = detect_bright_lesions(
        image,
        enhanced,
        fundus_mask,
        vessels,
        optic_disc,
    )
    dark_lesions = detect_dark_lesions(
        image,
        enhanced,
        fundus_mask,
        vessels,
        optic_disc,
    )

    features = extract_features(
        enhanced=enhanced,
        vessels=vessels,
        bright_lesions=bright_lesions,
        dark_lesions=dark_lesions,
        optic_disc=optic_disc,
        fundus_mask=fundus_mask,
    )
    quality = add_feature_quality_warnings(quality, features)
    result = classify_referable(features, quality)
    processed_images: dict[str, str] = {}

    if include_processed_images:
        overlay = create_overlay(image, vessels, bright_lesions, dark_lesions, optic_disc)
        processed_images = {
            "original": encode_png(image),
            "green_channel": encode_png(green),
            "enhanced": encode_png(denoised),
            "vesselness": encode_png(vesselness),
            "vessels": encode_png(vessels),
            "lesion_overlay": encode_png(overlay),
        }

    return PipelineOutput(
        quality=quality,
        features=features,
        result=result,
        processed_images=processed_images,
    )


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
    cropped = crop_center_square(image)
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


def create_fundus_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, mask = cv2.threshold(blurred, 12, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = keep_largest_component(mask)

    return mask


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
    if extent > 0.92 and fundus_area_ratio > 0.8:
        warnings.append("Retinal image appears to be a full-frame crop.")

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

    return QualityReport(
        is_acceptable=quality.is_acceptable,
        blur_score=quality.blur_score,
        brightness_mean=quality.brightness_mean,
        contrast_std=quality.contrast_std,
        fundus_area_ratio=quality.fundus_area_ratio,
        warnings=unique_warnings(warnings),
    )


def correct_illumination(green: np.ndarray, mask: np.ndarray) -> np.ndarray:
    background = cv2.GaussianBlur(green, (0, 0), sigmaX=25, sigmaY=25)
    corrected = cv2.divide(green, background, scale=128)
    corrected[mask == 0] = 0

    return corrected


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def frangi_vesselness(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    responses: list[np.ndarray] = []

    for kernel_size in (9, 13, 17, 23):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        responses.append(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel))

    for length in (13, 19, 25):
        for angle in (0, 30, 60, 90, 120, 150):
            kernel = create_line_kernel(length, angle)
            responses.append(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel))

    vesselness = np.maximum.reduce(responses)
    vesselness = cv2.GaussianBlur(vesselness, (3, 3), 0)
    vesselness = normalize_uint8(vesselness)
    vesselness[mask == 0] = 0

    return vesselness


def segment_vessels(vesselness: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked_values = vesselness[mask > 0]

    if masked_values.size == 0:
        return np.zeros_like(vesselness)

    otsu_threshold, _ = cv2.threshold(
        vesselness,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    percentile_threshold = float(np.percentile(masked_values, 88))
    threshold = max(10.0, min(float(otsu_threshold), percentile_threshold))
    vessels = np.zeros_like(vesselness)
    vessels[(vesselness >= threshold) & (mask > 0)] = 255
    vessels[mask == 0] = 0
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    vessels = cv2.morphologyEx(vessels, cv2.MORPH_CLOSE, kernel_close)
    vessels = cv2.morphologyEx(vessels, cv2.MORPH_OPEN, kernel_open)

    return remove_small_components(vessels, min_area=8, max_area=25000)


def detect_optic_disc(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    masked = lightness.copy()
    masked[mask == 0] = 0
    threshold = np.percentile(masked[mask > 0], 98) if np.any(mask > 0) else 255
    _, bright = cv2.threshold(masked, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
    bright = keep_largest_component(bright)

    return bright


def detect_bright_lesions(
    image: np.ndarray,
    enhanced: np.ndarray,
    mask: np.ndarray,
    vessels: np.ndarray,
    optic_disc: np.ndarray,
) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue = image[:, :, 0].astype(np.int16)
    green = image[:, :, 1].astype(np.int16)
    red = image[:, :, 2].astype(np.int16)
    background = cv2.morphologyEx(
        lightness,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
    )
    top_hat = cv2.subtract(lightness, background)
    enhanced_top_hat = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    )
    combined = cv2.addWeighted(top_hat, 0.75, enhanced_top_hat, 0.25, 0)
    interior_mask = create_interior_fundus_mask(mask, margin_ratio=0.045)
    candidate_mask = (
        (interior_mask > 0)
        & (build_exclusion_mask(vessels, optic_disc, vessel_size=5, disc_size=41) == 0)
    )
    yellow_candidate = (
        (red > 70)
        & (green > 60)
        & (red >= blue + 8)
        & (green >= blue + 4)
        & (hsv[:, :, 1] > 22)
    )
    threshold = np.percentile(combined[candidate_mask], 98.4) if np.any(candidate_mask) else 255
    bright = np.zeros_like(enhanced)
    bright[(combined >= threshold) & yellow_candidate & candidate_mask] = 255
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    return filter_lesion_components(
        bright,
        min_area=8,
        max_area=1800,
        min_circularity=0.16,
        max_aspect_ratio=4.0,
        min_solidity=0.35,
    )


def detect_dark_lesions(
    image: np.ndarray,
    enhanced: np.ndarray,
    mask: np.ndarray,
    vessels: np.ndarray,
    optic_disc: np.ndarray,
) -> np.ndarray:
    responses = [
        cv2.morphologyEx(
            enhanced,
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
        )
        for size in (7, 11, 15, 21)
    ]
    response = np.maximum.reduce(responses)
    response = cv2.GaussianBlur(response, (3, 3), 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue = image[:, :, 0].astype(np.int16)
    green = image[:, :, 1].astype(np.int16)
    red = image[:, :, 2].astype(np.int16)
    masked_green = green[mask > 0]
    median_green = float(np.median(masked_green)) if masked_green.size else 0.0
    interior_mask = create_interior_fundus_mask(mask, margin_ratio=0.035)
    candidate_mask = (
        (interior_mask > 0)
        & (build_exclusion_mask(vessels, optic_disc, vessel_size=9, disc_size=35) == 0)
    )
    red_dark_candidate = (
        (green <= median_green + 8.0)
        & (red >= blue + 3)
        & (hsv[:, :, 1] >= 18)
    )
    threshold = np.percentile(response[candidate_mask], 98.7) if np.any(candidate_mask) else 255
    dark = np.zeros_like(enhanced)
    dark[(response >= threshold) & red_dark_candidate & candidate_mask] = 255
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    return filter_lesion_components(
        dark,
        min_area=4,
        max_area=1800,
        min_circularity=0.18,
        max_aspect_ratio=3.2,
        min_solidity=0.34,
    )


def extract_features(
    enhanced: np.ndarray,
    vessels: np.ndarray,
    bright_lesions: np.ndarray,
    dark_lesions: np.ndarray,
    optic_disc: np.ndarray,
    fundus_mask: np.ndarray,
) -> FeatureReport:
    fundus_area = max(int(np.count_nonzero(fundus_mask)), 1)
    vessel_area = int(np.count_nonzero(vessels))
    bright_area = int(np.count_nonzero(bright_lesions))
    dark_area = int(np.count_nonzero(dark_lesions))
    bright_count = count_components(bright_lesions)
    dark_count = count_components(dark_lesions)
    microaneurysm_count, hemorrhage_count = count_dark_lesions(dark_lesions)
    pixels = enhanced[fundus_mask > 0]

    return FeatureReport(
        fundus_area=fundus_area,
        vessel_density=float(vessel_area / fundus_area),
        vessel_area=vessel_area,
        bright_lesion_area=bright_area,
        dark_lesion_area=dark_area,
        bright_lesion_count=bright_count,
        dark_lesion_count=dark_count,
        microaneurysm_count=microaneurysm_count,
        hemorrhage_candidate_count=hemorrhage_count,
        optic_disc_area=int(np.count_nonzero(optic_disc)),
        mean_intensity=float(np.mean(pixels)) if pixels.size else 0.0,
        intensity_std=float(np.std(pixels)) if pixels.size else 0.0,
        texture_contrast=estimate_texture_contrast(enhanced, fundus_mask),
        spatial_features=extract_spatial_features(
            enhanced,
            vessels,
            bright_lesions,
            dark_lesions,
            fundus_mask,
        ),
    )


def extract_spatial_features(
    enhanced: np.ndarray,
    vessels: np.ndarray,
    bright_lesions: np.ndarray,
    dark_lesions: np.ndarray,
    fundus_mask: np.ndarray,
) -> list[float]:
    height, width = enhanced.shape[:2]
    values: list[float] = []

    for row in range(SPATIAL_GRID_SIZE):
        y0 = int(round((row / SPATIAL_GRID_SIZE) * height))
        y1 = int(round(((row + 1) / SPATIAL_GRID_SIZE) * height))

        for col in range(SPATIAL_GRID_SIZE):
            x0 = int(round((col / SPATIAL_GRID_SIZE) * width))
            x1 = int(round(((col + 1) / SPATIAL_GRID_SIZE) * width))
            tile_mask = fundus_mask[y0:y1, x0:x1] > 0
            tile_area = max(int(np.count_nonzero(tile_mask)), 1)
            tile_pixels = enhanced[y0:y1, x0:x1][tile_mask]

            values.extend(
                [
                    float(tile_area / max(tile_mask.size, 1)),
                    float(np.mean(tile_pixels) / 255.0) if tile_pixels.size else 0.0,
                    float(np.std(tile_pixels) / 128.0) if tile_pixels.size else 0.0,
                    float(np.count_nonzero(vessels[y0:y1, x0:x1][tile_mask]) / tile_area),
                    float(
                        np.count_nonzero(bright_lesions[y0:y1, x0:x1][tile_mask])
                        / tile_area,
                    ),
                    float(
                        np.count_nonzero(dark_lesions[y0:y1, x0:x1][tile_mask])
                        / tile_area,
                    ),
                ],
            )

    return values


def count_dark_lesions(mask: np.ndarray) -> tuple[int, int]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    microaneurysm_count = 0
    hemorrhage_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        circularity, aspect_ratio, solidity = contour_shape_metrics(contour)

        if 4 <= area <= 95 and circularity >= 0.28 and aspect_ratio <= 2.4:
            microaneurysm_count += 1
        elif 95 < area <= 1800 and solidity >= 0.34 and aspect_ratio <= 3.2:
            hemorrhage_count += 1

    return microaneurysm_count, hemorrhage_count


def count_components(mask: np.ndarray) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    component_count = 0

    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] > 0:
            component_count += 1

    return component_count


def estimate_texture_contrast(gray: np.ndarray, mask: np.ndarray) -> float:
    pixels = gray[mask > 0]

    if pixels.size < 2:
        return 0.0

    hist = cv2.calcHist([pixels], [0], None, [32], [0, 256]).flatten()
    probabilities = hist / max(float(np.sum(hist)), 1.0)
    indices = np.arange(32, dtype=np.float32)
    mean = float(np.sum(indices * probabilities))
    return float(np.sum(((indices - mean) ** 2) * probabilities))


def estimate_vessel_hint(green: np.ndarray, mask: np.ndarray) -> float:
    candidate_pixels = green[mask > 0]

    if candidate_pixels.size == 0:
        return 0.0

    blackhat = cv2.morphologyEx(
        green,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
    )
    threshold = max(8.0, float(np.percentile(blackhat[mask > 0], 92)))
    candidate_count = int(np.count_nonzero((blackhat >= threshold) & (mask > 0)))

    return float(candidate_count / candidate_pixels.size)


def create_line_kernel(length: int, angle: int) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.uint8)
    center = length // 2
    cv2.line(kernel, (1, center), (length - 2, center), 1, 1)

    if angle == 0:
        return kernel

    rotation = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    rotated = cv2.warpAffine(kernel, rotation, (length, length))
    rotated[rotated > 0] = 1

    return rotated


def erode_binary_mask(mask: np.ndarray, size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.erode(mask, kernel, iterations=1)


def create_interior_fundus_mask(mask: np.ndarray, margin_ratio: float) -> np.ndarray:
    if not np.any(mask > 0):
        return np.zeros_like(mask)

    margin = max(8.0, min(mask.shape[:2]) * margin_ratio)
    distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    interior = np.zeros_like(mask)
    interior[distance >= margin] = 255

    return interior


def build_exclusion_mask(
    vessels: np.ndarray,
    optic_disc: np.ndarray,
    vessel_size: int,
    disc_size: int,
) -> np.ndarray:
    vessel_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (vessel_size, vessel_size),
    )
    disc_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (disc_size, disc_size),
    )
    vessel_exclusion = cv2.dilate(vessels, vessel_kernel, iterations=1)
    disc_exclusion = cv2.dilate(optic_disc, disc_kernel, iterations=1)

    return cv2.bitwise_or(vessel_exclusion, disc_exclusion)


def filter_lesion_components(
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


def classify_referable(features: FeatureReport, quality: QualityReport) -> ScreeningResult:
    if not quality.is_acceptable:
        return ScreeningResult(
            classification="Image not suitable for DR screening",
            referable=False,
            dr_probability=0.0,
            reason=", ".join(quality.warnings),
            disclaimer=(
                "Screening support only. Retake with a clear retinal image before "
                "reviewing diabetic retinopathy features."
            ),
        )

    dr_probability, reasons = estimate_dr_probability(features)
    referable = dr_probability >= CLASSICAL_REFERABLE_THRESHOLD
    classification = "Referable DR suspected" if referable else "Non-referable"

    return ScreeningResult(
        classification=classification,
        referable=referable,
        dr_probability=dr_probability,
        reason=", ".join(reasons) if reasons else "low lesion-candidate score",
        disclaimer=(
            "Screening support only. This result is not a medical diagnosis and "
            "must be reviewed by a qualified eye-care professional."
        ),
    )


def estimate_dr_probability(features: FeatureReport) -> tuple[float, list[str]]:
    fundus_area = max(float(features.fundus_area), 1.0)
    bright_ratio = features.bright_lesion_area / fundus_area
    dark_ratio = features.dark_lesion_area / fundus_area
    microaneurysm_density = features.microaneurysm_count / (fundus_area / 100000.0)
    hemorrhage_density = features.hemorrhage_candidate_count / (fundus_area / 100000.0)
    dark_count_density = microaneurysm_density + hemorrhage_density
    spatial = summarize_spatial_features(features.spatial_features)

    score = 4.0
    reasons: list[str] = []

    microaneurysm_score = scaled_score(microaneurysm_density, start=6.0, end=18.0, weight=4.0)
    hemorrhage_score = scaled_score(hemorrhage_density, start=0.9, end=3.5, weight=8.0)
    dark_area_score = scaled_score(dark_ratio, start=0.0045, end=0.011, weight=14.0)
    raw_bright_score = scaled_score(bright_ratio, start=0.003, end=0.010, weight=50.0)
    dark_tile_score = scaled_score(float(spatial["dark_tiles"]), start=3.0, end=8.0, weight=7.0)
    bright_tile_score = scaled_score(float(spatial["bright_tiles"]), start=1.0, end=6.0, weight=16.0)
    texture_score = scaled_score(features.texture_contrast, start=28.0, end=60.0, weight=4.0)
    intensity_score = 0.0
    vessel_score = 0.0
    red_lesion_score = (
        microaneurysm_score
        + hemorrhage_score
        + dark_area_score
        + dark_tile_score
    )
    bright_context = 1.0
    bright_area_score = raw_bright_score

    score += red_lesion_score
    score += bright_area_score
    score += bright_tile_score * bright_context
    score += texture_score
    score += intensity_score
    score += vessel_score

    if microaneurysm_score >= 8.0:
        reasons.append("microaneurysm candidate density is elevated")
    if hemorrhage_score >= 8.0:
        reasons.append("hemorrhage candidate density is elevated")
    if dark_area_score >= 7.0:
        reasons.append("dark lesion candidate area is elevated")
    if dark_tile_score >= 5.0:
        reasons.append("dark lesion candidates appear in multiple retinal regions")
    if raw_bright_score >= 7.0 and bright_context == 1.0:
        reasons.append("bright lesion candidate area is elevated")
    elif raw_bright_score >= 7.0:
        reasons.append("bright candidates found without strong red-lesion support")
    if bright_tile_score >= 5.0 and bright_context == 1.0:
        reasons.append("bright lesion candidates appear in multiple retinal regions")
    if texture_score >= 4.0:
        reasons.append("retinal texture variation is elevated")

    strong_lesion_signals = sum(
        value >= threshold
        for value, threshold in (
            (microaneurysm_score, 8.0),
            (hemorrhage_score, 8.0),
            (dark_area_score, 7.0),
            (dark_tile_score, 5.0),
            (bright_area_score, 7.0),
        )
    )

    weak_red_evidence = red_lesion_score < 10.0 and dark_count_density < 4.0

    if strong_lesion_signals == 0:
        score = min(score, 35.0)
    elif strong_lesion_signals == 1 and weak_red_evidence and bright_ratio < 0.0035:
        score = min(score, 48.0)

    if bright_ratio < 0.0025 and dark_ratio < 0.0055:
        score = min(score, 40.0)
    elif bright_ratio < 0.0035 and dark_ratio < 0.0065 and hemorrhage_density < 1.5:
        score = min(score, 48.0)
    if hemorrhage_density < 0.25 and bright_ratio < 0.008 and dark_ratio < 0.006:
        score = min(score, 48.0)
    if (
        spatial["central_bright_tiles"] == 0
        and spatial["central_dark_tiles"] == 0
        and spatial["edge_bright_tiles"] + spatial["edge_dark_tiles"] >= 5
    ):
        score = min(score, 45.0)
        reasons.append("lesion candidates are mostly near the retinal edge")

    if features.vessel_density < 0.006:
        score *= 0.75
        reasons.append("retinal vessel evidence is weak")
    elif features.vessel_density < 0.012 and score > 55.0:
        score = 55.0
        reasons.append("retinal vessel evidence is weak")

    if spatial["covered_tiles"] < 6 and score >= CLASSICAL_REFERABLE_THRESHOLD:
        score = min(score, 58.0)
        reasons.append("retinal field coverage is limited")

    return round(float(np.clip(score, 1.0, 97.0)), 1), reasons


def summarize_spatial_features(spatial_features: list[float]) -> dict[str, float]:
    stride = len(SPATIAL_FEATURE_KINDS)
    dark_tiles = 0
    bright_tiles = 0
    vessel_tiles = 0
    covered_tiles = 0
    edge_bright_tiles = 0
    edge_dark_tiles = 0
    central_bright_tiles = 0
    central_dark_tiles = 0
    max_bright_ratio = 0.0
    max_dark_ratio = 0.0

    for tile_index, index in enumerate(range(0, len(spatial_features), stride)):
        if index + stride > len(spatial_features):
            break

        row = tile_index // SPATIAL_GRID_SIZE
        col = tile_index % SPATIAL_GRID_SIZE
        is_edge_tile = (
            row == 0
            or col == 0
            or row == SPATIAL_GRID_SIZE - 1
            or col == SPATIAL_GRID_SIZE - 1
        )
        coverage, _, _, vessel_ratio, bright_ratio, dark_ratio = spatial_features[
            index : index + stride
        ]

        if coverage >= 0.18:
            covered_tiles += 1
        if coverage >= 0.18 and vessel_ratio >= 0.018:
            vessel_tiles += 1
        if coverage >= 0.18 and bright_ratio >= 0.0045:
            bright_tiles += 1
            if is_edge_tile:
                edge_bright_tiles += 1
            else:
                central_bright_tiles += 1
        if coverage >= 0.18 and dark_ratio >= 0.0022:
            dark_tiles += 1
            if is_edge_tile:
                edge_dark_tiles += 1
            else:
                central_dark_tiles += 1
        if coverage >= 0.18:
            max_bright_ratio = max(max_bright_ratio, float(bright_ratio))
            max_dark_ratio = max(max_dark_ratio, float(dark_ratio))

    return {
        "dark_tiles": dark_tiles,
        "bright_tiles": bright_tiles,
        "vessel_tiles": vessel_tiles,
        "covered_tiles": covered_tiles,
        "edge_bright_tiles": edge_bright_tiles,
        "edge_dark_tiles": edge_dark_tiles,
        "central_bright_tiles": central_bright_tiles,
        "central_dark_tiles": central_dark_tiles,
        "max_bright_ratio": max_bright_ratio,
        "max_dark_ratio": max_dark_ratio,
    }


def scaled_score(value: float, start: float, end: float, weight: float) -> float:
    if value <= start:
        return 0.0
    if value >= end:
        return weight

    return float(((value - start) / (end - start)) * weight)


def feature_vector_from_report(features: FeatureReport) -> list[float]:
    return [
        float(features.fundus_area),
        float(features.vessel_density),
        float(features.vessel_area),
        float(features.bright_lesion_area),
        float(features.dark_lesion_area),
        float(features.bright_lesion_count),
        float(features.dark_lesion_count),
        float(features.microaneurysm_count),
        float(features.hemorrhage_candidate_count),
        float(features.optic_disc_area),
        float(features.mean_intensity),
        float(features.intensity_std),
        float(features.texture_contrast),
    ]


def create_overlay(
    image: np.ndarray,
    vessels: np.ndarray,
    bright_lesions: np.ndarray,
    dark_lesions: np.ndarray,
    optic_disc: np.ndarray,
) -> np.ndarray:
    overlay = image.copy()
    overlay[vessels > 0] = blend_color(overlay[vessels > 0], np.array([255, 140, 0]))
    overlay[bright_lesions > 0] = blend_color(
        overlay[bright_lesions > 0],
        np.array([0, 255, 255]),
    )
    overlay[dark_lesions > 0] = blend_color(
        overlay[dark_lesions > 0],
        np.array([0, 0, 255]),
    )
    overlay[optic_disc > 0] = blend_color(
        overlay[optic_disc > 0],
        np.array([0, 255, 0]),
    )

    return overlay


def blend_color(pixels: np.ndarray, color: np.ndarray) -> np.ndarray:
    return np.clip((pixels.astype(np.float32) * 0.45) + (color * 0.55), 0, 255).astype(
        np.uint8,
    )


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


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def encode_png(image: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", image)

    if not success:
        raise ValueError("Failed to encode processed image.")

    encoded = base64.b64encode(buffer).decode("ascii")
    return f"data:image/png;base64,{encoded}"
