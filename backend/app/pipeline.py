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


def analyze_image(image_bytes: bytes) -> PipelineOutput:
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
    bright_lesions = detect_bright_lesions(image, enhanced, fundus_mask, optic_disc)
    dark_lesions = detect_dark_lesions(enhanced, fundus_mask, vessels, optic_disc)

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
    overlay = create_overlay(image, vessels, bright_lesions, dark_lesions, optic_disc)

    return PipelineOutput(
        quality=quality,
        features=features,
        result=result,
        processed_images={
            "original": encode_png(image),
            "green_channel": encode_png(green),
            "enhanced": encode_png(denoised),
            "vesselness": encode_png(vesselness),
            "vessels": encode_png(vessels),
            "lesion_overlay": encode_png(overlay),
        },
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
    warnings.extend(assess_retinal_field(image, mask, fundus_area_ratio))

    if blur_score < 80.0:
        warnings.append("Image may be blurry.")
    if brightness_mean < 35.0:
        warnings.append("Image may be underexposed.")
    if brightness_mean > 215.0:
        warnings.append("Image may be overexposed.")
    if contrast_std < 18.0:
        warnings.append("Image contrast may be too low.")
    if fundus_area_ratio < 0.2:
        warnings.append("Detected retinal field is too small.")

    return QualityReport(
        is_acceptable=len(warnings) == 0,
        blur_score=blur_score,
        brightness_mean=brightness_mean,
        contrast_std=contrast_std,
        fundus_area_ratio=fundus_area_ratio,
        warnings=warnings,
    )


def assess_retinal_field(
    image: np.ndarray,
    mask: np.ndarray,
    fundus_area_ratio: float,
) -> list[str]:
    warnings: list[str] = []
    contour = get_largest_contour(mask)

    if contour is None:
        return ["Retinal field could not be detected."]

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

    if fundus_area_ratio > 0.88 or (touches_border and fundus_area_ratio > 0.7):
        warnings.append("Retinal field boundary could not be separated from background.")
    if circularity < 0.45:
        warnings.append("Detected retinal field is not round enough.")
    if extent > 0.9:
        warnings.append("Detected field fills the crop like a background surface.")

    masked_pixels = image[mask > 0]
    if masked_pixels.size:
        blue_mean = float(np.mean(masked_pixels[:, 0]))
        green_mean = float(np.mean(masked_pixels[:, 1]))
        red_mean = float(np.mean(masked_pixels[:, 2]))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation_mean = float(np.mean(hsv[:, :, 1][mask > 0]))
        red_orange_score = red_mean - max(green_mean, blue_mean)

        if saturation_mean < 18.0 or red_orange_score < -5.0:
            warnings.append("Image does not match expected retinal color.")

    return warnings


def add_feature_quality_warnings(
    quality: QualityReport,
    features: FeatureReport,
) -> QualityReport:
    warnings = list(quality.warnings)

    if features.vessel_density < 0.001:
        warnings.append("Retinal vessel pattern is not visible.")

    return QualityReport(
        is_acceptable=len(warnings) == 0,
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
    gray_float = gray.astype(np.float32) / 255.0
    vesselness = np.zeros_like(gray_float, dtype=np.float32)

    for sigma in (1.0, 2.0, 4.0):
        smoothed = cv2.GaussianBlur(gray_float, (0, 0), sigma)
        dxx = cv2.Sobel(smoothed, cv2.CV_32F, 2, 0, ksize=3)
        dxy = cv2.Sobel(smoothed, cv2.CV_32F, 1, 1, ksize=3)
        dyy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 2, ksize=3)

        trace = dxx + dyy
        determinant_term = np.sqrt(np.maximum((dxx - dyy) ** 2 + 4 * dxy**2, 0))
        lambda1 = 0.5 * (trace - determinant_term)
        lambda2 = 0.5 * (trace + determinant_term)

        swap = np.abs(lambda1) > np.abs(lambda2)
        lambda1, lambda2 = (
            np.where(swap, lambda2, lambda1),
            np.where(swap, lambda1, lambda2),
        )

        beta = 0.5
        gamma = 15.0
        rb = (lambda1 / (lambda2 + 1e-6)) ** 2
        s2 = lambda1**2 + lambda2**2
        response = np.exp(-rb / (2 * beta**2)) * (1 - np.exp(-s2 / (2 * gamma**2)))
        response[lambda2 > 0] = 0
        vesselness = np.maximum(vesselness, response)

    vesselness = normalize_uint8(vesselness)
    vesselness[mask == 0] = 0

    return vesselness


def segment_vessels(vesselness: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked_values = vesselness[mask > 0]

    if masked_values.size == 0:
        return np.zeros_like(vesselness)

    _, vessels = cv2.threshold(vesselness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    vessels[mask == 0] = 0
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vessels = cv2.morphologyEx(vessels, cv2.MORPH_CLOSE, kernel_close)
    vessels = cv2.morphologyEx(vessels, cv2.MORPH_OPEN, kernel_open)

    return vessels


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
    optic_disc: np.ndarray,
) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    background = cv2.morphologyEx(
        lightness,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    )
    top_hat = cv2.subtract(lightness, background)
    combined = cv2.addWeighted(top_hat, 0.7, enhanced, 0.3, 0)
    candidate_mask = (mask > 0) & (optic_disc == 0)
    threshold = np.percentile(combined[candidate_mask], 96) if np.any(candidate_mask) else 255
    bright = np.zeros_like(enhanced)
    bright[(combined >= threshold) & candidate_mask] = 255
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    return remove_small_components(bright, min_area=12, max_area=5000)


def detect_dark_lesions(
    enhanced: np.ndarray,
    mask: np.ndarray,
    vessels: np.ndarray,
    optic_disc: np.ndarray,
) -> np.ndarray:
    inverted = cv2.bitwise_not(enhanced)
    candidate_mask = (mask > 0) & (vessels == 0) & (optic_disc == 0)
    threshold = np.percentile(inverted[candidate_mask], 97) if np.any(candidate_mask) else 255
    dark = np.zeros_like(enhanced)
    dark[(inverted >= threshold) & candidate_mask] = 255
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    return remove_small_components(dark, min_area=5, max_area=2500)


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
    microaneurysm_count, hemorrhage_count = count_dark_lesions(dark_lesions)
    pixels = enhanced[fundus_mask > 0]

    return FeatureReport(
        vessel_density=float(vessel_area / fundus_area),
        vessel_area=vessel_area,
        bright_lesion_area=bright_area,
        dark_lesion_area=dark_area,
        microaneurysm_count=microaneurysm_count,
        hemorrhage_candidate_count=hemorrhage_count,
        optic_disc_area=int(np.count_nonzero(optic_disc)),
        mean_intensity=float(np.mean(pixels)) if pixels.size else 0.0,
        intensity_std=float(np.std(pixels)) if pixels.size else 0.0,
        texture_contrast=estimate_texture_contrast(enhanced, fundus_mask),
    )


def count_dark_lesions(mask: np.ndarray) -> tuple[int, int]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    microaneurysm_count = 0
    hemorrhage_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if 5 <= area <= 80:
            microaneurysm_count += 1
        elif area > 80:
            hemorrhage_count += 1

    return microaneurysm_count, hemorrhage_count


def estimate_texture_contrast(gray: np.ndarray, mask: np.ndarray) -> float:
    pixels = gray[mask > 0]

    if pixels.size < 2:
        return 0.0

    hist = cv2.calcHist([pixels], [0], None, [32], [0, 256]).flatten()
    probabilities = hist / max(float(np.sum(hist)), 1.0)
    indices = np.arange(32, dtype=np.float32)
    mean = float(np.sum(indices * probabilities))
    return float(np.sum(((indices - mean) ** 2) * probabilities))


def classify_referable(features: FeatureReport, quality: QualityReport) -> ScreeningResult:
    reasons: list[str] = []

    if not quality.is_acceptable:
        return ScreeningResult(
            classification="Image not suitable for DR screening",
            referable=False,
            reason=", ".join(quality.warnings),
            disclaimer=(
                "Screening support only. Retake with a clear retinal image before "
                "reviewing diabetic retinopathy features."
            ),
        )
    if features.microaneurysm_count >= 8:
        reasons.append("multiple microaneurysm candidates")
    if features.hemorrhage_candidate_count >= 2:
        reasons.append("hemorrhage candidates detected")
    if features.bright_lesion_area >= 450:
        reasons.append("bright lesion candidate area is elevated")
    if features.dark_lesion_area >= 350:
        reasons.append("dark lesion candidate area is elevated")
    if features.vessel_density > 0.15:
        reasons.append("vessel density exceeds screening threshold")
    if features.intensity_std > 45.0:
        reasons.append("retinal intensity variation exceeds threshold")
    if features.texture_contrast > 60.0:
        reasons.append("texture contrast exceeds threshold")

    referable = len(reasons) > 0
    classification = "Referable DR suspected" if referable else "Non-referable"

    return ScreeningResult(
        classification=classification,
        referable=referable,
        reason=", ".join(reasons) if reasons else "no rule threshold exceeded",
        disclaimer=(
            "Screening support only. This result is not a medical diagnosis and "
            "must be reviewed by a qualified eye-care professional."
        ),
    )


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
