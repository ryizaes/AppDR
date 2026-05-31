from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import config
from utils import ensure_dir


@dataclass(frozen=True)
class PreprocessingResult:
    original_bgr: np.ndarray
    normalized_bgr: np.ndarray
    fov_mask: np.ndarray
    optic_disc_mask: np.ndarray
    green_channel: np.ndarray
    illumination_corrected: np.ndarray
    clahe_green: np.ndarray
    denoised_green: np.ndarray


def preprocess_retinal_image(
    image_path: str | Path,
    debug: bool = False,
    debug_dir: str | Path | None = None,
    clahe_clip_limit: float | None = None,
) -> PreprocessingResult:
    """Load and preprocess one fundus image for classical feature extraction."""
    original = load_retinal_image(image_path)
    normalized = normalize_image_dimensions(original)
    fov_mask = build_fov_mask(normalized)
    optic_disc_mask = detect_optic_disc_mask(normalized, fov_mask)

    green = normalized[:, :, 1]
    green_filled = fill_outside_mask(green, fov_mask)
    illumination_corrected = normalize_illumination(green_filled, fov_mask)
    clahe_green = apply_clahe(illumination_corrected, clip_limit=clahe_clip_limit)
    denoised = denoise_green_channel(clahe_green)
    denoised[fov_mask == 0] = 0

    result = PreprocessingResult(
        original_bgr=original,
        normalized_bgr=normalized,
        fov_mask=fov_mask,
        optic_disc_mask=optic_disc_mask,
        green_channel=green,
        illumination_corrected=illumination_corrected,
        clahe_green=clahe_green,
        denoised_green=denoised,
    )

    if debug and debug_dir is not None:
        save_preprocessing_debug(Path(debug_dir), Path(image_path).stem, result)

    return result


def load_retinal_image(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise ValueError(f"Could not read image bytes from {path}") from exc

    if image is None or image.size == 0:
        raise ValueError(f"Corrupted or unsupported image file: {path}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a 3-channel retinal image: {path}")

    return image


def normalize_image_dimensions(image: np.ndarray) -> np.ndarray:
    cropped = crop_to_fundus_bounds(image)
    height, width = cropped.shape[:2]
    longest_side = max(height, width)

    if longest_side <= config.MAX_IMAGE_SIZE:
        return cropped.copy()

    scale = config.MAX_IMAGE_SIZE / float(longest_side)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(cropped, new_size, interpolation=cv2.INTER_AREA)


def crop_to_fundus_bounds(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, mask = cv2.threshold(
        blurred,
        config.FUNDUS_CROP_THRESHOLD,
        255,
        cv2.THRESH_BINARY,
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk_kernel(10))
    contour = largest_contour(mask)

    if contour is None:
        return image

    height, width = image.shape[:2]
    x, y, box_width, box_height = cv2.boundingRect(contour)
    margin = int(round(min(height, width) * config.FUNDUS_CROP_MARGIN_RATIO))
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(width, x + box_width + margin)
    y1 = min(height, y + box_height + margin)

    if x1 <= x0 or y1 <= y0:
        return image

    return image[y0:y1, x0:x1]


def build_fov_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, config.FOV_THRESHOLD, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk_kernel(10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk_kernel(2))
    return keep_largest_component(mask)


def detect_optic_disc_mask(image: np.ndarray, fov_mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    optic_disc_mask = np.zeros_like(gray)
    fov_pixels = gray[fov_mask > 0]

    if fov_pixels.size == 0:
        return optic_disc_mask

    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=17, sigmaY=17)
    threshold = float(np.percentile(blurred[fov_mask > 0], 95.0))
    bright = np.zeros_like(gray)
    bright[(blurred >= threshold) & (fov_mask > 0)] = 255
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, disk_kernel(8))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, disk_kernel(3))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fov_area = max(float(np.count_nonzero(fov_mask)), 1.0)

    best_contour = None
    best_score = 0.0

    for contour in contours:
        area = float(cv2.contourArea(contour))
        circularity, aspect_ratio, solidity = contour_shape_metrics(contour)

        if not 0.0015 * fov_area <= area <= 0.16 * fov_area:
            continue
        if circularity < 0.18 or aspect_ratio > 3.0 or solidity < 0.35:
            continue

        candidate = np.zeros_like(gray)
        cv2.drawContours(candidate, [contour], -1, 255, thickness=cv2.FILLED)
        mean_brightness = float(np.mean(gray[candidate > 0]))
        score = area * circularity * solidity * max(mean_brightness, 1.0)

        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return optic_disc_mask

    cv2.drawContours(optic_disc_mask, [best_contour], -1, 255, thickness=cv2.FILLED)
    optic_disc_mask = cv2.dilate(optic_disc_mask, disk_kernel(10), iterations=1)
    optic_disc_mask[fov_mask == 0] = 0
    return optic_disc_mask


def normalize_illumination(green: np.ndarray, fov_mask: np.ndarray) -> np.ndarray:
    background = cv2.GaussianBlur(
        green,
        (0, 0),
        sigmaX=config.ILLUMINATION_SIGMA,
        sigmaY=config.ILLUMINATION_SIGMA,
    )
    corrected = cv2.divide(green, background, scale=128)
    corrected[fov_mask == 0] = 0
    return corrected


def apply_clahe(gray: np.ndarray, clip_limit: float | None = None) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT if clip_limit is None else float(clip_limit),
        tileGridSize=config.CLAHE_TILE_GRID_SIZE,
    )
    return clahe.apply(gray)


def denoise_green_channel(gray: np.ndarray) -> np.ndarray:
    # Median filtering protects small lesions better than aggressive smoothing.
    median = cv2.medianBlur(gray, 3)
    return cv2.fastNlMeansDenoising(median, h=config.DENOISE_H)


def fill_outside_mask(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = gray.copy()
    pixels = gray[mask > 0]
    fill_value = int(np.median(pixels)) if pixels.size else 0
    output[mask == 0] = fill_value
    return output


def disk_kernel(radius: int) -> np.ndarray:
    size = (radius * 2) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    output = np.zeros_like(mask)
    output[labels == largest_label] = 255
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


def save_preprocessing_debug(
    debug_dir: Path,
    image_id: str,
    result: PreprocessingResult,
) -> None:
    output_dir = ensure_dir(debug_dir / image_id)
    cv2.imwrite(str(output_dir / "00_original.png"), result.original_bgr)
    cv2.imwrite(str(output_dir / "01_normalized.png"), result.normalized_bgr)
    cv2.imwrite(str(output_dir / "02_fov_mask.png"), result.fov_mask)
    cv2.imwrite(str(output_dir / "03_optic_disc_mask.png"), result.optic_disc_mask)
    cv2.imwrite(str(output_dir / "04_green_channel.png"), result.green_channel)
    cv2.imwrite(str(output_dir / "05_illumination_corrected.png"), result.illumination_corrected)
    cv2.imwrite(str(output_dir / "06_clahe_green.png"), result.clahe_green)
    cv2.imwrite(str(output_dir / "07_denoised_green.png"), result.denoised_green)
