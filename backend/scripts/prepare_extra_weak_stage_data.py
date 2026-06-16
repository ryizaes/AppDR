"""Inspect and prepare selected weak-stage extra images for AppDR.

This script does not retrain models. It safely inspects the three user-confirmed
Downloads datasets, selects only weak-stage classes, extracts the existing 203
handcrafted features for selected images, and appends them to the current
features_combined.csv table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
from feature_extraction import FeatureExtractionError, extract_feature_dict


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXTRA_ROOT = Path.home() / "Downloads" / "AppDR_extra_datasets"
DOWNLOADS = Path.home() / "Downloads"
TARGET_TOTALS = {1: 1500, 3: 1000, 4: 1500}
WEAK_CLASS_PRIORITY = [3, 1, 4]
MEDICAL_LABELS = config.CLASS_NAMES
METADATA_COLUMNS = [
    "image_id",
    "image_path",
    "source_dataset",
    "label",
    "medical_label",
    "image_sha256",
]


@dataclass(frozen=True)
class Candidate:
    source_dataset: str
    image_id: str
    path: Path
    label: int
    label_source: str


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    candidates, unclear_rows, inspection = inspect_datasets(args.extra_root, args.features_csv)
    write_inspection_reports(candidates, unclear_rows, inspection, results_dir)
    print_inspection_summary(inspection)
    if args.inspect_only:
        return

    selected, clean_rows, rejected_rows, duplicate_rows, quality_rows = select_and_filter(
        candidates=candidates,
        current_features_csv=args.features_csv,
        results_dir=results_dir,
    )
    write_selection_reports(
        selected=selected,
        clean_rows=clean_rows,
        rejected_rows=rejected_rows,
        duplicate_rows=duplicate_rows,
        quality_rows=quality_rows,
        results_dir=results_dir,
    )
    print_selection_summary(clean_rows, args.features_csv)
    if args.no_extract:
        return

    extra_features = extract_selected_features(clean_rows, args.workers, results_dir)
    build_balanced_feature_table(
        current_features_csv=args.features_csv,
        extra_features=extra_features,
        output_csv=args.output_csv,
        results_dir=results_dir,
    )


def inspect_datasets(
    extra_root: Path,
    current_features_csv: Path,
) -> tuple[list[Candidate], list[dict[str, Any]], dict[str, Any]]:
    candidates: list[Candidate] = []
    unclear: list[dict[str, Any]] = []
    inspection: dict[str, Any] = {
        "confirmed_mapping": {
            "archive.zip": "Diabetic Retinopathy Dataset by Sachin Kumar",
            "Imagenes.zip + idrid_labels.csv": "IDRiD grading",
            "content.zip": "Diabetic_Retinopathy_Balanced",
        },
        "extraction_root": str(extra_root),
        "current_counts": current_counts(current_features_csv),
        "datasets": {},
    }

    dataset1 = inspect_sachin(extra_root / "dataset1_sachin", unclear)
    dataset2 = inspect_idrid(extra_root / "dataset2_idrid", DOWNLOADS / "idrid_labels.csv", unclear)
    dataset3 = inspect_balanced(extra_root / "dataset3_balanced", unclear)

    for name, result in (
        ("dataset1_sachin", dataset1),
        ("dataset2_idrid", dataset2),
        ("dataset3_balanced", dataset3),
    ):
        candidates.extend(result["candidates"])
        inspection["datasets"][name] = {
            key: value for key, value in result.items() if key != "candidates"
        }

    useful_counts: Counter[int] = Counter(candidate.label for candidate in candidates)
    inspection["total_available_useful_extra_images"] = count_dict(useful_counts)
    return candidates, unclear, inspection


def inspect_sachin(root: Path, unclear: list[dict[str, Any]]) -> dict[str, Any]:
    candidates: list[Candidate] = []
    folder_counts: dict[str, int] = {}
    label_counts: Counter[int] = Counter()
    if not root.exists():
        return {"exists": False, "candidates": candidates, "raw_counts": count_dict(label_counts)}

    for folder in sorted([path for path in root.iterdir() if path.is_dir()]):
        label = map_text_label(folder.name)
        image_paths = list_images(folder)
        folder_counts[folder.name] = len(image_paths)
        if label is None:
            for path in image_paths:
                unclear.append(unclear_row("Sachin Kumar", path, folder.name, "Unmapped folder name"))
            continue
        for path in image_paths:
            candidates.append(
                Candidate("Sachin Kumar", path.stem, path, label, f"folder:{folder.name}"),
            )
            label_counts[label] += 1
    return {
        "exists": True,
        "labeling_type": "folder_based",
        "root": str(root),
        "folder_counts": folder_counts,
        "raw_counts": count_dict(label_counts),
        "class3_available": int(label_counts.get(3, 0)),
        "class1_available": int(label_counts.get(1, 0)),
        "class4_available": int(label_counts.get(4, 0)),
        "candidates": candidates,
    }


def inspect_idrid(root: Path, labels_csv: Path, unclear: list[dict[str, Any]]) -> dict[str, Any]:
    candidates: list[Candidate] = []
    label_counts: Counter[int] = Counter()
    if not root.exists() or not labels_csv.exists():
        return {"exists": False, "candidates": candidates, "raw_counts": count_dict(label_counts)}

    labels = pd.read_csv(labels_csv)
    image_column = detect_column(labels, ("id_code", "image", "image_id", "filename"))
    label_column = detect_column(labels, ("diagnosis", "dr_grade", "grade", "label"))
    image_root = root / "Imagenes"
    matched = 0
    missing = 0
    for record in labels.to_dict("records"):
        image_id = str(record.get(image_column, "")).strip()
        label = map_numeric_or_text_label(record.get(label_column))
        image_path = resolve_image_from_id(image_root, image_id)
        if label is None:
            unclear.append(unclear_row("IDRiD grading", image_path or image_root / image_id, image_id, "Unclear CSV label"))
            continue
        if image_path is None:
            missing += 1
            unclear.append(unclear_row("IDRiD grading", image_root / image_id, image_id, "Image file not found"))
            continue
        matched += 1
        candidates.append(Candidate("IDRiD grading", image_id, image_path, label, f"csv:{label_column}"))
        label_counts[label] += 1
    return {
        "exists": True,
        "labeling_type": "csv_based",
        "root": str(root),
        "csv": str(labels_csv),
        "csv_shape": [int(labels.shape[0]), int(labels.shape[1])],
        "csv_columns": [str(column) for column in labels.columns],
        "image_column": image_column,
        "label_column": label_column,
        "matched_images": matched,
        "missing_images": missing,
        "raw_counts": count_dict(label_counts),
        "class3_available": int(label_counts.get(3, 0)),
        "class1_available": int(label_counts.get(1, 0)),
        "class4_available": int(label_counts.get(4, 0)),
        "candidates": candidates,
    }


def inspect_balanced(root: Path, unclear: list[dict[str, Any]]) -> dict[str, Any]:
    candidates: list[Candidate] = []
    label_counts: Counter[int] = Counter()
    split_counts: dict[str, dict[str, int]] = {}
    data_root = root / "Diabetic_Balanced_Data"
    if not data_root.exists():
        return {"exists": False, "candidates": candidates, "raw_counts": count_dict(label_counts)}

    for split_dir in sorted([path for path in data_root.iterdir() if path.is_dir()]):
        split_counts[split_dir.name] = {}
        for label_dir in sorted([path for path in split_dir.iterdir() if path.is_dir()]):
            label = map_numeric_or_text_label(label_dir.name)
            image_paths = list_images(label_dir)
            split_counts[split_dir.name][label_dir.name] = len(image_paths)
            if label is None:
                for path in image_paths:
                    unclear.append(unclear_row("Diabetic_Retinopathy_Balanced", path, label_dir.name, "Unmapped class folder"))
                continue
            for path in image_paths:
                candidates.append(
                    Candidate("Diabetic_Retinopathy_Balanced", path.stem, path, label, f"folder:{label_dir.name}"),
                )
                label_counts[label] += 1
    return {
        "exists": True,
        "labeling_type": "folder_based_split",
        "root": str(root),
        "split_counts": split_counts,
        "raw_counts": count_dict(label_counts),
        "class3_available": int(label_counts.get(3, 0)),
        "class1_available": int(label_counts.get(1, 0)),
        "class4_available": int(label_counts.get(4, 0)),
        "candidates": candidates,
    }


def select_and_filter(
    candidates: list[Candidate],
    current_features_csv: Path,
    results_dir: Path,
) -> tuple[list[Candidate], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    current = pd.read_csv(current_features_csv)
    old_counts = Counter(current["label"].astype(int).tolist())
    current_hashes = set(str(value) for value in current.get("image_sha256", pd.Series(dtype=str)).dropna())
    needed = {
        label: max(0, TARGET_TOTALS[label] - int(old_counts.get(label, 0)))
        for label in TARGET_TOTALS
    }

    by_label: dict[int, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.label in needed and needed[candidate.label] > 0:
            by_label[candidate.label].append(candidate)

    selected: list[Candidate] = []
    clean_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    seen_dhashes_by_label: dict[int, list[int]] = defaultdict(list)

    for label in WEAK_CLASS_PRIORITY:
        label_candidates = sorted(by_label[label], key=candidate_sort_key)
        accepted_for_label = 0
        for candidate in label_candidates:
            if accepted_for_label >= needed[label]:
                break
            quality = inspect_image_quality(candidate.path)
            quality_rows.append(quality_row(candidate, quality))
            if not quality["readable"] or quality["tiny_or_invalid"]:
                rejected_rows.append(rejection_row(candidate, "invalid_or_unreadable", quality))
                continue
            if not quality["quality_pass"]:
                rejected_rows.append(rejection_row(candidate, "failed_quality_check", quality))
                continue
            image_hash = file_sha256(candidate.path)
            if image_hash in current_hashes:
                duplicate_rows.append(duplicate_row(candidate, "duplicate_of_current_dataset", image_hash))
                rejected_rows.append(rejection_row(candidate, "duplicate_of_current_dataset", quality, image_hash))
                continue
            if image_hash in seen_hashes:
                duplicate_rows.append(duplicate_row(candidate, "duplicate_within_extra_selection", image_hash))
                rejected_rows.append(rejection_row(candidate, "duplicate_within_extra_selection", quality, image_hash))
                continue
            dhash = int(quality.get("dhash", -1))
            near_duplicate = any(hamming_distance(dhash, old) <= 2 for old in seen_dhashes_by_label[label])
            if near_duplicate:
                duplicate_rows.append(duplicate_row(candidate, "near_duplicate_within_extra_selection", image_hash))
                rejected_rows.append(rejection_row(candidate, "near_duplicate_within_extra_selection", quality, image_hash))
                continue

            selected.append(candidate)
            seen_hashes.add(image_hash)
            seen_dhashes_by_label[label].append(dhash)
            accepted_for_label += 1
            clean_rows.append(
                {
                    "source_dataset": candidate.source_dataset,
                    "original_image_path": str(candidate.path),
                    "mapped_label": candidate.label,
                    "medical_label": MEDICAL_LABELS[candidate.label],
                    "reason_selected": selection_reason(candidate, old_counts),
                    "image_quality_status": quality["status"],
                    "duplicate_status": "unique",
                    "image_sha256": image_hash,
                    "dhash": dhash,
                    "blur_score": quality["blur_score"],
                    "brightness_mean": quality["brightness_mean"],
                    "contrast_std": quality["contrast_std"],
                    "width": quality["width"],
                    "height": quality["height"],
                },
            )
    return selected, clean_rows, rejected_rows, duplicate_rows, quality_rows


def extract_selected_features(
    clean_rows: list[dict[str, Any]],
    workers: int,
    results_dir: Path,
) -> pd.DataFrame:
    failures: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    tasks = [(row, len(config.FEATURE_NAMES)) for row in clean_rows]

    if workers <= 1:
        for index, task in enumerate(tasks, start=1):
            ok, payload = extract_worker(task)
            (output_rows if ok else failures).append(payload)
            print_progress(index, len(tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(extract_worker, task) for task in tasks]
            for index, future in enumerate(as_completed(futures), start=1):
                ok, payload = future.result()
                (output_rows if ok else failures).append(payload)
                print_progress(index, len(futures))
    print()
    write_csv(results_dir / "selected_extra_feature_failures.csv", failures)
    if not output_rows:
        raise RuntimeError("No selected extra images produced valid 203-feature rows.")

    table = pd.DataFrame(output_rows)
    duplicate_feature_mask = table.duplicated(subset=config.FEATURE_NAMES, keep="first")
    if duplicate_feature_mask.any():
        duplicate_rows = table.loc[duplicate_feature_mask, [
            "source_dataset", "image_path", "label", "image_sha256",
        ]]
        duplicate_rows.to_csv(results_dir / "selected_extra_duplicate_features.csv", index=False)
        table = table.loc[~duplicate_feature_mask].reset_index(drop=True)
    return table


def extract_worker(task: tuple[dict[str, Any], int]) -> tuple[bool, dict[str, Any]]:
    row, expected_feature_count = task
    path = Path(row["original_image_path"])
    cv2.setNumThreads(1)
    try:
        features = extract_feature_dict(path)
        if len(features) != expected_feature_count:
            raise ValueError(f"Expected {expected_feature_count} features, got {len(features)}")
        missing = [name for name in config.FEATURE_NAMES if name not in features]
        if missing:
            raise ValueError(f"Missing features: {missing[:10]}")
        output = {
            "image_id": path.stem,
            "image_path": str(path),
            "source_dataset": row["source_dataset"],
            "label": int(row["mapped_label"]),
            "medical_label": row["medical_label"],
            "image_sha256": row["image_sha256"],
        }
        output.update({name: clean_float(features[name]) for name in config.FEATURE_NAMES})
        return True, output
    except (FeatureExtractionError, OSError, ValueError) as exc:
        return False, {
            "source_dataset": row["source_dataset"],
            "image_path": str(path),
            "label": int(row["mapped_label"]),
            "error": str(exc),
        }


def build_balanced_feature_table(
    current_features_csv: Path,
    extra_features: pd.DataFrame,
    output_csv: Path,
    results_dir: Path,
) -> None:
    current = pd.read_csv(current_features_csv)
    if "medical_label" not in current.columns:
        current["medical_label"] = current["label"].astype(int).map(MEDICAL_LABELS)
    for column in METADATA_COLUMNS:
        if column not in current.columns:
            current[column] = ""
    combined = pd.concat([current, extra_features], ignore_index=True, sort=False)
    combined = combined[[*METADATA_COLUMNS, *config.FEATURE_NAMES]]
    combined = sanitize_feature_table(combined)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["image_sha256"], keep="first")
    combined = combined.drop_duplicates(subset=config.FEATURE_NAMES, keep="first")
    removed = before - len(combined)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    report = build_count_report(current, extra_features, combined, removed, output_csv)
    (results_dir / "balanced_dataset_count_report.json").write_text(
        json.dumps(to_jsonable(report), indent=2),
        encoding="utf-8",
    )
    (results_dir / "balanced_dataset_count_report.md").write_text(
        render_count_report(report),
        encoding="utf-8",
    )
    print(render_count_report(report))


def sanitize_feature_table(table: pd.DataFrame) -> pd.DataFrame:
    output = table.copy()
    for feature_name in config.FEATURE_NAMES:
        output[feature_name] = pd.to_numeric(output[feature_name], errors="coerce")
    values = output[config.FEATURE_NAMES].to_numpy(dtype=np.float64)
    values[~np.isfinite(values)] = np.nan
    output.loc[:, config.FEATURE_NAMES] = values
    return output.dropna(subset=config.FEATURE_NAMES, how="all").reset_index(drop=True)


def write_inspection_reports(
    candidates: list[Candidate],
    unclear_rows: list[dict[str, Any]],
    inspection: dict[str, Any],
    results_dir: Path,
) -> None:
    candidate_rows = [
        {
            "source_dataset": candidate.source_dataset,
            "image_path": str(candidate.path),
            "label": candidate.label,
            "medical_label": MEDICAL_LABELS[candidate.label],
            "label_source": candidate.label_source,
        }
        for candidate in candidates
    ]
    write_csv(results_dir / "extra_dataset_candidates.csv", candidate_rows)
    write_csv(results_dir / "extra_dataset_unclear_labels.csv", unclear_rows)
    (results_dir / "extra_dataset_inspection_report.json").write_text(
        json.dumps(to_jsonable(inspection), indent=2),
        encoding="utf-8",
    )
    (results_dir / "extra_dataset_inspection_report.md").write_text(
        render_inspection_report(inspection),
        encoding="utf-8",
    )


def write_selection_reports(
    selected: list[Candidate],
    clean_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
    results_dir: Path,
) -> None:
    manifest_rows = [
        {
            "source_dataset": row["source_dataset"],
            "original_image_path": row["original_image_path"],
            "mapped_label": row["mapped_label"],
            "medical_label": row["medical_label"],
            "reason_selected": row["reason_selected"],
            "image_quality_status": row["image_quality_status"],
            "duplicate_status": row["duplicate_status"],
        }
        for row in clean_rows
    ]
    write_csv(results_dir / "selected_extra_dataset_manifest.csv", manifest_rows)
    write_csv(results_dir / "selected_extra_clean.csv", clean_rows)
    write_csv(results_dir / "selected_extra_rejected.csv", rejected_rows)
    write_csv(results_dir / "duplicate_report.csv", duplicate_rows)
    write_csv(results_dir / "extra_image_quality_report.csv", quality_rows)


def render_inspection_report(inspection: dict[str, Any]) -> str:
    lines = [
        "# Extra Dataset Inspection Report",
        "",
        "Confirmed mapping:",
        "",
        "- `archive.zip` = Diabetic Retinopathy Dataset by Sachin Kumar",
        "- `Imagenes.zip` + `idrid_labels.csv` = IDRiD grading",
        "- `content.zip` = Diabetic_Retinopathy_Balanced",
        "",
        f"Current AppDR counts: `{inspection['current_counts']}`",
        "",
        "## Raw Extra Dataset Counts",
        "",
    ]
    for name, data in inspection["datasets"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- Labeling type: `{data.get('labeling_type')}`")
        lines.append(f"- Raw counts: `{data.get('raw_counts')}`")
        lines.append(f"- Class 1 available: `{data.get('class1_available', 0)}`")
        lines.append(f"- Class 3 available: `{data.get('class3_available', 0)}`")
        lines.append(f"- Class 4 available: `{data.get('class4_available', 0)}`")
        lines.append("")
    lines.append(f"Total useful extra counts: `{inspection['total_available_useful_extra_images']}`")
    lines.append("")
    lines.append("No training was started by this inspection step.")
    lines.append("")
    return "\n".join(lines)


def render_count_report(report: dict[str, Any]) -> str:
    lines = [
        "# Balanced Feature Table Count Report",
        "",
        f"Output feature table: `{report['output_csv']}`",
        "",
        f"Old counts: `{report['old_counts']}`",
        f"Extra selected counts: `{report['extra_selected_counts']}`",
        f"New combined counts: `{report['new_counts']}`",
        f"New class percentages: `{report['new_percentages']}`",
        "",
        f"Class 1 increase: `{report['class_increases'].get('1', 0)}`",
        f"Class 3 increase: `{report['class_increases'].get('3', 0)}`",
        f"Class 4 increase: `{report['class_increases'].get('4', 0)}`",
        f"Class 0 increase: `{report['class_increases'].get('0', 0)}`",
        f"Class 2 increase: `{report['class_increases'].get('2', 0)}`",
        "",
        f"Rows removed during final duplicate cleanup: `{report['duplicate_rows_removed_after_append']}`",
        "",
    ]
    return "\n".join(lines)


def print_inspection_summary(inspection: dict[str, Any]) -> None:
    print(render_inspection_report(inspection), flush=True)


def print_selection_summary(clean_rows: list[dict[str, Any]], features_csv: Path) -> None:
    old = current_counts(features_csv)
    extra = Counter(int(row["mapped_label"]) for row in clean_rows)
    new = {str(label): int(old.get(str(label), 0)) + int(extra.get(label, 0)) for label in config.CLASS_LABELS}
    print("Selected extra counts:", count_dict(extra), flush=True)
    print("Projected new counts:", new, flush=True)


def build_count_report(
    current: pd.DataFrame,
    extra: pd.DataFrame,
    combined: pd.DataFrame,
    removed: int,
    output_csv: Path,
) -> dict[str, Any]:
    old_counts = count_dict(Counter(current["label"].astype(int).tolist()))
    extra_counts = count_dict(Counter(extra["label"].astype(int).tolist()))
    new_counts = count_dict(Counter(combined["label"].astype(int).tolist()))
    total = max(int(len(combined)), 1)
    increases = {
        str(label): int(new_counts[str(label)] - old_counts[str(label)])
        for label in config.CLASS_LABELS
    }
    return {
        "output_csv": str(output_csv),
        "old_counts": old_counts,
        "extra_selected_counts": extra_counts,
        "new_counts": new_counts,
        "new_percentages": {
            str(label): round((new_counts[str(label)] / total) * 100.0, 2)
            for label in config.CLASS_LABELS
        },
        "class_increases": increases,
        "duplicate_rows_removed_after_append": int(removed),
        "feature_count": len(config.FEATURE_NAMES),
    }


def current_counts(features_csv: Path) -> dict[str, int]:
    if not features_csv.exists():
        return {str(label): 0 for label in config.CLASS_LABELS}
    table = pd.read_csv(features_csv, usecols=["label"])
    return count_dict(Counter(table["label"].astype(int).tolist()))


def candidate_sort_key(candidate: Candidate) -> tuple[int, str]:
    # Prefer high-label-certainty datasets, then original/non-augmented files.
    source_rank = {
        "IDRiD grading": 0,
        "Sachin Kumar": 1,
        "Diabetic_Retinopathy_Balanced": 2,
    }.get(candidate.source_dataset, 9)
    aug_rank = 1 if "_aug_" in candidate.path.name.lower() else 0
    split_rank = 0
    parts = {part.lower() for part in candidate.path.parts}
    if "train" in parts:
        split_rank = 0
    elif "val" in parts:
        split_rank = 1
    elif "test" in parts:
        split_rank = 2
    return (source_rank, aug_rank, split_rank, str(candidate.path).lower())


def selection_reason(candidate: Candidate, old_counts: Counter[int]) -> str:
    return (
        f"Selected to fill weak class {candidate.label} "
        f"({MEDICAL_LABELS[candidate.label]}); old count was "
        f"{int(old_counts.get(candidate.label, 0))}."
    )


def inspect_image_quality(path: Path) -> dict[str, Any]:
    base = {
        "readable": False,
        "tiny_or_invalid": True,
        "quality_pass": False,
        "status": "unreadable",
        "width": 0,
        "height": 0,
        "blur_score": 0.0,
        "brightness_mean": 0.0,
        "contrast_std": 0.0,
        "dhash": -1,
    }
    try:
        data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        return base
    if image is None:
        return base
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    tiny = width < 200 or height < 200
    warnings: list[str] = []
    if tiny:
        warnings.append("image_too_small")
    if blur < 8.0:
        warnings.append("blurry")
    if brightness < 12.0:
        warnings.append("too_dark")
    if brightness > 245.0:
        warnings.append("too_bright")
    if contrast < 5.0:
        warnings.append("low_contrast")
    quality_pass = not warnings
    return {
        "readable": True,
        "tiny_or_invalid": tiny,
        "quality_pass": quality_pass,
        "status": "acceptable" if quality_pass else ";".join(warnings),
        "width": int(width),
        "height": int(height),
        "blur_score": blur,
        "brightness_mean": brightness,
        "contrast_std": contrast,
        "dhash": dhash(gray),
    }


def dhash(gray: np.ndarray, hash_size: int = 8) -> int:
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return int(value)


def hamming_distance(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


def map_text_label(value: Any) -> int | None:
    text = normalize_label(value)
    direct = {
        "healthy": 0,
        "normal": 0,
        "no_dr": 0,
        "no_diabetic_retinopathy": 0,
        "no_apparent_dr": 0,
        "mild": 1,
        "mild_dr": 1,
        "moderate": 2,
        "moderate_dr": 2,
        "severe": 3,
        "severe_dr": 3,
        "proliferative": 4,
        "proliferative_dr": 4,
        "proliferate_dr": 4,
        "pdr": 4,
    }
    if text in direct:
        return direct[text]
    return map_numeric_or_text_label(value)


def map_numeric_or_text_label(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        numeric = int(value)
        if numeric in config.CLASS_LABELS:
            return numeric
    except (TypeError, ValueError):
        pass
    text = normalize_label(value)
    for label in config.CLASS_LABELS:
        if str(label) == text:
            return label
    return {
        "no_apparent_diabetic_retinopathy": 0,
        "mild_non_proliferative_diabetic_retinopathy": 1,
        "moderate_non_proliferative_diabetic_retinopathy": 2,
        "severe_non_proliferative_diabetic_retinopathy": 3,
        "proliferative_diabetic_retinopathy": 4,
    }.get(text)


def normalize_label(value: Any) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def detect_column(table: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    lookup = {str(column).lower().strip(): str(column) for column in table.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(f"Could not detect required column from {candidates}")


def resolve_image_from_id(root: Path, image_id: str) -> Path | None:
    name = Path(image_id).name
    candidates = [root / name]
    if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
        candidates.extend(root / f"{name}{extension}" for extension in sorted(IMAGE_EXTENSIONS))
    for candidate in candidates:
        if candidate.exists() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
            return candidate
    matches = [path for path in root.glob(f"{name}.*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    return matches[0] if matches else None


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_float(value: Any) -> float:
    number = float(value)
    return number if np.isfinite(number) else np.nan


def count_dict(counter: Counter[int]) -> dict[str, int]:
    return {str(label): int(counter.get(label, 0)) for label in config.CLASS_LABELS}


def quality_row(candidate: Candidate, quality: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_dataset": candidate.source_dataset,
        "image_path": str(candidate.path),
        "label": candidate.label,
        **quality,
    }


def rejection_row(
    candidate: Candidate,
    reason: str,
    quality: dict[str, Any],
    image_hash: str = "",
) -> dict[str, Any]:
    return {
        "source_dataset": candidate.source_dataset,
        "image_path": str(candidate.path),
        "mapped_label": candidate.label,
        "medical_label": MEDICAL_LABELS[candidate.label],
        "rejection_reason": reason,
        "image_quality_status": quality.get("status", ""),
        "image_sha256": image_hash,
    }


def duplicate_row(candidate: Candidate, status: str, image_hash: str) -> dict[str, Any]:
    return {
        "source_dataset": candidate.source_dataset,
        "image_path": str(candidate.path),
        "mapped_label": candidate.label,
        "duplicate_status": status,
        "image_sha256": image_hash,
    }


def unclear_row(source: str, path: Path, raw_label: Any, reason: str) -> dict[str, Any]:
    return {
        "source_dataset": source,
        "image_path": str(path),
        "raw_label": str(raw_label),
        "reason": reason,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(to_jsonable(rows))


def print_progress(done: int, total: int) -> None:
    if done == total or done % 25 == 0:
        print(f"\rExtracted features for {done}/{total} selected extra images", end="", flush=True)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Counter):
        return {str(key): int(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare selected extra weak-stage AppDR images.")
    parser.add_argument("--extra-root", type=Path, default=EXTRA_ROOT)
    parser.add_argument("--features-csv", type=Path, default=BACKEND_DIR / "features_combined.csv")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=BACKEND_DIR / "features_combined_balanced.csv",
    )
    parser.add_argument("--results-dir", type=Path, default=BACKEND_DIR / "results")
    parser.add_argument("--workers", type=int, default=config.DEFAULT_WORKERS)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--no-extract", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
