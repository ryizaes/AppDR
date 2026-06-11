"""Prepare the combined OIA-DDR/DDR + APTOS handcrafted-feature dataset.

This script keeps AppDR on its existing classical image-processing pipeline:
each readable retinal image is passed through feature_extraction.py and must
produce the configured 203 features in the same order used by inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
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


METADATA_COLUMNS = [
    "image_id",
    "image_path",
    "source_dataset",
    "label",
    "image_sha256",
]
IMAGE_COLUMN_CANDIDATES = (
    "id_code",
    "image",
    "image_id",
    "filename",
    "file_name",
    "path",
)
LABEL_COLUMN_CANDIDATES = (
    "diagnosis",
    "label",
    "grade",
    "dr_grade",
    "class",
)


def prepare_combined_dataset(
    downloads_dir: Path,
    output_csv: Path,
    report_path: Path,
    failures_path: Path,
    workers: int,
    limit_per_class: int | None,
    resume: bool,
) -> pd.DataFrame:
    output_csv = output_csv.resolve()
    report_path = report_path.resolve()
    failures_path = failures_path.resolve()
    datasets = [
        (
            "OIA-DDR",
            downloads_dir / "DR_grading.csv",
            downloads_dir / "DR_grading",
            True,
        ),
        (
            "APTOS",
            downloads_dir / "train.csv",
            downloads_dir / "train_images",
            True,
        ),
        (
            "APTOS-test-unlabeled",
            downloads_dir / "test.csv",
            downloads_dir / "test_images",
            False,
        ),
    ]

    report: dict[str, Any] = {
        "downloads_dir": str(downloads_dir),
        "expected_feature_count": len(config.FEATURE_NAMES),
        "class_names": {str(key): value for key, value in config.CLASS_NAMES.items()},
        "datasets": {},
    }
    labeled_samples: list[dict[str, Any]] = []

    for source_name, csv_path, images_dir, require_label in datasets:
        samples, dataset_report = collect_csv_dataset(
            source_name=source_name,
            csv_path=csv_path,
            images_dir=images_dir,
            require_label=require_label,
            limit_per_class=limit_per_class,
        )
        report["datasets"][source_name] = dataset_report
        if require_label:
            labeled_samples.extend(samples)

    if not labeled_samples:
        raise RuntimeError("No labeled dataset samples were found in Downloads.")

    partial_csv = Path(f"{output_csv}.partial")
    initial_rows = load_partial_rows(partial_csv) if resume else []
    completed_paths = {str(row.get("image_path")) for row in initial_rows}
    remaining_samples = [
        sample
        for sample in labeled_samples
        if str(sample["image_path"]) not in completed_paths
    ]
    if initial_rows:
        print(
            f"Resuming from {partial_csv}: {len(initial_rows)} completed rows, "
            f"{len(remaining_samples)} remaining.",
            flush=True,
        )

    rows, failures = extract_rows(
        remaining_samples,
        workers=workers,
        partial_csv=partial_csv,
        initial_rows=initial_rows,
    )
    if not rows:
        write_text(failures_path, "\n".join(failures))
        raise RuntimeError("Feature extraction failed for every labeled image.")

    dataframe = pd.DataFrame(rows)
    duplicate_hash_mask = dataframe.duplicated(subset=["image_sha256"], keep="first")
    duplicate_hash_removed = int(duplicate_hash_mask.sum())
    if duplicate_hash_removed:
        dataframe = dataframe.loc[~duplicate_hash_mask].reset_index(drop=True)

    duplicate_row_mask = dataframe.duplicated(
        subset=[*config.FEATURE_NAMES, "label"],
        keep="first",
    )
    duplicate_rows_removed = int(duplicate_row_mask.sum())
    if duplicate_rows_removed:
        dataframe = dataframe.loc[~duplicate_row_mask].reset_index(drop=True)

    dataframe = dataframe[[*METADATA_COLUMNS, *config.FEATURE_NAMES]]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_csv, index=False)
    if partial_csv.exists():
        partial_csv.unlink()
    write_text(failures_path, "\n".join(failures))

    combined_counts = label_counts(dataframe["label"].astype(int).tolist())
    total = int(sum(combined_counts.values()))
    report["combined"] = {
        "feature_csv": str(output_csv),
        "total_labeled_rows_after_cleaning": total,
        "class_counts": combined_counts,
        "class_percentages": {
            str(label): round((count / total) * 100.0, 2) if total else 0.0
            for label, count in combined_counts.items()
        },
        "duplicate_image_hash_rows_removed": duplicate_hash_removed,
        "exact_duplicate_feature_label_rows_removed": duplicate_rows_removed,
        "feature_count": len(config.FEATURE_NAMES),
        "feature_names_saved": True,
        "failures_path": str(failures_path),
        "failure_count": len(failures),
        "checkpoint_resume_enabled": bool(resume),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")

    print_report(report)
    return dataframe


def collect_csv_dataset(
    source_name: str,
    csv_path: Path,
    images_dir: Path,
    require_label: bool,
    limit_per_class: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not csv_path.exists():
        return [], {
            "csv_path": str(csv_path),
            "images_dir": str(images_dir),
            "exists": False,
            "error": "CSV file was not found.",
        }
    if not images_dir.exists():
        return [], {
            "csv_path": str(csv_path),
            "images_dir": str(images_dir),
            "exists": False,
            "error": "Image folder was not found.",
        }

    table = pd.read_csv(csv_path)
    image_column = detect_column(table, IMAGE_COLUMN_CANDIDATES)
    label_column = detect_column(table, LABEL_COLUMN_CANDIDATES)
    if image_column is None:
        raise ValueError(f"{csv_path} does not contain an image filename column.")
    if require_label and label_column is None:
        raise ValueError(f"{csv_path} does not contain a supervised label column.")

    report: dict[str, Any] = {
        "csv_path": str(csv_path),
        "images_dir": str(images_dir),
        "exists": True,
        "shape": [int(table.shape[0]), int(table.shape[1])],
        "columns": [str(column) for column in table.columns],
        "image_column": image_column,
        "label_column": label_column,
        "is_labeled": label_column is not None,
        "label_counts_csv": {},
        "label_values_valid_0_to_4": True,
        "missing_image_count": 0,
        "missing_image_preview": [],
        "valid_path_count": 0,
        "limit_per_class": limit_per_class,
    }

    label_counts_csv: Counter[int] = Counter()
    missing_images: list[str] = []
    samples: list[dict[str, Any]] = []
    kept_per_class: dict[int, int] = defaultdict(int)

    for row in table.to_dict("records"):
        image_id = str(row[image_column]).strip()
        label: int | None = None
        if label_column is not None:
            try:
                label = int(row[label_column])
            except (TypeError, ValueError):
                report["label_values_valid_0_to_4"] = False
                continue
            if label not in config.CLASS_LABELS:
                report["label_values_valid_0_to_4"] = False
                continue
            label_counts_csv[label] += 1
            if limit_per_class is not None and kept_per_class[label] >= limit_per_class:
                continue

        image_path = resolve_image_from_id(images_dir, image_id)
        if image_path is None:
            missing_images.append(image_id)
            continue

        if label is not None:
            kept_per_class[label] += 1
            samples.append(
                {
                    "source_dataset": source_name,
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "label": label,
                }
            )

    report["label_counts_csv"] = label_counts_csv
    report["missing_image_count"] = len(missing_images)
    report["missing_image_preview"] = missing_images[:20]
    report["valid_path_count"] = len(samples)
    report["valid_path_label_counts"] = Counter(sample["label"] for sample in samples)
    return samples, report


def extract_rows(
    samples: list[dict[str, Any]],
    workers: int,
    partial_csv: Path,
    initial_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = list(initial_rows)
    failures: list[str] = []
    tasks = [(sample, len(config.FEATURE_NAMES)) for sample in samples]

    if not tasks:
        return rows, failures

    if workers <= 1:
        for index, task in enumerate(tasks, start=1):
            ok, payload = extract_sample_worker(task)
            if ok:
                rows.append(payload)
            else:
                failures.append(payload["error"])
            maybe_save_partial_rows(rows, partial_csv, index, len(tasks))
            print_progress(index, len(tasks))
        print()
        return rows, failures

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(extract_sample_worker, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            ok, payload = future.result()
            if ok:
                rows.append(payload)
            else:
                failures.append(payload["error"])
            maybe_save_partial_rows(rows, partial_csv, index, len(futures))
            print_progress(index, len(futures))
    print()
    return rows, failures


def extract_sample_worker(task: tuple[dict[str, Any], int]) -> tuple[bool, dict[str, Any]]:
    sample, expected_feature_count = task
    cv2.setNumThreads(1)
    image_path = Path(sample["image_path"])

    try:
        image_bytes = image_path.read_bytes()
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        if cv2.imdecode(image_array, cv2.IMREAD_COLOR) is None:
            raise ValueError("Unreadable or corrupt image.")

        feature_values = extract_feature_dict(image_path)
        if len(feature_values) != expected_feature_count:
            raise ValueError(
                f"Expected {expected_feature_count} features, got {len(feature_values)}."
            )
        missing_features = [
            name for name in config.FEATURE_NAMES if name not in feature_values
        ]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features[:10]}")

        row: dict[str, Any] = {
            "image_id": sample["image_id"],
            "image_path": str(image_path),
            "source_dataset": sample["source_dataset"],
            "label": int(sample["label"]),
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        }
        row.update({name: float(feature_values[name]) for name in config.FEATURE_NAMES})
        return True, row
    except (FeatureExtractionError, OSError, ValueError) as exc:
        return False, {
            "error": (
                f"{sample['source_dataset']}\t{sample['label']}\t"
                f"{image_path}\t{exc}"
            )
        }


def detect_column(table: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower_to_original = {str(column).lower(): str(column) for column in table.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def resolve_image_from_id(images_dir: Path, image_id: str) -> Path | None:
    image_name = Path(image_id).name
    image_path = Path(image_name)
    candidates: list[Path] = []
    if image_path.suffix.lower() in config.IMAGE_EXTENSIONS:
        candidates.append(images_dir / image_path.name)
    else:
        candidates.extend(
            images_dir / f"{image_name}{extension}"
            for extension in sorted(config.IMAGE_EXTENSIONS)
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = [
        path
        for path in images_dir.glob(f"{image_name}.*")
        if path.suffix.lower() in config.IMAGE_EXTENSIONS
    ]
    return matches[0] if matches else None


def label_counts(labels: list[int]) -> dict[str, int]:
    counts = Counter(int(label) for label in labels)
    return {str(label): int(counts.get(label, 0)) for label in config.CLASS_LABELS}


def print_progress(done: int, total: int) -> None:
    if done == total or done % 50 == 0:
        print(f"\rExtracted features for {done}/{total} images", end="", flush=True)


def maybe_save_partial_rows(
    rows: list[dict[str, Any]],
    partial_csv: Path,
    done: int,
    total: int,
) -> None:
    if done == total or done % 250 == 0:
        try:
            partial_csv.parent.mkdir(parents=True, exist_ok=True)
            temp_csv = partial_csv.with_name(f"{partial_csv.name}.tmp")
            pd.DataFrame(rows).to_csv(temp_csv, index=False)
            temp_csv.replace(partial_csv)
        except OSError as exc:
            print(
                f"\nWarning: checkpoint write failed at {done}/{total}: {exc}",
                flush=True,
            )


def load_partial_rows(partial_csv: Path) -> list[dict[str, Any]]:
    if not partial_csv.exists():
        return []
    return pd.read_csv(partial_csv).to_dict("records")


def print_report(report: dict[str, Any]) -> None:
    print("Dataset preparation complete.", flush=True)
    for source_name, dataset_report in report["datasets"].items():
        print(
            f"{source_name}: columns={dataset_report.get('columns', [])}, "
            f"image_column={dataset_report.get('image_column')}, "
            f"label_column={dataset_report.get('label_column')}, "
            f"label_counts={dataset_report.get('label_counts_csv', {})}, "
            f"missing={dataset_report.get('missing_image_count', 0)}",
            flush=True,
        )
    combined = report["combined"]
    print(f"Combined labeled count: {combined['total_labeled_rows_after_cleaning']}")
    print(f"Combined class counts: {combined['class_counts']}")
    print(f"Combined class percentages: {combined['class_percentages']}")
    print(f"Feature CSV: {combined['feature_csv']}")
    print(f"Failures: {combined['failure_count']} ({combined['failures_path']})")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Counter):
        return {str(key): int(item) for key, item in value.items()}
    if isinstance(value, defaultdict):
        return {str(key): int(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build combined AppDR 203-feature CSV from Downloads datasets.",
    )
    parser.add_argument(
        "--downloads-dir",
        type=Path,
        default=Path.home() / "Downloads",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=BACKEND_DIR / "features_combined.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=BACKEND_DIR / "results" / "dataset_preparation_report.json",
    )
    parser.add_argument(
        "--failures",
        type=Path,
        default=BACKEND_DIR / "results" / "dataset_preparation_failures.txt",
    )
    parser.add_argument("--workers", type=int, default=config.DEFAULT_WORKERS)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing .partial checkpoint next to the output CSV.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Optional smoke limit applied separately to each labeled source dataset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_combined_dataset(
        downloads_dir=args.downloads_dir,
        output_csv=args.output_csv,
        report_path=args.report,
        failures_path=args.failures,
        workers=args.workers,
        limit_per_class=args.limit_per_class,
        resume=args.resume,
    )
