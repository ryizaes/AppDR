import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

import config
from feature_extraction import FeatureExtractionError, extract_feature_dict
from utils import ensure_dir, list_image_files, print_class_distribution, progress_bar, save_text


def build_dataset(
    data_dir: str | Path = config.DATA_DIR,
    output_csv: str | Path = config.FEATURES_CSV,
    failed_samples_path: str | Path = config.FAILED_SAMPLES_TXT,
    labels_csv: str | Path | None = None,
    images_dir: str | Path | None = None,
    debug: bool = False,
    debug_dir: str | Path | None = None,
    workers: int = config.DEFAULT_WORKERS,
) -> pd.DataFrame:
    """Extract handcrafted features from a stage-folder or APTOS-style dataset."""
    samples = collect_samples(
        data_dir=Path(data_dir),
        labels_csv=Path(labels_csv) if labels_csv is not None else None,
        images_dir=Path(images_dir) if images_dir is not None else None,
    )
    if not samples:
        raise RuntimeError(
            "No labeled images found. Expected either data/Stage_0 ... data/Stage_4 "
            "or an APTOS-style train.csv with train_images/.",
        )

    rows: list[dict[str, Any]] = []
    failed: list[str] = []

    if workers <= 1:
        for image_path, label in progress_bar(samples, total=len(samples), prefix="Extracting features"):
            ok, payload = extract_sample_worker((image_path, label, debug, debug_dir))
            if ok:
                rows.append(payload)
            else:
                failed.append(payload["error"])
    else:
        tasks = [(image_path, label, debug, debug_dir) for image_path, label in samples]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(extract_sample_worker, task) for task in tasks]
            for future in progress_bar(as_completed(futures), total=len(futures), prefix="Extracting features"):
                ok, payload = future.result()
                if ok:
                    rows.append(payload)
                else:
                    failed.append(payload["error"])

    if not rows:
        save_text(failed_samples_path, "\n".join(failed))
        raise RuntimeError("Feature extraction failed for every image. See failed_samples.txt.")

    dataframe = pd.DataFrame(rows, columns=config.CSV_COLUMNS)
    output_path = Path(output_csv)
    ensure_dir(output_path.parent)
    dataframe.to_csv(output_path, index=False)
    save_text(failed_samples_path, "\n".join(failed))

    print(f"Saved feature table: {output_path}")
    print(f"Saved failed samples: {failed_samples_path} ({len(failed)} failures)")
    print_class_distribution(dataframe["label"].astype(int).tolist())

    return dataframe


def extract_sample_worker(task: tuple[Path, int, bool, str | Path | None]) -> tuple[bool, dict[str, Any]]:
    image_path, label, debug, debug_dir = task
    cv2.setNumThreads(1)

    try:
        row = extract_feature_dict(
            image_path,
            debug=debug,
            debug_dir=debug_dir,
        )
        row["label"] = int(label)
        return True, row
    except (FeatureExtractionError, OSError, ValueError) as exc:
        return False, {"error": f"{image_path}\t{exc}"}


def collect_samples(
    data_dir: Path,
    labels_csv: Path | None = None,
    images_dir: Path | None = None,
) -> list[tuple[Path, int]]:
    if labels_csv is not None:
        resolved_images_dir = images_dir if images_dir is not None else labels_csv.parent / config.APTOS_TRAIN_IMAGES_DIR
        return collect_csv_samples(labels_csv, resolved_images_dir)

    stage_samples = collect_stage_samples(data_dir)
    if stage_samples:
        return stage_samples

    candidate_csv = data_dir / config.APTOS_TRAIN_CSV
    candidate_images = data_dir / config.APTOS_TRAIN_IMAGES_DIR
    if candidate_csv.exists():
        return collect_csv_samples(candidate_csv, candidate_images)

    return []


def collect_stage_samples(data_dir: Path) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []

    for folder_name, label in config.STAGE_FOLDERS.items():
        stage_dir = data_dir / folder_name
        image_paths = list_image_files(stage_dir)
        if not image_paths:
            print(f"Warning: no images found in {stage_dir}")

        samples.extend((image_path, label) for image_path in image_paths)

    return samples


def collect_csv_samples(labels_csv: Path, images_dir: Path) -> list[tuple[Path, int]]:
    if not labels_csv.exists():
        raise FileNotFoundError(f"Label CSV not found: {labels_csv}")

    if not images_dir.exists():
        zip_hint = images_dir.with_suffix(".zip")
        if zip_hint.exists():
            raise FileNotFoundError(
                f"Image folder not found: {images_dir}. Found {zip_hint.name}; "
                "extract the ZIP first so images are available as normal files.",
            )
        raise FileNotFoundError(f"Image folder not found: {images_dir}")

    labels = pd.read_csv(labels_csv)
    required_columns = {config.APTOS_IMAGE_ID_COLUMN, config.APTOS_LABEL_COLUMN}
    missing = sorted(required_columns - set(labels.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    samples: list[tuple[Path, int]] = []
    missing_images: list[str] = []

    for row in labels.to_dict("records"):
        image_id = str(row[config.APTOS_IMAGE_ID_COLUMN])
        label = int(row[config.APTOS_LABEL_COLUMN])
        image_path = resolve_image_from_id(images_dir, image_id)

        if image_path is None:
            missing_images.append(image_id)
            continue

        samples.append((image_path, label))

    if missing_images:
        preview = ", ".join(missing_images[:10])
        print(
            f"Warning: {len(missing_images)} CSV rows had no matching image in {images_dir}. "
            f"First missing IDs: {preview}",
        )

    return samples


def resolve_image_from_id(images_dir: Path, image_id: str) -> Path | None:
    image_path = Path(image_id)
    candidates: list[Path] = []

    if image_path.suffix.lower() in config.IMAGE_EXTENSIONS:
        candidates.append(images_dir / image_path.name)
    else:
        candidates.extend(images_dir / f"{image_id}{extension}" for extension in config.IMAGE_EXTENSIONS)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = [
        path
        for path in images_dir.glob(f"{image_id}.*")
        if path.suffix.lower() in config.IMAGE_EXTENSIONS
    ]
    return matches[0] if matches else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build features.csv from either data/Stage_0 ... data/Stage_4 "
            "or an APTOS-style train.csv plus train_images/ folder."
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--csv", type=Path, default=None, help="Label CSV, e.g. Downloads/train.csv")
    parser.add_argument("--images-dir", type=Path, default=None, help="Image folder, e.g. Downloads/train_images")
    parser.add_argument("--output-csv", type=Path, default=config.FEATURES_CSV)
    parser.add_argument("--failed-samples", type=Path, default=config.FAILED_SAMPLES_TXT)
    parser.add_argument("--workers", type=int, default=config.DEFAULT_WORKERS)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-dir", type=Path, default=config.RESULTS_DIR / "debug_features")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        data_dir=args.data_dir,
        output_csv=args.output_csv,
        failed_samples_path=args.failed_samples,
        labels_csv=args.csv,
        images_dir=args.images_dir,
        debug=args.debug,
        debug_dir=args.debug_dir if args.debug else None,
        workers=args.workers,
    )
