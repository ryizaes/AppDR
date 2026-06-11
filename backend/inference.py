import argparse
import csv
import json
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

import config
from dataset_builder import resolve_image_from_id
from feature_extraction import extract_feature_dict
from utils import ensure_dir, progress_bar, save_text


def predict_stage(
    image_path: str | Path,
    model_path: str | Path = config.BEST_MODEL_PATH,
) -> dict[str, Any]:
    """Predict the encoded DR grade from one fundus image using the saved ML pipeline."""
    model = load_model(model_path)
    feature_names = load_model_feature_names()
    feature_dict = extract_feature_dict(image_path)
    return predict_from_feature_dict(feature_dict, model, feature_names)


def load_model(model_path: str | Path) -> Any:
    # best_model.pkl stores the complete scikit-learn Pipeline:
    # StandardScaler plus the selected classifier. Keeping them together ensures
    # inference uses exactly the same scaling learned during training.
    with Path(model_path).open("rb") as file:
        return pickle.load(file)


def load_model_feature_names(
    metadata_path: str | Path = config.METADATA_PATH,
) -> list[str]:
    try:
        with Path(metadata_path).open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except Exception:
        return list(config.FEATURE_NAMES)

    names = metadata.get("feature_names") if isinstance(metadata, dict) else None
    if isinstance(names, list) and all(isinstance(name, str) for name in names):
        return list(names)

    return list(config.FEATURE_NAMES)


def predict_from_feature_dict(
    feature_dict: dict[str, float],
    model: Any,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    selected_features = feature_names or config.FEATURE_NAMES
    feature_vector = np.array(
        [[float(feature_dict.get(name, 0.0)) for name in selected_features]],
        dtype=np.float64,
    )
    prediction = int(model.predict(feature_vector)[0])
    probabilities = prediction_probabilities(model, feature_vector)

    if probabilities:
        confidence = float(max(probabilities.values()))
    else:
        confidence = 1.0

    return {
        "predicted_class": prediction,
        "medical_label": config.CLASS_NAMES.get(prediction, "Unknown"),
        "confidence": confidence,
        "probabilities": probabilities,
        "feature_vector": {
            name: float(feature_dict.get(name, 0.0))
            for name in selected_features
        },
    }


def predict_dataset(
    csv_path: str | Path,
    images_dir: str | Path,
    model_path: str | Path = config.BEST_MODEL_PATH,
    output_csv: str | Path = config.RESULTS_DIR / "test_predictions.csv",
    failed_samples_path: str | Path = config.RESULTS_DIR / "test_failed_samples.txt",
    workers: int = config.DEFAULT_WORKERS,
) -> pd.DataFrame:
    """Run inference for an unlabeled APTOS-style test.csv plus test_images/."""
    table = pd.read_csv(csv_path)
    if config.APTOS_IMAGE_ID_COLUMN not in table.columns:
        raise ValueError(f"CSV must contain column: {config.APTOS_IMAGE_ID_COLUMN}")

    model = load_model(model_path)
    feature_names = load_model_feature_names()
    rows: list[dict[str, Any]] = []
    failed: list[str] = []
    image_root = Path(images_dir)
    records = table.to_dict("records")

    if workers <= 1:
        feature_results = (
            extract_test_sample_worker((index, record, image_root))
            for index, record in enumerate(records)
        )
        iterator = progress_bar(feature_results, total=len(records), prefix="Predicting test images")
        for ok, payload in iterator:
            append_prediction_or_failure(ok, payload, model, feature_names, rows, failed)
    else:
        tasks = [(index, record, image_root) for index, record in enumerate(records)]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(extract_test_sample_worker, task) for task in tasks]
            for future in progress_bar(as_completed(futures), total=len(futures), prefix="Predicting test images"):
                ok, payload = future.result()
                append_prediction_or_failure(ok, payload, model, feature_names, rows, failed)

    output_path = Path(output_csv)
    ensure_dir(output_path.parent)
    dataframe = pd.DataFrame(sorted(rows, key=lambda row: int(row["_row_order"])))
    dataframe = dataframe.drop(columns=["_row_order"], errors="ignore")
    dataframe.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    save_text(failed_samples_path, "\n".join(failed))
    print(f"Saved predictions: {output_path}")
    print(f"Saved failed test samples: {failed_samples_path} ({len(failed)} failures)")
    return dataframe


def extract_test_sample_worker(task: tuple[int, dict[str, Any], Path]) -> tuple[bool, dict[str, Any]]:
    row_order, record, image_root = task
    cv2.setNumThreads(1)
    image_id = str(record[config.APTOS_IMAGE_ID_COLUMN])
    image_path = resolve_image_from_id(image_root, image_id)

    if image_path is None:
        return False, {"error": f"{image_id}\tmissing image", "row_order": row_order}

    try:
        return True, {
            "row_order": row_order,
            "image_id": image_id,
            "features": extract_feature_dict(image_path),
        }
    except Exception as exc:
        return False, {"error": f"{image_id}\t{image_path}\t{exc}"}


def append_prediction_or_failure(
    ok: bool,
    payload: dict[str, Any],
    model: Any,
    feature_names: list[str],
    rows: list[dict[str, Any]],
    failed: list[str],
) -> None:
    if not ok:
        failed.append(payload["error"])
        return

    result = predict_from_feature_dict(payload["features"], model, feature_names)
    row: dict[str, Any] = {
        "_row_order": payload["row_order"],
        config.APTOS_IMAGE_ID_COLUMN: payload["image_id"],
        "predicted_class": result["predicted_class"],
        "medical_label": result["medical_label"],
        "confidence": result["confidence"],
    }
    for label in config.CLASS_LABELS:
        row[f"prob_stage_{label}"] = result["probabilities"].get(label, 0.0)
    for feature_name in feature_names:
        row[feature_name] = result["feature_vector"][feature_name]
    rows.append(row)


def prediction_probabilities(model: Any, feature_vector: np.ndarray) -> dict[int, float]:
    if not hasattr(model, "predict_proba"):
        return {}

    probabilities = model.predict_proba(feature_vector)[0]
    classes = model_classes(model)
    return {
        int(class_label): float(probability)
        for class_label, probability in zip(classes, probabilities)
    }


def model_classes(model: Any) -> list[int]:
    classes = getattr(model, "classes_", None)

    if classes is None and hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier")
        classes = getattr(classifier, "classes_", None)

    if classes is None:
        return list(config.CLASS_LABELS)

    return [int(label) for label in classes]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict DR grade for one image or an APTOS-style test CSV.")
    parser.add_argument("image_path", type=Path, nargs="?")
    parser.add_argument("--model-path", type=Path, default=config.BEST_MODEL_PATH)
    parser.add_argument("--csv", type=Path, default=None, help="Unlabeled CSV, e.g. Downloads/test.csv")
    parser.add_argument("--images-dir", type=Path, default=None, help="Image folder, e.g. Downloads/test_images")
    parser.add_argument("--output-csv", type=Path, default=config.RESULTS_DIR / "test_predictions.csv")
    parser.add_argument("--failed-samples", type=Path, default=config.RESULTS_DIR / "test_failed_samples.txt")
    parser.add_argument("--workers", type=int, default=config.DEFAULT_WORKERS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.csv is not None:
        if args.images_dir is None:
            raise SystemExit("--images-dir is required when --csv is used.")
        predict_dataset(
            csv_path=args.csv,
            images_dir=args.images_dir,
            model_path=args.model_path,
            output_csv=args.output_csv,
            failed_samples_path=args.failed_samples,
            workers=args.workers,
        )
    elif args.image_path is not None:
        result = predict_stage(args.image_path, model_path=args.model_path)
        print(json.dumps(result, indent=2))
    else:
        raise SystemExit("Provide either image_path or --csv with --images-dir.")
