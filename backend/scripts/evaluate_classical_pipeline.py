import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.pipeline import analyze_image


def evaluate(csv_path: Path, workers: int) -> None:
    rows = read_label_rows(csv_path)
    labels: list[int] = []
    binary_predictions: list[int] = []
    stage_predictions: list[int] = []
    probabilities: list[float] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(analyze_row, csv_path.parent, row)
            for row in rows
        ]

        for future in as_completed(futures):
            label, binary_prediction, stage_prediction, probability = future.result()
            labels.append(label)
            binary_predictions.append(binary_prediction)
            stage_predictions.append(stage_prediction)
            probabilities.append(probability)
            completed += 1

            if completed % 100 == 0 or completed == len(rows):
                print(f"Evaluated {completed}/{len(rows)}")

    label_array = np.array(labels, dtype=np.int64)
    binary_label_array = (label_array > 0).astype(np.int64)
    binary_prediction_array = np.array(binary_predictions, dtype=np.int64)
    stage_prediction_array = np.array(stage_predictions, dtype=np.int64)
    probability_array = np.array(probabilities, dtype=np.float32)

    print("Binary accuracy:", round(float(np.mean(binary_prediction_array == binary_label_array)), 4))
    print("Stage accuracy:", round(float(np.mean(stage_prediction_array == label_array)), 4))
    print("ROC AUC:", round(float(binary_auc(binary_label_array, probability_array)), 4))
    print("Binary confusion matrix:")
    print(binary_confusion_matrix(binary_label_array, binary_prediction_array))
    print("Stage confusion matrix:")
    print(stage_confusion_matrix(label_array, stage_prediction_array))
    print(binary_classification_report(binary_label_array, binary_prediction_array))


def read_label_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("CSV has no header.")

        rows = list(reader)

    if "image_path" in reader.fieldnames and "label" in reader.fieldnames:
        return rows
    if "id_code" in reader.fieldnames and "diagnosis" in reader.fieldnames:
        return [
            {
                "image_path": f"train_images/{row['id_code']}.png",
                "label": row["diagnosis"],
            }
            for row in rows
        ]

    raise ValueError("CSV must contain image_path,label or id_code,diagnosis columns.")


def analyze_row(base_dir: Path, row: dict[str, str]) -> tuple[int, int, int, float]:
    image_path = resolve_image_path(base_dir, row["image_path"])
    label = int(row["label"])
    output = analyze_image(image_path.read_bytes(), include_processed_images=False)
    binary_prediction = 1 if output.result.referable else 0
    stage_prediction = output.result.stage if output.result.stage is not None else 0
    return label, binary_prediction, stage_prediction, output.result.dr_probability


def resolve_image_path(base_dir: Path, value: str) -> Path:
    image_path = Path(value)

    if not image_path.is_absolute():
        image_path = base_dir / image_path

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return image_path


def binary_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positive = probabilities[labels == 1]
    negative = probabilities[labels == 0]

    if positive.size == 0 or negative.size == 0:
        return 0.0

    comparisons = positive[:, None] - negative[None, :]
    wins = np.count_nonzero(comparisons > 0)
    ties = np.count_nonzero(comparisons == 0)

    return float((wins + (0.5 * ties)) / comparisons.size)


def binary_confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [
                int(np.count_nonzero((labels == 0) & (predictions == 0))),
                int(np.count_nonzero((labels == 0) & (predictions == 1))),
            ],
            [
                int(np.count_nonzero((labels == 1) & (predictions == 0))),
                int(np.count_nonzero((labels == 1) & (predictions == 1))),
            ],
        ],
        dtype=np.int64,
    )


def stage_confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    matrix = np.zeros((5, 5), dtype=np.int64)

    for label, prediction in zip(labels, predictions):
        if 0 <= label <= 4 and 0 <= prediction <= 4:
            matrix[label, prediction] += 1

    return matrix


def binary_classification_report(labels: np.ndarray, predictions: np.ndarray) -> str:
    lines = ["class,precision,recall,f1,support"]

    for class_id, name in ((0, "healthy"), (1, "dr")):
        true_positive = np.count_nonzero((labels == class_id) & (predictions == class_id))
        false_positive = np.count_nonzero((labels != class_id) & (predictions == class_id))
        false_negative = np.count_nonzero((labels == class_id) & (predictions != class_id))
        support = np.count_nonzero(labels == class_id)
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-6)
        lines.append(f"{name},{precision:.4f},{recall:.4f},{f1:.4f},{support}")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the deterministic classical AppDR pipeline.",
    )
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.csv, args.workers)
