"""Generate beginner-friendly evaluation reports for saved AppDR models."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
from train import (
    RANDOM_STATE,
    TEST_SIZE,
    load_and_validate_dataset,
    predict_proba_aligned,
    predictions_from_probabilities,
    remap_labels_to_binary_referable,
)


CLASS_NAMES = config.CLASS_NAMES
BINARY_CLASS_NAMES = {
    0: "Non-referable DR",
    1: "Referable DR",
}
DISCLAIMER = (
    "This app is a screening support tool only and does not provide a final "
    "medical diagnosis. Please consult an ophthalmologist for confirmation."
)


def main() -> None:
    args = parse_args()
    report = build_report(args.features_csv, args.results_dir)
    write_reports(report, args.results_dir)
    print(report["markdown"])


def build_report(features_csv: Path, results_dir: Path) -> dict[str, Any]:
    multiclass = evaluate_model(
        features_csv=features_csv,
        model_dir=results_dir,
        binary_referable=False,
    )
    binary = evaluate_model(
        features_csv=features_csv,
        model_dir=results_dir / "binary",
        binary_referable=True,
    )
    markdown = render_markdown(multiclass, binary)
    csv_rows = build_csv_rows(multiclass, binary)
    return {
        "disclaimer": DISCLAIMER,
        "multiclass": multiclass,
        "binary_referable": binary,
        "markdown": markdown,
        "csv_rows": csv_rows,
    }


def evaluate_model(
    features_csv: Path,
    model_dir: Path,
    binary_referable: bool,
) -> dict[str, Any]:
    model_path = model_dir / "best_model.pkl"
    metadata_path = model_dir / "best_model_metadata.json"
    metrics_path = model_dir / "metrics.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    table, feature_names, y_values, data_quality = load_and_validate_dataset(features_csv)
    if binary_referable:
        y_values = remap_labels_to_binary_referable(y_values)
        labels = [0, 1]
        class_names = BINARY_CLASS_NAMES
    else:
        labels = list(config.CLASS_LABELS)
        class_names = CLASS_NAMES

    x_values = table[feature_names].to_numpy(dtype=np.float64)
    x_values[~np.isfinite(x_values)] = np.nan
    indices = table.index.to_numpy(dtype=int)
    (
        _x_train_validation,
        x_test,
        _y_train_validation,
        y_test,
        _train_validation_indices,
        test_indices,
    ) = train_test_split(
        x_values,
        y_values,
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_values,
    )

    with model_path.open("rb") as file:
        model = pickle.load(file)

    probabilities = predict_proba_aligned(model, x_test, labels)
    threshold = load_threshold(model_dir) if binary_referable else None
    predictions = predictions_from_probabilities(
        model=model,
        x_values=x_test,
        probabilities=probabilities,
        problem_type="binary" if binary_referable else "multiclass",
        threshold=threshold,
        class_labels=labels,
    )

    matrix = confusion_matrix(y_test, predictions, labels=labels)
    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=[class_names[label] for label in labels],
        output_dict=True,
        zero_division=0,
    )
    per_class = build_per_class_rows(matrix, report, labels, class_names)
    summary = build_summary(y_test, predictions, labels, binary_referable)
    return {
        "model_dir": str(model_dir),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "metrics_path": str(metrics_path),
        "features_csv": str(features_csv),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "test_count": int(len(y_test)),
        "test_indices_preview": [int(index) for index in test_indices[:20]],
        "threshold": threshold,
        "data_quality": data_quality,
        "per_class": per_class,
        "summary": summary,
        "confusion_matrix": {
            "labels": [int(label) for label in labels],
            "label_names": [class_names[label] for label in labels],
            "rows_true_columns_predicted": matrix.tolist(),
        },
    }


def build_per_class_rows(
    matrix: np.ndarray,
    report: dict[str, Any],
    labels: list[int],
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, label in enumerate(labels):
        label_name = class_names[label]
        stats = report.get(label_name, {})
        support = int(matrix[row_index].sum())
        correct = int(matrix[row_index, row_index])
        common = common_misclassification(matrix, row_index, labels, class_names)
        rows.append(
            {
                "class": int(label),
                "label": label_name,
                "test_images": support,
                "correct_predictions": correct,
                "precision": float(stats.get("precision", 0.0)),
                "recall": float(stats.get("recall", 0.0)),
                "f1_score": float(stats.get("f1-score", 0.0)),
                "common_misclassification": common,
            },
        )
    return rows


def common_misclassification(
    matrix: np.ndarray,
    row_index: int,
    labels: list[int],
    class_names: dict[int, str],
) -> dict[str, Any] | None:
    row = matrix[row_index].copy()
    row[row_index] = 0
    if int(row.sum()) == 0:
        return None
    pred_index = int(np.argmax(row))
    count = int(row[pred_index])
    if count == 0:
        return None
    return {
        "predicted_class": int(labels[pred_index]),
        "predicted_label": class_names[labels[pred_index]],
        "count": count,
    }


def build_summary(
    y_true: np.ndarray,
    predictions: np.ndarray,
    labels: list[int],
    binary_referable: bool,
) -> dict[str, float]:
    if binary_referable:
        average = "binary"
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        referable_recall = recall_score(y_true, predictions, pos_label=1, zero_division=0)
        return {
            "accuracy": float(accuracy_score(y_true, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "referable_recall": float(referable_recall),
            "average": average,
        }

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_precision": float(
            precision_score(y_true, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1_score": float(
            f1_score(y_true, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1_score": float(
            f1_score(y_true, predictions, labels=labels, average="weighted", zero_division=0)
        ),
    }


def load_threshold(model_dir: Path) -> float | None:
    threshold_path = model_dir / "optimal_threshold.json"
    if not threshold_path.exists():
        return None
    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    value = payload.get("threshold") if isinstance(payload, dict) else None
    return float(value) if value is not None else None


def render_markdown(multiclass: dict[str, Any], binary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# AppDR Evaluation Report")
    lines.append("")
    lines.append(DISCLAIMER)
    lines.append("")
    lines.append("## MULTICLASS DR GRADING RESULTS")
    lines.append("")
    lines.append(f"Overall accuracy: {pct(multiclass['summary']['accuracy'])}")
    lines.append("")
    lines.append("### Per-stage results")
    lines.append("")
    for row in multiclass["per_class"]:
        lines.extend(render_stage(row))
    lines.append("### Summary")
    lines.append("")
    lines.append(f"Balanced accuracy: {pct(multiclass['summary']['balanced_accuracy'])}")
    lines.append(f"Macro precision: {pct(multiclass['summary']['macro_precision'])}")
    lines.append(f"Macro recall: {pct(multiclass['summary']['macro_recall'])}")
    lines.append(f"Macro F1-score: {pct(multiclass['summary']['macro_f1_score'])}")
    lines.append(f"Weighted F1-score: {pct(multiclass['summary']['weighted_f1_score'])}")
    lines.append("")
    lines.extend(render_confusion_matrix(multiclass))
    lines.append("")
    lines.append("## BINARY REFERABLE DR SCREENING RESULTS")
    lines.append("")
    lines.append(f"Overall accuracy: {pct(binary['summary']['accuracy'])}")
    lines.append(f"Balanced accuracy: {pct(binary['summary']['balanced_accuracy'])}")
    lines.append(f"Precision: {pct(binary['summary']['precision'])}")
    lines.append(f"Recall: {pct(binary['summary']['recall'])}")
    lines.append(f"F1-score: {pct(binary['summary']['f1_score'])}")
    lines.append(f"Referable recall: {pct(binary['summary']['referable_recall'])}")
    lines.append("")
    for row in binary["per_class"]:
        lines.append(f"### {row['label']}")
        lines.append("")
        lines.append(f"Test images: {row['test_images']}")
        lines.append(f"Correct predictions: {row['correct_predictions']}")
        lines.append(f"Precision: {pct(row['precision'])}")
        lines.append(f"Recall: {pct(row['recall'])}")
        lines.append(f"F1-score: {pct(row['f1_score'])}")
        lines.append("")
    lines.extend(render_confusion_matrix(binary))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    class3 = next(row for row in multiclass["per_class"] if row["class"] == 3)
    lines.append(
        "The binary referable model is stronger for screening referable disease "
        f"(referable recall {pct(binary['summary']['referable_recall'])}) than the "
        "multiclass model is for exact five-class grading."
    )
    lines.append(
        "Class 3 / Severe non-proliferative diabetic retinopathy remains the "
        f"weakest exact-grade class: {class3['correct_predictions']} of "
        f"{class3['test_images']} test images were correctly predicted "
        f"({pct(class3['recall'])} recall)."
    )
    lines.append(
        "These results should be used as screening-support performance numbers, "
        "not as proof that the app can make final medical diagnoses."
    )
    lines.append("")
    return "\n".join(lines)


def render_stage(row: dict[str, Any]) -> list[str]:
    lines = [
        f"#### Class {row['class']} — {row['label']}",
        "",
        f"Test images: {row['test_images']}",
        f"Correct predictions: {row['correct_predictions']}",
        f"Precision: {pct(row['precision'])}",
        f"Recall: {pct(row['recall'])}",
        f"F1-score: {pct(row['f1_score'])}",
    ]
    common = row.get("common_misclassification")
    if common:
        lines.append(
            "Most common misclassification: "
            f"predicted as class {common['predicted_class']} — "
            f"{common['predicted_label']} ({common['count']} images)."
        )
    else:
        lines.append("Most common misclassification: none in this test split.")
    lines.append("")
    return lines


def render_confusion_matrix(section: dict[str, Any]) -> list[str]:
    matrix = section["confusion_matrix"]["rows_true_columns_predicted"]
    labels = section["confusion_matrix"]["labels"]
    names = section["confusion_matrix"]["label_names"]
    lines = ["### Confusion matrix", ""]
    lines.append("Rows are true labels. Columns are predicted labels.")
    lines.append("")
    header = "| True \\ Predicted | " + " | ".join(str(label) for label in labels) + " |"
    separator = "|---|" + "|".join("---" for _ in labels) + "|"
    lines.append(header)
    lines.append(separator)
    for label, name, row in zip(labels, names, matrix):
        lines.append(f"| {label} — {name} | " + " | ".join(str(int(value)) for value in row) + " |")
    return lines


def build_csv_rows(multiclass: dict[str, Any], binary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section_name, section in (("multiclass", multiclass), ("binary_referable", binary)):
        for row in section["per_class"]:
            rows.append(
                {
                    "section": section_name,
                    "class": row["class"],
                    "label": row["label"],
                    "test_images": row["test_images"],
                    "correct_predictions": row["correct_predictions"],
                    "precision_percent": round(row["precision"] * 100.0, 2),
                    "recall_percent": round(row["recall"] * 100.0, 2),
                    "f1_percent": round(row["f1_score"] * 100.0, 2),
                    "common_misclassification": json.dumps(
                        row.get("common_misclassification"),
                        ensure_ascii=False,
                    ),
                },
            )
    return rows


def write_reports(report: dict[str, Any], results_dir: Path) -> None:
    markdown_path = results_dir / "evaluation_report.md"
    json_path = results_dir / "evaluation_report.json"
    csv_path = results_dir / "evaluation_report.csv"
    markdown_path.write_text(report["markdown"], encoding="utf-8")
    json_payload = {
        key: value
        for key, value in report.items()
        if key not in {"markdown", "csv_rows"}
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "section",
            "class",
            "label",
            "test_images",
            "correct_predictions",
            "precision_percent",
            "recall_percent",
            "f1_percent",
            "common_misclassification",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["csv_rows"])


def pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AppDR evaluation report.")
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=BACKEND_DIR / "features_combined.csv",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=BACKEND_DIR / "results",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
