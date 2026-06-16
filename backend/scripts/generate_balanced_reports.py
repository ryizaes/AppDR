"""Write balanced weak-stage experiment reports without replacing production artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BACKEND_DIR / "results"
MULTICLASS_DIR = RESULTS_DIR / "experimental_balanced_medium"
BINARY_DIR = MULTICLASS_DIR / "binary"

CLASS_NAMES = {
    0: "No apparent diabetic retinopathy",
    1: "Mild non-proliferative diabetic retinopathy",
    2: "Moderate non-proliferative diabetic retinopathy",
    3: "Severe non-proliferative diabetic retinopathy",
    4: "Proliferative diabetic retinopathy",
}
BINARY_NAMES = {
    0: "Non-referable DR",
    1: "Referable DR",
}
OLD = {
    "class_1_recall": 32.99,
    "class_3_recall": 30.95,
    "class_4_recall": 61.51,
    "macro_f1": 50.77,
    "balanced_accuracy": 53.12,
    "binary_referable_recall": 93.73,
    "binary_false_negatives": 88,
}
DISCLAIMER = (
    "This app is a screening support tool only and does not provide a final "
    "medical diagnosis. Please consult an ophthalmologist for confirmation."
)


def main() -> None:
    multiclass = load_metrics(MULTICLASS_DIR)
    binary = load_metrics(BINARY_DIR)
    report = build_report(multiclass, binary)
    write_outputs(report)
    print(report["markdown"])


def load_metrics(path: Path) -> dict[str, Any]:
    metrics_path = path / "metrics.json"
    metadata_path = path / "best_model_metadata.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["metadata"] = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    return payload


def build_report(multiclass: dict[str, Any], binary: dict[str, Any]) -> dict[str, Any]:
    multiclass_section = section_from_metrics(multiclass, CLASS_NAMES)
    binary_section = section_from_metrics(binary, BINARY_NAMES)
    binary_matrix = binary_section["confusion_matrix"]
    false_positives = int(binary_matrix[0][1])
    false_negatives = int(binary_matrix[1][0])
    referable_caught = int(binary_matrix[1][1])
    non_referable_correct = int(binary_matrix[0][0])
    comparison = {
        "old_class_1_recall_percent": OLD["class_1_recall"],
        "new_class_1_recall_percent": pct_value(multiclass_section["per_class"]["1"]["recall"]),
        "old_class_3_recall_percent": OLD["class_3_recall"],
        "new_class_3_recall_percent": pct_value(multiclass_section["per_class"]["3"]["recall"]),
        "old_class_4_recall_percent": OLD["class_4_recall"],
        "new_class_4_recall_percent": pct_value(multiclass_section["per_class"]["4"]["recall"]),
        "old_macro_f1_percent": OLD["macro_f1"],
        "new_macro_f1_percent": pct_value(multiclass_section["summary"]["macro_f1_score"]),
        "old_balanced_accuracy_percent": OLD["balanced_accuracy"],
        "new_balanced_accuracy_percent": pct_value(multiclass_section["summary"]["balanced_accuracy"]),
        "old_binary_referable_recall_percent": OLD["binary_referable_recall"],
        "new_binary_referable_recall_percent": pct_value(binary_section["per_class"]["1"]["recall"]),
        "old_binary_false_negatives": OLD["binary_false_negatives"],
        "new_binary_false_negatives": false_negatives,
    }
    payload = {
        "disclaimer": DISCLAIMER,
        "training_mode": "medium",
        "production_replacement": {
            "multiclass_replaced": False,
            "binary_replaced": False,
            "reason": (
                "Class 3 recall improved, but multiclass macro F1 and overall accuracy "
                "fell below the current production baseline. The balanced models are kept "
                "as experimental artifacts."
            ),
        },
        "multiclass": multiclass_section,
        "binary_referable": binary_section,
        "screening_safety_summary": {
            "referable_cases_correctly_caught": referable_caught,
            "referable_cases_missed": false_negatives,
            "non_referable_cases_correctly_cleared": non_referable_correct,
            "non_referable_cases_over_referred": false_positives,
            "interpretation": (
                "The binary model is stronger for safety-oriented referable DR screening "
                "than the 5-class model is for exact grade assignment."
            ),
        },
        "old_vs_new_comparison": comparison,
    }
    payload["markdown"] = render_markdown(payload)
    payload["csv_rows"] = build_csv_rows(payload)
    return payload


def section_from_metrics(metrics: dict[str, Any], names: dict[int, str]) -> dict[str, Any]:
    holdout = metrics["holdout"]
    matrix = holdout["confusion_matrix"]
    report = holdout["classification_report"]
    per_class: dict[str, dict[str, Any]] = {}
    for label, label_name in names.items():
        stats = report[str(label)]
        support = int(stats["support"])
        correct = int(matrix[label][label])
        common = common_misclassification(matrix, label, names)
        per_class[str(label)] = {
            "class": label,
            "label": label_name,
            "test_images": support,
            "correct_predictions": correct,
            "precision": float(stats["precision"]),
            "recall": float(stats["recall"]),
            "f1_score": float(stats["f1-score"]),
            "common_misclassification": common,
        }
    return {
        "model_name": metrics["best_model_name"],
        "model_dir": str(MULTICLASS_DIR if len(names) == 5 else BINARY_DIR),
        "summary": {
            "accuracy": float(holdout["accuracy"]),
            "balanced_accuracy": float(holdout["balanced_accuracy"]),
            "macro_precision": float(holdout.get("precision", 0.0)),
            "macro_recall": float(holdout.get("recall", 0.0)),
            "macro_f1_score": float(holdout.get("f1", 0.0)),
            "weighted_f1_score": float(
                report.get("weighted avg", {}).get("f1-score", holdout.get("f1", 0.0))
            ),
        },
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def common_misclassification(
    matrix: list[list[int]],
    class_index: int,
    names: dict[int, str],
) -> dict[str, Any] | None:
    row = list(matrix[class_index])
    row[class_index] = 0
    if sum(row) == 0:
        return None
    predicted = max(range(len(row)), key=lambda index: row[index])
    if row[predicted] == 0:
        return None
    return {
        "predicted_class": predicted,
        "predicted_label": names[predicted],
        "count": int(row[predicted]),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Balanced Weak-Stage Experiment Report",
        "",
        DISCLAIMER,
        "",
        "## MULTICLASS DR GRADING RESULTS",
        "",
        f"Final model selected: {report['multiclass']['model_name']}",
        f"Overall accuracy: {pct(report['multiclass']['summary']['accuracy'])}",
        f"Balanced accuracy: {pct(report['multiclass']['summary']['balanced_accuracy'])}",
        f"Macro precision: {pct(report['multiclass']['summary']['macro_precision'])}",
        f"Macro recall: {pct(report['multiclass']['summary']['macro_recall'])}",
        f"Macro F1-score: {pct(report['multiclass']['summary']['macro_f1_score'])}",
        f"Weighted F1-score: {pct(report['multiclass']['summary']['weighted_f1_score'])}",
        "",
        "### Per-stage results",
        "",
    ]
    for label in range(5):
        row = report["multiclass"]["per_class"][str(label)]
        lines.extend(render_class(row))
    lines.extend(render_matrix(report["multiclass"]["confusion_matrix"], CLASS_NAMES))
    binary = report["binary_referable"]
    safety = report["screening_safety_summary"]
    lines.extend(
        [
            "",
            "## BINARY REFERABLE DR SCREENING RESULTS",
            "",
            f"Final model selected: {binary['model_name']}",
            f"Overall accuracy: {pct(binary['summary']['accuracy'])}",
            f"Balanced accuracy: {pct(binary['summary']['balanced_accuracy'])}",
            f"Precision: {pct(binary['per_class']['1']['precision'])}",
            f"Recall: {pct(binary['per_class']['1']['recall'])}",
            f"F1-score: {pct(binary['per_class']['1']['f1_score'])}",
            f"Referable recall: {pct(binary['per_class']['1']['recall'])}",
            f"Non-referable recall: {pct(binary['per_class']['0']['recall'])}",
            f"False positives: {safety['non_referable_cases_over_referred']}",
            f"False negatives: {safety['referable_cases_missed']}",
            "",
        ]
    )
    lines.extend(render_matrix(binary["confusion_matrix"], BINARY_NAMES))
    lines.extend(
        [
            "",
            "## SCREENING SAFETY SUMMARY",
            "",
            f"Referable cases correctly caught: {safety['referable_cases_correctly_caught']}",
            f"Referable cases missed: {safety['referable_cases_missed']}",
            f"Non-referable cases over-referred: {safety['non_referable_cases_over_referred']}",
            safety["interpretation"],
            "",
            "## OLD VS NEW COMPARISON",
            "",
        ]
    )
    comparison = report["old_vs_new_comparison"]
    for key, value in comparison.items():
        label = key.replace("_", " ").capitalize()
        suffix = "%" if key.endswith("_percent") else ""
        lines.append(f"{label}: {value}{suffix}")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "Keep the current production multiclass model. The balanced experiment improved Class 3 recall but reduced macro F1 and exact-grade reliability. Keep these balanced models as experimental candidates unless a later longer run improves macro F1 without damaging screening behavior.",
        ]
    )
    return "\n".join(lines)


def render_class(row: dict[str, Any]) -> list[str]:
    lines = [
        f"#### Class {row['class']} - {row['label']}",
        "",
        f"Test images: {row['test_images']}",
        f"Correct predictions: {row['correct_predictions']}",
        f"Precision: {pct(row['precision'])}",
        f"Recall: {pct(row['recall'])}",
        f"F1-score: {pct(row['f1_score'])}",
    ]
    common = row["common_misclassification"]
    if common:
        lines.append(
            f"Most common misclassification: class {common['predicted_class']} - "
            f"{common['predicted_label']} ({common['count']} images)."
        )
    else:
        lines.append("Most common misclassification: none in this split.")
    lines.append("")
    return lines


def render_matrix(matrix: list[list[int]], names: dict[int, str]) -> list[str]:
    labels = list(names)
    lines = [
        "### Confusion matrix",
        "",
        "Rows are true labels. Columns are predicted labels.",
        "",
        "| True \\ Predicted | " + " | ".join(str(label) for label in labels) + " |",
        "|---|" + "|".join("---" for _ in labels) + "|",
    ]
    for label, row in zip(labels, matrix):
        values = " | ".join(str(int(value)) for value in row)
        lines.append(f"| {label} - {names[label]} | {values} |")
    return lines


def build_csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section_name in ("multiclass", "binary_referable"):
        for row in report[section_name]["per_class"].values():
            rows.append(
                {
                    "section": section_name,
                    "class": row["class"],
                    "label": row["label"],
                    "test_images": row["test_images"],
                    "correct_predictions": row["correct_predictions"],
                    "precision_percent": pct_value(row["precision"]),
                    "recall_percent": pct_value(row["recall"]),
                    "f1_percent": pct_value(row["f1_score"]),
                    "common_misclassification": json.dumps(
                        row["common_misclassification"],
                        ensure_ascii=True,
                    ),
                }
            )
    return rows


def write_outputs(report: dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "evaluation_report_balanced.md").write_text(
        report["markdown"],
        encoding="utf-8",
    )
    json_payload = {k: v for k, v in report.items() if k not in {"markdown", "csv_rows"}}
    (RESULTS_DIR / "evaluation_report_balanced.json").write_text(
        json.dumps(json_payload, indent=2),
        encoding="utf-8",
    )
    with (RESULTS_DIR / "evaluation_report_balanced.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
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
    with (RESULTS_DIR / "confusion_matrix_balanced.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["section", "true_class", "predicted_class", "count"])
        for section_name, labels in (("multiclass", CLASS_NAMES), ("binary_referable", BINARY_NAMES)):
            matrix = report[section_name]["confusion_matrix"]
            for true_label in labels:
                for predicted_label in labels:
                    writer.writerow(
                        [
                            section_name,
                            true_label,
                            predicted_label,
                            matrix[true_label][predicted_label],
                        ]
                    )
    (RESULTS_DIR / "screening_report_balanced.md").write_text(
        render_screening_report(report),
        encoding="utf-8",
    )


def render_screening_report(report: dict[str, Any]) -> str:
    binary = report["binary_referable"]
    safety = report["screening_safety_summary"]
    return "\n".join(
        [
            "# Screening Safety Report - Balanced Experiment",
            "",
            DISCLAIMER,
            "",
            f"Model: {binary['model_name']}",
            f"Referable recall: {pct(binary['per_class']['1']['recall'])}",
            f"Non-referable recall: {pct(binary['per_class']['0']['recall'])}",
            f"False negatives: {safety['referable_cases_missed']}",
            f"False positives: {safety['non_referable_cases_over_referred']}",
            "",
            "The balanced binary experiment catches more referable cases than the current baseline, but it over-refers more non-referable cases. Keep it experimental until the production tradeoff is accepted.",
        ]
    )


def pct(value: float) -> str:
    return f"{pct_value(value):.2f}%"


def pct_value(value: float) -> float:
    return round(float(value) * 100.0, 2)


if __name__ == "__main__":
    main()
