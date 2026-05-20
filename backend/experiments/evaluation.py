import math
from collections import Counter
from typing import Any

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)


CLASS_LABELS = [0, 1, 2, 3, 4]


def evaluate_predictions(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        zero_division=0,
    )
    return {
        "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0)),
        "minority_recall": minority_recall(y_true, recall),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.tolist(),
    }


def classification_report_rows(y_true: list[int], y_pred: list[int]) -> list[dict[str, Any]]:
    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )
    rows: list[dict[str, Any]] = []

    for label in CLASS_LABELS:
        values = report[str(label)]
        rows.append(
            {
                "class": label,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1-score"],
                "support": values["support"],
            },
        )

    return rows


def confusion(y_true: list[int], y_pred: list[int]) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)


def minority_recall(y_true: list[int], recall_by_class: np.ndarray) -> float:
    counts = Counter(y_true)
    nonzero_counts = [counts[label] for label in CLASS_LABELS if counts[label] > 0]

    if not nonzero_counts:
        return 0.0

    median_support = float(np.median(nonzero_counts))
    minority_labels = [
        label
        for label in CLASS_LABELS
        if counts[label] > 0 and counts[label] <= median_support
    ]

    if not minority_labels:
        return 0.0

    return float(np.mean([recall_by_class[label] for label in minority_labels]))


def mean_confidence_interval(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "std": 0.0}

    array = np.array(values, dtype=np.float64)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    margin = 1.96 * std / math.sqrt(len(array)) if len(array) > 1 else 0.0

    return {
        "mean": mean,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "std": std,
    }


def most_confused_pairs(matrix: np.ndarray, limit: int = 10) -> list[dict[str, int]]:
    pairs: list[dict[str, int]] = []

    for true_label in CLASS_LABELS:
        for predicted_label in CLASS_LABELS:
            if true_label == predicted_label:
                continue

            count = int(matrix[true_label, predicted_label])

            if count > 0:
                pairs.append(
                    {
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": count,
                    },
                )

    return sorted(pairs, key=lambda row: row["count"], reverse=True)[:limit]


def fisher_feature_ranking(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
) -> list[dict[str, float | str]]:
    rankings: list[dict[str, float | str]] = []
    overall_mean = np.mean(feature_matrix, axis=0)

    for feature_index, name in enumerate(feature_names):
        between = 0.0
        within = 0.0

        for label in CLASS_LABELS:
            class_values = feature_matrix[labels == label, feature_index]

            if class_values.size == 0:
                continue

            class_mean = float(np.mean(class_values))
            between += float(class_values.size * ((class_mean - overall_mean[feature_index]) ** 2))
            within += float(np.sum((class_values - class_mean) ** 2))

        rankings.append(
            {
                "feature": name,
                "fisher_score": float(between / max(within, 1e-9)),
            },
        )

    return sorted(rankings, key=lambda row: float(row["fisher_score"]), reverse=True)
