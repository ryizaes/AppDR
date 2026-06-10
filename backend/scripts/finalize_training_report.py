"""Validate trained models and write a combined training summary."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config

RANDOM_STATE = 42
TEST_SIZE = 0.20
BINARY_THRESHOLD = 0.20
STAGE_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}


def load_table(features_csv: Path) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    table = pd.read_csv(features_csv)
    feature_names = [name for name in config.FEATURE_NAMES if name in table.columns]
    if len(feature_names) != len(config.FEATURE_NAMES):
        raise ValueError("features.csv is missing required 203-feature columns.")
    duplicate_mask = table.duplicated(subset=[*feature_names, "label"], keep="first")
    table = table.loc[~duplicate_mask].reset_index(drop=True)
    x_values = table[feature_names].to_numpy(dtype=np.float64)
    y_values = table["label"].to_numpy(dtype=np.int64)
    return table, feature_names, x_values, y_values


def binary_labels(y_values: np.ndarray) -> np.ndarray:
    return np.asarray([0 if int(value) <= 1 else 1 for value in y_values], dtype=np.int64)


def predict_binary(model: object, x_test: np.ndarray, threshold: float) -> np.ndarray:
    probabilities = model.predict_proba(x_test)
    classes = [int(label) for label in getattr(model, "classes_", [0, 1])]
    positive_index = classes.index(1) if 1 in classes else -1
    positive_probability = probabilities[:, positive_index]
    return (positive_probability >= threshold).astype(np.int64)


def summarize_multiclass(model_path: Path, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    with model_path.open("rb") as file:
        model = pickle.load(file)
    predictions = model.predict(x_test)
    report = classification_report(
        y_test,
        predictions,
        labels=config.CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=config.CLASS_LABELS)
    per_stage = {
        str(stage): {
            "name": STAGE_NAMES[stage],
            "support": int(report[str(stage)]["support"]),
            "precision": float(report[str(stage)]["precision"]),
            "recall": float(report[str(stage)]["recall"]),
            "f1": float(report[str(stage)]["f1-score"]),
        }
        for stage in config.CLASS_LABELS
        if str(stage) in report
    }
    return {
        "accuracy": float(report["accuracy"]),
        "f1_macro": float(f1_score(y_test, predictions, average="macro", labels=config.CLASS_LABELS, zero_division=0)),
        "recall_macro": float(recall_score(y_test, predictions, average="macro", labels=config.CLASS_LABELS, zero_division=0)),
        "per_stage": per_stage,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def summarize_binary(model_path: Path, x_test: np.ndarray, y_test: np.ndarray, threshold: float) -> dict:
    with model_path.open("rb") as file:
        model = pickle.load(file)
    predictions = predict_binary(model, x_test, threshold)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    return {
        "threshold": threshold,
        "accuracy": float(report["accuracy"]),
        "f1": float(report["1"]["f1-score"]),
        "referable_recall": float(report["1"]["recall"]),
        "referable_precision": float(report["1"]["precision"]),
        "false_negatives": int(matrix[1, 0]),
        "false_positives": int(matrix[0, 1]),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def main() -> None:
    features_csv = config.FEATURES_CSV
    _, feature_names, x_values, y_values = load_table(features_csv)
    x_train, x_test, y_train, y_test = train_test_split(
        x_values,
        y_values,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_values,
    )
    y_test_binary = binary_labels(y_test)

    multiclass_path = config.RESULTS_DIR / "best_model.pkl"
    binary_path = config.RESULTS_DIR / "binary" / "best_model.pkl"
    threshold_path = config.RESULTS_DIR / "binary" / "optimal_threshold.json"
    threshold = BINARY_THRESHOLD
    if threshold_path.exists():
        payload = json.loads(threshold_path.read_text(encoding="utf-8"))
        if payload.get("threshold") is not None:
            threshold = float(payload["threshold"])

    summary = {
        "feature_count": len(feature_names),
        "sample_count": int(len(y_values)),
        "stage_distribution": {
            str(stage): int(np.sum(y_values == stage))
            for stage in config.CLASS_LABELS
        },
        "multiclass_model": str(multiclass_path.relative_to(BACKEND_DIR)),
        "binary_model": str(binary_path.relative_to(BACKEND_DIR)),
        "multiclass_holdout": summarize_multiclass(multiclass_path, x_test, y_test),
        "binary_holdout": summarize_binary(binary_path, x_test, y_test_binary, threshold),
        "status": "complete",
    }

    output_path = config.RESULTS_DIR / "training_complete_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
