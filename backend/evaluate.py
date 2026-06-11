import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

import config
from utils import ensure_dir, read_feature_table, save_json, save_text


def evaluate_model(
    model: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    results_dir: str | Path = config.RESULTS_DIR,
    model_name: str = "best_model",
) -> dict[str, Any]:
    output_dir = ensure_dir(results_dir)
    y_pred = model.predict(x_test)
    probabilities = prediction_probabilities(model, x_test)
    matrix = confusion_matrix(y_test, y_pred, labels=config.CLASS_LABELS)
    report_text = classification_report(
        y_test,
        y_pred,
        labels=config.CLASS_LABELS,
        target_names=[config.CLASS_NAMES[label] for label in config.CLASS_LABELS],
        zero_division=0,
    )
    sensitivity_specificity = sensitivity_specificity_by_class(matrix)

    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(
            precision_score(y_test, y_pred, labels=config.CLASS_LABELS, average="macro", zero_division=0),
        ),
        "recall_macro": float(
            recall_score(y_test, y_pred, labels=config.CLASS_LABELS, average="macro", zero_division=0),
        ),
        "f1_macro": float(
            f1_score(y_test, y_pred, labels=config.CLASS_LABELS, average="macro", zero_division=0),
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_test, y_pred, labels=config.CLASS_LABELS)),
        "medical_metrics": sensitivity_specificity,
    }

    if probabilities is not None:
        roc_metrics = plot_roc_curves(
            y_test,
            probabilities,
            model_classes(model),
            output_dir / "roc_curves.png",
        )
        metrics["roc_auc"] = roc_metrics

    save_json(output_dir / "metrics.json", metrics)
    save_text(output_dir / "classification_report.txt", report_text)
    save_confusion_matrix_csv(matrix, output_dir / "confusion_matrix.csv")
    plot_confusion_matrix(matrix, output_dir / "confusion_matrix.png")
    save_prediction_rows(y_test, y_pred, probabilities, output_dir / "holdout_predictions.csv")

    return metrics


def prediction_probabilities(model: Any, x_values: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(x_values)
        except Exception:
            return None
    return None


def model_classes(model: Any) -> list[int]:
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier")
        classes = getattr(classifier, "classes_", None)
    if classes is None:
        return config.CLASS_LABELS
    return [int(label) for label in classes]


def sensitivity_specificity_by_class(matrix: np.ndarray) -> dict[str, dict[str, float]]:
    total = int(matrix.sum())
    metrics: dict[str, dict[str, float]] = {}

    for index, label in enumerate(config.CLASS_LABELS):
        true_positive = int(matrix[index, index])
        false_negative = int(matrix[index, :].sum() - true_positive)
        false_positive = int(matrix[:, index].sum() - true_positive)
        true_negative = int(total - true_positive - false_negative - false_positive)

        sensitivity = true_positive / max(true_positive + false_negative, 1)
        specificity = true_negative / max(true_negative + false_positive, 1)
        metrics[str(label)] = {
            "stage_name": config.CLASS_NAMES[label],
            "sensitivity_tpr": float(sensitivity),
            "specificity_tnr": float(specificity),
        }

    return metrics


def plot_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    labels = [str(label) for label in config.CLASS_LABELS]
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axis,
    )
    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted DR grade")
    axis.set_ylabel("True DR grade")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_roc_curves(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: list[int],
    output_path: Path,
) -> dict[str, float]:
    y_binary = label_binarize(y_true, classes=classes)

    if y_binary.shape[1] != probabilities.shape[1]:
        return {}

    figure, axis = plt.subplots(figsize=(8, 6))
    auc_scores: dict[str, float] = {}

    for class_index, class_label in enumerate(classes):
        truth = y_binary[:, class_index]
        if np.unique(truth).size < 2:
            continue

        fpr, tpr, _ = roc_curve(truth, probabilities[:, class_index])
        class_auc = roc_auc_score(truth, probabilities[:, class_index])
        auc_scores[str(class_label)] = float(class_auc)
        axis.plot(fpr, tpr, linewidth=2, label=f"Class {class_label} AUC={class_auc:.3f}")

    if not auc_scores:
        plt.close(figure)
        return {}

    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axis.set_title("One-vs-Rest ROC Curves")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return auc_scores


def save_feature_importance(
    model: Any,
    output_png: str | Path,
    output_csv: str | Path | None = None,
    feature_names: list[str] | None = None,
) -> list[dict[str, float | str]]:
    classifier = model.named_steps.get("classifier") if hasattr(model, "named_steps") else model
    importances = getattr(classifier, "feature_importances_", None)
    if importances is None:
        return []

    names = feature_names or config.FEATURE_NAMES
    rows = sorted(
        (
            {"feature": feature, "importance": float(importance)}
            for feature, importance in zip(names, importances)
        ),
        key=lambda row: float(row["importance"]),
        reverse=True,
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    names = [str(row["feature"]) for row in rows]
    values = [float(row["importance"]) for row in rows]
    axis.barh(names[::-1], values[::-1], color="#0E7C7B")
    axis.set_title("Random Forest Feature Importance")
    axis.set_xlabel("Mean Decrease in Impurity")
    figure.tight_layout()
    figure.savefig(output_png, dpi=180)
    plt.close(figure)

    if output_csv is not None:
        save_rows_csv(rows, Path(output_csv), fieldnames=["feature", "importance"])

    return rows


def explain_feature_importance(rows: list[dict[str, float | str]]) -> str:
    if not rows:
        return "Feature importance is available only for RandomForestClassifier."

    lines = ["Random Forest ranked the handcrafted retinal features as follows:"]
    for index, row in enumerate(rows, start=1):
        lines.append(f"{index}. {row['feature']}: {float(row['importance']):.4f}")
    lines.append(
        "Higher-ranked features contributed more to reducing impurity across the forest. "
        "In this thesis pipeline, these rankings show which lesion, vessel, or texture "
        "measurements most influenced the supervised DR-grade classifier.",
    )
    return "\n".join(lines)


def save_confusion_matrix_csv(matrix: np.ndarray, output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\pred", *config.CLASS_LABELS])
        for label, row in zip(config.CLASS_LABELS, matrix.tolist()):
            writer.writerow([label, *row])


def save_prediction_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None,
    output_path: Path,
) -> None:
    fieldnames = ["true_label", "predicted_label"]
    if probabilities is not None:
        fieldnames.extend([f"prob_stage_{label}" for label in config.CLASS_LABELS])

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for index, (true_label, predicted_label) in enumerate(zip(y_true, y_pred)):
            row: dict[str, Any] = {
                "true_label": int(true_label),
                "predicted_label": int(predicted_label),
            }
            if probabilities is not None:
                for class_index, label in enumerate(config.CLASS_LABELS):
                    if class_index < probabilities.shape[1]:
                        row[f"prob_stage_{label}"] = float(probabilities[index, class_index])
            writer.writerow(row)


def save_rows_csv(rows: list[dict[str, Any]], output_path: Path, fieldnames: list[str]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_saved_model(
    features_csv: str | Path = config.FEATURES_CSV,
    model_path: str | Path = config.BEST_MODEL_PATH,
    results_dir: str | Path = config.RESULTS_DIR,
) -> dict[str, Any]:
    table = read_feature_table(features_csv)
    feature_names = load_model_feature_names()
    x_values = table[feature_names].to_numpy(dtype=np.float64)
    y_values = table["label"].to_numpy(dtype=np.int64)
    _, x_test, _, y_test = train_test_split(
        x_values,
        y_values,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_values,
    )
    with Path(model_path).open("rb") as file:
        model = pickle.load(file)
    return evaluate_model(model, x_test, y_test, results_dir=results_dir)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved handcrafted-feature DR model.")
    parser.add_argument("--features-csv", type=Path, default=config.FEATURES_CSV)
    parser.add_argument("--model-path", type=Path, default=config.BEST_MODEL_PATH)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = evaluate_saved_model(
        features_csv=args.features_csv,
        model_path=args.model_path,
        results_dir=args.results_dir,
    )
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
