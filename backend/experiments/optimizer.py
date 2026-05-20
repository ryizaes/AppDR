import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from experiments.config import (
    FusionParams,
    ImageProcessingParams,
    dataclass_to_json_dict,
)
from experiments.dataset import DatasetItem
from experiments.evaluation import (
    classification_report_rows,
    confusion,
    evaluate_predictions,
    fisher_feature_ranking,
    mean_confidence_interval,
    most_confused_pairs,
)
from experiments.features import (
    extract_handcrafted_features,
    feature_names,
    numeric_feature_vector,
)
from experiments.fusion import predict_stage
from experiments.visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    save_preprocessing_outputs,
)


def run_grid_search(
    items: list[DatasetItem],
    processing_grid: list[ImageProcessingParams],
    fusion_grid: list[FusionParams],
    folds: int,
    output_dir: Path,
    max_combinations: int | None = None,
    save_failed_examples: int = 12,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = np.array([item.label for item in items], dtype=np.int64)
    adjusted_folds = adjust_fold_count(labels, folds)
    experiment_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    combo_id = 0

    for processing_index, processing_params in enumerate(processing_grid, start=1):
        print(f"Extracting features for processing config {processing_index}/{len(processing_grid)}")
        feature_rows = [
            extract_handcrafted_features(item.image_path, processing_params).features
            for item in items
        ]

        for fusion_index, fusion_params in enumerate(fusion_grid, start=1):
            combo_id += 1

            if max_combinations is not None and combo_id > max_combinations:
                break

            predictions, combo_fold_rows = cross_validate_combo(
                feature_rows,
                labels,
                fusion_params,
                adjusted_folds,
                combo_id,
            )
            metrics = evaluate_predictions(labels.tolist(), predictions)
            row = {
                "combo_id": combo_id,
                "processing_index": processing_index,
                "fusion_index": fusion_index,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_recall": metrics["macro_recall"],
                "minority_recall": metrics["minority_recall"],
                "processing_params": json.dumps(dataclass_to_json_dict(processing_params)),
                "fusion_params": json.dumps(dataclass_to_json_dict(fusion_params)),
            }
            experiment_rows.append(row)
            fold_rows.extend(combo_fold_rows)

            if best is None or is_better(metrics, best["metrics"]):
                best = {
                    "combo_id": combo_id,
                    "processing_params": processing_params,
                    "fusion_params": fusion_params,
                    "feature_rows": feature_rows,
                    "predictions": predictions,
                    "metrics": metrics,
                    "fold_rows": combo_fold_rows,
                }

        if max_combinations is not None and combo_id >= max_combinations:
            break

    if best is None:
        raise RuntimeError("No experiment combinations were evaluated.")

    save_experiment_outputs(
        items=items,
        labels=labels,
        best=best,
        experiment_rows=experiment_rows,
        fold_rows=fold_rows,
        output_dir=output_dir,
        save_failed_examples=save_failed_examples,
    )

    return {
        "best_combo_id": best["combo_id"],
        "metrics": best["metrics"],
        "output_dir": str(output_dir),
    }


def cross_validate_combo(
    feature_rows: list[dict[str, float | int | str]],
    labels: np.ndarray,
    fusion_params: FusionParams,
    folds: int,
    combo_id: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    predictions = np.zeros_like(labels)
    fold_rows: list[dict[str, Any]] = []

    for fold_index, (_, validation_indices) in enumerate(
        splitter.split(np.zeros(len(labels)), labels),
        start=1,
    ):
        fold_predictions = [
            predict_stage(feature_rows[index], fusion_params)
            for index in validation_indices
        ]
        predictions[validation_indices] = fold_predictions
        fold_metrics = evaluate_predictions(
            labels[validation_indices].tolist(),
            fold_predictions,
        )
        fold_rows.append(
            {
                "combo_id": combo_id,
                "fold": fold_index,
                "accuracy": fold_metrics["accuracy"],
                "macro_f1": fold_metrics["macro_f1"],
                "balanced_accuracy": fold_metrics["balanced_accuracy"],
                "macro_recall": fold_metrics["macro_recall"],
                "minority_recall": fold_metrics["minority_recall"],
            },
        )

    return predictions.tolist(), fold_rows


def is_better(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    candidate_key = (
        candidate["macro_f1"],
        candidate["minority_recall"],
        candidate["balanced_accuracy"],
        candidate["accuracy"],
    )
    incumbent_key = (
        incumbent["macro_f1"],
        incumbent["minority_recall"],
        incumbent["balanced_accuracy"],
        incumbent["accuracy"],
    )

    return candidate_key > incumbent_key


def adjust_fold_count(labels: np.ndarray, requested_folds: int) -> int:
    _, counts = np.unique(labels, return_counts=True)
    max_folds = int(np.min(counts))
    folds = max(2, min(requested_folds, max_folds))

    if folds < requested_folds:
        print(f"Using {folds} folds because the smallest class has {max_folds} samples.")

    return folds


def save_experiment_outputs(
    items: list[DatasetItem],
    labels: np.ndarray,
    best: dict[str, Any],
    experiment_rows: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
    output_dir: Path,
    save_failed_examples: int,
) -> None:
    predictions = best["predictions"]
    matrix = confusion(labels.tolist(), predictions)
    feature_matrix = np.array(
        [numeric_feature_vector(features) for features in best["feature_rows"]],
        dtype=np.float64,
    )
    ranking = fisher_feature_ranking(feature_matrix, labels, feature_names())
    best_fold_metrics = best["fold_rows"]
    summary = {
        "best_combo_id": best["combo_id"],
        "best_metrics": best["metrics"],
        "macro_f1_ci": mean_confidence_interval([row["macro_f1"] for row in best_fold_metrics]),
        "balanced_accuracy_ci": mean_confidence_interval(
            [row["balanced_accuracy"] for row in best_fold_metrics],
        ),
        "minority_recall_ci": mean_confidence_interval(
            [row["minority_recall"] for row in best_fold_metrics],
        ),
        "most_confused_pairs": most_confused_pairs(matrix),
    }
    best_config = {
        "processing_params": dataclass_to_json_dict(best["processing_params"]),
        "fusion_params": dataclass_to_json_dict(best["fusion_params"]),
        "summary": summary,
    }

    write_csv(output_dir / "experiment_log.csv", experiment_rows)
    write_csv(output_dir / "fold_metrics.csv", fold_rows)
    write_csv(output_dir / "classification_report.csv", classification_report_rows(labels.tolist(), predictions))
    write_csv(output_dir / "feature_importance.csv", ranking)
    write_csv(output_dir / "failed_predictions.csv", failed_prediction_rows(items, labels.tolist(), predictions))
    write_matrix_csv(output_dir / "confusion_matrix.csv", matrix)
    (output_dir / "best_config.json").write_text(
        json.dumps(best_config, indent=2),
        encoding="utf-8",
    )
    (output_dir / "cv_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    plot_confusion_matrix(matrix, output_dir / "confusion_matrix.png")
    plot_feature_importance(ranking, output_dir / "feature_importance.png")
    save_failed_visuals(
        items,
        labels.tolist(),
        predictions,
        best["processing_params"],
        output_dir / "failed_examples",
        limit=save_failed_examples,
    )


def failed_prediction_rows(
    items: list[DatasetItem],
    labels: list[int],
    predictions: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for item, label, prediction in zip(items, labels, predictions):
        if label == prediction:
            continue

        rows.append(
            {
                "image_id": item.image_id,
                "image_path": str(item.image_path),
                "true_label": label,
                "predicted_label": prediction,
            },
        )

    return rows


def save_failed_visuals(
    items: list[DatasetItem],
    labels: list[int],
    predictions: list[int],
    processing_params: ImageProcessingParams,
    output_dir: Path,
    limit: int,
) -> None:
    saved = 0

    for item, label, prediction in zip(items, labels, predictions):
        if label == prediction:
            continue
        if saved >= limit:
            return

        extraction = extract_handcrafted_features(
            item.image_path,
            processing_params,
            include_images=True,
        )
        save_preprocessing_outputs(
            output_dir,
            item.image_id,
            label,
            prediction,
            {**extraction.processed, **extraction.masks},
        )
        saved += 1


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\pred", 0, 1, 2, 3, 4])

        for index, row in enumerate(matrix.tolist()):
            writer.writerow([index, *row])
