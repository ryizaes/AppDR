"""Train the AppDR ML-only diabetic retinopathy classifier.

This module uses only the existing 203 handcrafted retinal features.  It does
not read images, alter OpenCV preprocessing, or use neural networks.
"""

from __future__ import annotations

import argparse
import ast
import csv
import inspect
import json
import math
import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import shap
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

import config
from predict import (
    ENGINEERED_FEATURE_NAMES,
    ClinicalFeatureEngineer,
    FeatureNameFrame,
    FeatureNameSelector,
)


try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - handled through dependency report.
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - handled through dependency report.
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - handled through dependency report.
    CatBoostClassifier = None


RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
CV_FOLDS = 5
MIN_OPTUNA_TRIALS = 50
FEATURE_COUNTS: list[int | str | None] = ["all_203", None, 150, 100, 75, 50]
FEATURE_SELECTION_METHODS = ["mutual_information", "random_forest", "shap"]
BINARY_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
NEAR_ZERO_VARIANCE_THRESHOLD = 1e-8
MAX_SHAP_BACKGROUND = 40
MAX_SHAP_EVAL_SAMPLES = 250
EPSILON = 1e-12
BINARY_REFERABLE_SOURCE_LABELS = [0, 1, 2, 3, 4]
BINARY_REFERABLE_MAPPING = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}
BINARY_REFERABLE_CLASS_NAMES = {
    0: "Non-Referable",
    1: "Referable",
}


@dataclass
class CandidateResult:
    name: str
    pipeline: BaseEstimator
    cv_metrics: dict[str, dict[str, float]]
    best_params: dict[str, Any]
    candidate_type: str
    calibration: dict[str, Any] | None = None


def train_models(
    features_csv: str | Path = config.FEATURES_CSV,
    results_dir: str | Path = config.RESULTS_DIR,
    n_trials: int = MIN_OPTUNA_TRIALS,
    binary_referable: bool = False,
    resume_completed: bool = False,
    smoke: bool = False,
    skip_interpretability: bool = False,
) -> dict[str, Any]:
    """Run the complete training, selection, and artifact export workflow."""
    if n_trials < MIN_OPTUNA_TRIALS and not smoke:
        raise ValueError(f"Optuna must run at least {MIN_OPTUNA_TRIALS} trials per model.")

    np.random.seed(RANDOM_STATE)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    output_dir = ensure_dir(results_dir)
    mode = "binary referable" if binary_referable else "multiclass DR grading"
    print(f"Loading dataset for {mode}: {features_csv}", flush=True)

    table, feature_names, y_values, data_quality = load_and_validate_dataset(
        features_csv,
    )
    save_json(output_dir / "data_quality_report.json", data_quality)

    label_remapping: dict[str, Any] | None = None
    if binary_referable:
        detect_problem_type(y_values)
        source_distribution = build_class_distribution(
            y_values,
            "multiclass",
            BINARY_REFERABLE_SOURCE_LABELS,
        )
        y_values = remap_labels_to_binary_referable(y_values)
        label_remapping = {
            "mode": "binary_referable",
            "source": "multiclass_dr_grades",
            "rule": "DR grades 0-1 map to Non-Referable (0); grades 2-4 map to Referable (1).",
            "mapping": {str(key): value for key, value in BINARY_REFERABLE_MAPPING.items()},
            "class_names": {str(key): value for key, value in BINARY_REFERABLE_CLASS_NAMES.items()},
            "source_stage_distribution": source_distribution,
        }
        save_json(output_dir / "label_remapping.json", label_remapping)

    problem_type, class_labels = detect_problem_type(y_values)
    class_distribution = build_class_distribution(
        y_values,
        problem_type,
        class_labels,
        class_names=BINARY_REFERABLE_CLASS_NAMES if binary_referable else None,
    )
    if label_remapping is not None:
        class_distribution["label_remapping"] = label_remapping
    save_json(output_dir / "class_distribution.json", class_distribution)
    print(
        f"Validated {len(y_values)} samples, {len(feature_names)} features, "
        f"problem={problem_type}",
        flush=True,
    )

    x_values = table[feature_names].to_numpy(dtype=np.float64)
    x_values[~np.isfinite(x_values)] = np.nan
    sample_indices = table.index.to_numpy(dtype=int)

    validate_enough_samples_for_split(y_values, class_labels)
    (
        x_train_validation,
        x_test,
        y_train_validation,
        y_test,
        train_validation_indices,
        test_indices,
    ) = train_test_split(
        x_values,
        y_values,
        sample_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_values,
    )
    (
        x_train,
        x_validation,
        y_train,
        y_validation,
        train_indices,
        validation_indices,
    ) = train_test_split(
        x_train_validation,
        y_train_validation,
        train_validation_indices,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_validation,
    )
    print(
        "Created stratified split: "
        f"train={len(y_train)}, validation={len(y_validation)}, test={len(y_test)}",
        flush=True,
    )
    validate_enough_samples_for_cv(y_train, class_labels, context="training split")
    split_report = build_split_report(
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        class_labels=class_labels,
    )
    save_json(output_dir / "split_report.json", split_report)

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scaler_name, engineering_enabled, preprocessing_report = select_scaler_and_engineering(
        x_train=x_train,
        y_train=y_train,
        feature_names=feature_names,
        problem_type=problem_type,
        class_labels=class_labels,
        cv=cv,
    )
    save_json(output_dir / "preprocessing_report.json", preprocessing_report)
    print(
        f"Selected scaler={scaler_name}, engineered_features={engineering_enabled}",
        flush=True,
    )

    full_feature_names = build_available_feature_names(feature_names, engineering_enabled)
    selected_features, selection_report = select_optimal_features(
        x_train=x_train,
        y_train=y_train,
        feature_names=feature_names,
        full_feature_names=full_feature_names,
        engineering_enabled=engineering_enabled,
        scaler_name=scaler_name,
        problem_type=problem_type,
        class_labels=class_labels,
        cv=cv,
        output_dir=output_dir,
    )
    print(
        f"Selected {len(selected_features)} features via "
        f"{selection_report['best_method']}",
        flush=True,
    )
    save_json(
        output_dir / "selected_features.json",
        {
            "selection_method": selection_report["best_method"],
            "requested_feature_count": selection_report["best_requested_count"],
            "selected_feature_count": len(selected_features),
            "engineered_features_enabled": bool(engineering_enabled),
            "available_feature_count": len(full_feature_names),
            "selected_features": selected_features,
            "engineered_feature_names": ENGINEERED_FEATURE_NAMES
            if engineering_enabled
            else [],
        },
    )
    save_json(
        output_dir / "feature_order.json",
        {
            "feature_order": feature_names,
            "feature_count": len(feature_names),
            "full_feature_order_after_engineering": full_feature_names,
            "random_state": RANDOM_STATE,
        },
    )

    optimized_candidates = optimize_all_models(
        x_train=x_train,
        y_train=y_train,
        feature_names=feature_names,
        full_feature_names=full_feature_names,
        selected_features=selected_features,
        scaler_name=scaler_name,
        engineering_enabled=engineering_enabled,
        problem_type=problem_type,
        class_labels=class_labels,
        cv=cv,
        n_trials=n_trials,
        output_dir=output_dir,
        resume_completed=resume_completed,
    )

    ensemble_candidates = build_ensemble_candidates(
        optimized_candidates,
        x_train=x_train,
        y_train=y_train,
        feature_names=feature_names,
        full_feature_names=full_feature_names,
        selected_features=selected_features,
        scaler_name=scaler_name,
        engineering_enabled=engineering_enabled,
        problem_type=problem_type,
        class_labels=class_labels,
        cv=cv,
    )

    all_candidates = [*optimized_candidates, *ensemble_candidates]
    print(f"Built {len(ensemble_candidates)} ensemble candidates", flush=True)
    calibration_candidates, calibration_report = evaluate_calibration_candidates(
        ranked_candidates(all_candidates)[:3],
        x_train=x_train,
        y_train=y_train,
        feature_names=feature_names,
        full_feature_names=full_feature_names,
        selected_features=selected_features,
        scaler_name=scaler_name,
        engineering_enabled=engineering_enabled,
        problem_type=problem_type,
        class_labels=class_labels,
        cv=cv,
    )
    all_candidates.extend(calibration_candidates)
    save_json(output_dir / "calibration_report.json", calibration_report)
    print(f"Evaluated {len(calibration_candidates)} calibration candidates", flush=True)

    ranked = ranked_candidates(all_candidates)
    save_model_comparison(ranked, output_dir / "model_comparison_results.csv")
    best_candidate = ranked[0]

    optimal_threshold = optimize_binary_threshold(
        best_candidate.pipeline,
        x_train,
        y_train,
        cv,
        class_labels,
    ) if problem_type == "binary" else {
        "problem_type": "multiclass",
        "threshold": None,
        "message": "Threshold optimization is used only for binary classification.",
    }
    save_json(output_dir / "optimal_threshold.json", optimal_threshold)

    validation_model = clone(best_candidate.pipeline)
    fit_estimator(validation_model, x_train, y_train)
    validation_metrics = evaluate_holdout_model(
        model=validation_model,
        x_test=x_validation,
        y_test=y_validation,
        sample_indices=validation_indices,
        problem_type=problem_type,
        class_labels=class_labels,
        threshold=optimal_threshold.get("threshold"),
        output_dir=output_dir,
        split_name="validation",
    )

    x_final_train = np.vstack([x_train, x_validation])
    y_final_train = np.concatenate([y_train, y_validation])
    best_model = clone(best_candidate.pipeline)
    fit_estimator(best_model, x_final_train, y_final_train)
    save_pickle(output_dir / "best_model.pkl", best_model)
    save_pipeline_artifacts(best_model, output_dir)

    holdout_metrics = evaluate_holdout_model(
        model=best_model,
        x_test=x_test,
        y_test=y_test,
        sample_indices=test_indices,
        problem_type=problem_type,
        class_labels=class_labels,
        threshold=optimal_threshold.get("threshold"),
        output_dir=output_dir,
        split_name="test",
    )

    if skip_interpretability:
        explainer, importance_rows = None, []
    else:
        explainer, importance_rows = build_interpretability_artifacts(
            model=best_model,
            x_reference=x_final_train,
            x_eval=x_test,
            y_eval=y_test,
            problem_type=problem_type,
            class_labels=class_labels,
            output_dir=output_dir,
        )
    save_pickle(output_dir / "explainer.pkl", explainer)
    save_feature_importance(importance_rows, output_dir / "feature_importance.csv")

    metrics_payload = {
        "problem_type": problem_type,
        "binary_referable": bool(binary_referable),
        "label_remapping": label_remapping,
        "best_model_name": best_candidate.name,
        "best_candidate_type": best_candidate.candidate_type,
        "best_params": best_candidate.best_params,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "validation_size_of_train_validation": VALIDATION_SIZE,
        "cv_folds": CV_FOLDS,
        "optuna_trials_per_model": n_trials,
        "smoke_run": bool(smoke),
        "interpretability_skipped": bool(skip_interpretability),
        "scaler": scaler_name,
        "engineered_features_enabled": bool(engineering_enabled),
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "class_labels": class_labels,
        "class_distribution": class_distribution,
        "data_quality": data_quality,
        "preprocessing_report": preprocessing_report,
        "feature_selection_report": selection_report,
        "split_report": split_report,
        "calibration_report": calibration_report,
        "cross_validation": {
            candidate.name: candidate.cv_metrics
            for candidate in ranked
        },
        "model_ranking": [
            candidate_summary(candidate, rank=index + 1)
            for index, candidate in enumerate(ranked)
        ],
        "holdout": holdout_metrics,
        "validation": validation_metrics,
        "optimal_threshold": optimal_threshold,
        "artifact_paths": artifact_path_report(),
    }
    save_json(output_dir / "metrics.json", metrics_payload)
    save_json(
        output_dir / "best_model_metadata.json",
        {
            "best_model_name": best_candidate.name,
            "problem_type": problem_type,
            "binary_referable": bool(binary_referable),
            "label_remapping": label_remapping,
            "feature_names": feature_names,
            "selected_features": selected_features,
            "selected_feature_count": len(selected_features),
            "engineered_features_enabled": bool(engineering_enabled),
            "engineered_feature_names": ENGINEERED_FEATURE_NAMES
            if engineering_enabled
            else [],
            "scaler": scaler_name,
            "class_labels": class_labels,
            "random_state": RANDOM_STATE,
            "threshold": optimal_threshold.get("threshold"),
            "artifact_paths": artifact_path_report(),
            "interpretability_skipped": bool(skip_interpretability),
        },
    )

    print(f"Best model: {best_candidate.name}")
    print(f"Validation F1: {validation_metrics['f1']:.4f}")
    print(f"Holdout F1: {holdout_metrics['f1']:.4f}")
    print(f"Holdout recall: {holdout_metrics['recall']:.4f}")
    print(f"Artifacts saved under: {Path(results_dir)}")

    return metrics_payload


def load_and_validate_dataset(
    features_csv: str | Path,
) -> tuple[pd.DataFrame, list[str], np.ndarray, dict[str, Any]]:
    table = pd.read_csv(features_csv)
    if "label" not in table.columns:
        raise ValueError("Feature CSV must contain a 'label' column.")

    feature_names = resolve_feature_names(table)
    feature_frame = table[feature_names].apply(pd.to_numeric, errors="coerce")
    labels = table["label"].to_numpy()

    try:
        y_values = labels.astype(np.int64)
    except ValueError as exc:
        raise ValueError("Labels must be integer class values.") from exc

    report = build_data_quality_report(table, feature_frame, feature_names)
    feature_frame, sanitization_report, keep_mask = sanitize_feature_frame(feature_frame)
    if not np.all(keep_mask):
        table = table.loc[keep_mask].reset_index(drop=True)
        y_values = y_values[keep_mask.to_numpy()]
    report["sanitization"] = sanitization_report
    exact_duplicate_mask = table.duplicated(subset=[*feature_names, "label"], keep="first")
    exact_duplicate_count = int(exact_duplicate_mask.sum())
    if exact_duplicate_count:
        table = table.loc[~exact_duplicate_mask].reset_index(drop=True)
        feature_frame = feature_frame.loc[~exact_duplicate_mask].reset_index(drop=True)
        y_values = y_values[~exact_duplicate_mask.to_numpy()]

    # Prevent train/test leakage from repeated extracted feature vectors, even
    # when duplicate rows disagree on label. There is no raw image identifier in
    # features.csv, so exact feature-vector duplication is the safest available
    # proxy for duplicated images or duplicated extraction output.
    duplicate_feature_mask = feature_frame.duplicated(keep="first")
    duplicate_feature_count = int(duplicate_feature_mask.sum())
    if duplicate_feature_count:
        table = table.loc[~duplicate_feature_mask].reset_index(drop=True)
        feature_frame = feature_frame.loc[~duplicate_feature_mask].reset_index(drop=True)
        y_values = y_values[~duplicate_feature_mask.to_numpy()]

    report["exact_duplicate_rows_removed"] = exact_duplicate_count
    report["duplicate_feature_rows_removed"] = duplicate_feature_count

    for name in feature_names:
        table[name] = feature_frame[name]

    return table, feature_names, y_values, report


def resolve_feature_names(table: pd.DataFrame) -> list[str]:
    if all(name in table.columns for name in config.FEATURE_NAMES):
        return list(config.FEATURE_NAMES)

    numeric_candidates = [
        column
        for column in table.columns
        if column != "label" and pd.api.types.is_numeric_dtype(table[column])
    ]
    if len(numeric_candidates) == len(config.FEATURE_NAMES):
        return numeric_candidates

    missing = [name for name in config.FEATURE_NAMES if name not in table.columns]
    raise ValueError(
        "Feature CSV must contain the existing 203 retinal feature columns. "
        f"Missing examples: {missing[:10]}",
    )


def build_data_quality_report(
    table: pd.DataFrame,
    features: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, Any]:
    values = features.to_numpy(dtype=np.float64)
    finite_values = np.where(np.isfinite(values), values, np.nan)
    nan_mask = np.isnan(values)
    inf_mask = np.isinf(values)

    constant_features: list[str] = []
    near_zero_variance_features: list[str] = []
    feature_variances: dict[str, float] = {}
    for index, name in enumerate(feature_names):
        column = finite_values[:, index]
        non_missing = column[~np.isnan(column)]
        if non_missing.size == 0 or np.unique(non_missing).size <= 1:
            constant_features.append(name)
            feature_variances[name] = 0.0
            continue
        variance = float(np.nanvar(column))
        feature_variances[name] = variance
        value_counts = pd.Series(non_missing).value_counts(normalize=True, dropna=True)
        dominant_ratio = float(value_counts.iloc[0]) if not value_counts.empty else 1.0
        unique_ratio = float(np.unique(non_missing).size / max(non_missing.size, 1))
        if variance <= NEAR_ZERO_VARIANCE_THRESHOLD or (
            unique_ratio < 0.01 and dominant_ratio > 0.95
        ):
            near_zero_variance_features.append(name)

    duplicate_feature_mask = features.duplicated(keep=False)
    exact_duplicate_mask = table.duplicated(subset=[*feature_names, "label"], keep=False)

    return {
        "row_count": int(table.shape[0]),
        "feature_count": int(len(feature_names)),
        "nan_cell_count": int(nan_mask.sum()),
        "inf_cell_count": int(inf_mask.sum()),
        "nan_features": {
            feature_names[index]: int(nan_mask[:, index].sum())
            for index in range(len(feature_names))
            if int(nan_mask[:, index].sum()) > 0
        },
        "inf_features": {
            feature_names[index]: int(inf_mask[:, index].sum())
            for index in range(len(feature_names))
            if int(inf_mask[:, index].sum()) > 0
        },
        "duplicate_feature_row_count": int(duplicate_feature_mask.sum()),
        "exact_duplicate_row_count": int(exact_duplicate_mask.sum()),
        "duplicate_feature_row_indices_preview": [
            int(index)
            for index in np.flatnonzero(duplicate_feature_mask.to_numpy())[:50]
        ],
        "constant_feature_count": len(constant_features),
        "constant_features": constant_features,
        "near_zero_variance_feature_count": len(near_zero_variance_features),
        "near_zero_variance_features": near_zero_variance_features,
        "variance_summary": {
            "min": float(np.nanmin(list(feature_variances.values()))),
            "median": float(np.nanmedian(list(feature_variances.values()))),
            "max": float(np.nanmax(list(feature_variances.values()))),
        },
    }


def sanitize_feature_frame(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], pd.Series]:
    raw_values = features.to_numpy(dtype=np.float64)
    nonfinite_mask = ~np.isfinite(raw_values)
    cleaned = features.replace([np.inf, -np.inf], np.nan).copy()
    valid_value_counts = cleaned.notna().sum(axis=1)
    keep_mask = valid_value_counts > 0
    cleaned = cleaned.loc[keep_mask].reset_index(drop=True)
    cleaned_values = cleaned.to_numpy(dtype=np.float64)
    remaining_inf = int(np.isinf(cleaned_values).sum())
    finite_values = np.where(np.isfinite(cleaned_values), cleaned_values, np.nan)
    report = {
        "nan_cells_before": int(np.isnan(raw_values).sum()),
        "inf_cells_before": int(np.isinf(raw_values).sum()),
        "nonfinite_cells_before": int(nonfinite_mask.sum()),
        "rows_removed_all_features_missing": int((~keep_mask).sum()),
        "nan_cells_after": int(np.isnan(cleaned_values).sum()),
        "inf_cells_after": remaining_inf,
        "max_abs_finite_after": float(np.nanmax(np.abs(finite_values)))
        if np.any(np.isfinite(finite_values))
        else 0.0,
    }
    if remaining_inf:
        raise ValueError("Feature sanitization failed: infinite values remain.")
    return cleaned, report, keep_mask


def detect_problem_type(y_values: np.ndarray) -> tuple[str, list[int]]:
    unique_labels = sorted(int(value) for value in np.unique(y_values))
    if len(unique_labels) < 2:
        raise ValueError("At least two classes are required for supervised training.")
    if set(unique_labels).issubset({0, 1}):
        return "binary", [0, 1]
    if set(unique_labels).issubset({0, 1, 2, 3, 4}):
        return "multiclass", [0, 1, 2, 3, 4]
    raise ValueError(
        "Labels must be binary {0, 1} or multiclass DR grades {0, 1, 2, 3, 4}.",
    )


def remap_labels_to_binary_referable(y_values: np.ndarray) -> np.ndarray:
    unique = {int(value) for value in np.unique(y_values)}
    if not unique.issubset(set(BINARY_REFERABLE_SOURCE_LABELS)):
        raise ValueError(
            "Binary referable remapping requires DR grade labels in {0, 1, 2, 3, 4}.",
        )

    return np.asarray(
        [BINARY_REFERABLE_MAPPING[int(value)] for value in y_values],
        dtype=np.int64,
    )


def build_class_distribution(
    y_values: np.ndarray,
    problem_type: str,
    class_labels: list[int],
    class_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    total = int(y_values.size)
    counts = {
        str(label): int(np.sum(y_values == label))
        for label in class_labels
    }
    nonzero = [count for count in counts.values() if count > 0]
    max_to_min = float(max(nonzero) / max(min(nonzero), 1)) if nonzero else 0.0
    payload = {
        "problem_type": problem_type,
        "total_samples": total,
        "class_counts": counts,
        "class_percentages": {
            label: float(count / max(total, 1))
            for label, count in counts.items()
        },
        "max_to_min_ratio": max_to_min,
        "is_imbalanced": bool(max_to_min >= 1.5),
        "imbalance_strategy": [
            "class_weight='balanced' where supported",
            "sample_weight from compute_sample_weight during model fitting",
            "scale_pos_weight for binary XGBoost",
            "BalancedRandomForestClassifier for the balanced forest candidate",
        ],
    }
    if class_names is not None:
        payload["class_names"] = {
            str(label): class_names.get(label, str(label))
            for label in class_labels
        }
    return payload


def build_split_report(
    y_train: np.ndarray,
    y_validation: np.ndarray,
    y_test: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    class_labels: list[int],
) -> dict[str, Any]:
    return {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "validation_size_of_train_validation": VALIDATION_SIZE,
        "train_count": int(len(y_train)),
        "validation_count": int(len(y_validation)),
        "test_count": int(len(y_test)),
        "train_class_counts": {
            str(label): int(np.sum(y_train == label)) for label in class_labels
        },
        "validation_class_counts": {
            str(label): int(np.sum(y_validation == label)) for label in class_labels
        },
        "test_class_counts": {
            str(label): int(np.sum(y_test == label)) for label in class_labels
        },
        "train_indices_preview": [int(index) for index in train_indices[:20]],
        "validation_indices_preview": [int(index) for index in validation_indices[:20]],
        "test_indices_preview": [int(index) for index in test_indices[:20]],
        "leakage_check": {
            "train_validation_overlap": bool(
                set(map(int, train_indices)).intersection(map(int, validation_indices))
            ),
            "train_test_overlap": bool(
                set(map(int, train_indices)).intersection(map(int, test_indices))
            ),
            "validation_test_overlap": bool(
                set(map(int, validation_indices)).intersection(map(int, test_indices))
            ),
        },
    }


def validate_enough_samples_for_split(y_values: np.ndarray, labels: list[int]) -> None:
    counts = {label: int(np.sum(y_values == label)) for label in labels}
    too_small = {label: count for label, count in counts.items() if 0 < count < 2}
    if too_small:
        raise ValueError(
            f"Every present class needs at least two samples for stratified split: {too_small}",
        )


def validate_enough_samples_for_cv(
    y_values: np.ndarray,
    labels: list[int],
    context: str,
) -> None:
    counts = {label: int(np.sum(y_values == label)) for label in labels}
    too_small = {label: count for label, count in counts.items() if 0 < count < CV_FOLDS}
    if too_small:
        raise ValueError(
            f"Stratified {CV_FOLDS}-fold CV requires at least {CV_FOLDS} "
            f"samples per present class in the {context}: {too_small}",
        )


def select_scaler_and_engineering(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    problem_type: str,
    class_labels: list[int],
    cv: StratifiedKFold,
) -> tuple[str, bool, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scaler_name in ("standard", "robust"):
        for engineering_enabled in (False, True):
            classifier = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=3000,
                random_state=RANDOM_STATE,
                solver="lbfgs",
            )
            pipeline = Pipeline(
                steps=[
                    (
                        "feature_engineer",
                        ClinicalFeatureEngineer(feature_names, enabled=engineering_enabled),
                    ),
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", build_scaler(scaler_name)),
                    ("classifier", classifier),
                ],
            )
            metrics = cross_validate_estimator(
                pipeline,
                x_train,
                y_train,
                cv,
                problem_type,
                class_labels,
            )
            rows.append(
                {
                    "scaler": scaler_name,
                    "engineered_features_enabled": bool(engineering_enabled),
                    "metrics": metrics,
                    "rank_key": metric_rank_key(metrics),
                },
            )

    rows.sort(key=lambda row: row["rank_key"], reverse=True)
    best = rows[0]
    return (
        str(best["scaler"]),
        bool(best["engineered_features_enabled"]),
        {
            "selection_basis": "5-fold CV LogisticRegression, ranked by F1 then recall",
            "candidates": rows,
            "selected_scaler": str(best["scaler"]),
            "engineered_features_enabled": bool(best["engineered_features_enabled"]),
        },
    )


def build_available_feature_names(
    feature_names: list[str],
    engineering_enabled: bool,
) -> list[str]:
    names = list(feature_names)
    if engineering_enabled:
        names.extend(ENGINEERED_FEATURE_NAMES)
    return names


def select_optimal_features(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    full_feature_names: list[str],
    engineering_enabled: bool,
    scaler_name: str,
    problem_type: str,
    class_labels: list[int],
    cv: StratifiedKFold,
    output_dir: Path,
) -> tuple[list[str], dict[str, Any]]:
    engineered_train = ClinicalFeatureEngineer(
        feature_names,
        enabled=engineering_enabled,
    ).fit_transform(x_train)
    imputed_train = SimpleImputer(strategy="median").fit_transform(engineered_train)

    rankings = {
        method: rank_features(method, imputed_train, y_train, full_feature_names)
        for method in FEATURE_SELECTION_METHODS
    }
    save_feature_rankings(rankings, output_dir / "feature_selection_rankings.csv")

    experiments: list[dict[str, Any]] = []
    for method, ranked_features in rankings.items():
        for requested_count in FEATURE_COUNTS:
            selected = feature_subset_from_ranking(
                ranked_features,
                requested_count,
                feature_names,
            )
            classifier = ExtraTreesClassifier(
                n_estimators=250,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=cpu_jobs(),
            )
            pipeline = make_pipeline(
                classifier=classifier,
                feature_names=feature_names,
                full_feature_names=full_feature_names,
                selected_features=selected,
                scaler_name=scaler_name,
                engineering_enabled=engineering_enabled,
            )
            metrics = cross_validate_estimator(
                pipeline,
                x_train,
                y_train,
                cv,
                problem_type,
                class_labels,
            )
            experiments.append(
                {
                    "method": method,
                    "requested_count": "all" if requested_count is None else requested_count,
                    "selected_feature_count": len(selected),
                    "selected_features": selected,
                    "metrics": metrics,
                    "rank_key": metric_rank_key(metrics),
                },
            )

    experiments.sort(key=lambda row: row["rank_key"], reverse=True)
    best = experiments[0]
    save_feature_selection_experiments(
        experiments,
        output_dir / "feature_selection_experiments.csv",
    )
    return list(best["selected_features"]), {
        "best_method": best["method"],
        "best_requested_count": best["requested_count"],
        "best_selected_feature_count": best["selected_feature_count"],
        "experiments": experiments,
    }


def rank_features(
    method: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    if method == "mutual_information":
        scores = mutual_info_classif(
            x_train,
            y_train,
            random_state=RANDOM_STATE,
            discrete_features=False,
        )
    elif method == "random_forest":
        selector_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=cpu_jobs(),
        )
        selector_model.fit(x_train, y_train, sample_weight=balanced_weights(y_train))
        scores = selector_model.feature_importances_
    elif method == "shap":
        scores = shap_feature_scores(x_train, y_train)
    else:
        raise ValueError(f"Unsupported feature selection method: {method}")

    clean_scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(-clean_scores)
    return [
        {
            "rank": int(rank),
            "feature": str(feature_names[index]),
            "score": float(clean_scores[index]),
        }
        for rank, index in enumerate(order, start=1)
    ]


def shap_feature_scores(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    selector_model = ExtraTreesClassifier(
        n_estimators=250,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=cpu_jobs(),
    )
    selector_model.fit(x_train, y_train, sample_weight=balanced_weights(y_train))
    sample = deterministic_sample(x_train, MAX_SHAP_EVAL_SAMPLES)
    explainer = shap.TreeExplainer(selector_model)
    raw_values = explainer.shap_values(sample)
    return mean_abs_shap_by_feature(raw_values)


def feature_subset_from_ranking(
    ranked_features: list[dict[str, Any]],
    requested_count: int | str | None,
    original_feature_names: list[str],
) -> list[str]:
    if requested_count == "all_203":
        return list(original_feature_names)
    if requested_count is None:
        count = len(ranked_features)
    else:
        count = min(int(requested_count), len(ranked_features))
    return [str(row["feature"]) for row in ranked_features[:count]]


def optimize_all_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    full_feature_names: list[str],
    selected_features: list[str],
    scaler_name: str,
    engineering_enabled: bool,
    problem_type: str,
    class_labels: list[int],
    cv: StratifiedKFold,
    n_trials: int,
    output_dir: Path,
    resume_completed: bool,
) -> list[CandidateResult]:
    model_names = [
        "Logistic Regression",
        "Random Forest",
        "Extra Trees",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "HistGradientBoosting",
        "SVM RBF",
        "Balanced Random Forest",
    ]
    candidates: list[CandidateResult] = []
    optuna_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        if not dependency_available(model_name):
            optuna_rows.append(
                {
                    "model": model_name,
                    "status": "skipped_missing_dependency",
                    "best_value": None,
                    "best_params": {},
                },
            )
            continue

        if resume_completed:
            resumed = load_completed_optuna_candidate(
                model_name=model_name,
                x_train=x_train,
                y_train=y_train,
                feature_names=feature_names,
                full_feature_names=full_feature_names,
                selected_features=selected_features,
                scaler_name=scaler_name,
                engineering_enabled=engineering_enabled,
                problem_type=problem_type,
                class_labels=class_labels,
                cv=cv,
                n_trials=n_trials,
                output_dir=output_dir,
            )
            if resumed is not None:
                candidate, row = resumed
                candidates.append(candidate)
                optuna_rows.append(row)
                continue

        print(f"Optimizing {model_name} ({n_trials} Optuna trials)", flush=True)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            study_name=safe_name(model_name),
        )
        progress_path = output_dir / f"optuna_progress_{safe_name(model_name)}.csv"
        initialize_optuna_progress(progress_path)

        def objective(trial: optuna.Trial) -> float:
            classifier = build_classifier(
                model_name,
                trial,
                problem_type,
                y_train,
                enable_probability=model_name != "SVM RBF",
            )
            pipeline = make_pipeline(
                classifier=classifier,
                feature_names=feature_names,
                full_feature_names=full_feature_names,
                selected_features=selected_features,
                scaler_name=scaler_name,
                engineering_enabled=engineering_enabled,
            )
            metrics = cross_validate_estimator(
                pipeline,
                x_train,
                y_train,
                cv,
                problem_type,
                class_labels,
            )
            f1_mean = metrics["f1"]["mean"]
            recall_mean = metrics["recall"]["mean"]
            trial.set_user_attr("metrics", metrics)
            trial.set_user_attr("f1", f1_mean)
            trial.set_user_attr("recall", recall_mean)
            return float(f1_mean + (recall_mean * 1e-4))

        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=1,
            show_progress_bar=False,
            callbacks=[
                lambda current_study, frozen_trial, path=progress_path: append_optuna_progress(
                    path,
                    current_study,
                    frozen_trial,
                ),
            ],
        )
        best_trial = study.best_trial
        best_classifier = build_classifier(
            model_name,
            optuna.trial.FixedTrial(best_trial.params),
            problem_type,
            y_train,
        )
        best_pipeline = make_pipeline(
            classifier=best_classifier,
            feature_names=feature_names,
            full_feature_names=full_feature_names,
            selected_features=selected_features,
            scaler_name=scaler_name,
            engineering_enabled=engineering_enabled,
        )
        best_metrics = cross_validate_estimator(
            best_pipeline,
            x_train,
            y_train,
            cv,
            problem_type,
            class_labels,
        )
        candidates.append(
            CandidateResult(
                name=model_name,
                pipeline=best_pipeline,
                cv_metrics=best_metrics,
                best_params=to_jsonable(best_trial.params),
                candidate_type="optimized_model",
            ),
        )
        optuna_rows.append(
            {
                "model": model_name,
                "status": "completed",
                "best_value": float(best_trial.value),
                "best_params": json.dumps(to_jsonable(best_trial.params)),
                "f1_mean": best_metrics["f1"]["mean"],
                "recall_mean": best_metrics["recall"]["mean"],
            },
        )
        save_optuna_trials(study, output_dir / f"optuna_trials_{safe_name(model_name)}.csv")

    write_rows(output_dir / "optuna_summary.csv", optuna_rows)
    if not candidates:
        raise RuntimeError("No model candidates were trained. Check ML dependencies.")
    return candidates


def load_completed_optuna_candidate(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    full_feature_names: list[str],
    selected_features: list[str],
    scaler_name: str,
    engineering_enabled: bool,
    problem_type: str,
    class_labels: list[int],
    cv: StratifiedKFold,
    n_trials: int,
    output_dir: Path,
) -> tuple[CandidateResult, dict[str, Any]] | None:
    trials_path = output_dir / f"optuna_trials_{safe_name(model_name)}.csv"
    if not trials_path.exists():
        return None

    trials = pd.read_csv(trials_path)
    if "state" not in trials.columns or "value" not in trials.columns:
        return None

    completed = trials[trials["state"] == "COMPLETE"].copy()
    if len(completed) < n_trials:
        return None

    completed["value"] = pd.to_numeric(completed["value"], errors="coerce")
    completed = completed.dropna(subset=["value"])
    if completed.empty:
        return None

    best_row = completed.loc[completed["value"].idxmax()]
    best_params = extract_optuna_params(best_row)
    print(f"Resuming completed {model_name} Optuna study", flush=True)

    best_classifier = build_classifier(
        model_name,
        optuna.trial.FixedTrial(best_params),
        problem_type,
        y_train,
    )
    best_pipeline = make_pipeline(
        classifier=best_classifier,
        feature_names=feature_names,
        full_feature_names=full_feature_names,
        selected_features=selected_features,
        scaler_name=scaler_name,
        engineering_enabled=engineering_enabled,
    )
    best_metrics = parse_stored_metrics(best_row.get("user_attrs_metrics"))
    if best_metrics is None:
        best_metrics = cross_validate_estimator(
            best_pipeline,
            x_train,
            y_train,
            cv,
            problem_type,
            class_labels,
        )
    candidate = CandidateResult(
        name=model_name,
        pipeline=best_pipeline,
        cv_metrics=best_metrics,
        best_params=to_jsonable(best_params),
        candidate_type="optimized_model",
    )
    row = {
        "model": model_name,
        "status": "resumed_completed",
        "best_value": float(best_row["value"]),
        "best_params": json.dumps(to_jsonable(best_params)),
        "f1_mean": best_metrics["f1"]["mean"],
        "recall_mean": best_metrics["recall"]["mean"],
    }
    return candidate, row


def extract_optuna_params(row: pd.Series) -> dict[str, Any]:
    params: dict[str, Any] = {}
    integer_params = {
        "depth",
        "iterations",
        "max_depth",
        "max_iter",
        "max_leaf_nodes",
        "min_child_samples",
        "min_samples_leaf",
        "min_samples_split",
        "n_estimators",
        "num_leaves",
    }
    for column, value in row.items():
        if not str(column).startswith("params_"):
            continue

        name = str(column).removeprefix("params_")
        if pd.isna(value):
            params[name] = None
        elif name in integer_params:
            params[name] = int(value)
        elif isinstance(value, np.generic):
            params[name] = value.item()
        else:
            params[name] = value
    return params


def parse_stored_metrics(value: Any) -> dict[str, dict[str, float]] | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    return {
        str(metric): {
            str(stat): float(number)
            for stat, number in stats.items()
        }
        for metric, stats in parsed.items()
        if isinstance(stats, dict)
    }


def dependency_available(model_name: str) -> bool:
    if model_name == "XGBoost":
        return XGBClassifier is not None
    if model_name == "LightGBM":
        return LGBMClassifier is not None
    if model_name == "CatBoost":
        return CatBoostClassifier is not None
    return True


def build_classifier(
    model_name: str,
    trial: optuna.trial.BaseTrial,
    problem_type: str,
    y_train: np.ndarray,
    enable_probability: bool = True,
) -> BaseEstimator:
    if model_name == "Logistic Regression":
        return LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 100.0, log=True),
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 120, 320, step=40),
            max_depth=trial.suggest_categorical("max_depth", [None, 6, 10, 14, 18]),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 12),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=cpu_jobs(),
        )
    if model_name == "Extra Trees":
        return ExtraTreesClassifier(
            n_estimators=trial.suggest_int("n_estimators", 120, 360, step=40),
            max_depth=trial.suggest_categorical("max_depth", [None, 6, 10, 14, 18]),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 12),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=cpu_jobs(),
        )
    if model_name == "XGBoost":
        if XGBClassifier is None:
            raise ImportError("xgboost is required for the XGBoost candidate.")
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 320, step=40),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
            "objective": "binary:logistic" if problem_type == "binary" else "multi:softprob",
            "eval_metric": "logloss" if problem_type == "binary" else "mlogloss",
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
            "n_jobs": cpu_jobs(),
            "verbosity": 0,
        }
        if problem_type == "binary":
            params["scale_pos_weight"] = binary_scale_pos_weight(y_train)
        else:
            params["num_class"] = len(np.unique(y_train))
        return XGBClassifier(**params)
    if model_name == "LightGBM":
        if LGBMClassifier is None:
            raise ImportError("lightgbm is required for the LightGBM candidate.")
        return LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 80, 360, step=40),
            num_leaves=trial.suggest_int("num_leaves", 15, 63),
            max_depth=trial.suggest_categorical("max_depth", [-1, 4, 6, 10]),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            subsample=trial.suggest_float("subsample", 0.65, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.65, 1.0),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 60),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
            class_weight="balanced",
            objective="binary" if problem_type == "binary" else "multiclass",
            random_state=RANDOM_STATE,
            n_jobs=cpu_jobs(),
            verbose=-1,
        )
    if model_name == "CatBoost":
        if CatBoostClassifier is None:
            raise ImportError("catboost is required for the CatBoost candidate.")
        return CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 80, 240, step=40),
            depth=trial.suggest_int("depth", 3, 5),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 12.0),
            loss_function="Logloss" if problem_type == "binary" else "MultiClass",
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE,
            thread_count=cpu_jobs(),
            verbose=False,
            allow_writing_files=False,
        )
    if model_name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            max_iter=trial.suggest_int("max_iter", 80, 320, step=40),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 10, 63),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 50),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    if model_name == "SVM RBF":
        return SVC(
            C=trial.suggest_float("C", 1e-2, 100.0, log=True),
            gamma=trial.suggest_float("gamma", 1e-4, 1.0, log=True),
            kernel="rbf",
            probability=enable_probability,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    if model_name == "Balanced Random Forest":
        return BalancedRandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 120, 320, step=40),
            max_depth=trial.suggest_categorical("max_depth", [None, 6, 10, 14, 18]),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 12),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            random_state=RANDOM_STATE,
            n_jobs=cpu_jobs(),
        )
    raise ValueError(f"Unsupported model: {model_name}")


def build_ensemble_candidates(
    optimized_candidates: list[CandidateResult],
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    full_feature_names: list[str],
    selected_features: list[str],
    scaler_name: str,
    engineering_enabled: bool,
    problem_type: str,
    class_labels: list[int],
    cv: StratifiedKFold,
) -> list[CandidateResult]:
    required = ["Random Forest", "XGBoost", "CatBoost", "SVM RBF"]
    lookup = {candidate.name: candidate for candidate in optimized_candidates}
    if not all(name in lookup for name in required):
        return []

    base_estimators = [
        ("rf", clone(lookup["Random Forest"].pipeline.named_steps["classifier"])),
        ("xgb", clone(lookup["XGBoost"].pipeline.named_steps["classifier"])),
        ("cat", clone(lookup["CatBoost"].pipeline.named_steps["classifier"])),
        ("svm", clone(lookup["SVM RBF"].pipeline.named_steps["classifier"])),
    ]
    ensembles = [
        (
            "Soft Voting Ensemble",
            VotingClassifier(
                estimators=base_estimators,
                voting="soft",
                n_jobs=1,
            ),
        ),
        (
            "Stacking Ensemble",
            StackingClassifier(
                estimators=base_estimators,
                final_estimator=LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
                stack_method="predict_proba",
                cv=StratifiedKFold(
                    n_splits=CV_FOLDS,
                    shuffle=True,
                    random_state=RANDOM_STATE,
                ),
                n_jobs=1,
            ),
        ),
    ]

    results: list[CandidateResult] = []
    for name, classifier in ensembles:
        print(f"Evaluating {name}", flush=True)
        pipeline = make_pipeline(
            classifier=classifier,
            feature_names=feature_names,
            full_feature_names=full_feature_names,
            selected_features=selected_features,
            scaler_name=scaler_name,
            engineering_enabled=engineering_enabled,
        )
        metrics = cross_validate_estimator(
            pipeline,
            x_train,
            y_train,
            cv,
            problem_type,
            class_labels,
        )
        results.append(
            CandidateResult(
                name=name,
                pipeline=pipeline,
                cv_metrics=metrics,
                best_params={},
                candidate_type="ensemble",
            ),
        )
    return results


def evaluate_calibration_candidates(
    top_candidates: list[CandidateResult],
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    full_feature_names: list[str],
    selected_features: list[str],
    scaler_name: str,
    engineering_enabled: bool,
    problem_type: str,
    class_labels: list[int],
    cv: StratifiedKFold,
) -> tuple[list[CandidateResult], dict[str, Any]]:
    calibrated_results: list[CandidateResult] = []
    rows: list[dict[str, Any]] = []
    for candidate in top_candidates:
        base_classifier = candidate.pipeline.named_steps["classifier"]
        for method in ("sigmoid", "isotonic"):
            print(f"Evaluating {candidate.name} with {method} calibration", flush=True)
            classifier = CalibratedClassifierCV(
                estimator=clone(base_classifier),
                method=method,
                cv=StratifiedKFold(
                    n_splits=min(3, CV_FOLDS),
                    shuffle=True,
                    random_state=RANDOM_STATE,
                ),
            )
            pipeline = make_pipeline(
                classifier=classifier,
                feature_names=feature_names,
                full_feature_names=full_feature_names,
                selected_features=selected_features,
                scaler_name=scaler_name,
                engineering_enabled=engineering_enabled,
            )
            try:
                metrics = cross_validate_estimator(
                    pipeline,
                    x_train,
                    y_train,
                    cv,
                    problem_type,
                    class_labels,
                )
            except Exception as exc:
                rows.append(
                    {
                        "base_model": candidate.name,
                        "method": method,
                        "status": "failed",
                        "error": str(exc),
                    },
                )
                continue

            result = CandidateResult(
                name=f"{candidate.name} Calibrated {method.title()}",
                pipeline=pipeline,
                cv_metrics=metrics,
                best_params=candidate.best_params,
                candidate_type="calibrated_model",
                calibration={"base_model": candidate.name, "method": method},
            )
            calibrated_results.append(result)
            rows.append(
                {
                    "base_model": candidate.name,
                    "method": method,
                    "status": "completed",
                    "roc_auc_mean": metrics.get("roc_auc", {}).get("mean", 0.0),
                    "brier_score_mean": metrics.get("brier_score", {}).get("mean", 1.0),
                    "f1_mean": metrics["f1"]["mean"],
                    "recall_mean": metrics["recall"]["mean"],
                },
            )

    completed = [row for row in rows if row.get("status") == "completed"]
    completed.sort(
        key=lambda row: (
            float(row.get("roc_auc_mean", 0.0)),
            -float(row.get("brier_score_mean", 1.0)),
        ),
        reverse=True,
    )
    return calibrated_results, {
        "selection_basis": "highest ROC-AUC, then lowest Brier score, using training CV",
        "top_calibration": completed[0] if completed else None,
        "calibration_candidates": rows,
    }


def make_pipeline(
    classifier: BaseEstimator,
    feature_names: list[str],
    full_feature_names: list[str],
    selected_features: list[str],
    scaler_name: str,
    engineering_enabled: bool,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "feature_engineer",
                ClinicalFeatureEngineer(feature_names, enabled=engineering_enabled),
            ),
            (
                "feature_selector",
                FeatureNameSelector(full_feature_names, selected_features),
            ),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", build_scaler(scaler_name)),
            ("feature_frame", FeatureNameFrame(selected_features)),
            ("classifier", classifier),
        ],
    )


def build_scaler(scaler_name: str) -> BaseEstimator:
    if scaler_name == "standard":
        return StandardScaler()
    if scaler_name == "robust":
        return RobustScaler()
    raise ValueError(f"Unsupported scaler: {scaler_name}")


def cross_validate_estimator(
    estimator: BaseEstimator,
    x_values: np.ndarray,
    y_values: np.ndarray,
    cv: StratifiedKFold,
    problem_type: str,
    class_labels: list[int],
) -> dict[str, dict[str, float]]:
    fold_rows: list[dict[str, float]] = []
    for train_index, valid_index in cv.split(x_values, y_values):
        model = clone(estimator)
        x_train_fold = x_values[train_index]
        y_train_fold = y_values[train_index]
        x_valid_fold = x_values[valid_index]
        y_valid_fold = y_values[valid_index]
        fit_estimator(model, x_train_fold, y_train_fold)
        probabilities = predict_proba_aligned(model, x_valid_fold, class_labels)
        predictions = predictions_from_probabilities(
            model,
            x_valid_fold,
            probabilities,
            problem_type,
            threshold=None,
            class_labels=class_labels,
        )
        fold_rows.append(
            compute_metrics(
                y_valid_fold,
                predictions,
                probabilities,
                problem_type,
                class_labels,
            ),
        )

    return summarize_fold_metrics(fold_rows)


def fit_estimator(estimator: BaseEstimator, x_values: np.ndarray, y_values: np.ndarray) -> None:
    weights = balanced_weights(y_values)
    try:
        classifier = estimator.named_steps["classifier"] if hasattr(estimator, "named_steps") else estimator
        if supports_sample_weight(classifier):
            estimator.fit(x_values, y_values, classifier__sample_weight=weights)
            return
    except Exception:
        pass
    estimator.fit(x_values, y_values)


def supports_sample_weight(estimator: BaseEstimator) -> bool:
    try:
        return "sample_weight" in inspect.signature(estimator.fit).parameters
    except (TypeError, ValueError):
        return False


def balanced_weights(y_values: np.ndarray) -> np.ndarray:
    return compute_sample_weight(class_weight="balanced", y=y_values)


def predict_proba_aligned(
    model: BaseEstimator,
    x_values: np.ndarray,
    class_labels: list[int],
) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None

    try:
        probabilities = np.asarray(model.predict_proba(x_values), dtype=np.float64)
    except Exception:
        return None

    model_labels = model_classes(model)
    aligned = np.zeros((x_values.shape[0], len(class_labels)), dtype=np.float64)
    for source_index, label in enumerate(model_labels):
        if label in class_labels and source_index < probabilities.shape[1]:
            aligned[:, class_labels.index(label)] = probabilities[:, source_index]

    row_sums = aligned.sum(axis=1, keepdims=True)
    aligned = np.divide(
        aligned,
        np.maximum(row_sums, EPSILON),
        out=aligned,
        where=row_sums > 0,
    )
    return aligned


def model_classes(model: BaseEstimator) -> list[int]:
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier")
        classes = getattr(classifier, "classes_", None)
    if classes is None:
        return list(config.CLASS_LABELS)
    return [int(label) for label in classes]


def predictions_from_probabilities(
    model: BaseEstimator,
    x_values: np.ndarray,
    probabilities: np.ndarray | None,
    problem_type: str,
    threshold: float | None,
    class_labels: list[int],
) -> np.ndarray:
    if probabilities is None:
        return np.asarray(model.predict(x_values), dtype=np.int64)
    if problem_type == "binary" and threshold is not None:
        positive_index = class_labels.index(1) if 1 in class_labels else -1
        negative_label = 0 if 0 in class_labels else class_labels[0]
        positive_label = 1 if 1 in class_labels else class_labels[-1]
        return np.where(
            probabilities[:, positive_index] >= float(threshold),
            positive_label,
            negative_label,
        ).astype(np.int64)
    return np.asarray([class_labels[index] for index in np.argmax(probabilities, axis=1)], dtype=np.int64)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None,
    problem_type: str,
    class_labels: list[int],
) -> dict[str, float]:
    average = "binary" if problem_type == "binary" else "macro"
    labels_present = [label for label in class_labels if np.any(y_true == label)]
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                average=average,
                labels=labels_present if problem_type != "binary" else None,
                zero_division=0,
            ),
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                average=average,
                labels=labels_present if problem_type != "binary" else None,
                zero_division=0,
            ),
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                average=average,
                labels=labels_present if problem_type != "binary" else None,
                zero_division=0,
            ),
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=class_labels)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(y_true, y_pred, labels=class_labels, weights="quadratic"),
        ),
        "mean_absolute_stage_error": float(np.mean(np.abs(y_true - y_pred))),
    }
    if probabilities is not None:
        metrics["roc_auc"] = safe_roc_auc(y_true, probabilities, problem_type, class_labels)
        metrics["brier_score"] = multiclass_brier_score(y_true, probabilities, class_labels)
    else:
        metrics["roc_auc"] = 0.0
        metrics["brier_score"] = 1.0
    return metrics


def summarize_fold_metrics(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted(rows[0])
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
        }
    return summary


def safe_roc_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    problem_type: str,
    class_labels: list[int],
) -> float:
    try:
        if problem_type == "binary":
            positive_index = class_labels.index(1) if 1 in class_labels else -1
            if np.unique(y_true).size < 2:
                return 0.0
            return float(roc_auc_score(y_true, probabilities[:, positive_index]))

        present_labels = [label for label in class_labels if np.any(y_true == label)]
        if len(present_labels) < 2:
            return 0.0
        label_indices = [class_labels.index(label) for label in present_labels]
        return float(
            roc_auc_score(
                y_true,
                probabilities[:, label_indices],
                labels=present_labels,
                multi_class="ovr",
                average="macro",
            ),
        )
    except ValueError:
        return 0.0


def multiclass_brier_score(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_labels: list[int],
) -> float:
    if len(class_labels) == 2 and set(class_labels).issubset({0, 1}):
        positive_index = class_labels.index(1)
        try:
            return float(brier_score_loss(y_true, probabilities[:, positive_index]))
        except ValueError:
            return 1.0
    y_binary = label_binarize(y_true, classes=class_labels)
    if y_binary.shape[1] == 1:
        y_binary = np.column_stack([1 - y_binary[:, 0], y_binary[:, 0]])
    return float(np.mean(np.sum((probabilities - y_binary) ** 2, axis=1)))


def metric_rank_key(metrics: dict[str, dict[str, float]]) -> tuple[float, float, float]:
    return (
        float(metrics["f1"]["mean"]),
        float(metrics["recall"]["mean"]),
        float(metrics.get("quadratic_weighted_kappa", metrics["cohen_kappa"])["mean"]),
    )


def ranked_candidates(candidates: list[CandidateResult]) -> list[CandidateResult]:
    return sorted(candidates, key=lambda candidate: metric_rank_key(candidate.cv_metrics), reverse=True)


def optimize_binary_threshold(
    model: BaseEstimator,
    x_train: np.ndarray,
    y_train: np.ndarray,
    cv: StratifiedKFold,
    class_labels: list[int],
) -> dict[str, Any]:
    probabilities = np.zeros((x_train.shape[0], len(class_labels)), dtype=np.float64)
    for train_index, valid_index in cv.split(x_train, y_train):
        fold_model = clone(model)
        fit_estimator(fold_model, x_train[train_index], y_train[train_index])
        fold_probabilities = predict_proba_aligned(
            fold_model,
            x_train[valid_index],
            class_labels,
        )
        if fold_probabilities is None:
            raise ValueError("Threshold optimization requires probability estimates.")
        probabilities[valid_index] = fold_probabilities

    rows: list[dict[str, Any]] = []
    positive_index = class_labels.index(1)
    for threshold in BINARY_THRESHOLDS:
        predictions = np.where(probabilities[:, positive_index] >= threshold, 1, 0)
        matrix = confusion_matrix(y_train, predictions, labels=class_labels)
        false_negatives = int(matrix[class_labels.index(1), class_labels.index(0)])
        row = {
            "threshold": float(threshold),
            "recall": float(recall_score(y_train, predictions, zero_division=0)),
            "f1": float(f1_score(y_train, predictions, zero_division=0)),
            "precision": float(precision_score(y_train, predictions, zero_division=0)),
            "false_negatives": false_negatives,
            "false_positives": int(matrix[class_labels.index(0), class_labels.index(1)]),
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["recall"],
            row["f1"],
            -row["false_negatives"],
            -row["false_positives"],
        ),
        reverse=True,
    )
    best = rows[0]
    return {
        "problem_type": "binary",
        "threshold": float(best["threshold"]),
        "selection_rule": "maximize recall, then F1-score, then minimize false negatives",
        "threshold_candidates": rows,
    }


def evaluate_holdout_model(
    model: BaseEstimator,
    x_test: np.ndarray,
    y_test: np.ndarray,
    sample_indices: np.ndarray,
    problem_type: str,
    class_labels: list[int],
    threshold: float | None,
    output_dir: Path,
    split_name: str = "test",
) -> dict[str, Any]:
    probabilities = predict_proba_aligned(model, x_test, class_labels)
    predictions = predictions_from_probabilities(
        model,
        x_test,
        probabilities,
        problem_type,
        threshold,
        class_labels,
    )
    metrics = compute_metrics(y_test, predictions, probabilities, problem_type, class_labels)
    metrics["split"] = split_name
    matrix = confusion_matrix(y_test, predictions, labels=class_labels)
    metrics["confusion_matrix"] = matrix.astype(int).tolist()
    metrics["classification_report"] = classification_report(
        y_test,
        predictions,
        labels=class_labels,
        zero_division=0,
        output_dict=True,
    )
    prefix = "" if split_name == "test" else f"{safe_name(split_name)}_"
    save_confusion_matrix(matrix, class_labels, output_dir / f"{prefix}confusion_matrix.csv")
    save_misclassified_cases(
        sample_indices,
        y_test,
        predictions,
        probabilities,
        class_labels,
        output_dir / f"{prefix}misclassified_cases.csv",
    )
    confusion_summary = build_confusion_summary(matrix, class_labels)
    save_json(output_dir / f"{prefix}confusion_summary.json", confusion_summary)
    metrics["confusion_summary"] = confusion_summary
    return metrics


def save_misclassified_cases(
    sample_indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None,
    class_labels: list[int],
    output_path: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for row_index, (sample_index, true_label, predicted_label) in enumerate(
        zip(sample_indices, y_true, y_pred),
    ):
        if int(true_label) == int(predicted_label):
            continue
        probability = 0.0
        confidence = 0.0
        row: dict[str, Any] = {
            "sample_index": int(sample_index),
            "true_label": int(true_label),
            "predicted_label": int(predicted_label),
        }
        if probabilities is not None:
            predicted_index = class_labels.index(int(predicted_label))
            probability = float(probabilities[row_index, predicted_index])
            confidence = float(np.max(probabilities[row_index]))
            for class_index, label in enumerate(class_labels):
                row[f"probability_class_{label}"] = float(probabilities[row_index, class_index])
        row["probability"] = probability
        row["confidence_score"] = confidence
        rows.append(row)

    fieldnames = [
        "sample_index",
        "true_label",
        "predicted_label",
        "confidence_score",
        "probability",
        *[f"probability_class_{label}" for label in class_labels],
    ]
    write_rows(output_path, rows, fieldnames=fieldnames)


def build_confusion_summary(matrix: np.ndarray, class_labels: list[int]) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for true_index, true_label in enumerate(class_labels):
        for pred_index, pred_label in enumerate(class_labels):
            if true_label == pred_label:
                continue
            count = int(matrix[true_index, pred_index])
            if count > 0:
                pairs.append(
                    {
                        "true_label": int(true_label),
                        "predicted_label": int(pred_label),
                        "count": count,
                        "stage_distance": int(abs(true_label - pred_label)),
                    },
                )
    pairs.sort(key=lambda row: (row["count"], row["stage_distance"]), reverse=True)
    return {
        "most_common_confusions": pairs[:10],
        "severity_aware_note": (
            "Quadratic weighted kappa and mean absolute stage error penalize "
            "larger DR-stage distances more strongly than adjacent-stage errors."
        ),
    }


def build_interpretability_artifacts(
    model: BaseEstimator,
    x_reference: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    problem_type: str,
    class_labels: list[int],
    output_dir: Path,
) -> tuple[Any, list[dict[str, Any]]]:
    transformed_reference, selected_names, classifier = transformed_matrix_and_classifier(
        model,
        x_reference,
    )
    transformed_eval, _, _ = transformed_matrix_and_classifier(model, x_eval)

    explainer = build_shap_explainer(classifier, transformed_reference)
    shap_scores = np.zeros(len(selected_names), dtype=np.float64)
    if explainer is not None:
        shap_sample = deterministic_sample(transformed_eval, MAX_SHAP_EVAL_SAMPLES)
        try:
            if hasattr(explainer, "shap_values"):
                raw_values = explainer.shap_values(shap_sample)
            else:
                explanation = explainer(shap_sample)
                raw_values = getattr(explanation, "values", explanation)
            shap_scores = mean_abs_shap_by_feature(raw_values)
        except Exception as exc:
            save_json(output_dir / "shap_error.json", {"error": str(exc)})

    permutation_scores, permutation_stds = permutation_scores_for_classifier(
        classifier,
        transformed_eval,
        y_eval,
        problem_type,
    )
    model_scores = model_native_importance(classifier, len(selected_names))
    rows = [
        {
            "feature": str(name),
            "shap_mean_abs": float(shap_scores[index]) if index < len(shap_scores) else 0.0,
            "permutation_importance_mean": float(permutation_scores[index])
            if index < len(permutation_scores)
            else 0.0,
            "permutation_importance_std": float(permutation_stds[index])
            if index < len(permutation_stds)
            else 0.0,
            "model_native_importance": float(model_scores[index])
            if index < len(model_scores)
            else 0.0,
        }
        for index, name in enumerate(selected_names)
    ]
    rows.sort(
        key=lambda row: (
            row["shap_mean_abs"],
            row["permutation_importance_mean"],
            row["model_native_importance"],
        ),
        reverse=True,
    )
    return explainer, rows


def transformed_matrix_and_classifier(
    model: BaseEstimator,
    x_values: np.ndarray,
) -> tuple[Any, list[str], BaseEstimator]:
    if not hasattr(model, "steps"):
        return x_values, list(config.FEATURE_NAMES), model

    values = x_values
    names = list(config.FEATURE_NAMES)
    classifier = model.named_steps["classifier"]
    for step_name, step in model.steps:
        if step_name == "classifier":
            break
        values = step.transform(values)
        if hasattr(step, "get_feature_names_out"):
            names = [str(name) for name in step.get_feature_names_out(names)]
    return values, names, classifier


def build_shap_explainer(classifier: BaseEstimator, x_background: np.ndarray) -> Any | None:
    background = deterministic_sample(x_background, MAX_SHAP_BACKGROUND)
    try:
        if is_tree_based(classifier):
            return shap.TreeExplainer(classifier)
        summary = shap.kmeans(background, min(MAX_SHAP_BACKGROUND, background.shape[0]))
        return shap.KernelExplainer(classifier.predict_proba, summary)
    except Exception:
        try:
            masker = shap.maskers.Independent(background)
            return shap.Explainer(classifier.predict_proba, masker, seed=RANDOM_STATE)
        except Exception:
            return None


def is_tree_based(classifier: BaseEstimator) -> bool:
    if isinstance(classifier, CalibratedClassifierCV):
        return False
    name = classifier.__class__.__name__.lower()
    tree_tokens = ("forest", "xgb", "lgbm", "catboost", "gradientboosting", "extratrees")
    return hasattr(classifier, "feature_importances_") or any(token in name for token in tree_tokens)


def mean_abs_shap_by_feature(raw_values: Any) -> np.ndarray:
    if isinstance(raw_values, list):
        arrays = [np.asarray(values, dtype=np.float64) for values in raw_values]
        return np.mean([np.mean(np.abs(values), axis=0) for values in arrays], axis=0)

    values = np.asarray(raw_values, dtype=np.float64)
    if values.ndim == 3:
        return np.mean(np.abs(values), axis=(0, 2))
    if values.ndim == 2:
        return np.mean(np.abs(values), axis=0)
    return np.abs(values)


def permutation_scores_for_classifier(
    classifier: BaseEstimator,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    problem_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    scoring = "f1" if problem_type == "binary" else "f1_macro"
    try:
        result = permutation_importance(
            classifier,
            x_eval,
            y_eval,
            scoring=scoring,
            n_repeats=8,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        return result.importances_mean, result.importances_std
    except Exception:
        zeros = np.zeros(x_eval.shape[1], dtype=np.float64)
        return zeros, zeros


def model_native_importance(classifier: BaseEstimator, feature_count: int) -> np.ndarray:
    importances = getattr(classifier, "feature_importances_", None)
    if importances is not None:
        return np.asarray(importances, dtype=np.float64)

    coefficients = getattr(classifier, "coef_", None)
    if coefficients is not None:
        values = np.asarray(coefficients, dtype=np.float64)
        if values.ndim == 2:
            return np.mean(np.abs(values), axis=0)
        return np.abs(values)

    return np.zeros(feature_count, dtype=np.float64)


def save_pipeline_artifacts(model: BaseEstimator, output_dir: Path) -> None:
    if not hasattr(model, "named_steps"):
        return
    for step_name, filename in (
        ("imputer", "imputer.pkl"),
        ("scaler", "scaler.pkl"),
    ):
        step = model.named_steps.get(step_name)
        if step is not None:
            save_pickle(output_dir / filename, step)


def save_feature_importance(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "feature",
        "shap_mean_abs",
        "permutation_importance_mean",
        "permutation_importance_std",
        "model_native_importance",
    ]
    write_rows(output_path, rows, fieldnames=fieldnames)


def save_feature_rankings(
    rankings: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for method, ranking in rankings.items():
        for row in ranking:
            rows.append(
                {
                    "method": method,
                    "rank": row["rank"],
                    "feature": row["feature"],
                    "score": row["score"],
                },
            )
    write_rows(output_path, rows)


def save_feature_selection_experiments(
    experiments: list[dict[str, Any]],
    output_path: Path,
) -> None:
    rows = [
        {
            "method": experiment["method"],
            "requested_count": experiment["requested_count"],
            "selected_feature_count": experiment["selected_feature_count"],
            "f1_mean": experiment["metrics"]["f1"]["mean"],
            "f1_std": experiment["metrics"]["f1"]["std"],
            "recall_mean": experiment["metrics"]["recall"]["mean"],
            "recall_std": experiment["metrics"]["recall"]["std"],
            "roc_auc_mean": experiment["metrics"]["roc_auc"]["mean"],
            "balanced_accuracy_mean": experiment["metrics"]["balanced_accuracy"]["mean"],
            "cohen_kappa_mean": experiment["metrics"]["cohen_kappa"]["mean"],
            "quadratic_weighted_kappa_mean": experiment["metrics"]["quadratic_weighted_kappa"]["mean"],
        }
        for experiment in experiments
    ]
    write_rows(output_path, rows)


def save_model_comparison(candidates: list[CandidateResult], output_path: Path) -> None:
    rows = [candidate_summary(candidate, rank=index + 1) for index, candidate in enumerate(candidates)]
    write_rows(output_path, rows)


def candidate_summary(candidate: CandidateResult, rank: int) -> dict[str, Any]:
    metrics = candidate.cv_metrics
    return {
        "rank": int(rank),
        "model": candidate.name,
        "candidate_type": candidate.candidate_type,
        "f1_mean": metrics["f1"]["mean"],
        "f1_std": metrics["f1"]["std"],
        "recall_mean": metrics["recall"]["mean"],
        "recall_std": metrics["recall"]["std"],
        "precision_mean": metrics["precision"]["mean"],
        "accuracy_mean": metrics["accuracy"]["mean"],
        "roc_auc_mean": metrics["roc_auc"]["mean"],
        "balanced_accuracy_mean": metrics["balanced_accuracy"]["mean"],
        "cohen_kappa_mean": metrics["cohen_kappa"]["mean"],
        "quadratic_weighted_kappa_mean": metrics["quadratic_weighted_kappa"]["mean"],
        "brier_score_mean": metrics["brier_score"]["mean"],
        "best_params": json.dumps(to_jsonable(candidate.best_params)),
        "calibration": json.dumps(to_jsonable(candidate.calibration or {})),
    }


def save_optuna_trials(study: optuna.Study, output_path: Path) -> None:
    dataframe = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    dataframe.to_csv(output_path, index=False)


def initialize_optuna_progress(output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "trial",
                "state",
                "value",
                "best_value",
                "f1",
                "recall",
                "params",
            ],
        )
        writer.writeheader()


def append_optuna_progress(
    output_path: Path,
    study: optuna.Study,
    trial: optuna.trial.FrozenTrial,
) -> None:
    metrics = trial.user_attrs.get("metrics", {})
    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "trial",
                "state",
                "value",
                "best_value",
                "f1",
                "recall",
                "params",
            ],
        )
        writer.writerow(
            {
                "trial": int(trial.number),
                "state": str(trial.state.name),
                "value": none_or_float(trial.value),
                "best_value": best_study_value(study),
                "f1": nested_metric(metrics, "f1"),
                "recall": nested_metric(metrics, "recall"),
                "params": json.dumps(to_jsonable(trial.params)),
            },
        )


def nested_metric(metrics: Any, name: str) -> float | None:
    if not isinstance(metrics, dict):
        return None
    payload = metrics.get(name)
    if not isinstance(payload, dict):
        return None
    return none_or_float(payload.get("mean"))


def none_or_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def best_study_value(study: optuna.Study) -> float | None:
    try:
        return none_or_float(study.best_value)
    except ValueError:
        return None


def save_confusion_matrix(matrix: np.ndarray, labels: list[int], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, matrix.tolist()):
            writer.writerow([label, *row])


def write_rows(
    output_path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    ensure_dir(output_path.parent)
    if not rows and fieldnames is None:
        output_path.write_text("", encoding="utf-8")
        return
    names = fieldnames or list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(to_jsonable(rows))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def save_pickle(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("wb") as file:
        pickle.dump(payload, file)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def deterministic_sample(values: np.ndarray, max_rows: int) -> np.ndarray:
    if values.shape[0] <= max_rows:
        return values
    rng = np.random.default_rng(RANDOM_STATE)
    indices = np.sort(rng.choice(values.shape[0], size=max_rows, replace=False))
    if isinstance(values, pd.DataFrame):
        return values.iloc[indices].reset_index(drop=True)
    return values[indices]


def binary_scale_pos_weight(y_values: np.ndarray) -> float:
    negative = max(int(np.sum(y_values == 0)), 1)
    positive = max(int(np.sum(y_values == 1)), 1)
    return float(negative / positive)


def cpu_jobs() -> int:
    return max(1, min(4, (os.cpu_count() or 2) - 1))


def safe_name(name: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in name).strip("_")


def artifact_path_report() -> dict[str, str]:
    return {
        "best_model": "results/best_model.pkl",
        "imputer": "results/imputer.pkl",
        "scaler": "results/scaler.pkl",
        "explainer": "results/explainer.pkl",
        "selected_features": "results/selected_features.json",
        "feature_order": "results/feature_order.json",
        "metrics": "results/metrics.json",
        "optimal_threshold": "results/optimal_threshold.json",
        "class_distribution": "results/class_distribution.json",
        "data_quality_report": "results/data_quality_report.json",
        "feature_importance": "results/feature_importance.csv",
        "misclassified_cases": "results/misclassified_cases.csv",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the ML-only AppDR handcrafted-feature classifier.",
    )
    parser.add_argument("--features-csv", type=Path, default=config.FEATURES_CSV)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--trials", type=int, default=MIN_OPTUNA_TRIALS)
    parser.add_argument(
        "--binary-referable",
        action="store_true",
        help=(
            "Collapse multiclass DR grades into binary screening labels: "
            "grades 0-1 -> Non-Referable (0), grades 2-4 -> Referable (1)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse already completed Optuna trial CSVs from the results directory.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Allow a short validation run with fewer than 50 Optuna trials.",
    )
    parser.add_argument(
        "--skip-interpretability",
        action="store_true",
        help="Skip slow SHAP/permutation artifacts while still exporting the model and metrics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    results_dir = args.results_dir
    if results_dir is None:
        results_dir = config.RESULTS_DIR / "binary" if args.binary_referable else config.RESULTS_DIR
    train_models(
        features_csv=args.features_csv,
        results_dir=results_dir,
        n_trials=args.trials,
        binary_referable=args.binary_referable,
        resume_completed=args.resume,
        smoke=args.smoke,
        skip_interpretability=args.skip_interpretability,
    )
