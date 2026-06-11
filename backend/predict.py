"""Production prediction utilities for the AppDR handcrafted-feature model.

The public entry point is ``predict(features_203: list) -> dict``.  The
function expects the existing 203 retinal features in the order defined by
``config.FEATURE_NAMES`` and returns JSON-serializable screening outputs.
"""

from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import config


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
MODEL_PATH = RESULTS_DIR / "best_model.pkl"
EXPLAINER_PATH = RESULTS_DIR / "explainer.pkl"
FEATURE_ORDER_PATH = RESULTS_DIR / "feature_order.json"
SELECTED_FEATURES_PATH = RESULTS_DIR / "selected_features.json"
OPTIMAL_THRESHOLD_PATH = RESULTS_DIR / "optimal_threshold.json"
CONFIDENCE_REVIEW_THRESHOLD = 0.60
RANDOM_STATE = 42

ENGINEERED_FEATURE_NAMES = [
    "clinical_lesion_burden_score",
    "clinical_exudate_burden_score",
    "clinical_hemorrhage_burden_score",
    "clinical_vessel_abnormality_score",
    "clinical_microaneurysm_density_score",
    "clinical_exudate_to_microaneurysm_ratio",
    "clinical_hemorrhage_to_vessel_ratio",
    "clinical_total_lesion_count",
    "clinical_total_lesion_area",
    "clinical_referable_pathology_score",
]


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add clinically meaningful ratios and burden scores from known features."""

    def __init__(
        self,
        input_feature_names: list[str] | None = None,
        enabled: bool = True,
    ) -> None:
        self.input_feature_names = input_feature_names
        self.enabled = enabled

    def fit(self, X: Any, y: Any = None) -> "ClinicalFeatureEngineer":
        self.input_feature_names_ = list(
            self.input_feature_names or config.FEATURE_NAMES,
        )
        self.output_feature_names_ = list(self.input_feature_names_)
        if self.enabled:
            self.output_feature_names_.extend(ENGINEERED_FEATURE_NAMES)
        return self

    def transform(self, X: Any) -> np.ndarray:
        feature_names = list(
            getattr(self, "input_feature_names_", self.input_feature_names or config.FEATURE_NAMES),
        )
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("Feature matrix must be two-dimensional.")
        if values.shape[1] != len(feature_names):
            raise ValueError(
                f"Expected {len(feature_names)} input features, got {values.shape[1]}.",
            )
        if not self.enabled:
            return values

        engineered = build_clinical_engineered_features(values, feature_names)
        return np.column_stack([values, engineered])

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if hasattr(self, "output_feature_names_"):
            return np.asarray(self.output_feature_names_, dtype=object)

        names = list(self.input_feature_names or config.FEATURE_NAMES)
        if self.enabled:
            names.extend(ENGINEERED_FEATURE_NAMES)
        return np.asarray(names, dtype=object)


class FeatureNameSelector(BaseEstimator, TransformerMixin):
    """Select deployment features by stable feature name."""

    def __init__(
        self,
        available_feature_names: list[str] | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        self.available_feature_names = available_feature_names
        self.selected_features = selected_features

    def fit(self, X: Any, y: Any = None) -> "FeatureNameSelector":
        available = list(self.available_feature_names or config.FEATURE_NAMES)
        selected = list(self.selected_features or available)
        lookup = {name: index for index, name in enumerate(available)}
        missing = [name for name in selected if name not in lookup]
        if missing:
            raise ValueError(f"Selected features are unavailable: {missing}")
        self.available_feature_names_ = available
        self.selected_features_ = selected
        self.selected_indices_ = np.asarray([lookup[name] for name in selected], dtype=int)
        return self

    def transform(self, X: Any) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("Feature matrix must be two-dimensional.")
        indices = getattr(self, "selected_indices_", None)
        if indices is None:
            self.fit(values)
            indices = self.selected_indices_
        return values[:, indices]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        selected = getattr(self, "selected_features_", self.selected_features)
        if selected is None:
            selected = self.available_feature_names or config.FEATURE_NAMES
        return np.asarray(list(selected), dtype=object)


class FeatureNameFrame(BaseEstimator, TransformerMixin):
    """Emit a DataFrame with stable names for estimators that track columns."""

    def __init__(self, feature_names: list[str] | None = None) -> None:
        self.feature_names = feature_names

    def fit(self, X: Any, y: Any = None) -> "FeatureNameFrame":
        self.feature_names_ = list(self.feature_names or config.FEATURE_NAMES)
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("Feature matrix must be two-dimensional.")
        feature_names = list(getattr(self, "feature_names_", self.feature_names or []))
        if values.shape[1] != len(feature_names):
            raise ValueError(
                f"Expected {len(feature_names)} named features, got {values.shape[1]}."
            )
        return pd.DataFrame(values, columns=feature_names)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(
            list(getattr(self, "feature_names_", self.feature_names or config.FEATURE_NAMES)),
            dtype=object,
        )


def build_clinical_engineered_features(
    values: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    """Create deterministic clinical burden features from the 203 inputs."""
    lookup = {name: index for index, name in enumerate(feature_names)}

    def col(name: str) -> np.ndarray:
        index = lookup.get(name)
        if index is None:
            return np.zeros(values.shape[0], dtype=np.float64)
        return np.nan_to_num(values[:, index], nan=0.0, posinf=0.0, neginf=0.0)

    ma_count = col("ma_count")
    ma_area = col("ma_area")
    ma_density = col("ma_density") + col("ma_density_per_retinal_area")
    exudate_count = col("exudate_count") + col("hard_exudate_count") + col("soft_exudate_count")
    exudate_area = col("exudate_area") + col("hard_exudate_area") + col("soft_exudate_area")
    exudate_density = col("exudate_density") + col("hard_exudate_coverage_pct")
    hemorrhage_count = col("hemorrhage_count")
    hemorrhage_area = col("hemorrhage_area")
    hemorrhage_density = col("hemorrhage_density")
    cotton_wool_count = col("cotton_wool_count")
    cotton_wool_area = col("cotton_wool_area")
    vessel_density = col("vessel_density") + col("vessel_area_ratio")
    vessel_length = col("vessel_length") + col("vessel_skeleton_length")
    vessel_complexity = (
        col("vessel_complexity_score")
        + col("vessel_fragmentation_index")
        + col("vessel_tortuosity_mean")
        + col("vessel_tortuosity_std")
    )
    advanced_signal = (
        col("advanced_dr_indicator_score")
        + col("neovascularization_likelihood_score")
        + col("referable_lesion_score")
        + col("stage_progression_score")
    )

    total_lesion_count = ma_count + exudate_count + hemorrhage_count + cotton_wool_count
    total_lesion_area = ma_area + exudate_area + hemorrhage_area + cotton_wool_area
    lesion_burden = np.log1p(np.maximum(total_lesion_area, 0.0)) + np.log1p(
        np.maximum(total_lesion_count, 0.0),
    )
    exudate_burden = np.log1p(np.maximum(exudate_area, 0.0)) + exudate_density
    hemorrhage_burden = np.log1p(np.maximum(hemorrhage_area, 0.0)) + hemorrhage_density
    vessel_abnormality = vessel_complexity + safe_ratio(hemorrhage_area, vessel_length)
    ma_density_score = ma_density + safe_ratio(ma_count, total_lesion_count)
    exudate_to_ma = safe_ratio(exudate_area, ma_area) + safe_ratio(exudate_count, ma_count)
    hemorrhage_to_vessel = safe_ratio(hemorrhage_area, vessel_length) + safe_ratio(
        hemorrhage_count,
        vessel_density,
    )
    referable_pathology = (
        lesion_burden
        + 1.4 * hemorrhage_burden
        + 1.2 * exudate_burden
        + 1.6 * advanced_signal
    )

    return np.column_stack(
        [
            lesion_burden,
            exudate_burden,
            hemorrhage_burden,
            vessel_abnormality,
            ma_density_score,
            exudate_to_ma,
            hemorrhage_to_vessel,
            total_lesion_count,
            total_lesion_area,
            referable_pathology,
        ],
    )


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        np.maximum(np.abs(denominator), 1.0),
        out=np.zeros_like(numerator, dtype=np.float64),
        where=np.isfinite(denominator),
    )


def predict(features_203: list) -> dict:
    """Predict DR class from exactly 203 extracted retinal features."""
    feature_order = load_feature_order()
    feature_vector = validate_features(features_203, expected_length=len(feature_order))
    model = load_model()
    probabilities = predict_probabilities(model, feature_vector)
    problem_type = infer_problem_type(probabilities)
    threshold = load_optimal_threshold()

    if problem_type == "binary" and threshold is not None:
        predicted_class = binary_class_from_threshold(probabilities, threshold)
    else:
        predicted_class = max(probabilities, key=probabilities.get)

    prediction_probability = float(probabilities.get(int(predicted_class), 0.0))
    confidence_score = prediction_probability
    review_recommendation = (
        "Manual Review Recommended"
        if confidence_score < CONFIDENCE_REVIEW_THRESHOLD
        else "Clinical Review Required"
    )
    positive, negative = explain_local_prediction(model, feature_vector, int(predicted_class))

    return {
        "predicted_class": int(predicted_class),
        "prediction_probability": float(prediction_probability),
        "confidence_score": float(confidence_score),
        "review_recommendation": str(review_recommendation),
        "probabilities": {
            str(int(label)): float(probability)
            for label, probability in sorted(probabilities.items())
        },
        "top_positive_contributing_features": positive,
        "top_negative_contributing_features": negative,
    }


def validate_features(features: list, expected_length: int) -> np.ndarray:
    if not isinstance(features, list):
        raise ValueError("features_203 must be a list of numeric values.")
    if len(features) != expected_length:
        raise ValueError(f"Expected {expected_length} features, received {len(features)}.")

    try:
        values = np.asarray([float(value) for value in features], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("All features must be numeric.") from exc

    if not np.all(np.isfinite(values)):
        raise ValueError("Feature vector contains NaN or infinite values.")

    return values.reshape(1, -1)


@lru_cache(maxsize=1)
def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


@lru_cache(maxsize=1)
def load_explainer() -> Any | None:
    if not EXPLAINER_PATH.exists():
        return None
    with EXPLAINER_PATH.open("rb") as file:
        return pickle.load(file)


@lru_cache(maxsize=1)
def load_feature_order() -> list[str]:
    payload = load_json(FEATURE_ORDER_PATH, default={})
    names = payload.get("feature_order") if isinstance(payload, dict) else None
    if isinstance(names, list) and all(isinstance(name, str) for name in names):
        return list(names)
    return list(config.FEATURE_NAMES)


@lru_cache(maxsize=1)
def load_selected_features() -> list[str]:
    payload = load_json(SELECTED_FEATURES_PATH, default={})
    names = payload.get("selected_features") if isinstance(payload, dict) else None
    if isinstance(names, list) and all(isinstance(name, str) for name in names):
        return list(names)
    return list(config.FEATURE_NAMES)


@lru_cache(maxsize=1)
def load_optimal_threshold() -> float | None:
    payload = load_json(OPTIMAL_THRESHOLD_PATH, default={})
    value = payload.get("threshold") if isinstance(payload, dict) else None
    if value is None:
        return None
    return float(value)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def predict_probabilities(model: Any, feature_vector: np.ndarray) -> dict[int, float]:
    if not hasattr(model, "predict_proba"):
        predicted = int(model.predict(feature_vector)[0])
        return {predicted: 1.0}

    probabilities = np.asarray(model.predict_proba(feature_vector)[0], dtype=np.float64)
    classes = model_classes(model)
    return {
        int(label): float(probability)
        for label, probability in zip(classes, probabilities)
    }


def model_classes(model: Any) -> list[int]:
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier")
        classes = getattr(classifier, "classes_", None)
    if classes is None:
        return [0, 1]
    return [int(label) for label in classes]


def infer_problem_type(probabilities: dict[int, float]) -> str:
    return "binary" if len(probabilities) == 2 and set(probabilities).issubset({0, 1}) else "multiclass"


def binary_class_from_threshold(probabilities: dict[int, float], threshold: float) -> int:
    positive_label = 1 if 1 in probabilities else max(probabilities)
    negative_label = 0 if 0 in probabilities else min(probabilities)
    return int(positive_label if probabilities.get(positive_label, 0.0) >= threshold else negative_label)


def explain_local_prediction(
    model: Any,
    feature_vector: np.ndarray,
    predicted_class: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    explainer = load_explainer()
    if explainer is None:
        return [], []

    transformed, feature_names = transform_for_explanation(model, feature_vector)
    try:
        if hasattr(explainer, "shap_values"):
            raw_values = explainer.shap_values(transformed)
        else:
            explanation = explainer(transformed)
            raw_values = getattr(explanation, "values", explanation)
        shap_values = extract_class_shap_values(
            raw_values,
            predicted_class=predicted_class,
            classes=model_classes(model),
        )
    except Exception:
        return [], []

    rows = [
        {
            "feature": str(feature_names[index]),
            "feature_value": float(transformed[0, index]),
            "shap_value": float(value),
        }
        for index, value in enumerate(shap_values)
        if np.isfinite(value)
    ]
    positives = sorted(
        [row for row in rows if row["shap_value"] > 0.0],
        key=lambda row: row["shap_value"],
        reverse=True,
    )[:5]
    negatives = sorted(
        [row for row in rows if row["shap_value"] < 0.0],
        key=lambda row: row["shap_value"],
    )[:5]
    return positives, negatives


def transform_for_explanation(model: Any, feature_vector: np.ndarray) -> tuple[np.ndarray, list[str]]:
    if not hasattr(model, "steps"):
        return feature_vector, load_selected_features()

    values = feature_vector
    feature_names = load_feature_order()
    for step_name, step in model.steps:
        if step_name == "classifier":
            break
        values = step.transform(values)
        if hasattr(step, "get_feature_names_out"):
            feature_names = [str(name) for name in step.get_feature_names_out(feature_names)]

    return np.asarray(values, dtype=np.float64), feature_names


def extract_class_shap_values(
    raw_values: Any,
    predicted_class: int,
    classes: list[int],
) -> np.ndarray:
    class_index = classes.index(predicted_class) if predicted_class in classes else 0

    if isinstance(raw_values, list):
        values = np.asarray(raw_values[class_index], dtype=np.float64)
        return values[0] if values.ndim == 2 else values

    values = np.asarray(raw_values, dtype=np.float64)
    if values.ndim == 3:
        if values.shape[2] == len(classes):
            return values[0, :, class_index]
        return values[class_index, 0, :]
    if values.ndim == 2:
        return values[0]
    return values


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict DR class from a JSON feature vector.")
    parser.add_argument("features_json", help="JSON list containing exactly 203 numeric features.")
    args = parser.parse_args()
    parsed_features = json.loads(args.features_json)
    print(json.dumps(predict(parsed_features), indent=2))
