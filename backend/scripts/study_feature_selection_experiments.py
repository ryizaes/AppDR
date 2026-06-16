"""Study-style classical ML feature audit and selection experiments for AppDR.

This script intentionally stays inside the existing 203 handcrafted-feature
pipeline. It does not use CNNs, learned image embeddings, or deep segmentation.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import pickle
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.naive_bayes import GaussianNB

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
except Exception:  # pragma: no cover - optional dependency fallback.
    BalancedRandomForestClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency fallback.
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - optional dependency fallback.
    CatBoostClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency in some installs.
    XGBClassifier = None

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config


RESULTS_DIR = BACKEND_DIR / "results" / "study_feature_selection"
RANDOM_STATE = config.RANDOM_STATE
FEATURE_COUNTS: tuple[int | str, ...] = (50, 75, 100, 150, "all_203")
MODELS = ("logistic_regression", "random_forest", "xgboost", "svm_rbf")
QUALITY_FEATURE_HINTS = ("quality_", "brightness", "blur", "sharpness", "contrast", "snr")
BINARY_THRESHOLDS = np.round(np.arange(0.30, 0.71, 0.05), 2)
PRODUCTION_BASELINE = {
    "multiclass_accuracy": 0.6798,
    "multiclass_balanced_accuracy": 0.5312,
    "multiclass_macro_f1": 0.5077,
    "class_1_recall": 0.3299,
    "class_3_recall": 0.3095,
    "class_4_recall": 0.6151,
    "binary_accuracy": 0.7932,
    "binary_balanced_accuracy": 0.8087,
    "binary_f1": 0.7995,
    "binary_referable_recall": 0.9373,
    "binary_false_negatives": 88,
}
BALANCED_BASELINE = {
    "multiclass_accuracy": 0.5457,
    "multiclass_balanced_accuracy": 0.5721,
    "multiclass_macro_f1": 0.4790,
    "class_1_recall": 0.5733,
    "class_3_recall": 0.6800,
    "class_4_recall": 0.6300,
    "binary_accuracy": 0.7468,
    "binary_balanced_accuracy": 0.7648,
    "binary_f1": 0.7762,
    "binary_referable_recall": 0.9646,
    "binary_false_negatives": 56,
}


@dataclass
class ExperimentResult:
    problem: str
    model_name: str
    feature_set: str
    feature_count: int
    metrics: dict[str, Any]
    selected_features: list[str]
    model_path: str


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    backup_production_artifacts(output_dir)

    table, feature_names = load_feature_table(args.features_csv)
    audit = audit_features(table, feature_names)
    write_json(output_dir / "feature_audit.json", audit)
    write_feature_audit_csv(output_dir / "feature_audit_removed_features.csv", audit)

    feature_frame = table[feature_names].apply(pd.to_numeric, errors="coerce")
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    labels = table["label"].astype(int).to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        feature_frame,
        labels,
        test_size=config.TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    importance = compute_feature_importance(
        x_train,
        y_train,
        x_test,
        y_test,
        feature_names,
        audit,
        args.skip_shap,
    )
    write_importance(output_dir / "feature_importance_study.csv", importance)

    ranked_features = build_ranked_features(importance, audit, x_train)
    write_json(output_dir / "ranked_feature_sets.json", ranked_features)

    multiclass_results = run_multiclass_experiments(
        x_train,
        x_test,
        y_train,
        y_test,
        ranked_features,
        output_dir,
        args.svm_train_limit,
    )
    binary_results, threshold_report = run_binary_experiments(
        x_train,
        x_test,
        y_train,
        y_test,
        ranked_features,
        output_dir,
        args.svm_train_limit,
    )
    subclassifier_report = run_class_3_vs_4_experiment(
        x_train,
        x_test,
        y_train,
        y_test,
        ranked_features,
        multiclass_results,
        output_dir,
    )

    best_multiclass = select_best_multiclass(multiclass_results)
    best_binary = select_best_binary(binary_results)
    replacement = decide_replacement(best_multiclass, best_binary)
    report = build_report(
        audit=audit,
        importance=importance,
        ranked_features=ranked_features,
        multiclass_results=multiclass_results,
        binary_results=binary_results,
        threshold_report=threshold_report,
        subclassifier_report=subclassifier_report,
        best_multiclass=best_multiclass,
        best_binary=best_binary,
        replacement=replacement,
        features_csv=args.features_csv,
    )
    write_outputs(output_dir, report, multiclass_results, binary_results)
    print(report["markdown"])


def load_feature_table(features_csv: Path) -> tuple[pd.DataFrame, list[str]]:
    table = pd.read_csv(features_csv)
    missing = [name for name in config.FEATURE_NAMES if name not in table.columns]
    if missing:
        raise ValueError(f"Feature table is missing {len(missing)} expected features.")
    if "label" not in table.columns:
        raise ValueError("Feature table must include a label column.")
    return table, list(config.FEATURE_NAMES)


def audit_features(table: pd.DataFrame, feature_names: list[str]) -> dict[str, Any]:
    features = table[feature_names].apply(pd.to_numeric, errors="coerce")
    values = features.to_numpy(dtype=np.float64)
    finite = np.isfinite(values)
    nonfinite_by_feature = {
        feature: int((~finite[:, index]).sum())
        for index, feature in enumerate(feature_names)
    }
    clean = features.replace([np.inf, -np.inf], np.nan)
    medians = clean.median(numeric_only=True).fillna(0.0)
    clean = clean.fillna(medians)
    variances = clean.var(axis=0)
    nunique = clean.nunique(dropna=False)
    constant = [feature for feature in feature_names if int(nunique[feature]) <= 1]
    near_constant: list[str] = []
    for feature in feature_names:
        dominant_ratio = float(clean[feature].value_counts(normalize=True, dropna=False).iloc[0])
        if feature not in constant and (float(variances[feature]) <= 1e-8 or dominant_ratio >= 0.995):
            near_constant.append(feature)

    corr_frame = clean.drop(columns=constant + near_constant, errors="ignore")
    corr_pairs: list[dict[str, Any]] = []
    correlated_remove: set[str] = set()
    if corr_frame.shape[1] > 1:
        corr = corr_frame.corr(method="spearman").abs()
        columns = list(corr.columns)
        for i, left in enumerate(columns):
            for right in columns[i + 1 :]:
                value = float(corr.at[left, right])
                if value >= 0.985:
                    remove = choose_correlation_drop(left, right, variances)
                    correlated_remove.add(remove)
                    corr_pairs.append(
                        {
                            "feature_a": left,
                            "feature_b": right,
                            "spearman_abs_correlation": value,
                            "suggested_remove": remove,
                        }
                    )

    false_positive_risk = identify_false_positive_risk_features(clean, table["label"].astype(int))
    removed = [
        {"feature": feature, "reason": "constant"}
        for feature in constant
    ] + [
        {"feature": feature, "reason": "near_constant"}
        for feature in near_constant
    ] + [
        {"feature": feature, "reason": "highly_correlated_duplicate"}
        for feature in sorted(correlated_remove)
        if feature not in set(constant + near_constant)
    ]
    return {
        "row_count": int(len(table)),
        "feature_count": int(len(feature_names)),
        "nan_count": int(np.isnan(values).sum()),
        "pos_inf_count": int(np.isposinf(values).sum()),
        "neg_inf_count": int(np.isneginf(values).sum()),
        "nonfinite_rows": int((~finite).any(axis=1).sum()),
        "nonfinite_by_feature": nonfinite_by_feature,
        "constant_features": constant,
        "near_constant_features": near_constant,
        "highly_correlated_pairs": corr_pairs[:500],
        "highly_correlated_pair_count": len(corr_pairs),
        "suggested_removed_features": removed,
        "false_positive_risk_features": false_positive_risk,
    }


def choose_correlation_drop(left: str, right: str, variances: pd.Series) -> str:
    left_priority = feature_priority(left)
    right_priority = feature_priority(right)
    if left_priority != right_priority:
        return left if left_priority < right_priority else right
    return left if float(variances[left]) < float(variances[right]) else right


def feature_priority(feature: str) -> int:
    lower = feature.lower()
    if lower.startswith("quality_") or any(token in lower for token in QUALITY_FEATURE_HINTS):
        return 0
    if "color" in lower or lower.startswith(("rgb_", "hsv_", "lab_")):
        return 1
    if "texture" in lower or "glcm" in lower or "lbp" in lower:
        return 2
    return 3


def identify_false_positive_risk_features(
    clean: pd.DataFrame,
    labels: pd.Series,
) -> list[dict[str, Any]]:
    normal = clean.loc[labels == 0]
    referable = clean.loc[labels >= 2]
    rows: list[dict[str, Any]] = []
    if normal.empty or referable.empty:
        return rows
    for feature in clean.columns:
        if not (feature.startswith("quality_") or any(token in feature for token in QUALITY_FEATURE_HINTS)):
            continue
        normal_mean = float(normal[feature].mean())
        referable_mean = float(referable[feature].mean())
        pooled = float(clean[feature].std()) or 1.0
        rows.append(
            {
                "feature": feature,
                "normal_mean": normal_mean,
                "referable_mean": referable_mean,
                "standardized_gap": abs(normal_mean - referable_mean) / pooled,
                "reason": (
                    "Quality/acquisition feature may reflect image capture rather "
                    "than pathology and can cause false positive shifts."
                ),
            }
        )
    rows.sort(key=lambda row: row["standardized_gap"], reverse=True)
    return rows[:20]


def compute_feature_importance(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: list[str],
    audit: dict[str, Any],
    skip_shap: bool,
) -> dict[str, dict[str, float]]:
    usable = [
        feature for feature in feature_names
        if feature not in {row["feature"] for row in audit["suggested_removed_features"]}
    ]
    x_train_clean = x_train[usable].fillna(x_train[usable].median(numeric_only=True).fillna(0.0))
    x_test_clean = x_test[usable].fillna(x_train_clean.median(numeric_only=True).fillna(0.0))
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    rf = RandomForestClassifier(
        n_estimators=240,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    rf.fit(x_train_clean, y_train, sample_weight=sample_weight)
    rf_importance = normalize_importance(dict(zip(usable, rf.feature_importances_)))

    xgb_importance: dict[str, float] = {}
    if XGBClassifier is not None:
        xgb = XGBClassifier(
            objective="multi:softprob",
            num_class=5,
            n_estimators=220,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=4,
            tree_method="hist",
        )
        xgb.fit(x_train_clean, y_train, sample_weight=sample_weight)
        xgb_importance = normalize_importance(dict(zip(usable, xgb.feature_importances_)))

    perm_importance: dict[str, float] = {}
    top_for_perm = sorted(
        usable,
        key=lambda feature: rf_importance.get(feature, 0.0) + xgb_importance.get(feature, 0.0),
        reverse=True,
    )[:100]
    perm_model = RandomForestClassifier(
        n_estimators=160,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE + 1,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    perm_model.fit(x_train_clean[top_for_perm], y_train, sample_weight=sample_weight)
    perm = permutation_importance(
        perm_model,
        x_test_clean[top_for_perm],
        y_test,
        scoring="f1_macro",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    perm_importance = normalize_importance(dict(zip(top_for_perm, np.maximum(perm.importances_mean, 0.0))))

    shap_importance: dict[str, float] = {}
    if not skip_shap:
        shap_importance = try_shap_importance(rf, x_train_clean, usable)

    combined: dict[str, dict[str, float]] = {}
    for feature in usable:
        combined_score = (
            0.40 * rf_importance.get(feature, 0.0)
            + 0.40 * xgb_importance.get(feature, 0.0)
            + 0.20 * perm_importance.get(feature, 0.0)
        )
        if shap_importance:
            combined_score = 0.32 * rf_importance.get(feature, 0.0) + 0.32 * xgb_importance.get(feature, 0.0) + 0.18 * perm_importance.get(feature, 0.0) + 0.18 * shap_importance.get(feature, 0.0)
        combined[feature] = {
            "random_forest": float(rf_importance.get(feature, 0.0)),
            "xgboost": float(xgb_importance.get(feature, 0.0)),
            "permutation": float(perm_importance.get(feature, 0.0)),
            "shap": float(shap_importance.get(feature, 0.0)),
            "combined": float(combined_score),
        }
    return combined


def try_shap_importance(
    model: RandomForestClassifier,
    x_train: pd.DataFrame,
    features: list[str],
) -> dict[str, float]:
    try:
        import shap
        background = x_train[features].sample(
            n=min(200, len(x_train)),
            random_state=RANDOM_STATE,
        )
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(background, check_additivity=False)
        if isinstance(values, list):
            scores = np.mean([np.abs(item).mean(axis=0) for item in values], axis=0)
        else:
            scores = np.abs(values).mean(axis=tuple(range(values.ndim - 1)))
        return normalize_importance(dict(zip(features, np.asarray(scores).ravel())))
    except Exception:
        return {}


def normalize_importance(values: dict[str, float]) -> dict[str, float]:
    cleaned = {key: max(float(value), 0.0) for key, value in values.items()}
    total = sum(cleaned.values())
    if total <= 0:
        return {key: 0.0 for key in cleaned}
    return {key: value / total for key, value in cleaned.items()}


def build_ranked_features(
    importance: dict[str, dict[str, float]],
    audit: dict[str, Any],
    x_train: pd.DataFrame,
) -> dict[str, Any]:
    removed = {row["feature"] for row in audit["suggested_removed_features"]}
    ordered = [
        feature for feature, _ in sorted(
            importance.items(),
            key=lambda item: item[1]["combined"],
            reverse=True,
        )
        if feature not in removed
    ]
    corr = x_train[ordered].corr(method="spearman").abs() if ordered else pd.DataFrame()
    sets: dict[str, list[str]] = {}
    for count in FEATURE_COUNTS:
        limit = len(ordered) if count == "all_203" else int(count)
        selected: list[str] = []
        for feature in ordered:
            if len(selected) >= limit:
                break
            if any(float(corr.at[feature, chosen]) >= 0.985 for chosen in selected):
                continue
            selected.append(feature)
        sets[str(count)] = selected
    return {
        "ranking": ordered,
        "feature_sets": sets,
        "removed_features": audit["suggested_removed_features"],
    }


def run_multiclass_experiments(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    ranked_features: dict[str, Any],
    output_dir: Path,
    svm_train_limit: int,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    for set_name, features in ranked_features["feature_sets"].items():
        for model_name in MODELS:
            if model_name == "xgboost" and XGBClassifier is None:
                continue
            result = fit_evaluate_model(
                problem="multiclass",
                model_name=model_name,
                feature_set=set_name,
                features=features,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                output_dir=output_dir,
                svm_train_limit=svm_train_limit,
            )
            results.append(result)
            print(
                f"multiclass {model_name} {set_name}: "
                f"macro_f1={result.metrics['macro_f1']:.4f} "
                f"bal_acc={result.metrics['balanced_accuracy']:.4f} "
                f"class3_recall={result.metrics['per_class']['3']['recall']:.4f}",
                flush=True,
            )
    return results


def run_binary_experiments(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train_multi: np.ndarray,
    y_test_multi: np.ndarray,
    ranked_features: dict[str, Any],
    output_dir: Path,
    svm_train_limit: int,
) -> tuple[list[ExperimentResult], list[dict[str, Any]]]:
    y_train = remap_binary(y_train_multi)
    y_test = remap_binary(y_test_multi)
    results: list[ExperimentResult] = []
    thresholds: list[dict[str, Any]] = []
    for set_name, features in ranked_features["feature_sets"].items():
        for model_name in MODELS:
            if model_name == "xgboost" and XGBClassifier is None:
                continue
            result = fit_evaluate_model(
                problem="binary",
                model_name=model_name,
                feature_set=set_name,
                features=features,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
                output_dir=output_dir,
                svm_train_limit=svm_train_limit,
            )
            results.append(result)
            thresholds.extend(
                threshold_sweep(
                    result,
                    x_test[features],
                    y_test,
                    output_dir,
                )
            )
            print(
                f"binary {model_name} {set_name}: "
                f"f1={result.metrics['f1']:.4f} "
                f"referable_recall={result.metrics['referable_recall']:.4f} "
                f"false_negatives={result.metrics['false_negatives']}",
                flush=True,
            )
    return results, thresholds


def fit_evaluate_model(
    problem: str,
    model_name: str,
    feature_set: str,
    features: list[str],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    svm_train_limit: int,
) -> ExperimentResult:
    estimator = build_estimator(model_name, problem)
    model_dir = ensure_dir(output_dir / "models")
    model_path = model_dir / f"{problem}_{model_name}_{feature_set}.pkl"
    if model_path.exists():
        with model_path.open("rb") as file:
            estimator = pickle.load(file)
        metrics = evaluate_estimator(estimator, x_test[features], y_test, problem)
        metrics["calibration"] = "loaded_existing"
        return ExperimentResult(
            problem=problem,
            model_name=model_name,
            feature_set=feature_set,
            feature_count=len(features),
            metrics=metrics,
            selected_features=features,
            model_path=str(model_path),
        )

    x_fit = x_train[features]
    y_fit = y_train
    if model_name == "svm_rbf" and len(x_fit) > svm_train_limit:
        rng = np.random.default_rng(RANDOM_STATE)
        indices = balanced_sample_indices(y_fit, svm_train_limit, rng)
        x_fit = x_fit.iloc[indices]
        y_fit = y_fit[indices]
    fit_with_weights(estimator, x_fit, y_fit)
    metrics = evaluate_estimator(estimator, x_test[features], y_test, problem)
    metrics["calibration"] = "none"
    with model_path.open("wb") as file:
        pickle.dump(estimator, file)
    return ExperimentResult(
        problem=problem,
        model_name=model_name,
        feature_set=feature_set,
        feature_count=len(features),
        metrics=metrics,
        selected_features=features,
        model_path=str(model_path),
    )


def build_estimator(model_name: str, problem: str) -> Pipeline:
    if model_name == "logistic_regression":
        classifier: BaseEstimator = LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )
        return preserve_feature_names(Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]))
    if model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=260,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        return preserve_feature_names(Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ]))
    if model_name == "extra_trees":
        classifier = ExtraTreesClassifier(
            n_estimators=260,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        return preserve_feature_names(Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ]))
    if model_name == "balanced_random_forest":
        if BalancedRandomForestClassifier is None:
            raise ImportError("imbalanced-learn is required for Balanced Random Forest.")
        classifier = BalancedRandomForestClassifier(
            n_estimators=240,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        return preserve_feature_names(Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ]))
    if model_name == "xgboost":
        objective = "binary:logistic" if problem == "binary" else "multi:softprob"
        kwargs = {
            "objective": objective,
            "n_estimators": 240,
            "max_depth": 5,
            "learning_rate": 0.06,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
            "eval_metric": "logloss" if problem == "binary" else "mlogloss",
            "random_state": RANDOM_STATE,
            "n_jobs": 4,
            "tree_method": "hist",
        }
        if problem == "multiclass":
            kwargs["num_class"] = 5
        classifier = XGBClassifier(**kwargs)
        return preserve_feature_names(Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ]))
    if model_name == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError("lightgbm is required for the LightGBM candidate.")
        classifier = LGBMClassifier(
            objective="binary" if problem == "binary" else "multiclass",
            n_estimators=260,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=4,
            verbose=-1,
        )
        return preserve_feature_names(Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ]))
    if model_name == "catboost":
        if CatBoostClassifier is None:
            raise ImportError("catboost is required for the CatBoost candidate.")
        classifier = CatBoostClassifier(
            iterations=220,
            depth=6,
            learning_rate=0.06,
            loss_function="Logloss" if problem == "binary" else "MultiClass",
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        )
        return preserve_feature_names(Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ]))
    if model_name == "histgradientboosting":
        classifier = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=260,
            max_leaf_nodes=45,
            l2_regularization=0.01,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        return preserve_feature_names(Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ]))
    if model_name == "naive_bayes":
        classifier = GaussianNB()
        return preserve_feature_names(Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]))
    if model_name == "svm_rbf":
        classifier = SVC(
            C=4.0,
            gamma="scale",
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        )
        return preserve_feature_names(Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]))
    raise ValueError(model_name)


def preserve_feature_names(estimator: Pipeline) -> Pipeline:
    """Keep DataFrame column names through preprocessing for estimators that validate feature names."""
    try:
        estimator.set_output(transform="pandas")
    except (AttributeError, ValueError):
        pass
    return estimator


def fit_with_weights(estimator: Pipeline, x_values: pd.DataFrame, y_values: np.ndarray) -> None:
    weights = compute_sample_weight(class_weight="balanced", y=y_values)
    classifier = estimator.named_steps["classifier"]
    try:
        fit_signature = inspect.signature(classifier.fit)
    except (TypeError, ValueError):
        fit_signature = None
    if fit_signature is not None and "sample_weight" in fit_signature.parameters:
        estimator.fit(x_values, y_values, classifier__sample_weight=weights)
    else:
        estimator.fit(x_values, y_values)


def evaluate_estimator(
    estimator: BaseEstimator,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    problem: str,
) -> dict[str, Any]:
    probabilities = (
        estimator.predict_proba(x_test)
        if hasattr(estimator, "predict_proba")
        else None
    )
    if problem == "binary" and probabilities is not None:
        predictions = (probabilities[:, 1] >= 0.5).astype(int)
    else:
        predictions = estimator.predict(x_test)
    labels = [0, 1] if problem == "binary" else [0, 1, 2, 3, 4]
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    base = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "classification_report": report,
        "per_class": {
            str(label): {
                "precision": float(report[str(label)]["precision"]),
                "recall": float(report[str(label)]["recall"]),
                "f1": float(report[str(label)]["f1-score"]),
                "support": int(report[str(label)]["support"]),
                "correct": int(matrix[index, index]),
            }
            for index, label in enumerate(labels)
        },
    }
    if problem == "binary":
        base.update(
            {
                "precision": float(precision_score(y_test, predictions, zero_division=0)),
                "recall": float(recall_score(y_test, predictions, zero_division=0)),
                "f1": float(f1_score(y_test, predictions, zero_division=0)),
                "referable_recall": float(recall_score(y_test, predictions, pos_label=1, zero_division=0)),
                "non_referable_recall": float(recall_score(y_test, predictions, pos_label=0, zero_division=0)),
                "false_negatives": int(matrix[1, 0]),
                "false_positives": int(matrix[0, 1]),
            }
        )
        base["selection_score"] = (
            0.45 * base["referable_recall"]
            + 0.25 * base["balanced_accuracy"]
            + 0.20 * base["f1"]
            - 0.10 * (base["false_negatives"] / max(int(np.sum(y_test == 1)), 1))
        )
    else:
        macro_f1 = float(f1_score(y_test, predictions, labels=labels, average="macro", zero_division=0))
        macro_recall = float(recall_score(y_test, predictions, labels=labels, average="macro", zero_division=0))
        base.update(
            {
                "macro_f1": macro_f1,
                "macro_recall": macro_recall,
                "macro_precision": float(precision_score(y_test, predictions, labels=labels, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y_test, predictions, labels=labels, average="weighted", zero_division=0)),
            }
        )
        base["selection_score"] = (
            0.30 * macro_f1
            + 0.25 * base["balanced_accuracy"]
            + 0.15 * base["per_class"]["3"]["recall"]
            + 0.10 * base["per_class"]["1"]["recall"]
            + 0.10 * base["per_class"]["4"]["recall"]
            + 0.10 * min(
                base["per_class"]["0"]["recall"],
                base["per_class"]["2"]["recall"],
            )
        )
    return base


def calibrate_if_better(
    estimator: BaseEstimator,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    problem: str,
) -> tuple[BaseEstimator | None, dict[str, Any] | None]:
    try:
        calibrated = CalibratedClassifierCV(
            clone(estimator),
            method="isotonic",
            cv=3,
        )
        calibrated.fit(x_train, y_train)
        metrics = evaluate_estimator(calibrated, x_test, y_test, problem)
        return calibrated, metrics
    except Exception:
        return None, None


def threshold_sweep(
    result: ExperimentResult,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    output_dir: Path,
) -> list[dict[str, Any]]:
    with Path(result.model_path).open("rb") as file:
        model = pickle.load(file)
    if not hasattr(model, "predict_proba"):
        return []
    probs = model.predict_proba(x_test)[:, 1]
    rows: list[dict[str, Any]] = []
    for threshold in BINARY_THRESHOLDS:
        pred = (probs >= threshold).astype(int)
        matrix = confusion_matrix(y_test, pred, labels=[0, 1])
        rows.append(
            {
                "model_name": result.model_name,
                "feature_set": result.feature_set,
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(y_test, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "f1": float(f1_score(y_test, pred, zero_division=0)),
                "referable_recall": float(recall_score(y_test, pred, pos_label=1, zero_division=0)),
                "non_referable_recall": float(recall_score(y_test, pred, pos_label=0, zero_division=0)),
                "false_negatives": int(matrix[1, 0]),
                "false_positives": int(matrix[0, 1]),
                "uncertain_below_60_pct": float(np.mean(np.maximum(probs, 1 - probs) < 0.60)),
            }
        )
    write_csv(output_dir / "binary_threshold_sweep.csv", rows)
    return rows


def run_class_3_vs_4_experiment(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    ranked_features: dict[str, Any],
    multiclass_results: list[ExperimentResult],
    output_dir: Path,
) -> dict[str, Any]:
    best = select_best_multiclass(multiclass_results)
    features = best.selected_features
    mask_train = np.isin(y_train, [3, 4])
    mask_test = np.isin(y_test, [3, 4])
    if int(mask_train.sum()) < 20 or int(mask_test.sum()) < 10:
        return {"status": "skipped", "reason": "Not enough Class 3/4 samples."}
    sub_model = build_estimator("xgboost" if XGBClassifier is not None else "random_forest", "binary")
    sub_y_train = np.where(y_train[mask_train] == 4, 1, 0)
    sub_y_test = np.where(y_test[mask_test] == 4, 1, 0)
    fit_with_weights(sub_model, x_train.loc[mask_train, features], sub_y_train)
    sub_metrics = evaluate_estimator(sub_model, x_test.loc[mask_test, features], sub_y_test, "binary")

    with Path(best.model_path).open("rb") as file:
        main_model = pickle.load(file)
    main_pred = main_model.predict(x_test[features])
    refined_pred = main_pred.copy()
    candidate_mask = np.isin(main_pred, [3, 4])
    if candidate_mask.any():
        sub_pred = sub_model.predict(x_test.loc[candidate_mask, features])
        refined_pred[candidate_mask] = np.where(sub_pred == 1, 4, 3)

    before = confusion_matrix(y_test, main_pred, labels=[3, 4])
    after = confusion_matrix(y_test, refined_pred, labels=[3, 4])
    before_all = evaluate_predictions(y_test, main_pred)
    after_all = evaluate_predictions(y_test, refined_pred)
    model_path = output_dir / "models" / "class_3_vs_4_subclassifier.pkl"
    with model_path.open("wb") as file:
        pickle.dump(sub_model, file)
    return {
        "status": "completed",
        "base_model": f"{best.model_name}_{best.feature_set}",
        "feature_count": len(features),
        "model_path": str(model_path),
        "subclassifier_metrics": sub_metrics,
        "class_3_4_confusion_before": before.astype(int).tolist(),
        "class_3_4_confusion_after": after.astype(int).tolist(),
        "main_metrics_before": before_all,
        "main_metrics_after": after_all,
        "reduced_3_vs_4_confusion": int(after[0, 1] + after[1, 0]) < int(before[0, 1] + before[1, 0]),
    }


def evaluate_predictions(y_true: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    labels = [0, 1, 2, 3, 4]
    matrix = confusion_matrix(y_true, pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=labels, average="macro", zero_division=0)),
        "class_3_recall": float(recall_score(y_true, pred, labels=labels, average=None, zero_division=0)[3]),
        "class_4_recall": float(recall_score(y_true, pred, labels=labels, average=None, zero_division=0)[4]),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def select_best_multiclass(results: list[ExperimentResult]) -> ExperimentResult:
    return max(results, key=lambda item: item.metrics["selection_score"])


def select_best_binary(results: list[ExperimentResult]) -> ExperimentResult:
    return max(results, key=lambda item: item.metrics["selection_score"])


def decide_replacement(best_multiclass: ExperimentResult, best_binary: ExperimentResult) -> dict[str, Any]:
    m = best_multiclass.metrics
    b = best_binary.metrics
    multiclass_ok = (
        (m["macro_f1"] > PRODUCTION_BASELINE["multiclass_macro_f1"]
         or m["balanced_accuracy"] > PRODUCTION_BASELINE["multiclass_balanced_accuracy"])
        and m["per_class"]["3"]["recall"] > PRODUCTION_BASELINE["class_3_recall"]
        and m["per_class"]["0"]["recall"] >= 0.70
        and m["per_class"]["2"]["recall"] >= 0.45
    )
    binary_ok = (
        b["referable_recall"] >= PRODUCTION_BASELINE["binary_referable_recall"]
        and b["false_negatives"] <= PRODUCTION_BASELINE["binary_false_negatives"]
    )
    return {
        "replace_production": bool(multiclass_ok and binary_ok),
        "multiclass_ok": bool(multiclass_ok),
        "binary_ok": bool(binary_ok),
        "reason": (
            "Replace only if exact grading improves macro F1 or balanced accuracy, "
            "Class 3 recall improves, binary referable recall stays high, and "
            "Class 0/Class 2 are not badly damaged."
        ),
    }


def build_report(
    audit: dict[str, Any],
    importance: dict[str, dict[str, float]],
    ranked_features: dict[str, Any],
    multiclass_results: list[ExperimentResult],
    binary_results: list[ExperimentResult],
    threshold_report: list[dict[str, Any]],
    subclassifier_report: dict[str, Any],
    best_multiclass: ExperimentResult,
    best_binary: ExperimentResult,
    replacement: dict[str, Any],
    features_csv: Path,
) -> dict[str, Any]:
    best_threshold = select_best_threshold(threshold_report)
    payload = {
        "features_csv": str(features_csv),
        "feature_audit": audit,
        "top_features": list(importance.keys())[:30],
        "ranked_feature_sets": ranked_features,
        "best_multiclass": result_to_dict(best_multiclass),
        "best_binary": result_to_dict(best_binary),
        "best_binary_threshold": best_threshold,
        "class_3_vs_4_subclassifier": subclassifier_report,
        "production_baseline": PRODUCTION_BASELINE,
        "balanced_baseline": BALANCED_BASELINE,
        "production_replacement": replacement,
        "multiclass_results": [result_to_dict(item) for item in multiclass_results],
        "binary_results": [result_to_dict(item) for item in binary_results],
    }
    payload["markdown"] = render_markdown(payload)
    return payload


def select_best_threshold(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            row["referable_recall"] >= PRODUCTION_BASELINE["binary_referable_recall"],
            row["referable_recall"],
            row["balanced_accuracy"],
            row["f1"],
            -row["false_negatives"],
            -row["false_positives"],
        ),
    )


def result_to_dict(result: ExperimentResult) -> dict[str, Any]:
    return {
        "problem": result.problem,
        "model_name": result.model_name,
        "feature_set": result.feature_set,
        "feature_count": result.feature_count,
        "metrics": result.metrics,
        "selected_features": result.selected_features,
        "model_path": result.model_path,
    }


def render_markdown(report: dict[str, Any]) -> str:
    best_m = report["best_multiclass"]
    best_b = report["best_binary"]
    lines = [
        "# AppDR Study Feature-Selection Experiment",
        "",
        "This experiment keeps the existing classical 203-feature AppDR pipeline. It does not use CNN, UNet, ResNet, YOLO, or other deep-learning image models.",
        "",
        "## Feature Audit",
        "",
        f"Rows: {report['feature_audit']['row_count']}",
        f"Feature count: {report['feature_audit']['feature_count']}",
        f"NaN values: {report['feature_audit']['nan_count']}",
        f"+inf values: {report['feature_audit']['pos_inf_count']}",
        f"-inf values: {report['feature_audit']['neg_inf_count']}",
        f"Constant features: {len(report['feature_audit']['constant_features'])}",
        f"Near-constant features: {len(report['feature_audit']['near_constant_features'])}",
        f"Highly correlated pairs: {report['feature_audit']['highly_correlated_pair_count']}",
        "",
        "## Best Multiclass Model",
        "",
        f"Model: {best_m['model_name']}",
        f"Feature set: {best_m['feature_set']} ({best_m['feature_count']} features)",
        f"Accuracy: {pct(best_m['metrics']['accuracy'])}",
        f"Balanced accuracy: {pct(best_m['metrics']['balanced_accuracy'])}",
        f"Macro F1: {pct(best_m['metrics']['macro_f1'])}",
        f"Class 1 recall: {pct(best_m['metrics']['per_class']['1']['recall'])}",
        f"Class 3 recall: {pct(best_m['metrics']['per_class']['3']['recall'])}",
        f"Class 4 recall: {pct(best_m['metrics']['per_class']['4']['recall'])}",
        "",
        "### Per-stage Recall/F1",
        "",
    ]
    for label, name in config.CLASS_NAMES.items():
        row = best_m["metrics"]["per_class"][str(label)]
        lines.append(
            f"- Class {label} - {name}: recall {pct(row['recall'])}, F1 {pct(row['f1'])}, support {row['support']}"
        )
    lines.extend(
        [
            "",
            "## Best Binary Screening Model",
            "",
            f"Model: {best_b['model_name']}",
            f"Feature set: {best_b['feature_set']} ({best_b['feature_count']} features)",
            f"Accuracy: {pct(best_b['metrics']['accuracy'])}",
            f"Balanced accuracy: {pct(best_b['metrics']['balanced_accuracy'])}",
            f"F1: {pct(best_b['metrics']['f1'])}",
            f"Referable recall: {pct(best_b['metrics']['referable_recall'])}",
            f"False negatives: {best_b['metrics']['false_negatives']}",
            "",
            "## Binary Threshold Result",
            "",
        ]
    )
    threshold = report["best_binary_threshold"]
    if threshold:
        lines.extend(
            [
                f"Model: {threshold['model_name']}",
                f"Feature set: {threshold['feature_set']}",
                f"Threshold: {threshold['threshold']:.2f}",
                f"Referable recall: {pct(threshold['referable_recall'])}",
                f"False negatives: {threshold['false_negatives']}",
                f"False positives: {threshold['false_positives']}",
                f"Low-confidence/uncertain share: {pct(threshold['uncertain_below_60_pct'])}",
            ]
        )
    sub = report["class_3_vs_4_subclassifier"]
    lines.extend(["", "## Class 3 vs Class 4 Sub-classifier", ""])
    if sub.get("status") == "completed":
        lines.extend(
            [
                f"Base model: {sub['base_model']}",
                f"Reduced 3-vs-4 confusion: {sub['reduced_3_vs_4_confusion']}",
                f"Before 3/4 confusion matrix: {sub['class_3_4_confusion_before']}",
                f"After 3/4 confusion matrix: {sub['class_3_4_confusion_after']}",
            ]
        )
    else:
        lines.append(f"Skipped: {sub.get('reason', 'unknown')}")
    lines.extend(
        [
            "",
            "## Old vs New",
            "",
            f"Production macro F1: {pct(PRODUCTION_BASELINE['multiclass_macro_f1'])}",
            f"New macro F1: {pct(best_m['metrics']['macro_f1'])}",
            f"Production balanced accuracy: {pct(PRODUCTION_BASELINE['multiclass_balanced_accuracy'])}",
            f"New balanced accuracy: {pct(best_m['metrics']['balanced_accuracy'])}",
            f"Production Class 3 recall: {pct(PRODUCTION_BASELINE['class_3_recall'])}",
            f"New Class 3 recall: {pct(best_m['metrics']['per_class']['3']['recall'])}",
            f"Production binary referable recall: {pct(PRODUCTION_BASELINE['binary_referable_recall'])}",
            f"New binary referable recall: {pct(best_b['metrics']['referable_recall'])}",
            f"Production binary false negatives: {PRODUCTION_BASELINE['binary_false_negatives']}",
            f"New binary false negatives: {best_b['metrics']['false_negatives']}",
            "",
            "## Production Decision",
            "",
            f"Production replaced: {report['production_replacement']['replace_production']}",
            report["production_replacement"]["reason"],
            "",
            "## Study-Based Rationale",
            "",
            "Related classical DR studies commonly use lesion measurements, exudate and microaneurysm features, blood-vessel features, texture descriptors such as GLCM/LBP, contrast enhancement such as CLAHE, and feature selection before classical classifiers. This experiment follows that pattern by auditing redundant/noisy handcrafted features, ranking them with Random Forest/XGBoost/permutation importance, comparing smaller feature sets, and adding a focused severe-NPDR-vs-PDR classifier while keeping the binary referable result as the safer main output.",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    output_dir: Path,
    report: dict[str, Any],
    multiclass_results: list[ExperimentResult],
    binary_results: list[ExperimentResult],
) -> None:
    write_json(output_dir / "study_feature_selection_report.json", {k: v for k, v in report.items() if k != "markdown"})
    (output_dir / "study_feature_selection_report.md").write_text(report["markdown"], encoding="utf-8")
    write_results_csv(output_dir / "model_comparison_study.csv", multiclass_results + binary_results)
    write_confusion_matrix(output_dir / "best_multiclass_confusion_matrix.csv", report["best_multiclass"]["metrics"]["confusion_matrix"])
    write_confusion_matrix(output_dir / "best_binary_confusion_matrix.csv", report["best_binary"]["metrics"]["confusion_matrix"])


def write_results_csv(path: Path, results: list[ExperimentResult]) -> None:
    rows = []
    for result in results:
        metrics = result.metrics
        row = {
            "problem": result.problem,
            "model_name": result.model_name,
            "feature_set": result.feature_set,
            "feature_count": result.feature_count,
            "accuracy": metrics.get("accuracy"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "binary_f1": metrics.get("f1"),
            "class_1_recall": metrics.get("per_class", {}).get("1", {}).get("recall"),
            "class_3_recall": metrics.get("per_class", {}).get("3", {}).get("recall"),
            "class_4_recall": metrics.get("per_class", {}).get("4", {}).get("recall"),
            "referable_recall": metrics.get("referable_recall"),
            "false_negatives": metrics.get("false_negatives"),
            "selection_score": metrics.get("selection_score"),
            "model_path": result.model_path,
        }
        rows.append(row)
    write_csv(path, rows)


def write_confusion_matrix(path: Path, matrix: list[list[int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(matrix)


def write_importance(path: Path, importance: dict[str, dict[str, float]]) -> None:
    rows = [
        {"feature": feature, **scores}
        for feature, scores in sorted(
            importance.items(),
            key=lambda item: item[1]["combined"],
            reverse=True,
        )
    ]
    write_csv(path, rows)


def write_feature_audit_csv(path: Path, audit: dict[str, Any]) -> None:
    write_csv(path, audit["suggested_removed_features"])


def backup_production_artifacts(output_dir: Path) -> None:
    backup_dir = ensure_dir(output_dir / "production_backup")
    artifacts = [
        BACKEND_DIR / "results" / "best_model.pkl",
        BACKEND_DIR / "results" / "best_model_metadata.json",
        BACKEND_DIR / "results" / "metrics.json",
        BACKEND_DIR / "results" / "binary" / "best_model.pkl",
        BACKEND_DIR / "results" / "binary" / "best_model_metadata.json",
        BACKEND_DIR / "results" / "binary" / "metrics.json",
    ]
    manifest = []
    for source in artifacts:
        if not source.exists():
            continue
        relative = source.relative_to(BACKEND_DIR / "results")
        target = backup_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest.append({"source": str(source), "backup": str(target)})
    write_json(backup_dir / "manifest.json", {"created_at": datetime.now().isoformat(), "artifacts": manifest})


def balanced_sample_indices(y_values: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    labels = np.unique(y_values)
    per_class = max(1, limit // len(labels))
    selected: list[int] = []
    for label in labels:
        candidates = np.flatnonzero(y_values == label)
        size = min(len(candidates), per_class)
        selected.extend(rng.choice(candidates, size=size, replace=False).tolist())
    if len(selected) < min(limit, len(y_values)):
        remaining = np.setdiff1d(np.arange(len(y_values)), np.array(selected), assume_unique=False)
        extra = min(len(remaining), min(limit, len(y_values)) - len(selected))
        selected.extend(rng.choice(remaining, size=extra, replace=False).tolist())
    return np.array(sorted(selected), dtype=int)


def remap_binary(labels: np.ndarray) -> np.ndarray:
    return np.where(np.isin(labels, [2, 3, 4]), 1, 0).astype(int)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def pct(value: float) -> str:
    return f"{float(value) * 100.0:.2f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AppDR study-style feature experiments.")
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=BACKEND_DIR / "features_combined_balanced.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
    )
    parser.add_argument(
        "--svm-train-limit",
        type=int,
        default=6000,
        help="Balanced training sample cap for SVM RBF to keep the study run practical.",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip optional SHAP importance and rely on RF/XGBoost/permutation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
