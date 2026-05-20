import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
from evaluate import evaluate_model, explain_feature_importance, save_feature_importance
from utils import ensure_dir, print_class_distribution, read_feature_table, save_json, save_text


def train_models(
    features_csv: str | Path = config.FEATURES_CSV,
    results_dir: str | Path = config.RESULTS_DIR,
) -> dict[str, Any]:
    output_dir = ensure_dir(results_dir)
    table = read_feature_table(features_csv)
    x_values = table[config.FEATURE_NAMES].to_numpy(dtype=np.float64)
    y_values = table["label"].to_numpy(dtype=np.int64)

    validate_training_labels(y_values, context="full dataset")
    print_class_distribution(y_values.tolist())

    # Leakage prevention: the holdout split is created before GridSearchCV.
    # StandardScaler lives inside each Pipeline, so scaling parameters are fit
    # only on the training fold during cross-validation.
    x_train, x_test, y_train, y_test = train_test_split(
        x_values,
        y_values,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_values,
    )
    validate_training_labels(y_train, context="training split")

    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )

    searches = {
        "RandomForestClassifier": build_random_forest_search(cv),
        "SVC": build_svc_search(cv),
    }

    comparison_rows: list[dict[str, Any]] = []

    for model_name, search in searches.items():
        print(f"\nRunning GridSearchCV for {model_name}")
        search.fit(x_train, y_train)
        save_grid_results(search, output_dir / f"gridsearch_results_{model_name}.csv")

        holdout_predictions = search.best_estimator_.predict(x_test)
        comparison_rows.append(
            {
                "model": model_name,
                "best_cv_f1_macro": float(search.best_score_),
                "holdout_f1_macro": float(
                    f1_score(
                        y_test,
                        holdout_predictions,
                        labels=config.CLASS_LABELS,
                        average="macro",
                        zero_division=0,
                    ),
                ),
                "holdout_balanced_accuracy": float(
                    balanced_accuracy_score(y_test, holdout_predictions),
                ),
                "best_params": search.best_params_,
            },
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "model_comparison_results.csv", index=False)

    best_model_name, best_search = select_best_search(searches)
    best_model = best_search.best_estimator_

    with (output_dir / "best_model.pkl").open("wb") as file:
        pickle.dump(best_model, file)

    metadata = {
        "best_model_name": best_model_name,
        "best_cv_f1_macro": float(best_search.best_score_),
        "best_parameters": best_search.best_params_,
        "feature_names": config.FEATURE_NAMES,
        "class_labels": config.CLASS_LABELS,
        "class_names": config.CLASS_NAMES,
        "random_state": config.RANDOM_STATE,
        "test_size": config.TEST_SIZE,
        "cv_folds": config.CV_FOLDS,
        "scoring": "f1_macro",
        "imbalance_handling": (
            "class_weight='balanced' is used because DR datasets commonly contain "
            "many more no-DR or moderate cases than severe/proliferative cases. "
            "Class weighting increases the penalty for errors on minority stages "
            "without duplicating images or leaking synthetic samples into test folds."
        ),
    }
    save_json(output_dir / "best_model_metadata.json", metadata)

    metrics = evaluate_model(
        best_model,
        x_test,
        y_test,
        results_dir=output_dir,
        model_name=best_model_name,
    )

    rf_search = searches["RandomForestClassifier"]
    importance_rows = save_feature_importance(
        rf_search.best_estimator_,
        output_dir / "feature_importance.png",
        output_dir / "feature_importance.csv",
    )
    save_text(output_dir / "feature_importance_explanation.txt", explain_feature_importance(importance_rows))

    print("\nBest model:", best_model_name)
    print("Best parameters:", best_search.best_params_)
    print(f"Best CV macro F1: {best_search.best_score_:.4f}")
    print(f"Holdout macro F1: {metrics['f1_macro']:.4f}")

    return {
        "best_model_name": best_model_name,
        "best_parameters": best_search.best_params_,
        "best_cv_f1_macro": float(best_search.best_score_),
        "holdout_metrics": metrics,
        "results_dir": str(output_dir),
    }


def build_random_forest_search(cv: StratifiedKFold) -> GridSearchCV:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=config.RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ],
    )
    return GridSearchCV(
        estimator=pipeline,
        param_grid=config.RF_PARAM_GRID,
        scoring="f1_macro",
        cv=cv,
        verbose=2,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )


def build_svc_search(cv: StratifiedKFold) -> GridSearchCV:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    probability=True,
                    class_weight="balanced",
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ],
    )
    return GridSearchCV(
        estimator=pipeline,
        param_grid=config.SVC_PARAM_GRID,
        scoring="f1_macro",
        cv=cv,
        verbose=2,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )


def select_best_search(searches: dict[str, GridSearchCV]) -> tuple[str, GridSearchCV]:
    return max(searches.items(), key=lambda item: float(item[1].best_score_))


def save_grid_results(search: GridSearchCV, output_path: Path) -> None:
    results = pd.DataFrame(search.cv_results_)
    results.to_csv(output_path, index=False)


def validate_training_labels(labels: np.ndarray, context: str) -> None:
    unique, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique.tolist(), counts.tolist()))
    missing_labels = [label for label in config.CLASS_LABELS if label not in label_counts]
    if missing_labels:
        raise ValueError(f"Missing DR stages in features.csv: {missing_labels}")

    too_small = {
        label: count
        for label, count in label_counts.items()
        if count < config.CV_FOLDS
    }
    if too_small:
        raise ValueError(
            "GridSearchCV is configured with cv=5, so each class needs at least "
            f"5 samples in the {context}. Too-small classes: {too_small}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train supervised DR stage classifiers.")
    parser.add_argument("--features-csv", type=Path, default=config.FEATURES_CSV)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_models(features_csv=args.features_csv, results_dir=args.results_dir)
