import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
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
    all_feature_names = resolve_training_feature_names(table)
    x_values = table[all_feature_names].to_numpy(dtype=np.float64)
    y_values = table["label"].to_numpy(dtype=np.int64)

    validate_training_labels(y_values, context="full dataset")
    print_class_distribution(y_values.tolist())

    # Leakage prevention: the holdout split is created before GridSearchCV.
    # StandardScaler lives inside each Pipeline, so scaling parameters are fit
    # only on the training fold during cross-validation.
    x_train_full, x_test_full, y_train, y_test = train_test_split(
        x_values,
        y_values,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_values,
    )
    validate_training_labels(y_train, context="training split")
    feature_names, selected_indices, selection_rows = select_informative_features(
        x_train_full,
        y_train,
        all_feature_names,
        output_dir=output_dir,
    )
    x_train = x_train_full[:, selected_indices]
    x_test = x_test_full[:, selected_indices]

    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )

    searches = {
        "RandomForestClassifier": build_random_forest_search(cv),
        "SVC": build_svc_search(cv),
        "HistGradientBoostingClassifier": build_hist_gradient_boosting_search(cv),
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
        "feature_names": feature_names,
        "all_feature_names": all_feature_names,
        "feature_selection": {
            "method": "consensus of mutual information, ANOVA F-score, and RFE",
            "selected_feature_count": len(feature_names),
            "candidate_feature_count": len(all_feature_names),
            "report_path": "feature_selection_report.csv",
        },
        "class_labels": config.CLASS_LABELS,
        "class_names": config.CLASS_NAMES,
        "random_state": config.RANDOM_STATE,
        "test_size": config.TEST_SIZE,
        "cv_folds": config.CV_FOLDS,
        "scoring": "f1_macro",
        "imbalance_handling": (
            "RandomForestClassifier and SVC use class_weight='balanced'. "
            "HistGradientBoostingClassifier uses SMOTE inside the cross-validation "
            "pipeline, so synthetic minority samples are created only from each "
            "training fold. This matters because DR datasets usually contain many "
            "more no-DR or moderate cases than severe/proliferative cases; imbalance "
            "handling reduces the tendency to ignore minority clinical stages while "
            "preventing leakage into validation or holdout data."
        ),
    }
    save_json(output_dir / "best_model_metadata.json", metadata)
    save_json(
        output_dir / "selected_features.json",
        {
            "selected_feature_count": len(feature_names),
            "selected_features": feature_names,
        },
    )

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
        feature_names=feature_names,
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
    pipeline = SklearnPipeline(
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


def resolve_training_feature_names(table: pd.DataFrame) -> list[str]:
    missing_expanded = [name for name in config.FEATURE_NAMES if name not in table.columns]
    if not missing_expanded:
        return list(config.FEATURE_NAMES)

    missing_legacy = [
        name
        for name in config.LEGACY_FEATURE_NAMES
        if name not in table.columns
    ]
    if not missing_legacy:
        print(
            "Using legacy six-feature table. Rebuild features.csv with "
            "dataset_builder.py to train the expanded 30-feature classifier.",
        )
        return list(config.LEGACY_FEATURE_NAMES)

    raise ValueError(
        "features.csv is missing required handcrafted feature columns. "
        f"Missing expanded columns: {missing_expanded[:8]}; "
        f"missing legacy columns: {missing_legacy}",
    )


def select_informative_features(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
) -> tuple[list[str], np.ndarray, list[dict[str, Any]]]:
    selected_count = min(config.SELECTED_FEATURE_COUNT, len(feature_names))

    if selected_count >= len(feature_names):
        indices = np.arange(len(feature_names), dtype=int)
        rows = [
            {
                "feature": feature,
                "selected": True,
                "consensus_rank": index + 1,
                "mutual_info": 0.0,
                "anova_f": 0.0,
                "rfe_rank": 1,
            }
            for index, feature in enumerate(feature_names)
        ]
        pd.DataFrame(rows).to_csv(output_dir / "feature_selection_report.csv", index=False)
        return list(feature_names), indices, rows

    print(
        f"Selecting {selected_count} of {len(feature_names)} handcrafted features "
        "using mutual information, ANOVA F-score, and RFE.",
    )

    mi_scores = np.nan_to_num(
        mutual_info_classif(
            x_train,
            y_train,
            random_state=config.RANDOM_STATE,
            discrete_features=False,
        ),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    anova_scores, _ = f_classif(x_train, y_train)
    anova_scores = np.nan_to_num(anova_scores, nan=0.0, posinf=0.0, neginf=0.0)

    rfe_estimator = RandomForestClassifier(
        n_estimators=75,
        max_depth=10,
        min_samples_split=5,
        random_state=config.RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    rfe = RFE(
        estimator=rfe_estimator,
        n_features_to_select=selected_count,
        step=0.2,
    )
    rfe.fit(x_train, y_train)

    mi_rank = descending_rank(mi_scores)
    anova_rank = descending_rank(anova_scores)
    rfe_rank = rfe.ranking_.astype(float)
    consensus_score = mi_rank + anova_rank + rfe_rank
    selected_indices = np.argsort(consensus_score)[:selected_count]
    selected_indices = np.array(sorted(selected_indices.tolist()), dtype=int)
    selected_set = set(selected_indices.tolist())
    consensus_order = np.argsort(consensus_score)
    consensus_rank_lookup = {
        int(index): rank
        for rank, index in enumerate(consensus_order.tolist(), start=1)
    }

    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(feature_names):
        rows.append(
            {
                "feature": feature,
                "selected": index in selected_set,
                "consensus_rank": consensus_rank_lookup[index],
                "consensus_score": float(consensus_score[index]),
                "mutual_info": float(mi_scores[index]),
                "mutual_info_rank": int(mi_rank[index]),
                "anova_f": float(anova_scores[index]),
                "anova_rank": int(anova_rank[index]),
                "rfe_rank": int(rfe.ranking_[index]),
                "rfe_selected": bool(rfe.support_[index]),
            },
        )

    rows = sorted(rows, key=lambda row: int(row["consensus_rank"]))
    pd.DataFrame(rows).to_csv(output_dir / "feature_selection_report.csv", index=False)

    selected_feature_names = [feature_names[index] for index in selected_indices]
    return selected_feature_names, selected_indices, rows


def descending_rank(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks.astype(float)


def build_svc_search(cv: StratifiedKFold) -> GridSearchCV:
    pipeline = SklearnPipeline(
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


def build_hist_gradient_boosting_search(cv: StratifiedKFold) -> GridSearchCV:
    # SMOTE is placed inside the imbalanced-learn Pipeline so oversampling is
    # fitted separately within each CV fold. This avoids leaking synthetic
    # minority-stage samples into validation data.
    pipeline = ImbalancedPipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "smote",
                SMOTE(random_state=config.RANDOM_STATE, k_neighbors=3),
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    random_state=config.RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ],
    )
    return GridSearchCV(
        estimator=pipeline,
        param_grid=config.HGB_PARAM_GRID,
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
