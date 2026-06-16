"""Hybrid CNN + AppDR handcrafted feature comparison.

Experiment only. This script reads existing AppDR production artifacts and the
previous capped Guanshuo-style CNN artifact, then writes separate hybrid results.
It does not update production or backend/frontend prediction behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
PREVIOUS_DIR = BACKEND_DIR / "results" / "aptos_guanshuo_close_comparison"
OUTPUT_DIR = BACKEND_DIR / "results" / "hybrid_cnn_features_comparison"
RANDOM_STATE = 42
OPTIMIZED_CNN_THRESHOLDS = [1.0, 1.4, 2.1, 3.1]
THRESHOLDS = [round(v, 2) for v in np.arange(0.05, 0.801, 0.05)]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config


@dataclass
class Result:
    problem: str
    model_name: str
    feature_set: str
    metrics: dict[str, Any]
    model: Any
    features: list[str]


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifests = load_manifests()
    copy_manifests(manifests)
    write_split_summary(manifests)

    if args.reuse_features and all((OUTPUT_DIR / f"hybrid_features_{split}.csv").exists() for split in ["train", "val", "test"]):
        hybrid_tables = {
            split: pd.read_csv(OUTPUT_DIR / f"hybrid_features_{split}.csv")
            for split in ["train", "val", "test"]
        }
        dictionary = pd.read_csv(OUTPUT_DIR / "hybrid_feature_dictionary.csv").to_dict(orient="records")
    else:
        production_grading, production_binary = load_production_pipelines()
        selected_75 = list(production_grading.named_steps["feature_selector"].selected_features)
        selected_100 = list(production_binary.named_steps["feature_selector"].selected_features)
        hybrid_tables, dictionary = build_hybrid_tables(manifests, production_grading, production_binary, selected_75, selected_100, args)
        for split, table in hybrid_tables.items():
            table.to_csv(OUTPUT_DIR / f"hybrid_features_{split}.csv", index=False)
        write_rows(OUTPUT_DIR / "hybrid_feature_dictionary.csv", dictionary)
        audit = audit_features(hybrid_tables, dictionary)
        write_audit(audit)

    feature_sets = build_feature_sets(dictionary)
    five_results = train_5class_models(hybrid_tables, feature_sets)
    binary_results, threshold_rows = train_binary_models(hybrid_tables, feature_sets)
    write_rows(OUTPUT_DIR / "hybrid_5class_model_comparison.csv", [result.metrics for result in five_results])
    write_rows(OUTPUT_DIR / "hybrid_binary_model_comparison.csv", [result.metrics for result in binary_results])
    write_rows(OUTPUT_DIR / "hybrid_binary_threshold_sweep.csv", threshold_rows)

    best_five = select_best_5class(five_results)
    best_binary, best_threshold_row = select_best_binary(binary_results, threshold_rows)
    export_best_models(best_five, best_binary)
    write_best_metric_files(best_five, best_binary, best_threshold_row)
    write_comparison_reports(best_five, best_binary, best_threshold_row)
    print(f"Created {OUTPUT_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--reuse-features", action="store_true")
    return parser.parse_args()


def load_manifests() -> dict[str, pd.DataFrame]:
    manifest_names = {
        "dataset": "dataset_manifest.csv",
        "train": "run_train_manifest.csv",
        "val": "run_val_manifest.csv",
        "test": "run_test_manifest.csv",
    }
    fallback_names = {
        "train": "train_manifest.csv",
        "val": "val_manifest.csv",
        "test": "test_manifest.csv",
    }
    manifests: dict[str, pd.DataFrame] = {}
    for split, name in manifest_names.items():
        path = PREVIOUS_DIR / name
        if not path.exists() and split in fallback_names:
            path = PREVIOUS_DIR / fallback_names[split]
        if not path.exists():
            raise FileNotFoundError(f"Missing required previous manifest: {path}")
        manifests[split] = pd.read_csv(path)
    return manifests


def copy_manifests(manifests: dict[str, pd.DataFrame]) -> None:
    manifests["dataset"].to_csv(OUTPUT_DIR / "dataset_manifest.csv", index=False)
    manifests["train"].to_csv(OUTPUT_DIR / "train_manifest.csv", index=False)
    manifests["val"].to_csv(OUTPUT_DIR / "val_manifest.csv", index=False)
    manifests["test"].to_csv(OUTPUT_DIR / "test_manifest.csv", index=False)


def write_split_summary(manifests: dict[str, pd.DataFrame]) -> None:
    lines = [
        "# Hybrid Split Summary",
        "",
        "Reused the effective capped split from `backend/results/aptos_guanshuo_close_comparison/`.",
        "This keeps AppDR, straight CNN, and hybrid comparisons on the same run manifests.",
        "Patient IDs are not available in APTOS 2019 train.csv, so this remains an image-level split.",
        "",
        "| Split | Rows | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ["train", "val", "test"]:
        frame = manifests[split]
        counts = frame["label"].value_counts().sort_index().to_dict()
        lines.append(
            f"| {split} | {len(frame)} | {counts.get(0, 0)} | {counts.get(1, 0)} | {counts.get(2, 0)} | {counts.get(3, 0)} | {counts.get(4, 0)} |"
        )
    (OUTPUT_DIR / "split_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_production_pipelines() -> tuple[Any, Any]:
    with (BACKEND_DIR / "results" / "best_model.pkl").open("rb") as file:
        grading = pickle.load(file)
    with (BACKEND_DIR / "results" / "binary" / "best_model.pkl").open("rb") as file:
        binary = pickle.load(file)
    return grading, binary


def build_hybrid_tables(
    manifests: dict[str, pd.DataFrame],
    production_grading: Any,
    production_binary: Any,
    selected_75: list[str],
    selected_100: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    feature_source = pd.read_csv(BACKEND_DIR / "features.csv")
    raw_features = feature_source[config.FEATURE_NAMES].copy()
    grading_selected_features = Pipeline(production_grading.steps[:-1]).transform(raw_features)
    binary_selected_features = Pipeline(production_binary.steps[:-1]).transform(raw_features)

    all_manifest = pd.concat(
        [
            manifests["train"].assign(split="train"),
            manifests["val"].assign(split="val"),
            manifests["test"].assign(split="test"),
        ],
        ignore_index=True,
    )
    cnn_table = extract_cnn_table(all_manifest, args)
    tables: dict[str, pd.DataFrame] = {}
    for split in ["train", "val", "test"]:
        split_manifest = manifests[split].reset_index(drop=True)
        row_index = split_manifest["row_index"].astype(int).to_numpy()
        base = split_manifest[["row_index", "image_id", "image_path", "label", "split"]].copy()
        handcrafted = raw_features.iloc[row_index].reset_index(drop=True)
        selected_grading = grading_selected_features.iloc[row_index].reset_index(drop=True)
        selected_binary = binary_selected_features.iloc[row_index].reset_index(drop=True)
        cnn_split = cnn_table[cnn_table["split"] == split].reset_index(drop=True)
        table = pd.concat(
            [
                base,
                handcrafted.add_prefix("appdr203__"),
                selected_grading[selected_75].reset_index(drop=True).add_prefix("selected75__"),
                selected_binary[selected_100].reset_index(drop=True).add_prefix("selected100__"),
                cnn_split.drop(columns=["row_index", "image_id", "image_path", "label", "split"], errors="ignore"),
            ],
            axis=1,
        )
        tables[split] = table

    dictionary = build_dictionary(tables["train"])
    return tables, dictionary


def extract_cnn_table(manifest: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    import timm

    checkpoint_path = PREVIOUS_DIR / "cnn_single_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing previous CNN model: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backbone = checkpoint["backbone"]
    input_size = int(checkpoint["input_size"])

    class GeM(nn.Module):
        def __init__(self, p: float = 3.0, eps: float = 1e-6):
            super().__init__()
            self.p = nn.Parameter(torch.ones(1) * p)
            self.eps = eps

        def forward(self, x):
            return torch.nn.functional.avg_pool2d(
                x.clamp(min=self.eps).pow(self.p),
                (x.size(-2), x.size(-1)),
            ).pow(1.0 / self.p).flatten(1)

    class RegressionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = timm.create_model(backbone, pretrained=False, num_classes=0, global_pool="")
            channels = int(getattr(self.encoder, "num_features", 0))
            self.pool = GeM()
            self.head = nn.Linear(channels, 1)

        def forward_features_and_output(self, x):
            features = self.encoder(x)
            if features.ndim == 2:
                pooled = features
            else:
                pooled = self.pool(features)
            output = self.head(pooled).squeeze(1)
            return pooled, output

    class ImageDataset(Dataset):
        def __init__(self, frame: pd.DataFrame):
            self.frame = frame.reset_index(drop=True)
            self.transform = transforms.Compose(
                [
                    transforms.Resize((input_size, input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        def __len__(self) -> int:
            return len(self.frame)

        def __getitem__(self, index: int):
            row = self.frame.iloc[index]
            image = Image.open(row["image_path"]).convert("RGB")
            return self.transform(image), index

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegressionModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(
        ImageDataset(manifest),
        batch_size=args.embedding_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device, non_blocking=True)
            embeddings, outputs = model.forward_features_and_output(images)
            embeddings_np = embeddings.detach().cpu().numpy()
            outputs_np = outputs.detach().cpu().numpy()
            for batch_pos, original_index in enumerate(indices.numpy().tolist()):
                base = manifest.iloc[int(original_index)].to_dict()
                pred_value = float(np.clip(outputs_np[batch_pos], 0.0, 4.0))
                class_probs = regression_class_probabilities(pred_value)
                pred_class = int(apply_thresholds(np.asarray([pred_value]), OPTIMIZED_CNN_THRESHOLDS)[0])
                row = {
                    "row_index": int(base["row_index"]),
                    "image_id": base["image_id"],
                    "image_path": base["image_path"],
                    "label": int(base["label"]),
                    "split": base["split"],
                    "cnn__continuous_grade": pred_value,
                    "cnn__predicted_class": pred_class,
                    "cnn__referable_probability": float(1.0 / (1.0 + math.exp(-(pred_value - OPTIMIZED_CNN_THRESHOLDS[1])))),
                    "cnn__confidence": float(max(class_probs)),
                    "cnn__uncertainty": float(1.0 - max(class_probs)),
                }
                for class_id, prob in enumerate(class_probs):
                    row[f"cnn__class_{class_id}_probability"] = float(prob)
                for dim, value in enumerate(embeddings_np[batch_pos]):
                    row[f"cnn_embedding__{dim:04d}"] = float(value)
                rows.append(row)
    return pd.DataFrame(rows)


def regression_class_probabilities(value: float) -> np.ndarray:
    centers = np.asarray([0, 1, 2, 3, 4], dtype=np.float64)
    logits = -np.square(centers - float(value))
    logits -= logits.max()
    probs = np.exp(logits)
    return probs / probs.sum()


def apply_thresholds(predictions: np.ndarray, thresholds: list[float]) -> np.ndarray:
    output = np.zeros(len(predictions), dtype=int)
    for threshold in thresholds:
        output += predictions > float(threshold)
    return np.clip(output, 0, 4)


def build_dictionary(table: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for column in table.columns:
        if column in {"row_index", "image_id", "image_path", "label", "split"}:
            continue
        if column.startswith("appdr203__"):
            group = "appdr_203_handcrafted"
            basis = "Existing AppDR 203-feature extractor output."
        elif column.startswith("selected75__"):
            group = "appdr_selected_75"
            basis = "Current production grading selected feature subset after AppDR feature engineering."
        elif column.startswith("selected100__"):
            group = "appdr_selected_100"
            basis = "Current production binary selected feature subset after AppDR feature engineering."
        elif column.startswith("cnn_embedding__"):
            group = "cnn_embedding"
            basis = "GeM pooled SEResNeXt50 embedding from previous Guanshuo-style CNN."
        else:
            group = "cnn_prediction"
            basis = "CNN regression output or derived probability/confidence feature."
        rows.append({"feature_name": column, "group": group, "basis": basis})
    return rows


def build_feature_sets(dictionary: list[dict[str, Any]]) -> dict[str, list[str]]:
    all_features = [row["feature_name"] for row in dictionary]
    appdr203 = [name for name in all_features if name.startswith("appdr203__")]
    selected75 = [name for name in all_features if name.startswith("selected75__")]
    selected100 = [name for name in all_features if name.startswith("selected100__")]
    cnn_pred = [name for name in all_features if name.startswith("cnn__")]
    cnn_embed = [name for name in all_features if name.startswith("cnn_embedding__")]
    return {
        "v1_appdr203_cnn_predictions": appdr203 + cnn_pred,
        "v2_appdr203_cnn_embedding": appdr203 + cnn_embed,
        "v3_selected75_cnn_predictions": selected75 + cnn_pred,
        "v3b_selected100_cnn_predictions": selected100 + cnn_pred,
        "v4_selected75_cnn_embedding": selected75 + cnn_embed,
        "v4b_selected100_cnn_embedding": selected100 + cnn_embed,
        "v5_cnn_prediction_only": cnn_pred,
        "v5b_cnn_embedding_only": cnn_embed,
    }


def audit_features(tables: dict[str, pd.DataFrame], dictionary: list[dict[str, Any]]) -> dict[str, Any]:
    feature_names = [row["feature_name"] for row in dictionary]
    combined = pd.concat([tables["train"][feature_names], tables["val"][feature_names], tables["test"][feature_names]], ignore_index=True)
    numeric = combined.apply(pd.to_numeric, errors="coerce")
    variances = numeric.var(numeric_only=True)
    return {
        "splits": {split: len(table) for split, table in tables.items()},
        "feature_count": len(feature_names),
        "nan_count": int(numeric.isna().sum().sum()),
        "pos_inf_count": int(np.isposinf(numeric.to_numpy(dtype=np.float64, copy=True)).sum()),
        "neg_inf_count": int(np.isneginf(numeric.to_numpy(dtype=np.float64, copy=True)).sum()),
        "constant_features": [str(name) for name, value in variances.items() if float(value) == 0.0],
        "extreme_value_features": [
            str(name)
            for name in feature_names
            if np.nanmax(np.abs(numeric[name].to_numpy(dtype=np.float64))) > 1e8
        ],
        "missing_cnn_output_count": int(numeric[[name for name in feature_names if name.startswith("cnn__")]].isna().sum().sum()),
    }


def write_audit(audit: dict[str, Any]) -> None:
    lines = [
        "# Hybrid Feature Audit",
        "",
        f"Split sizes: {audit['splits']}",
        f"Feature count: {audit['feature_count']}",
        f"NaN count: {audit['nan_count']}",
        f"Positive infinity count: {audit['pos_inf_count']}",
        f"Negative infinity count: {audit['neg_inf_count']}",
        f"Missing CNN output count: {audit['missing_cnn_output_count']}",
        f"Constant features: {len(audit['constant_features'])}",
        f"Extreme value features: {len(audit['extreme_value_features'])}",
        "",
        "Scaling is handled inside model pipelines where needed. Tree models use imputation only; linear/SVM models use imputation plus StandardScaler.",
    ]
    (OUTPUT_DIR / "feature_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = []
    for name in audit["constant_features"]:
        rows.append({"feature": name, "issue": "constant"})
    for name in audit["extreme_value_features"]:
        rows.append({"feature": name, "issue": "extreme_value"})
    write_rows(OUTPUT_DIR / "feature_audit.csv", rows or [{"feature": "", "issue": "none"}])


def train_5class_models(tables: dict[str, pd.DataFrame], feature_sets: dict[str, list[str]]) -> list[Result]:
    train = pd.concat([tables["train"], tables["val"]], ignore_index=True)
    test = tables["test"]
    y_train = train["label"].astype(int).to_numpy()
    y_test = test["label"].astype(int).to_numpy()
    results: list[Result] = []
    for feature_set, features in feature_sets.items():
        x_train = train[features]
        x_test = test[features]
        for model_name, model in models_for_feature_set(build_multiclass_models(), feature_set).items():
            try:
                fitted = clone(model)
                fitted.fit(x_train, y_train)
                pred = fitted.predict(x_test)
                metrics = five_class_metrics(y_test, pred)
                metrics.update({"model_name": model_name, "feature_set": feature_set, "feature_count": len(features)})
                results.append(Result("multiclass", model_name, feature_set, metrics, fitted, features))
            except Exception as error:
                results.append(Result("multiclass", model_name, feature_set, {"model_name": model_name, "feature_set": feature_set, "status": "failed", "error": str(error)[:300]}, None, features))
    return results


def train_binary_models(tables: dict[str, pd.DataFrame], feature_sets: dict[str, list[str]]) -> tuple[list[Result], list[dict[str, Any]]]:
    train = pd.concat([tables["train"], tables["val"]], ignore_index=True)
    test = tables["test"]
    y_train = (train["label"].astype(int).to_numpy() >= 2).astype(int)
    y_test = (test["label"].astype(int).to_numpy() >= 2).astype(int)
    results: list[Result] = []
    threshold_rows: list[dict[str, Any]] = []
    for feature_set, features in feature_sets.items():
        x_train = train[features]
        x_test = test[features]
        for model_name, model in models_for_feature_set(build_binary_models(), feature_set).items():
            try:
                fitted = clone(model)
                fitted.fit(x_train, y_train)
                probabilities = predict_binary_probability(fitted, x_test)
                pred_default = (probabilities >= 0.5).astype(int)
                metrics = binary_metrics(y_test, pred_default, probabilities, threshold=0.5)
                metrics.update({"model_name": model_name, "feature_set": feature_set, "feature_count": len(features)})
                results.append(Result("binary", model_name, feature_set, metrics, fitted, features))
                for threshold in THRESHOLDS:
                    pred = (probabilities >= threshold).astype(int)
                    row = binary_metrics(y_test, pred, probabilities, threshold=threshold)
                    row.update({"model_name": model_name, "feature_set": feature_set, "feature_count": len(features)})
                    threshold_rows.append(row)
            except Exception as error:
                results.append(Result("binary", model_name, feature_set, {"model_name": model_name, "feature_set": feature_set, "status": "failed", "error": str(error)[:300]}, None, features))
    return results, threshold_rows


def build_multiclass_models() -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))]),
        "random_forest": Pipeline([("imputer", SimpleImputer()), ("model", RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))]),
        "extra_trees": Pipeline([("imputer", SimpleImputer()), ("model", ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))]),
        "svm_rbf": Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("model", SVC(C=3.0, gamma="scale", class_weight="balanced", probability=True, random_state=RANDOM_STATE))]),
    }
    if XGBClassifier is not None:
        models["xgboost"] = Pipeline([("imputer", SimpleImputer()), ("model", XGBClassifier(n_estimators=160, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8, objective="multi:softprob", eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=2))])
    if LGBMClassifier is not None:
        models["lightgbm"] = Pipeline([("imputer", SimpleImputer()), ("model", LGBMClassifier(n_estimators=180, learning_rate=0.05, num_leaves=15, class_weight="balanced", random_state=RANDOM_STATE, verbose=-1))])
    if CatBoostClassifier is not None:
        models["catboost"] = Pipeline([("imputer", SimpleImputer()), ("model", CatBoostClassifier(iterations=120, depth=4, learning_rate=0.05, loss_function="MultiClass", verbose=False, random_seed=RANDOM_STATE))])
    if all(name in models for name in ["xgboost", "lightgbm", "random_forest", "svm_rbf"]):
        models["stacking_xgb_lgbm_rf_svm_lr"] = StackingClassifier(
            estimators=[
                ("xgb", models["xgboost"]),
                ("lgbm", models["lightgbm"]),
                ("rf", models["random_forest"]),
                ("svm", models["svm_rbf"]),
            ],
            final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
            stack_method="predict_proba",
            n_jobs=None,
        )
    return models


def models_for_feature_set(models: dict[str, Any], feature_set: str) -> dict[str, Any]:
    """Avoid very slow/high-risk combinations on 2k-dim embedding-only tables."""
    if "embedding" not in feature_set:
        return models
    allowed = {"logistic_regression", "random_forest", "extra_trees", "xgboost", "lightgbm"}
    return {name: model for name, model in models.items() if name in allowed}


def build_binary_models() -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))]),
        "random_forest": Pipeline([("imputer", SimpleImputer()), ("model", RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))]),
        "extra_trees": Pipeline([("imputer", SimpleImputer()), ("model", ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))]),
        "svm_rbf": Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("model", SVC(C=3.0, gamma="scale", class_weight="balanced", probability=True, random_state=RANDOM_STATE))]),
        "calibrated_svm": Pipeline([
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            (
                "model",
                CalibratedClassifierCV(
                    estimator=SVC(C=2.0, gamma="scale", class_weight="balanced", probability=False, random_state=RANDOM_STATE),
                    cv=3,
                    method="sigmoid",
                ),
            ),
        ]),
    }
    if XGBClassifier is not None:
        models["xgboost"] = Pipeline([("imputer", SimpleImputer()), ("model", XGBClassifier(n_estimators=160, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=2))])
        models["calibrated_xgboost"] = Pipeline([
            ("imputer", SimpleImputer()),
            (
                "model",
                CalibratedClassifierCV(
                    estimator=XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=2),
                    cv=3,
                    method="sigmoid",
                ),
            ),
        ])
    if LGBMClassifier is not None:
        models["lightgbm"] = Pipeline([("imputer", SimpleImputer()), ("model", LGBMClassifier(n_estimators=180, learning_rate=0.05, num_leaves=15, class_weight="balanced", random_state=RANDOM_STATE, verbose=-1))])
        models["calibrated_lightgbm"] = Pipeline([
            ("imputer", SimpleImputer()),
            (
                "model",
                CalibratedClassifierCV(
                    estimator=LGBMClassifier(n_estimators=120, learning_rate=0.05, num_leaves=15, class_weight="balanced", random_state=RANDOM_STATE, verbose=-1),
                    cv=3,
                    method="sigmoid",
                ),
            ),
        ])
    return models


def predict_binary_probability(model: Any, x_values: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_values)
        return np.asarray(proba)[:, 1]
    pred = model.predict(x_values)
    return np.asarray(pred, dtype=float)


def five_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4], output_dict=True, zero_division=0)
    row = {
        "status": "ok",
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    for label in [0, 1, 2, 3, 4]:
        metrics = report.get(str(label), {})
        row[f"class_{label}_precision"] = metrics.get("precision", 0.0)
        row[f"class_{label}_recall"] = metrics.get("recall", 0.0)
        row[f"class_{label}_f1"] = metrics.get("f1-score", 0.0)
        row[f"class_{label}_support"] = metrics.get("support", 0.0)
    return row


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None, threshold: float) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = ""
    if scores is not None and len(np.unique(y_true)) == 2:
        try:
            auc = roc_auc_score(y_true, scores)
        except Exception:
            auc = ""
    return {
        "status": "ok",
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "referable_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "non_referable_recall": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "auc": auc,
    }


def select_best_5class(results: list[Result]) -> Result:
    valid = [r for r in results if r.metrics.get("status") == "ok"]
    return max(valid, key=lambda r: (float(r.metrics.get("macro_f1", 0)), float(r.metrics.get("balanced_accuracy", 0))))


def select_best_binary(results: list[Result], threshold_rows: list[dict[str, Any]]) -> tuple[Result, dict[str, Any]]:
    valid_thresholds = [row for row in threshold_rows if row.get("status") == "ok"]
    best_row = max(
        valid_thresholds,
        key=lambda row: (
            float(row.get("referable_recall", 0)),
            -float(row.get("false_negatives", 999)),
            float(row.get("balanced_accuracy", 0)),
            float(row.get("f1", 0)),
        ),
    )
    matching = [
        result for result in results
        if result.model_name == best_row["model_name"] and result.feature_set == best_row["feature_set"]
    ]
    return matching[0], best_row


def export_best_models(best_five: Result, best_binary: Result) -> None:
    with (OUTPUT_DIR / "hybrid_5class_best_model.pkl").open("wb") as file:
        pickle.dump({"model": best_five.model, "features": best_five.features, "metrics": best_five.metrics}, file)
    with (OUTPUT_DIR / "hybrid_binary_best_model.pkl").open("wb") as file:
        pickle.dump({"model": best_binary.model, "features": best_binary.features, "metrics": best_binary.metrics}, file)


def write_best_metric_files(best_five: Result, best_binary: Result, best_threshold_row: dict[str, Any]) -> None:
    write_json(OUTPUT_DIR / "hybrid_5class_metrics.json", best_five.metrics)
    write_json(OUTPUT_DIR / "hybrid_binary_metrics.json", best_threshold_row)
    test = pd.read_csv(OUTPUT_DIR / "hybrid_features_test.csv")
    y_test_5 = test["label"].astype(int).to_numpy()
    pred_5 = best_five.model.predict(test[best_five.features])
    write_confusion(OUTPUT_DIR / "hybrid_5class_confusion_matrix.csv", y_test_5, pred_5, labels=[0, 1, 2, 3, 4])
    per_class = []
    report = classification_report(y_test_5, pred_5, labels=[0, 1, 2, 3, 4], output_dict=True, zero_division=0)
    for label in [0, 1, 2, 3, 4]:
        row = {"class": label}
        row.update(report.get(str(label), {}))
        per_class.append(row)
    write_rows(OUTPUT_DIR / "hybrid_5class_per_class_metrics.csv", per_class)
    y_test_bin = (y_test_5 >= 2).astype(int)
    scores = predict_binary_probability(best_binary.model, test[best_binary.features])
    pred_bin = (scores >= float(best_threshold_row["threshold"])).astype(int)
    write_confusion(OUTPUT_DIR / "hybrid_binary_confusion_matrix.csv", y_test_bin, pred_bin, labels=[0, 1])


def write_confusion(path: Path, y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for idx, values in enumerate(matrix):
        row = {"actual_class": labels[idx]}
        for pred_label, value in zip(labels, values):
            row[f"predicted_{pred_label}"] = int(value)
        rows.append(row)
    write_rows(path, rows)


def write_comparison_reports(best_five: Result, best_binary: Result, best_threshold_row: dict[str, Any]) -> None:
    prev = PREVIOUS_DIR
    appdr_5 = read_first(prev / "appdr_current_5class_metrics.csv")
    cnn_5 = read_first(prev / "cnn_single_5class_metrics.csv")
    appdr_binary = read_first(prev / "appdr_current_binary_metrics.csv")
    cnn_binary = read_first(prev / "cnn_single_binary_metrics.csv")
    best_feature = {
        "Model": "AppDR best experimental feature-based grading",
        "Input type": "handcrafted feature-vector",
        "Uses handcrafted features?": True,
        "Uses CNN?": False,
        "Accuracy": 0.6700,
        "Balanced accuracy": 0.6113,
        "Macro F1": 0.5799,
        "Class 1 recall": 0.4533,
        "Class 3 recall": 0.6000,
        "Class 4 recall": 0.6433,
        "Notes": "Reported prior experiment, not rerun on capped test.",
    }
    comparison_5 = [
        normalize_5_row(appdr_5, "AppDR production XGBoost", "203 handcrafted features", True, False, "Same effective capped test manifest."),
        best_feature,
        normalize_5_row(cnn_5, "Straight Guanshuo-style CNN", "CNN image model", False, True, "Previous capped CNN run."),
        normalize_5_row(best_five.metrics, f"Hybrid {best_five.model_name}", best_five.feature_set, True, True, "Best hybrid 5-class model."),
    ]
    write_rows(OUTPUT_DIR / "comparison_5class_all.csv", comparison_5)
    best_binary_prior = {
        "Model": "AppDR best experimental binary screening",
        "Input type": "expanded handcrafted feature-vector",
        "Accuracy": 0.7512,
        "Referable recall": 0.9697,
        "False negatives": 48,
        "False positives": 817,
        "Notes": "Reported prior experiment, not rerun on capped test.",
    }
    comparison_binary = [
        normalize_binary_row(appdr_binary, "AppDR production SVM", "203 handcrafted features", "Same effective capped test manifest."),
        best_binary_prior,
        normalize_binary_row(cnn_binary, "Straight CNN binary result", "CNN regression converted to binary", "Previous capped CNN run."),
        normalize_binary_row(cnn_binary, "5-class CNN converted to binary", "CNN class >= 2", "Same as straight CNN binary conversion."),
        normalize_binary_row(best_threshold_row, f"Hybrid {best_threshold_row['model_name']}", best_threshold_row["feature_set"], "Best hybrid threshold sweep row."),
    ]
    write_rows(OUTPUT_DIR / "comparison_binary_all.csv", comparison_binary)
    report = build_report(best_five, best_binary, best_threshold_row, comparison_5, comparison_binary)
    (OUTPUT_DIR / "hybrid_vs_appdr_vs_cnn_report.md").write_text(report["markdown"], encoding="utf-8")
    write_json(OUTPUT_DIR / "hybrid_vs_appdr_vs_cnn_report.json", report)
    (OUTPUT_DIR / "final_recommendation.md").write_text(report["recommendation"], encoding="utf-8")


def normalize_5_row(row: dict[str, Any], model: str, input_type: str, uses_handcrafted: bool, uses_cnn: bool, notes: str) -> dict[str, Any]:
    out = {"Model": model, "Input type": input_type, "Uses handcrafted features?": uses_handcrafted, "Uses CNN?": uses_cnn}
    mapping = {
        "Accuracy": "accuracy",
        "Balanced accuracy": "balanced_accuracy",
        "Macro precision": "macro_precision",
        "Macro recall": "macro_recall",
        "Macro F1": "macro_f1",
        "Weighted F1": "weighted_f1",
        "Class 0 precision": "class_0_precision",
        "Class 0 recall": "class_0_recall",
        "Class 1 precision": "class_1_precision",
        "Class 1 recall": "class_1_recall",
        "Class 2 precision": "class_2_precision",
        "Class 2 recall": "class_2_recall",
        "Class 3 precision": "class_3_precision",
        "Class 3 recall": "class_3_recall",
        "Class 4 precision": "class_4_precision",
        "Class 4 recall": "class_4_recall",
    }
    for label, key in mapping.items():
        out[label] = row.get(key, "")
    out["Confusion matrix path"] = str(OUTPUT_DIR / "hybrid_5class_confusion_matrix.csv") if model.startswith("Hybrid") else ""
    out["Notes"] = notes
    return out


def normalize_binary_row(row: dict[str, Any], model: str, input_type: str, notes: str) -> dict[str, Any]:
    out = {"Model": model, "Input type": input_type}
    for label, key in [
        ("Accuracy", "accuracy"),
        ("Balanced accuracy", "balanced_accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("Referable recall", "referable_recall"),
        ("Non-referable recall", "non_referable_recall"),
        ("False negatives", "false_negatives"),
        ("False positives", "false_positives"),
        ("AUC", "auc"),
        ("Threshold", "threshold"),
    ]:
        out[label] = row.get(key, "")
    out["Notes"] = notes
    return out


def build_report(
    best_five: Result,
    best_binary: Result,
    best_threshold: dict[str, Any],
    comparison_5: list[dict[str, Any]],
    comparison_binary: list[dict[str, Any]],
) -> dict[str, Any]:
    hybrid_5 = best_five.metrics
    hybrid_bin = best_threshold
    appdr_5 = comparison_5[0]
    cnn_5 = comparison_5[2]
    appdr_bin = comparison_binary[0]
    cnn_bin = comparison_binary[2]
    beat_appdr_5 = ffloat(hybrid_5.get("macro_f1")) > ffloat(appdr_5.get("Macro F1"))
    beat_cnn_5 = ffloat(hybrid_5.get("macro_f1")) > ffloat(cnn_5.get("Macro F1"))
    hybrid_fp = int(float(hybrid_bin.get("false_positives", 999)))
    appdr_fp = int(float(appdr_bin.get("False positives", 999)))
    cnn_fp = int(float(cnn_bin.get("False positives", 999)))
    beat_appdr_bin_sensitivity = ffloat(hybrid_bin.get("referable_recall")) >= ffloat(appdr_bin.get("Referable recall")) and int(float(hybrid_bin.get("false_negatives", 999))) <= int(float(appdr_bin.get("False negatives", 999)))
    beat_cnn_bin_sensitivity = ffloat(hybrid_bin.get("referable_recall")) >= ffloat(cnn_bin.get("Referable recall")) and int(float(hybrid_bin.get("false_negatives", 999))) <= int(float(cnn_bin.get("False negatives", 999)))
    false_positives_higher_than_appdr = hybrid_fp > appdr_fp
    false_positives_higher_than_cnn = hybrid_fp > cnn_fp
    recommend_integration_candidate = (
        beat_appdr_5
        and beat_cnn_5
        and ffloat(hybrid_5.get("class_4_recall")) > 0.1
        and not false_positives_higher_than_appdr
    )
    markdown = "\n".join(
        [
            "# Hybrid CNN + AppDR Features Comparison",
            "",
            "Experiment only. Production was not updated.",
            "",
            "## Dataset",
            "",
            "Reused the effective capped APTOS split from the previous Guanshuo-style CNN run.",
            "",
            "## Best Hybrid 5-Class Model",
            "",
            f"Model: {best_five.model_name}",
            f"Feature set: {best_five.feature_set}",
            f"Accuracy: {pct(hybrid_5.get('accuracy'))}",
            f"Balanced accuracy: {pct(hybrid_5.get('balanced_accuracy'))}",
            f"Macro precision: {pct(hybrid_5.get('macro_precision'))}",
            f"Macro recall: {pct(hybrid_5.get('macro_recall'))}",
            f"Macro F1: {pct(hybrid_5.get('macro_f1'))}",
            f"Class 1 recall: {pct(hybrid_5.get('class_1_recall'))}",
            f"Class 3 recall: {pct(hybrid_5.get('class_3_recall'))}",
            f"Class 4 recall: {pct(hybrid_5.get('class_4_recall'))}",
            "",
            "## Best Hybrid Binary Model",
            "",
            f"Model: {best_threshold.get('model_name')} / {best_threshold.get('feature_set')}",
            f"Threshold: {best_threshold.get('threshold')}",
            f"Accuracy: {pct(hybrid_bin.get('accuracy'))}",
            f"Referable recall: {pct(hybrid_bin.get('referable_recall'))}",
            f"False negatives: {hybrid_bin.get('false_negatives')}",
            f"False positives: {hybrid_bin.get('false_positives')}",
            "",
            "## Decision",
            "",
            f"Hybrid beat AppDR production macro F1 on this capped split: {beat_appdr_5}",
            f"Hybrid beat straight CNN macro F1 on this capped split: {beat_cnn_5}",
            f"Hybrid improved AppDR binary sensitivity on this capped split: {beat_appdr_bin_sensitivity}",
            f"Hybrid improved straight CNN binary sensitivity on this capped split: {beat_cnn_bin_sensitivity}",
            f"Hybrid false positives higher than AppDR: {false_positives_higher_than_appdr} ({hybrid_fp} vs {appdr_fp})",
            f"Hybrid false positives higher than straight CNN: {false_positives_higher_than_cnn} ({hybrid_fp} vs {cnn_fp})",
            "",
            "Do not replace production. This run is capped and experimental. Hybrid may be a next integration candidate only after a larger/full split repeat confirms grading safety and keeps false positives usable.",
        ]
    ) + "\n"
    recommendation = "\n".join(
        [
            "# Final Recommendation",
            "",
            "Do not update production.",
            "",
            f"Hybrid integration candidate later: {recommend_integration_candidate}",
            "",
            "Reason: the hybrid experiment is promising only if it improves both grading safety and screening safety on a larger, fair validation/test setup. This run reused the capped CNN split and should not be overclaimed.",
        ]
    ) + "\n"
    return {
        "best_5class": hybrid_5,
        "best_binary": hybrid_bin,
        "beat_appdr_5class_macro_f1": beat_appdr_5,
        "beat_cnn_5class_macro_f1": beat_cnn_5,
        "beat_appdr_binary_sensitivity": beat_appdr_bin_sensitivity,
        "beat_cnn_binary_sensitivity": beat_cnn_bin_sensitivity,
        "false_positives_higher_than_appdr": false_positives_higher_than_appdr,
        "false_positives_higher_than_cnn": false_positives_higher_than_cnn,
        "recommendation": recommendation,
        "markdown": markdown,
    }


def read_first(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return rows[0] if rows else {}


def ffloat(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "n/a"


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
