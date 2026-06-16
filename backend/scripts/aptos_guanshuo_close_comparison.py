"""Close Guanshuo Xu APTOS-style CNN comparison for AppDR.

Experiment only. This script does not update production artifacts, does not
modify backend/frontend prediction behavior, and does not remove the 203-feature
pipeline.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import pickle
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
APTOS_DIR = BACKEND_DIR / "images" / "aptos2019"
OUTPUT_DIR = BACKEND_DIR / "results" / "aptos_guanshuo_close_comparison"
RANDOM_STATE = 42
DEFAULT_THRESHOLDS = [0.5, 1.5, 2.5, 3.5]
GUANSHUO_THRESHOLDS = [0.7, 1.5, 2.5, 3.5]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

APPDR_PRODUCTION_5CLASS = {
    "model_name": "AppDR production XGBoost",
    "accuracy": 0.6798,
    "balanced_accuracy": 0.5312,
    "macro_precision": 0.4927,
    "macro_recall": 0.5312,
    "macro_f1": 0.5077,
}
APPDR_PRODUCTION_BINARY = {
    "model_name": "AppDR production SVM RBF",
    "accuracy": 0.7932,
    "referable_recall": 0.9373,
    "false_negatives": 88,
    "f1": 0.7995,
}
BEST_FEATURE_GRADING = {
    "model_name": "Study-expanded LightGBM top150",
    "accuracy": 0.6700,
    "balanced_accuracy": 0.6113,
    "macro_f1": 0.5799,
    "class_1_recall": 0.4533,
    "class_3_recall": 0.6000,
    "class_4_recall": 0.6433,
}


@dataclass
class TrainingResult:
    status: str
    model_name: str
    backbone: str
    input_size: int
    pretrained: bool
    gem_pooling: bool
    reason: str


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    dataset_manifest = build_dataset_manifest()
    train_manifest, val_manifest, test_manifest = split_dataset(dataset_manifest)
    write_dataframe(OUTPUT_DIR / "dataset_manifest.csv", dataset_manifest)
    write_dataframe(OUTPUT_DIR / "train_manifest.csv", train_manifest)
    write_dataframe(OUTPUT_DIR / "val_manifest.csv", val_manifest)
    write_dataframe(OUTPUT_DIR / "test_manifest.csv", test_manifest)
    write_dataset_summary(dataset_manifest, train_manifest, val_manifest, test_manifest)
    write_split_summary(train_manifest, val_manifest, test_manifest)

    run_train_manifest = apply_run_limit(train_manifest, args.train_limit)
    run_val_manifest = apply_run_limit(val_manifest, args.val_limit)
    run_test_manifest = apply_run_limit(test_manifest, args.test_limit)
    write_dataframe(OUTPUT_DIR / "run_train_manifest.csv", run_train_manifest)
    write_dataframe(OUTPUT_DIR / "run_val_manifest.csv", run_val_manifest)
    write_dataframe(OUTPUT_DIR / "run_test_manifest.csv", run_test_manifest)

    appdr_5, appdr_binary = evaluate_appdr_current(run_test_manifest)
    write_rows(OUTPUT_DIR / "appdr_current_5class_metrics.csv", [appdr_5])
    write_rows(OUTPUT_DIR / "appdr_current_binary_metrics.csv", [appdr_binary])

    config = build_cnn_config(args)
    write_json(OUTPUT_DIR / "cnn_single_model_config.json", config)

    if args.report_only:
        existing_cnn = read_first_row(OUTPUT_DIR / "cnn_single_5class_metrics.csv")
        status = existing_cnn.get("status", "not_run")
        cnn_result = TrainingResult(
            status="trained" if status == "trained" else "not_run",
            model_name="Guanshuo-style single CNN",
            backbone="seresnext50_32x4d" if status == "trained" else "",
            input_size=args.input_size,
            pretrained=True if status == "trained" else False,
            gem_pooling=True if status == "trained" else False,
            reason="" if status == "trained" else "No existing trained CNN metric row was found.",
        )
    else:
        cnn_result = run_cnn_level1(args, run_train_manifest, run_val_manifest, run_test_manifest, config)
    write_required_reports(cnn_result, appdr_5, appdr_binary)
    print(f"Created {OUTPUT_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=0, help="Optional development cap for train rows; 0 means full train split.")
    parser.add_argument("--val-limit", type=int, default=0, help="Optional development cap for val rows; 0 means full val split.")
    parser.add_argument("--test-limit", type=int, default=0, help="Optional development cap for test rows; 0 means full test split.")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--force-skip-cnn", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Refresh comparison reports from existing CNN metric files without retraining.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_dataset_manifest() -> pd.DataFrame:
    train_csv = APTOS_DIR / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing {train_csv}")
    table = pd.read_csv(train_csv)
    rows: list[dict[str, Any]] = []
    for index, row in table.reset_index(drop=True).iterrows():
        image_id = str(row["id_code"])
        image_path = APTOS_DIR / "train_images" / f"{image_id}.png"
        rows.append(
            {
                "row_index": int(index),
                "image_id": image_id,
                "image_path": str(image_path),
                "label": int(row["diagnosis"]),
                "source_dataset": "APTOS 2019 train",
                "patient_id": "",
                "split_group": image_id,
                "exists": image_path.exists(),
            }
        )
    frame = pd.DataFrame(rows)
    return frame[frame["exists"]].reset_index(drop=True)


def split_dataset(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(
        dataset,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=dataset["label"],
    )
    train, val = train_test_split(
        train_val,
        test_size=0.15 / 0.85,
        random_state=RANDOM_STATE,
        stratify=train_val["label"],
    )
    return (
        train.reset_index(drop=True).assign(split="train"),
        val.reset_index(drop=True).assign(split="val"),
        test.reset_index(drop=True).assign(split="test"),
    )


def write_dataset_summary(dataset: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    lines = [
        "# Dataset Summary",
        "",
        "Dataset: APTOS 2019 labeled train images.",
        f"Total labeled images found: {len(dataset)}",
        "Patient IDs: not available, so split is image-level and stratified by class.",
        "Unlabeled APTOS test images: not used for evaluation or pseudo-labeling.",
        "External IDRiD/Messidor/OIA-DDR data: not used in this first close comparison.",
        "",
        "## Class Counts",
        "",
        "| Split | 0 | 1 | 2 | 3 | 4 | Total |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, frame in [("all", dataset), ("train", train), ("val", val), ("test", test)]:
        counts = frame["label"].value_counts().sort_index().to_dict()
        lines.append(
            f"| {name} | {counts.get(0, 0)} | {counts.get(1, 0)} | {counts.get(2, 0)} | {counts.get(3, 0)} | {counts.get(4, 0)} | {len(frame)} |"
        )
    (OUTPUT_DIR / "dataset_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_split_summary(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    lines = [
        "# Split Summary",
        "",
        "Split policy: 70% train, 15% validation, 15% test, stratified by class, random_state=42.",
        "No patient_id column exists in APTOS train.csv, so patient-level splitting was not possible.",
        "",
        f"Train rows: {len(train)}",
        f"Validation rows: {len(val)}",
        f"Test rows: {len(test)}",
    ]
    (OUTPUT_DIR / "split_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_appdr_current(test_manifest: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    features_csv = BACKEND_DIR / "features.csv"
    model_path = BACKEND_DIR / "results" / "best_model.pkl"
    binary_path = BACKEND_DIR / "results" / "binary" / "best_model.pkl"
    if not (features_csv.exists() and model_path.exists() and binary_path.exists()):
        return static_appdr_5class_row("missing_artifact"), static_appdr_binary_row("missing_artifact")

    features = pd.read_csv(features_csv)
    labels = pd.read_csv(APTOS_DIR / "train.csv")["diagnosis"].astype(int).to_numpy()
    test_indices = test_manifest["row_index"].astype(int).to_numpy()
    x_test = features.drop(columns=["label"], errors="ignore").iloc[test_indices]
    y_test = labels[test_indices]

    with model_path.open("rb") as file:
        grading_artifact = pickle.load(file)
    with binary_path.open("rb") as file:
        binary_artifact = pickle.load(file)
    grading_model = grading_artifact.get("model") if isinstance(grading_artifact, dict) else grading_artifact
    binary_model = binary_artifact.get("model") if isinstance(binary_artifact, dict) else binary_artifact

    grading_pred = grading_model.predict(x_test)
    grading_prob = predict_proba_or_none(grading_model, x_test)
    binary_prob = predict_proba_or_none(binary_model, x_test)
    if binary_prob is not None and binary_prob.shape[1] > 1:
        binary_pred = (binary_prob[:, 1] >= 0.20).astype(int)
    else:
        binary_pred = binary_model.predict(x_test)
    y_binary = (y_test >= 2).astype(int)

    return (
        five_class_metrics(
            "AppDR production XGBoost on same APTOS split",
            y_test,
            grading_pred,
            grading_prob,
            status="evaluated_same_split",
        ),
        binary_metrics(
            "AppDR production SVM RBF on same APTOS split",
            y_binary,
            binary_pred,
            binary_prob[:, 1] if binary_prob is not None and binary_prob.shape[1] > 1 else None,
            threshold="0.20",
            status="evaluated_same_split",
        ),
    )


def static_appdr_5class_row(status: str) -> dict[str, Any]:
    return {
        "model_name": APPDR_PRODUCTION_5CLASS["model_name"],
        "status": status,
        **{k: v for k, v in APPDR_PRODUCTION_5CLASS.items() if k != "model_name"},
    }


def static_appdr_binary_row(status: str) -> dict[str, Any]:
    return {
        "model_name": APPDR_PRODUCTION_BINARY["model_name"],
        "status": status,
        **{k: v for k, v in APPDR_PRODUCTION_BINARY.items() if k != "model_name"},
    }


def predict_proba_or_none(model: Any, x_values: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x_values))
    return None


def build_cnn_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "basis": "Guanshuo Xu APTOS 2019 1st-place writeup",
        "gem_source": "https://github.com/filipradenovic/cnnimageretrieval-pytorch",
        "level": "Level 1 practical single-model reproduction",
        "preferred_backbone_order": [
            "seresnext50_32x4d",
            "seresnext101_32x4d",
            "inception_resnet_v2",
            "inception_v4",
            "resnet50",
            "efficientnet_b0",
            "efficientnet_b3",
            "mobilenetv3_large_100",
        ],
        "selected_input_size": args.input_size,
        "plain_resizing_only": True,
        "rgb_input": True,
        "normalization": "ImageNet mean/std",
        "loss": "SmoothL1Loss",
        "output": "single continuous DR grade value",
        "thresholds": {
            "default": DEFAULT_THRESHOLDS,
            "guanshuo_adjusted": GUANSHUO_THRESHOLDS,
            "optimized": "validation-only coordinate search",
        },
        "augmentations": {
            "contrast_range": 0.2,
            "brightness_range": 20,
            "hue_range": 10,
            "saturation_range": 20,
            "blur_and_sharpen": True,
            "rotate_range": 180,
            "scale_range": 0.2,
            "shear_range": 0.2,
            "shift_range": 0.2,
            "do_mirror": True,
        },
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "pretrained_requested": not args.no_pretrained,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "test_limit": args.test_limit,
        "limit_note": "0 means full split. Non-zero limits are compute-bounded smoke/comparison runs and must not be overclaimed.",
    }


def run_cnn_level1(
    args: argparse.Namespace,
    train_manifest: pd.DataFrame,
    val_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame,
    config: dict[str, Any],
) -> TrainingResult:
    if args.force_skip_cnn:
        return write_blocked_cnn_outputs("force_skip_cnn was set.", config)
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
        from PIL import Image, ImageFilter, ImageEnhance
        import timm
    except Exception as error:
        return write_blocked_cnn_outputs(f"Deep-learning import failed: {error}", config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = select_backbone(timm, config["preferred_backbone_order"])
    if backbone is None:
        return write_blocked_cnn_outputs("No requested timm backbone is available.", config)

    if args.train_limit > 0:
        train_manifest = stratified_cap(train_manifest, args.train_limit)
    if args.val_limit > 0:
        val_manifest = stratified_cap(val_manifest, args.val_limit)
    if args.test_limit > 0:
        test_manifest = stratified_cap(test_manifest, args.test_limit)

    class FundusDataset(Dataset):
        def __init__(self, frame: pd.DataFrame, train: bool):
            self.frame = frame.reset_index(drop=True)
            self.train = train
            self.base_transform = transforms.Compose(
                [
                    transforms.Resize((args.input_size, args.input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

        def __len__(self) -> int:
            return len(self.frame)

        def __getitem__(self, index: int):
            row = self.frame.iloc[index]
            image = Image.open(row["image_path"]).convert("RGB")
            if self.train:
                image = apply_fundus_safe_augmentation(image, ImageFilter, ImageEnhance)
            tensor = self.base_transform(image)
            label = torch.tensor(float(row["label"]), dtype=torch.float32)
            return tensor, label

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
        def __init__(self, name: str, pretrained_requested: bool):
            super().__init__()
            self.pretrained = pretrained_requested
            try:
                self.encoder = timm.create_model(
                    name,
                    pretrained=pretrained_requested,
                    num_classes=0,
                    global_pool="",
                )
            except Exception:
                self.pretrained = False
                self.encoder = timm.create_model(name, pretrained=False, num_classes=0, global_pool="")
            channels = int(getattr(self.encoder, "num_features", 0))
            if channels <= 0:
                raise ValueError(f"Could not infer feature channels for {name}")
            self.pool = GeM()
            self.head = nn.Linear(channels, 1)

        def forward(self, x):
            features = self.encoder(x)
            if features.ndim == 2:
                pooled = features
            else:
                pooled = self.pool(features)
            return self.head(pooled).squeeze(1)

    model = RegressionModel(backbone, not args.no_pretrained).to(device)
    pretrained_used = bool(model.pretrained)
    train_loader = DataLoader(
        FundusDataset(train_manifest, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        FundusDataset(val_manifest, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        FundusDataset(test_manifest, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    log_rows: list[dict[str, Any]] = []
    best_state = None
    best_val = math.inf
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images).clamp(0.0, 4.0)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu()) * len(labels)
            train_count += len(labels)
            if train_count % max(args.batch_size * 25, 1) == 0:
                write_rows(
                    OUTPUT_DIR / "cnn_single_training_progress.csv",
                    [
                        {
                            "epoch": epoch,
                            "processed_train_images": train_count,
                            "train_split_rows_this_run": len(train_manifest),
                            "elapsed_seconds": round(time.time() - start, 2),
                        }
                    ],
                )
        val_loss, val_predictions, val_labels = predict_regression(model, val_loader, device, criterion)
        val_thresholds = optimize_thresholds(val_labels, val_predictions)
        val_classes = apply_thresholds(val_predictions, val_thresholds)
        val_macro_f1 = f1_score(val_labels.astype(int), val_classes, average="macro", zero_division=0)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_count, 1),
            "val_loss": val_loss,
            "val_macro_f1_optimized_thresholds": val_macro_f1,
            "optimized_thresholds": json.dumps(val_thresholds),
            "elapsed_seconds": round(time.time() - start, 2),
        }
        log_rows.append(row)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        write_rows(OUTPUT_DIR / "cnn_single_training_log.csv", log_rows)

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "backbone": backbone,
            "input_size": args.input_size,
            "pretrained": pretrained_used,
            "gem_pooling": True,
            "config": config,
        },
        OUTPUT_DIR / "cnn_single_model.pt",
    )

    _, val_predictions, val_labels = predict_regression(model, val_loader, device, criterion)
    optimized = optimize_thresholds(val_labels, val_predictions)
    _, test_predictions, test_labels = predict_regression(model, test_loader, device, criterion)
    threshold_rows = []
    metric_rows = []
    for name, thresholds in [
        ("default_0.5_1.5_2.5_3.5", DEFAULT_THRESHOLDS),
        ("guanshuo_0.7_1.5_2.5_3.5", GUANSHUO_THRESHOLDS),
        ("validation_optimized", optimized),
    ]:
        pred_classes = apply_thresholds(test_predictions, thresholds)
        five = five_class_metrics(f"Guanshuo-style CNN single ({name})", test_labels, pred_classes, None, status="trained")
        binary = binary_metrics(
            f"Guanshuo-style CNN single ({name})",
            (test_labels >= 2).astype(int),
            (pred_classes >= 2).astype(int),
            score=test_predictions,
            threshold=str(thresholds),
            status="trained",
        )
        five["threshold_mode"] = name
        five["thresholds"] = json.dumps(thresholds)
        binary["threshold_mode"] = name
        binary["thresholds"] = json.dumps(thresholds)
        metric_rows.append(five)
        threshold_rows.append(
            {
                "threshold_mode": name,
                "thresholds": json.dumps(thresholds),
                "macro_f1": five["macro_f1"],
                "balanced_accuracy": five["balanced_accuracy"],
                "class_1_recall": five["class_1_recall"],
                "class_3_recall": five["class_3_recall"],
                "class_4_recall": five["class_4_recall"],
                "binary_referable_recall": binary["referable_recall"],
                "binary_false_negatives": binary["false_negatives"],
                "binary_false_positives": binary["false_positives"],
            }
        )
        if name == "validation_optimized":
            write_rows(OUTPUT_DIR / "cnn_single_5class_metrics.csv", [five])
            write_rows(OUTPUT_DIR / "cnn_single_binary_metrics.csv", [binary])
            write_confusion_matrix(OUTPUT_DIR / "cnn_single_confusion_matrix.csv", test_labels, pred_classes)
    write_rows(OUTPUT_DIR / "threshold_comparison.csv", threshold_rows)
    write_rows(OUTPUT_DIR / "cnn_all_threshold_5class_metrics.csv", metric_rows)
    return TrainingResult("trained", "Guanshuo-style single CNN", backbone, args.input_size, pretrained_used, True, "")


def select_backbone(timm_module: Any, order: list[str]) -> str | None:
    available = set(timm_module.list_models())
    for name in order:
        if name in available:
            return name
    return None


def stratified_cap(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0 or len(frame) <= limit:
        return frame
    parts = []
    per_class = max(1, limit // frame["label"].nunique())
    for _, group in frame.groupby("label"):
        parts.append(group.sample(min(len(group), per_class), random_state=RANDOM_STATE))
    capped = pd.concat(parts).sample(frac=1.0, random_state=RANDOM_STATE)
    if len(capped) > limit:
        capped = capped.sample(limit, random_state=RANDOM_STATE)
    return capped.reset_index(drop=True)


def apply_run_limit(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    limited = stratified_cap(frame, limit)
    return limited.reset_index(drop=True)


def apply_fundus_safe_augmentation(image: Any, image_filter: Any, image_enhance: Any) -> Any:
    if random.random() < 0.5:
        image = image.transpose(method=0)  # PIL.Image.FLIP_LEFT_RIGHT without importing enum.
    angle = random.uniform(-180, 180)
    image = image.rotate(angle)
    if random.random() < 0.8:
        image = image_enhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.8:
        image = image_enhance.Brightness(image).enhance(random.uniform(0.92, 1.08))
    if random.random() < 0.3:
        image = image.filter(image_filter.GaussianBlur(radius=random.uniform(0.1, 1.0)))
    if random.random() < 0.3:
        image = image.filter(image_filter.SHARPEN)
    return image


def predict_regression(model: Any, loader: Any, device: Any, criterion: Any) -> tuple[float, np.ndarray, np.ndarray]:
    import torch

    model.eval()
    losses = 0.0
    count = 0
    predictions: list[float] = []
    labels_all: list[float] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images).clamp(0.0, 4.0)
            loss = criterion(outputs, labels)
            losses += float(loss.detach().cpu()) * len(labels)
            count += len(labels)
            predictions.extend(outputs.detach().cpu().numpy().tolist())
            labels_all.extend(labels.detach().cpu().numpy().tolist())
    return losses / max(count, 1), np.asarray(predictions, dtype=np.float64), np.asarray(labels_all, dtype=np.int64)


def apply_thresholds(predictions: np.ndarray, thresholds: list[float]) -> np.ndarray:
    values = np.asarray(predictions, dtype=np.float64)
    output = np.zeros(len(values), dtype=int)
    for threshold in thresholds:
        output += values > float(threshold)
    return np.clip(output, 0, 4)


def optimize_thresholds(labels: np.ndarray, predictions: np.ndarray) -> list[float]:
    best = list(GUANSHUO_THRESHOLDS)
    best_score = cohen_kappa_score(labels.astype(int), apply_thresholds(predictions, best), weights="quadratic")
    grids = [
        np.arange(0.3, 1.01, 0.1),
        np.arange(1.1, 2.01, 0.1),
        np.arange(2.1, 3.01, 0.1),
        np.arange(3.1, 3.91, 0.1),
    ]
    for _ in range(2):
        for index, grid in enumerate(grids):
            for value in grid:
                candidate = list(best)
                candidate[index] = round(float(value), 2)
                if not all(candidate[i] < candidate[i + 1] for i in range(3)):
                    continue
                score = cohen_kappa_score(labels.astype(int), apply_thresholds(predictions, candidate), weights="quadratic")
                if score > best_score:
                    best = candidate
                    best_score = score
    return best


def write_blocked_cnn_outputs(reason: str, config: dict[str, Any]) -> TrainingResult:
    write_rows(OUTPUT_DIR / "cnn_single_training_log.csv", [{"status": "not_run", "reason": reason}])
    blocked_5 = blocked_metric_row("Guanshuo-style single CNN", reason)
    blocked_binary = blocked_metric_row("Guanshuo-style single CNN", reason)
    write_rows(OUTPUT_DIR / "cnn_single_5class_metrics.csv", [blocked_5])
    write_rows(OUTPUT_DIR / "cnn_single_binary_metrics.csv", [blocked_binary])
    write_rows(OUTPUT_DIR / "cnn_single_confusion_matrix.csv", [{"status": "not_run", "reason": reason}])
    write_rows(
        OUTPUT_DIR / "threshold_comparison.csv",
        [
            {"threshold_mode": "default", "thresholds": json.dumps(DEFAULT_THRESHOLDS), "status": "not_run", "reason": reason},
            {"threshold_mode": "guanshuo_adjusted", "thresholds": json.dumps(GUANSHUO_THRESHOLDS), "status": "not_run", "reason": reason},
            {"threshold_mode": "validation_optimized", "thresholds": "validation_only", "status": "not_run", "reason": reason},
        ],
    )
    return TrainingResult("not_run", "Guanshuo-style single CNN", "", int(config["selected_input_size"]), False, False, reason)


def blocked_metric_row(model_name: str, reason: str) -> dict[str, Any]:
    return {"model_name": model_name, "status": "not_run", "reason": reason}


def five_class_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    status: str,
) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4], output_dict=True, zero_division=0)
    row = {
        "model_name": model_name,
        "status": status,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "quadratic_weighted_kappa": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }
    for label in [0, 1, 2, 3, 4]:
        metrics = report.get(str(label), {})
        row[f"class_{label}_precision"] = metrics.get("precision", 0.0)
        row[f"class_{label}_recall"] = metrics.get("recall", 0.0)
        row[f"class_{label}_f1"] = metrics.get("f1-score", 0.0)
        row[f"class_{label}_support"] = metrics.get("support", 0.0)
    return row


def binary_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray | None,
    threshold: str,
    status: str,
) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    auc = ""
    if score is not None and len(np.unique(y_true)) == 2:
        try:
            auc = roc_auc_score(y_true, score)
        except Exception:
            auc = ""
    return {
        "model_name": model_name,
        "status": status,
        "threshold_used": threshold,
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


def write_confusion_matrix(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    rows = []
    for index, values in enumerate(matrix):
        row = {"actual_class": index}
        for pred_index, value in enumerate(values):
            row[f"predicted_{pred_index}"] = int(value)
        rows.append(row)
    write_rows(path, rows)


def write_required_reports(
    result: TrainingResult,
    appdr_5: dict[str, Any],
    appdr_binary: dict[str, Any],
) -> None:
    cnn_5 = read_first_row(OUTPUT_DIR / "cnn_single_5class_metrics.csv")
    cnn_binary = read_first_row(OUTPUT_DIR / "cnn_single_binary_metrics.csv")
    comparison_5 = [
        appdr_5,
        baseline_row_from_known("Best experimental feature-based grading", BEST_FEATURE_GRADING),
        cnn_5,
    ]
    comparison_binary = [
        appdr_binary,
        {
            "model_name": "Best experimental feature-based binary screening",
            "status": "reported_prior_experiment",
            "accuracy": 0.7512,
            "referable_recall": 0.9697,
            "false_negatives": 48,
            "false_positives": 817,
        },
        cnn_binary,
    ]
    write_rows(OUTPUT_DIR / "comparison_5class.csv", comparison_5)
    write_rows(OUTPUT_DIR / "comparison_binary.csv", comparison_binary)
    write_rows(OUTPUT_DIR / "cnn_ensemble_metrics.csv", [{"status": "not_implemented", "reason": "Level 2 ensemble requires a valid Level 1 result and more compute time."}])
    write_report_md(result, appdr_5, appdr_binary, cnn_5, cnn_binary)
    write_final_recommendation(result, appdr_5, appdr_binary, cnn_5, cnn_binary)


def baseline_row_from_known(model_name: str, values: dict[str, Any]) -> dict[str, Any]:
    row = {"model_name": model_name, "status": "reported_prior_experiment"}
    row.update(values)
    return row


def read_first_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return rows[0] if rows else {}


def metric_float(row: dict[str, Any], key: str) -> float | None:
    try:
        value = row.get(key, "")
        if value == "":
            return None
        return float(value)
    except Exception:
        return None


def answer_comparison(cnn: dict[str, Any], appdr: dict[str, Any], metric: str, higher: bool = True) -> str:
    cnn_value = metric_float(cnn, metric)
    appdr_value = metric_float(appdr, metric)
    if cnn_value is None:
        return "No. CNN metric is unavailable because the CNN was not successfully evaluated."
    if appdr_value is None:
        return "Cannot compare because the AppDR baseline metric is unavailable on this row."
    improved = cnn_value > appdr_value if higher else cnn_value < appdr_value
    return "Yes." if improved else "No."


def write_report_md(
    result: TrainingResult,
    appdr_5: dict[str, Any],
    appdr_binary: dict[str, Any],
    cnn_5: dict[str, Any],
    cnn_binary: dict[str, Any],
) -> None:
    answers = final_answers(result, appdr_5, appdr_binary, cnn_5, cnn_binary)
    lines = [
        "# AppDR vs Guanshuo Xu-Style CNN Close Comparison",
        "",
        "Experiment only. Production AppDR artifacts and app behavior were not changed.",
        "",
        "## Guanshuo Method Reproduced",
        "",
        "- Image-input CNN path using APTOS train images.",
        "- Plain RGB resizing only.",
        "- SEResNeXt50 preferred when available.",
        "- GeM pooling based on cnnimageretrieval-pytorch idea.",
        "- SmoothL1Loss regression-style DR grade output.",
        "- Threshold conversion to classes 0-4.",
        "- Default, Guanshuo adjusted, and validation-optimized threshold evaluation.",
        "",
        "## Simplifications",
        "",
        "- Level 1 single model first; Level 2/3 ensembles are not run by default.",
        "- Input size starts at 384 for compute safety.",
        "- If run limits are non-zero, the CNN metrics are a compute-bounded smoke comparison and not a full APTOS reproduction.",
        "- AppDR and CNN are compared on the same effective run_test_manifest when limits are used.",
        "- No pseudo-labeling in first comparison.",
        "- No APTOS public test evaluation because labels are unavailable.",
        "- External IDRiD/Messidor data are not used in this first run.",
        "",
        "## Level 1 Status",
        "",
        f"Status: {result.status}",
        f"Backbone: {result.backbone or 'n/a'}",
        f"Input size: {result.input_size}",
        f"ImageNet pretrained used: {result.pretrained}",
        f"GeM pooling used: {result.gem_pooling}",
        f"Reason/blocker: {result.reason or 'none'}",
        "",
        "## Required Questions",
        "",
        "| Question | Answer |",
        "| --- | --- |",
    ]
    for question, answer in answers:
        lines.append(f"| {question} | {answer} |")
    (OUTPUT_DIR / "appdr_vs_guanshuo_close_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def final_answers(
    result: TrainingResult,
    appdr_5: dict[str, Any],
    appdr_binary: dict[str, Any],
    cnn_5: dict[str, Any],
    cnn_binary: dict[str, Any],
) -> list[tuple[str, str]]:
    if result.status != "trained":
        unavailable = f"No. CNN was not fully trained/evaluated: {result.reason}"
    else:
        unavailable = ""
    cnn_fp = metric_float(cnn_binary, "false_positives")
    appdr_fp = metric_float(appdr_binary, "false_positives")
    if unavailable:
        fp_answer = unavailable
    elif cnn_fp is None or appdr_fp is None:
        fp_answer = "Cannot determine because false-positive counts are unavailable."
    elif cnn_fp > appdr_fp:
        fp_answer = f"Yes. CNN false positives were higher ({int(cnn_fp)} vs AppDR {int(appdr_fp)}) on the effective test manifest."
    else:
        fp_answer = f"No. CNN false positives were lower ({int(cnn_fp)} vs AppDR {int(appdr_fp)}) on the effective test manifest."

    return [
        ("What parts of Guanshuo Xu's method were reproduced?", "Plain resized image-input CNN, SEResNeXt-style backbone if available, GeM pooling, SmoothL1Loss regression, heavy fundus-safe augmentation, and threshold conversion."),
        ("What parts were simplified because of compute/data limitations?", "Single model at 384px first; no 8-model ensemble, no pseudo-labeling, no external IDRiD/Messidor fusion, and no public test evaluation."),
        ("Did the CNN beat AppDR production macro F1?", unavailable or answer_comparison(cnn_5, APPDR_PRODUCTION_5CLASS, "macro_f1")),
        ("Did the CNN beat AppDR best experimental macro F1?", unavailable or answer_comparison(cnn_5, BEST_FEATURE_GRADING, "macro_f1")),
        ("Did the CNN improve macro precision?", unavailable or answer_comparison(cnn_5, APPDR_PRODUCTION_5CLASS, "macro_precision")),
        ("Did the CNN improve macro recall?", unavailable or answer_comparison(cnn_5, APPDR_PRODUCTION_5CLASS, "macro_recall")),
        ("Did the CNN improve Class 1 recall?", unavailable or answer_comparison(cnn_5, {"class_1_recall": BEST_FEATURE_GRADING["class_1_recall"]}, "class_1_recall")),
        ("Did the CNN improve Class 3 recall?", unavailable or answer_comparison(cnn_5, {"class_3_recall": BEST_FEATURE_GRADING["class_3_recall"]}, "class_3_recall")),
        ("Did the CNN improve Class 4 recall?", unavailable or answer_comparison(cnn_5, {"class_4_recall": BEST_FEATURE_GRADING["class_4_recall"]}, "class_4_recall")),
        ("Did the CNN improve binary referable recall?", unavailable or answer_comparison(cnn_binary, appdr_binary, "referable_recall")),
        ("Did the CNN reduce false negatives?", unavailable or answer_comparison(cnn_binary, appdr_binary, "false_negatives", higher=False)),
        ("Did the CNN create too many false positives?", fp_answer),
        ("Is straight CNN justified as replacement?", "No. This is comparison-only and replacement requires all safety metrics to pass."),
        ("Should CNN remain experimental?", "Yes."),
        ("Should hybrid CNN + 203 handcrafted features be tested next?", "Yes, after a stable Level 1 CNN is available."),
    ]


def write_final_recommendation(
    result: TrainingResult,
    appdr_5: dict[str, Any],
    appdr_binary: dict[str, Any],
    cnn_5: dict[str, Any],
    cnn_binary: dict[str, Any],
) -> None:
    lines = [
        "# Final Recommendation",
        "",
        "Do not update production.",
        "",
        "Straight CNN should remain experimental unless it improves macro F1, preserves macro precision/recall, does not collapse Class 1 or Class 3 recall, keeps referable recall strong, and does not increase false negatives badly.",
        "",
    ]
    if result.status == "trained":
        lines.extend(
            [
                f"CNN macro F1: {cnn_5.get('macro_f1', 'n/a')}",
                f"AppDR production macro F1 on same split/report: {appdr_5.get('macro_f1', 'n/a')}",
                f"CNN referable recall: {cnn_binary.get('referable_recall', 'n/a')}",
                f"CNN false negatives: {cnn_binary.get('false_negatives', 'n/a')}",
                f"CNN false positives: {cnn_binary.get('false_positives', 'n/a')}",
                f"AppDR false positives on same effective test manifest: {appdr_binary.get('false_positives', 'n/a')}",
                "",
                "The CNN improved binary screening on this capped run, but macro F1, macro precision, macro recall, Class 1 recall, and Class 4 recall did not beat the AppDR grading baselines. It is not justified as a replacement.",
                "",
                "Next recommended experiment: hybrid CNN continuous score/probabilities plus the 203 handcrafted AppDR features.",
            ]
        )
    else:
        lines.append(f"CNN result unavailable: {result.reason}")
    (OUTPUT_DIR / "final_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


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
