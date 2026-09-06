"""Full-split hybrid CNN + AppDR handcrafted feature comparison.

Experiment only. This script does not update production artifacts, backend model
loading, frontend behavior, or the existing 203-feature pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
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
OUTPUT_DIR = BACKEND_DIR / "results" / "full_hybrid_cnn_appdr_full_training"
FULL_CNN_DIR = BACKEND_DIR / "results" / "full_cnn_vs_appdr_comparison"
PREVIOUS_HYBRID_DIR = BACKEND_DIR / "results" / "full_hybrid_cnn_appdr_comparison"
PREVIOUS_MAXIMIZED_DIR = BACKEND_DIR / "results" / "full_hybrid_cnn_appdr_maximized"
CNN_SOURCE_DIR = OUTPUT_DIR / "cnn_sources"
SOURCE_FEATURES = BACKEND_DIR / "features_combined_balanced.csv"
RANDOM_STATE = 42
CNN_THRESHOLDS = [1.0, 1.6, 2.1, 3.1]
BINARY_THRESHOLDS = [round(v, 2) for v in np.arange(0.05, 0.901, 0.01)]
APPDR_PRODUCTION_5 = {
    "accuracy": 0.6912159570387418,
    "balanced_accuracy": 0.5776602382211962,
    "macro_precision": 0.6111971479308388,
    "macro_recall": 0.5776602382211962,
    "macro_f1": 0.5835033629121716,
    "class_1_recall": 0.3244444444444444,
    "class_3_recall": 0.34,
    "class_4_recall": 0.7511111111111111,
}
APPDR_PRODUCTION_BINARY = {
    "referable_recall": 0.9570345408593092,
    "false_negatives": 51,
    "false_positives": 398,
    "f1": 0.8349871370819552,
    "balanced_accuracy": 0.8383764253592321,
}
BEST_EXPERIMENTAL_5 = {
    "accuracy": 0.67,
    "balanced_accuracy": 0.6113,
    "macro_f1": 0.5799,
    "class_1_recall": 0.4533,
    "class_3_recall": 0.60,
    "class_4_recall": 0.6433,
}

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config


@dataclass
class ModelResult:
    problem: str
    model_name: str
    feature_version: str
    metrics: dict[str, Any]
    model: Any
    features: list[str]
    threshold: float | None = None


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CNN_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(RANDOM_STATE)
    copy_previous_references()
    manifests = load_or_copy_manifests()
    write_split_summary(manifests)
    appdr_tables, selected_features = build_or_load_appdr_tables(manifests)
    cnn_sources = train_or_load_cnn_sources(manifests, args)
    cnn_tables = build_or_load_cnn_tables(manifests, args)
    hybrid_tables = {split: merge_tables(appdr_tables[split], cnn_tables[split]) for split in ["train", "val", "test"]}
    versions = build_feature_versions(hybrid_tables["train"], selected_features)
    write_feature_docs(hybrid_tables, versions)

    five_results = train_5class(hybrid_tables, versions, args)
    binary_results, threshold_rows = train_binary(hybrid_tables, versions, args)
    write_rows(OUTPUT_DIR / "hybrid_5class_model_comparison.csv", [row.metrics for row in five_results])
    write_rows(OUTPUT_DIR / "hybrid_5class_validation_leaderboard.csv", [row.metrics for row in five_results])
    write_rows(OUTPUT_DIR / "hybrid_5class_test_results.csv", [row.metrics for row in five_results])
    write_rows(OUTPUT_DIR / "hybrid_binary_model_comparison.csv", [row.metrics for row in binary_results])
    write_rows(OUTPUT_DIR / "hybrid_binary_validation_threshold_sweep.csv", threshold_rows)
    write_rows(OUTPUT_DIR / "hybrid_binary_test_results.csv", [row.metrics for row in binary_results])
    best_five = select_best_five(five_results)
    best_binary, best_threshold = select_best_binary(binary_results, threshold_rows)
    write_best_outputs(best_five, best_binary, best_threshold, hybrid_tables)
    write_calibration(best_binary, best_threshold, hybrid_tables)
    write_comparison_reports(best_five, best_binary, best_threshold)
    print(f"Created {OUTPUT_DIR}")


def copy_previous_references() -> None:
    reference_dir = OUTPUT_DIR / "previous_maximized_reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "maximized_hybrid_vs_appdr_report.md",
        "final_recommendation.md",
        "hybrid_5class_metrics.json",
        "hybrid_binary_metrics.json",
        "comparison_5class_all.csv",
        "comparison_binary_all.csv",
    ]:
        source = PREVIOUS_MAXIMIZED_DIR / name
        if source.exists():
            target = reference_dir / name
            if not target.exists():
                target.write_bytes(source.read_bytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--cnn-specs", default="efficientnet_b3:384:classification_ce", help="Comma specs: backbone:input_size:objective. objective=classification_ce|classification_focal|regression.")
    parser.add_argument("--cnn-epochs", type=int, default=50)
    parser.add_argument("--cnn-batch-size", type=int, default=8)
    parser.add_argument("--cnn-learning-rate", type=float, default=1e-4)
    parser.add_argument("--cnn-weight-decay", type=float, default=1e-4)
    parser.add_argument("--cnn-patience", type=int, default=3)
    parser.add_argument("--include-old-resnet-source", action="store_true", help="Also include previous one-epoch ResNet50 source as a weak baseline source.")
    parser.add_argument("--warm-start-efficientnet-b3", action="store_true", help="Resume EfficientNet-B3 from the previous maximized best checkpoint when available.")
    parser.add_argument("--resume-cnn-checkpoint", type=Path, default=None, help="Resume CNN training from a paused full-training checkpoint such as last_model.pt.")
    parser.add_argument("--skip-cnn-training", action="store_true", help="Use existing maximized CNN source checkpoints only.")
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--quick-models", action="store_true", help="Use a smaller model grid for runtime troubleshooting only.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_or_copy_manifests() -> dict[str, pd.DataFrame]:
    names = ["dataset_manifest.csv", "train_manifest.csv", "val_manifest.csv", "test_manifest.csv"]
    missing = [name for name in names if not (FULL_CNN_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing full CNN split files: {missing}")
    manifests = {
        "dataset": pd.read_csv(FULL_CNN_DIR / "dataset_manifest.csv"),
        "train": pd.read_csv(FULL_CNN_DIR / "train_manifest.csv"),
        "val": pd.read_csv(FULL_CNN_DIR / "val_manifest.csv"),
        "test": pd.read_csv(FULL_CNN_DIR / "test_manifest.csv"),
    }
    for split, frame in manifests.items():
        frame.to_csv(OUTPUT_DIR / f"{split}_manifest.csv" if split != "dataset" else OUTPUT_DIR / "dataset_manifest.csv", index=False)
    return manifests


def write_split_summary(manifests: dict[str, pd.DataFrame]) -> None:
    lines = [
        "# Full Hybrid Split Summary",
        "",
        "Reused the exact full train/validation/test manifests from `backend/results/full_cnn_vs_appdr_comparison/`.",
        "This keeps AppDR production, straight CNN, and full hybrid evaluation on the same held-out test split.",
        "No patient_id is available, so the split is image-level stratified by diagnosis.",
        "",
        "| Split | Rows | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ["train", "val", "test"]:
        counts = manifests[split]["diagnosis"].value_counts().sort_index().to_dict()
        lines.append(
            f"| {split} | {len(manifests[split])} | {counts.get(0, 0)} | {counts.get(1, 0)} | {counts.get(2, 0)} | {counts.get(3, 0)} | {counts.get(4, 0)} |"
        )
    (OUTPUT_DIR / "split_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_or_load_appdr_tables(manifests: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    paths = {split: OUTPUT_DIR / f"appdr_features_{split}.csv" for split in ["train", "val", "test"]}
    selected = load_selected_features()
    if all(path.exists() for path in paths.values()):
        return {split: pd.read_csv(path) for split, path in paths.items()}, selected
    source = pd.read_csv(SOURCE_FEATURES)
    tables = {}
    for split in ["train", "val", "test"]:
        manifest = manifests[split].reset_index(drop=True)
        row_index = manifest["source_row_index"].astype(int).to_numpy()
        features = source.iloc[row_index][config.FEATURE_NAMES].reset_index(drop=True).copy()
        meta = manifest[["source_row_index", "image_path", "source_dataset", "id_code", "diagnosis"]].reset_index(drop=True)
        table = pd.concat([meta, features], axis=1)
        table.to_csv(paths[split], index=False)
        tables[split] = table
    dictionary = [{"feature_name": name, "group": "appdr_203", "basis": "Current AppDR handcrafted retinal feature pipeline."} for name in config.FEATURE_NAMES]
    write_rows(OUTPUT_DIR / "appdr_feature_dictionary.csv", dictionary)
    return tables, selected


def load_selected_features() -> list[str]:
    try:
        with (BACKEND_DIR / "results" / "best_model.pkl").open("rb") as file:
            model = pickle.load(file)
        selector = model.named_steps.get("feature_selector")
        selected = list(getattr(selector, "selected_features", []))
        return [name for name in selected if name in config.FEATURE_NAMES] or list(config.FEATURE_NAMES[:75])
    except Exception:
        return list(config.FEATURE_NAMES[:75])


def train_or_load_cnn_sources(manifests: dict[str, pd.DataFrame], args: argparse.Namespace) -> list[dict[str, Any]]:
    summary_path = CNN_SOURCE_DIR / "cnn_training_summary.csv"
    existing = read_csv_rows(summary_path)
    existing_by_id = {row.get("source_id"): row for row in existing if row.get("source_id")}
    rows: list[dict[str, Any]] = list(existing)
    if args.include_old_resnet_source:
        old_path = FULL_CNN_DIR / "checkpoints" / "resnet50_384_regression_gem" / "best_model.pt"
        old_id = "old_resnet50_384_regression_gem_1epoch"
        if old_path.exists() and old_id not in existing_by_id:
            old_row = {
                "source_id": old_id,
                "backbone": "resnet50",
                "input_size": 384,
                "objective": "regression",
                "pooling": "gem",
                "checkpoint_path": str(old_path),
                "status": "reused_previous_one_epoch_source",
                "epochs_completed": 1,
                "best_val_macro_f1": "",
                "test_macro_f1": read_first(FULL_CNN_DIR / "cnn_5class_metrics.csv").get("macro_f1", ""),
                "note": "Previous one-epoch ResNet50 source included only as a weak baseline CNN source.",
            }
            rows.append(old_row)
            existing_by_id[old_id] = old_row
    if not args.skip_cnn_training:
        for spec in parse_cnn_specs(args.cnn_specs):
            if spec["source_id"] in existing_by_id and Path(existing_by_id[spec["source_id"]].get("checkpoint_path", "")).exists():
                continue
            try:
                row = train_one_cnn_source(spec, manifests, args)
            except Exception as error:
                row = {**spec, "status": "failed", "error": str(error)[:500], "checkpoint_path": ""}
            rows = [item for item in rows if item.get("source_id") != spec["source_id"]]
            rows.append(row)
            write_rows(summary_path, rows)
    if not rows:
        raise RuntimeError("No CNN sources available. Train a source or pass --include-old-resnet-source.")
    write_rows(summary_path, rows)
    return rows


def parse_cnn_specs(text: str) -> list[dict[str, Any]]:
    specs = []
    for raw in [part.strip() for part in text.split(",") if part.strip()]:
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid CNN spec: {raw}")
        backbone, input_size, objective = parts
        source_id = "efficientnet_b3_full_training" if backbone == "efficientnet_b3" and objective == "classification_ce" else f"{backbone}_{input_size}_{objective}_gem"
        specs.append({"source_id": source_id, "backbone": backbone, "input_size": int(input_size), "objective": objective, "pooling": "gem"})
    return specs


def train_one_cnn_source(spec: dict[str, Any], manifests: dict[str, pd.DataFrame], args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from torchvision import transforms
    import timm

    source_dir = CNN_SOURCE_DIR / spec["source_id"]
    source_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = source_dir / "best_model.pt"

    class FundusDataset(Dataset):
        def __init__(self, frame: pd.DataFrame, train_mode: bool):
            self.frame = frame.reset_index(drop=True)
            self.train_mode = train_mode
            self.transform = transforms.Compose(
                [
                    transforms.Resize((int(spec["input_size"]), int(spec["input_size"]))),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        def __len__(self) -> int:
            return len(self.frame)

        def __getitem__(self, index: int):
            row = self.frame.iloc[index]
            image = Image.open(row["image_path"]).convert("RGB")
            if self.train_mode:
                image = augment_image(image)
            label = int(row["diagnosis"])
            target = torch.tensor(float(label), dtype=torch.float32) if spec["objective"] == "regression" else torch.tensor(label, dtype=torch.long)
            return self.transform(image), target

    class GeM(nn.Module):
        def __init__(self, p: float = 3.0, eps: float = 1e-6):
            super().__init__()
            self.p = nn.Parameter(torch.ones(1) * p)
            self.eps = eps

        def forward(self, x):
            return torch.nn.functional.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p).flatten(1)

    class CnnSourceModel(nn.Module):
        def __init__(self):
            super().__init__()
            try:
                self.encoder = timm.create_model(spec["backbone"], pretrained=True, num_classes=0, global_pool="")
                pretrained = True
            except Exception:
                self.encoder = timm.create_model(spec["backbone"], pretrained=False, num_classes=0, global_pool="")
                pretrained = False
            self.pretrained = pretrained
            self.pool = GeM()
            out_dim = 1 if spec["objective"] == "regression" else 5
            self.head = nn.Linear(int(self.encoder.num_features), out_dim)

        def forward_features_and_output(self, x):
            features = self.encoder(x)
            pooled = features if features.ndim == 2 else self.pool(features)
            output = self.head(pooled)
            return pooled, output.squeeze(1) if spec["objective"] == "regression" else output

        def forward(self, x):
            return self.forward_features_and_output(x)[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CnnSourceModel().to(device)
    warm_start_path = PREVIOUS_MAXIMIZED_DIR / "cnn_sources" / "efficientnet_b3_384_classification_ce_gem" / "best_model.pt"
    warm_started = False
    if args.warm_start_efficientnet_b3 and spec["backbone"] == "efficientnet_b3" and warm_start_path.exists():
        warm = torch.load(warm_start_path, map_location=device, weights_only=False)
        model.load_state_dict(warm["state_dict"])
        warm_started = True
    labels = manifests["train"]["diagnosis"].astype(int).to_numpy()
    weights = class_weights(labels)
    sample_weights = manifests["train"]["diagnosis"].map(lambda label: weights[int(label)]).to_numpy()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(FundusDataset(manifests["train"], True), batch_size=args.cnn_batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(FundusDataset(manifests["val"], False), batch_size=args.cnn_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(FundusDataset(manifests["test"], False), batch_size=args.cnn_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    if spec["objective"] == "regression":
        criterion: Any = nn.SmoothL1Loss()
    else:
        ce = nn.CrossEntropyLoss(weight=torch.tensor([weights[idx] for idx in range(5)], dtype=torch.float32, device=device))
        criterion = FocalLoss(ce) if spec["objective"] == "classification_focal" else ce
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.cnn_learning_rate, weight_decay=args.cnn_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.cnn_epochs))
    scaler = GradScaler(enabled=torch.cuda.is_available())
    best_state = None
    best_val = -1.0
    stale = 0
    logs = read_csv_rows(source_dir / "training_log.csv")
    start_epoch = 1
    resume_path = args.resume_cnn_checkpoint
    resumed_from = ""
    if resume_path and spec["backbone"] == "efficientnet_b3" and resume_path.exists():
        resumed = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resumed["state_dict"])
        resumed_from = str(resume_path)
        start_epoch = int(resumed.get("epoch", len(logs))) + 1
        if "optimizer_state_dict" in resumed:
            optimizer.load_state_dict(resumed["optimizer_state_dict"])
        if "scheduler_state_dict" in resumed:
            scheduler.load_state_dict(resumed["scheduler_state_dict"])
        if "scaler_state_dict" in resumed:
            scaler.load_state_dict(resumed["scaler_state_dict"])
        if logs:
            best_val = max(ffloat(row.get("val_macro_f1")) for row in logs)
            if "stale_epochs" in resumed:
                stale = int(resumed["stale_epochs"])
            else:
                best_log_index = max(
                    range(len(logs)),
                    key=lambda index: ffloat(logs[index].get("val_macro_f1")),
                )
                stale = len(logs) - best_log_index - 1
            if checkpoint_path.exists():
                best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                best_state = {key: value.detach().cpu() for key, value in best_checkpoint["state_dict"].items()}
            else:
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        warm_started = False
    elif warm_started:
        warm_pred, _, warm_labels = predict_cnn_loader(model, val_loader, device, spec)
        best_val = f1_score(warm_labels, warm_pred, average="macro", zero_division=0)
        best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        torch.save({"state_dict": best_state, "spec": spec, "pretrained": model.pretrained, "warm_start_baseline": True}, checkpoint_path)
    for epoch in range(start_epoch, args.cnn_epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        start = time.time()
        for step, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=torch.cuda.is_available()):
                output = model(images)
                loss = criterion(output.clamp(0, 4), targets) if spec["objective"] == "regression" else criterion(output, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * len(targets)
            total_count += len(targets)
            if step == 1 or step % 150 == 0 or step == len(train_loader):
                print(json.dumps({"source_id": spec["source_id"], "epoch": epoch, "step": step, "total_steps": len(train_loader), "elapsed_seconds": round(time.time() - start, 2)}), flush=True)
        scheduler.step()
        val_pred, val_scores, val_labels = predict_cnn_loader(model, val_loader, device, spec)
        val_macro_f1 = f1_score(val_labels, val_pred, average="macro", zero_division=0)
        val_bal = balanced_accuracy_score(val_labels, val_pred)
        val_qwk = cohen_kappa_score(val_labels, val_pred, weights="quadratic")
        log_row = {
            "source_id": spec["source_id"],
            "epoch": epoch,
            "train_loss": total_loss / max(total_count, 1),
            "val_macro_f1": val_macro_f1,
            "val_balanced_accuracy": val_bal,
            "val_qwk": val_qwk,
            "elapsed_seconds": round(time.time() - start, 2),
        }
        logs.append(log_row)
        write_rows(source_dir / "training_log.csv", logs)
        if val_macro_f1 > best_val:
            best_val = val_macro_f1
            stale = 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "state_dict": best_state,
                    "spec": spec,
                    "pretrained": model.pretrained,
                    "epoch": epoch,
                    "best_val_macro_f1": best_val,
                    "stale_epochs": stale,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= args.cnn_patience:
                break
        epoch_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        epoch_checkpoint = {
            "state_dict": epoch_state,
            "spec": spec,
            "epoch": epoch,
            "best_val_macro_f1": best_val,
            "stale_epochs": stale,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        }
        torch.save(epoch_checkpoint, source_dir / f"epoch_{epoch:03d}.pt")
        torch.save(epoch_checkpoint, source_dir / "last_model.pt")
    if best_state is not None:
        model.load_state_dict(best_state)
    test_pred, _, test_labels = predict_cnn_loader(model, test_loader, device, spec)
    test_metrics = five_metrics(test_labels, test_pred)
    write_json(source_dir / "model_config.json", {**spec, "epochs_requested": args.cnn_epochs, "batch_size": args.cnn_batch_size, "pretrained": bool(getattr(model, "pretrained", False)), "warm_started_from_previous_maximized": warm_started, "warm_start_path": str(warm_start_path) if warm_started else "", "resumed_from": resumed_from, "start_epoch": start_epoch})
    write_json(source_dir / "config.json", {**spec, "epochs_requested": args.cnn_epochs, "batch_size": args.cnn_batch_size, "pretrained": bool(getattr(model, "pretrained", False)), "warm_started_from_previous_maximized": warm_started, "resumed_from": resumed_from})
    write_json(source_dir / "test_metrics.json", test_metrics)
    write_json(source_dir / "test_metrics_best_checkpoint.json", test_metrics)
    write_rows(source_dir / "validation_metrics_by_epoch.csv", logs)
    return {
        **spec,
        "checkpoint_path": str(checkpoint_path),
        "status": "trained",
        "epochs_completed": len(logs),
        "best_val_macro_f1": best_val,
        "test_macro_f1": test_metrics.get("macro_f1", ""),
        "test_balanced_accuracy": test_metrics.get("balanced_accuracy", ""),
        "test_class_1_recall": test_metrics.get("class_1_recall", ""),
        "test_class_3_recall": test_metrics.get("class_3_recall", ""),
        "test_class_4_recall": test_metrics.get("class_4_recall", ""),
    }


class FocalLoss:
    def __init__(self, ce_loss: Any, gamma: float = 2.0):
        self.ce_loss = ce_loss
        self.gamma = gamma

    def __call__(self, logits: Any, targets: Any) -> Any:
        import torch

        ce = self.ce_loss(logits, targets)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma) * ce


def predict_cnn_loader(model: Any, loader: Any, device: Any, spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    model.eval()
    preds, scores, labels_out = [], [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            output = model(images)
            if spec["objective"] == "regression":
                values = output.detach().cpu().numpy().clip(0, 4)
                pred = apply_thresholds(values, CNN_THRESHOLDS)
                scores.extend(values.tolist())
                labels = targets.detach().cpu().numpy().astype(int)
            else:
                probs = torch.softmax(output, dim=1).detach().cpu().numpy()
                pred = probs.argmax(axis=1)
                scores.extend(np.sum(probs[:, 2:], axis=1).tolist())
                labels = targets.detach().cpu().numpy().astype(int)
            preds.extend(pred.tolist())
            labels_out.extend(labels.tolist())
    return np.asarray(preds, dtype=int), np.asarray(scores, dtype=float), np.asarray(labels_out, dtype=int)


def augment_image(image: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    image = image.rotate(random.uniform(-180, 180))
    if random.random() < 0.75:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.75:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.92, 1.08))
    if random.random() < 0.20:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.8)))
    if random.random() < 0.20:
        image = image.filter(ImageFilter.SHARPEN)
    return image


def class_weights(labels: np.ndarray) -> dict[int, float]:
    counts = pd.Series(labels).value_counts().to_dict()
    total = len(labels)
    return {label: total / (5.0 * max(counts.get(label, 1), 1)) for label in range(5)}


def build_or_load_cnn_tables(manifests: dict[str, pd.DataFrame], args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    paths = {split: OUTPUT_DIR / f"cnn_features_{split}.csv" for split in ["train", "val", "test"]}
    if args.reuse_features and all(path.exists() for path in paths.values()):
        return {split: pd.read_csv(path) for split, path in paths.items()}
    sources = [
        row
        for row in read_csv_rows(CNN_SOURCE_DIR / "cnn_training_summary.csv")
        if row.get("status") not in {"failed", ""} and Path(row.get("checkpoint_path", "")).exists()
    ]
    if not sources:
        raise RuntimeError("No trained CNN sources found for feature extraction.")
    base_tables = {
        split: manifests[split][["source_row_index", "diagnosis"]].reset_index(drop=True).copy()
        for split in ["train", "val", "test"]
    }
    report_lines = ["# CNN Embedding PCA Report", ""]
    dictionary_rows = []
    for source in sources:
        source_id = str(source["source_id"])
        source_artifact_dir = CNN_SOURCE_DIR / source_id
        source_artifact_dir.mkdir(parents=True, exist_ok=True)
        raw_embeddings: dict[str, np.ndarray] = {}
        prediction_tables: dict[str, pd.DataFrame] = {}
        for split in ["train", "val", "test"]:
            prediction_tables[split], raw_embeddings[split] = infer_cnn_source_split(manifests[split], source, args)
        pca_components = min(args.pca_components, raw_embeddings["train"].shape[1], raw_embeddings["train"].shape[0])
        pca = PCA(n_components=pca_components, random_state=RANDOM_STATE)
        pca.fit(raw_embeddings["train"])
        with (source_artifact_dir / "embedding_pca.pkl").open("wb") as file:
            pickle.dump(pca, file)
        report_lines.extend(
            [
                f"## {source_id}",
                f"Raw embedding dimension: {raw_embeddings['train'].shape[1]}",
                f"PCA components: {pca.n_components_}",
                f"Cumulative explained variance: {float(np.sum(pca.explained_variance_ratio_)):.4f}",
                "",
            ]
        )
        for split in ["train", "val", "test"]:
            reduced = pca.transform(raw_embeddings[split])
            emb = pd.DataFrame(reduced, columns=[f"cnn_embed_{source_id}_pca128_{i:03d}" for i in range(reduced.shape[1])])
            base_tables[split] = pd.concat(
                [base_tables[split].reset_index(drop=True), prediction_tables[split].reset_index(drop=True), emb.reset_index(drop=True)],
                axis=1,
            )
        for col in prediction_tables["train"].columns:
            dictionary_rows.append({"feature_name": col, "group": "cnn_prediction", "basis": f"CNN source {source_id} prediction/logit/probability feature."})
        for i in range(pca_components):
            dictionary_rows.append({"feature_name": f"cnn_embed_{source_id}_pca128_{i:03d}", "group": "cnn_embedding_pca", "basis": f"Train-fitted PCA embedding feature from CNN source {source_id}."})
    (OUTPUT_DIR / "cnn_embedding_pca_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "cnn_calibration_report.md").write_text("# CNN Calibration Report\n\nCNN probability calibration was not used for final selection in this pass; hybrid binary calibration is reported separately.\n", encoding="utf-8")
    write_rows(OUTPUT_DIR / "cnn_threshold_tuning.csv", read_csv_rows(CNN_SOURCE_DIR / "cnn_training_summary.csv") or [{"status": "not_available"}])
    tables = base_tables
    for split in ["train", "val", "test"]:
        tables[split].to_csv(paths[split], index=False)
    cnn_dict = []
    for col in tables["train"].columns:
        if col.startswith("cnn_"):
            group = "cnn_embedding_pca" if col.startswith("cnn_embedding") else "cnn_prediction"
            cnn_dict.append({"feature_name": col, "group": group, "basis": "Maximized full-split CNN source feature."})
    write_rows(OUTPUT_DIR / "cnn_feature_dictionary.csv", dictionary_rows or cnn_dict)
    return tables


def infer_cnn_source_split(frame: pd.DataFrame, source: dict[str, Any], args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    import timm

    spec = {
        "source_id": source["source_id"],
        "backbone": source["backbone"],
        "input_size": int(float(source["input_size"])),
        "objective": source["objective"],
        "pooling": source.get("pooling", "gem"),
    }
    checkpoint_path = Path(source["checkpoint_path"])

    class FundusDataset(Dataset):
        def __init__(self, rows: pd.DataFrame):
            self.rows = rows.reset_index(drop=True)
            self.transform = transforms.Compose(
                [
                    transforms.Resize((spec["input_size"], spec["input_size"])),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows.iloc[index]
            return self.transform(Image.open(row["image_path"]).convert("RGB")), index

    class GeM(nn.Module):
        def __init__(self, p: float = 3.0, eps: float = 1e-6):
            super().__init__()
            self.p = nn.Parameter(torch.ones(1) * p)
            self.eps = eps

        def forward(self, x):
            return torch.nn.functional.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p).flatten(1)

    class CnnSourceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = timm.create_model(spec["backbone"], pretrained=False, num_classes=0, global_pool="")
            self.pool = GeM()
            out_dim = 1 if spec["objective"] == "regression" else 5
            self.head = nn.Linear(int(self.encoder.num_features), out_dim)

        def forward_features_and_output(self, x):
            features = self.encoder(x)
            pooled = features if features.ndim == 2 else self.pool(features)
            output = self.head(pooled)
            return pooled, output.squeeze(1) if spec["objective"] == "regression" else output

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CnnSourceModel().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    loader = DataLoader(FundusDataset(frame), batch_size=args.embedding_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    rows = []
    embeddings = []
    source_id = spec["source_id"]
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device, non_blocking=True)
            pooled, output = model.forward_features_and_output(images)
            pooled_np = pooled.detach().cpu().numpy()
            for batch_pos, original_index in enumerate(indices.numpy().tolist()):
                if spec["objective"] == "regression":
                    severity = float(output.detach().cpu().numpy()[batch_pos].clip(0, 4))
                    probs = pseudo_class_probabilities(severity)
                    logits = [math.log(max(prob, 1e-9)) for prob in probs]
                    pred_class = int(apply_thresholds(np.asarray([severity]), CNN_THRESHOLDS)[0])
                    referable = float(1.0 / (1.0 + math.exp(-(severity - CNN_THRESHOLDS[1]))))
                else:
                    logits = output.detach().cpu().numpy()[batch_pos].astype(float).tolist()
                    exp = np.exp(np.asarray(logits) - np.max(logits))
                    probs = (exp / exp.sum()).tolist()
                    pred_class = int(np.argmax(probs))
                    severity = float(sum(idx * prob for idx, prob in enumerate(probs)))
                    referable = float(sum(probs[2:]))
                top = sorted(probs, reverse=True)
                entropy = -sum(float(p) * math.log(max(float(p), 1e-9)) for p in probs)
                row = {
                    f"cnn_{source_id}__severity": severity,
                    f"cnn_{source_id}__predicted_class": pred_class,
                    f"cnn_{source_id}__referable_probability": referable,
                    f"cnn_{source_id}__max_probability": float(max(probs)),
                    f"cnn_{source_id}__entropy": float(entropy),
                    f"cnn_{source_id}__uncertainty": float(1.0 - max(probs)),
                    f"cnn_{source_id}__margin": float(top[0] - top[1]),
                }
                for idx, value in enumerate(probs):
                    row[f"cnn_{source_id}__prob_class_{idx}"] = float(value)
                for idx, value in enumerate(logits):
                    row[f"cnn_{source_id}__logit_class_{idx}"] = float(value)
                rows.append(row)
                embeddings.append(pooled_np[batch_pos])
    return pd.DataFrame(rows), np.asarray(embeddings, dtype=np.float32)


def infer_cnn_split(frame: pd.DataFrame, checkpoint_path: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    import timm

    class FundusDataset(Dataset):
        def __init__(self, rows: pd.DataFrame):
            self.rows = rows.reset_index(drop=True)
            self.transform = transforms.Compose(
                [
                    transforms.Resize((384, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows.iloc[index]
            image = Image.open(row["image_path"]).convert("RGB")
            return self.transform(image), index

    class GeM(nn.Module):
        def __init__(self, p: float = 3.0, eps: float = 1e-6):
            super().__init__()
            self.p = nn.Parameter(torch.ones(1) * p)
            self.eps = eps

        def forward(self, x):
            return torch.nn.functional.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p).flatten(1)

    class CnnModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = timm.create_model("resnet50", pretrained=False, num_classes=0, global_pool="")
            self.pool = GeM()
            self.head = nn.Linear(int(self.encoder.num_features), 1)

        def forward_features_and_output(self, x):
            features = self.encoder(x)
            pooled = self.pool(features)
            output = self.head(pooled).squeeze(1)
            return pooled, output

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CnnModel().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    loader = DataLoader(FundusDataset(frame), batch_size=args.embedding_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    rows = []
    embeddings = []
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device, non_blocking=True)
            pooled, outputs = model.forward_features_and_output(images)
            pooled_np = pooled.detach().cpu().numpy()
            outputs_np = outputs.detach().cpu().numpy().clip(0, 4)
            for batch_pos, original_index in enumerate(indices.numpy().tolist()):
                base = frame.iloc[int(original_index)]
                severity = float(outputs_np[batch_pos])
                probs = pseudo_class_probabilities(severity)
                pred_class = int(apply_thresholds(np.asarray([severity]), CNN_THRESHOLDS)[0])
                entropy = -sum(float(p) * math.log(max(float(p), 1e-9)) for p in probs)
                top = sorted(probs, reverse=True)
                rows.append(
                    {
                        "source_row_index": int(base["source_row_index"]),
                        "diagnosis": int(base["diagnosis"]),
                        "cnn_severity_output": severity,
                        "cnn_predicted_class": pred_class,
                        "cnn_referable_probability": float(1.0 / (1.0 + math.exp(-(severity - CNN_THRESHOLDS[1])))),
                        "cnn_max_probability": float(max(probs)),
                        "cnn_entropy": float(entropy),
                        "cnn_uncertainty": float(1.0 - max(probs)),
                        "cnn_margin": float(top[0] - top[1]),
                        **{f"cnn_prob_class_{idx}": float(value) for idx, value in enumerate(probs)},
                    }
                )
                embeddings.append(pooled_np[batch_pos])
    return pd.DataFrame(rows), np.asarray(embeddings, dtype=np.float32)


def pseudo_class_probabilities(value: float) -> list[float]:
    centers = np.arange(5, dtype=float)
    logits = -np.abs(centers - float(value))
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    return probs.tolist()


def apply_thresholds(values: np.ndarray, thresholds: list[float]) -> np.ndarray:
    out = np.zeros(len(values), dtype=int)
    for threshold in thresholds:
        out += values > threshold
    return np.clip(out, 0, 4)


def write_dimensionality_report(raw_embeddings: dict[str, np.ndarray], pca: PCA) -> None:
    lines = [
        "# Dimensionality Reduction Report",
        "",
        f"Raw CNN pooled embedding dimension: {raw_embeddings['train'].shape[1]}",
        f"PCA components used: {pca.n_components_}",
        f"Train/val/test embedding rows: {raw_embeddings['train'].shape[0]} / {raw_embeddings['val'].shape[0]} / {raw_embeddings['test'].shape[0]}",
        f"Cumulative explained variance: {float(np.sum(pca.explained_variance_ratio_)):.4f}",
        "",
        "Raw 2048-dim GeM embeddings were reduced before CSV export/modeling to control classical ML dimensionality and disk size.",
    ]
    (OUTPUT_DIR / "dimensionality_reduction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_tables(appdr: pd.DataFrame, cnn: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["source_row_index", "diagnosis"]
    merged = appdr.merge(cnn, on=key_cols, how="inner", validate="one_to_one")
    if len(merged) != len(appdr):
        raise ValueError("Hybrid merge lost rows")
    return merged


def build_feature_versions(table: pd.DataFrame, selected: list[str]) -> dict[str, list[str]]:
    appdr203 = [name for name in config.FEATURE_NAMES if name in table.columns]
    selected75 = [name for name in selected if name in table.columns]
    selected100 = [name for name in config.FEATURE_NAMES[:100] if name in table.columns]
    pred = [name for name in table.columns if name.startswith("cnn_") and "__" in name and not name.startswith("cnn_embed_")]
    embed128 = [name for name in table.columns if name.startswith("cnn_embed_")]
    uncertainty = [name for name in pred if any(token in name for token in ["entropy", "uncertainty", "margin", "max_probability"])]
    prob = [name for name in pred if any(token in name for token in ["prob_class", "referable_probability"])]
    source_ids = sorted({name.split("__", 1)[0].replace("cnn_", "") for name in pred})
    best_source = best_cnn_source_id()
    best_pred = [name for name in pred if name.startswith(f"cnn_{best_source}__")] if best_source else pred
    best_embed = [name for name in embed128 if name.startswith(f"cnn_embed_{best_source}_")] if best_source else embed128
    best_embed64 = best_embed[:64]
    embed64 = []
    for source_id in source_ids:
        embed64.extend([name for name in embed128 if name.startswith(f"cnn_embed_{source_id}_")][:64])
    return {
        "v1_appdr203_best_cnn_predictions": appdr203 + best_pred,
        "v2_appdr203_best_cnn_embedding_pca64": appdr203 + best_embed64,
        "v3_appdr203_best_cnn_embedding_pca128": appdr203 + best_embed,
        "v4_appdr203_best_cnn_predictions_embedding_pca64": appdr203 + best_pred + best_embed64,
        "v5_appdr203_best_cnn_predictions_embedding_pca128": appdr203 + best_pred + best_embed,
        "v6_selected75_best_cnn_predictions": selected75 + best_pred,
        "v7_selected75_best_cnn_predictions_embedding_pca128": selected75 + best_pred + best_embed,
        "v8_selected100_best_cnn_predictions_embedding_pca128": selected100 + best_pred + best_embed,
        "v9_appdr203_ensemble_cnn_predictions": appdr203 + pred,
        "v10_appdr203_ensemble_cnn_predictions_combined_pca": appdr203 + pred + embed64,
        "v11_appdr203_only_baseline": appdr203,
        "v12_cnn_only_predictions_embeddings": pred + embed64,
        "v13_appdr203_cnn_uncertainty": appdr203 + uncertainty,
        "v14_appdr203_cnn_binary_and_5class_probabilities": appdr203 + prob,
    }


def best_cnn_source_id() -> str:
    rows = [row for row in read_csv_rows(CNN_SOURCE_DIR / "cnn_training_summary.csv") if row.get("status") != "failed"]
    if not rows:
        return ""
    best = max(rows, key=lambda row: ffloat(row.get("best_val_macro_f1")) if row.get("best_val_macro_f1") not in {"", None} else ffloat(row.get("test_macro_f1")))
    return str(best.get("source_id", ""))


def write_feature_docs(tables: dict[str, pd.DataFrame], versions: dict[str, list[str]]) -> None:
    dictionary = []
    for name in versions[next(iter(versions))] if False else sorted({feature for features in versions.values() for feature in features}):
        if name in config.FEATURE_NAMES:
            group = "appdr_203"
            basis = "Current AppDR handcrafted retinal feature."
        elif name.startswith("cnn_embed_"):
            group = "cnn_embedding_pca"
            basis = "PCA-reduced GeM pooled CNN embedding."
        else:
            group = "cnn_prediction"
            basis = "CNN severity prediction, pseudo-probability, confidence, or uncertainty feature."
        dictionary.append({"feature_name": name, "group": group, "basis": basis})
    write_rows(OUTPUT_DIR / "hybrid_feature_dictionary.csv", dictionary)
    lines = ["# Hybrid Feature Versions", ""]
    for version, features in versions.items():
        lines.append(f"- `{version}`: {len(features)} features")
    (OUTPUT_DIR / "hybrid_feature_versions_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit_rows = []
    audit_lines = ["# Feature Audit", ""]
    train = tables["train"]
    for version, features in versions.items():
        values = train[features].replace([np.inf, -np.inf], np.nan)
        nunique = values.nunique(dropna=False)
        constant = int((nunique <= 1).sum())
        near_constant = int((values.std(numeric_only=True) < 1e-8).sum())
        missing = int(values.isna().sum().sum())
        duplicate = int(values.T.duplicated().sum())
        high_corr = high_corr_count(values)
        row = {
            "feature_version": version,
            "feature_count": len(features),
            "nan_values": missing,
            "constant_features": constant,
            "near_constant_features": near_constant,
            "duplicate_features": duplicate,
            "high_correlation_pairs_abs_gt_0_995": high_corr,
            "scaling_needed": any(name.startswith("cnn_") for name in features),
        }
        audit_rows.append(row)
        audit_lines.append(f"- `{version}`: {len(features)} features, NaN values {missing}, constants {constant}, high-corr pairs {high_corr}")
    write_rows(OUTPUT_DIR / "feature_audit.csv", audit_rows)
    (OUTPUT_DIR / "feature_audit.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")


def high_corr_count(values: pd.DataFrame) -> int:
    if values.shape[1] > 400:
        return -1
    corr = values.corr(numeric_only=True).abs().to_numpy()
    if corr.size == 0:
        return 0
    upper = np.triu(corr, k=1)
    return int(np.nansum(upper > 0.995))


def train_5class(tables: dict[str, pd.DataFrame], versions: dict[str, list[str]], args: argparse.Namespace) -> list[ModelResult]:
    train, val, test = tables["train"], tables["val"], tables["test"]
    y_train = train["diagnosis"].astype(int).to_numpy()
    y_val = val["diagnosis"].astype(int).to_numpy()
    y_test = test["diagnosis"].astype(int).to_numpy()
    results = []
    for version, features in versions.items():
        for model_name, model in filter_models(version, five_class_models(args), args).items():
            try:
                fitted = clone(model).fit(train[features], y_train)
                val_pred = fitted.predict(val[features])
                test_pred = fitted.predict(test[features])
                metrics = five_metrics(y_test, test_pred)
                metrics.update(
                    {
                        "model_name": model_name,
                        "feature_version": version,
                        "feature_count": len(features),
                        "selection_metric_val_macro_f1": f1_score(y_val, val_pred, average="macro", zero_division=0),
                        "selection_metric_val_balanced_accuracy": balanced_accuracy_score(y_val, val_pred),
                        "status": "ok",
                    }
                )
                results.append(ModelResult("5class", model_name, version, metrics, fitted, features))
            except Exception as error:
                results.append(ModelResult("5class", model_name, version, {"model_name": model_name, "feature_version": version, "status": "failed", "error": str(error)[:500]}, None, features))
    return results


def train_binary(tables: dict[str, pd.DataFrame], versions: dict[str, list[str]], args: argparse.Namespace) -> tuple[list[ModelResult], list[dict[str, Any]]]:
    train, val, test = tables["train"], tables["val"], tables["test"]
    y_train = (train["diagnosis"].astype(int).to_numpy() >= 2).astype(int)
    y_val = (val["diagnosis"].astype(int).to_numpy() >= 2).astype(int)
    y_test = (test["diagnosis"].astype(int).to_numpy() >= 2).astype(int)
    results = []
    threshold_rows = []
    for version, features in versions.items():
        for model_name, model in filter_models(version, binary_models(args), args).items():
            try:
                fitted = clone(model).fit(train[features], y_train)
                val_scores = binary_scores(fitted, val[features])
                test_scores = binary_scores(fitted, test[features])
                val_best = choose_binary_threshold(y_val, val_scores)
                val_pred = (val_scores >= val_best).astype(int)
                test_pred = (test_scores >= val_best).astype(int)
                val_metrics = binary_metrics(y_val, val_pred, val_scores, val_best)
                metrics = binary_metrics(y_test, test_pred, test_scores, val_best)
                metrics.update(
                    {
                        "model_name": model_name,
                        "feature_version": version,
                        "feature_count": len(features),
                        "selection_metric_val_referable_recall": val_metrics["referable_recall"],
                        "selection_metric_val_f1": val_metrics["f1"],
                        "selection_metric_val_balanced_accuracy": val_metrics["balanced_accuracy"],
                        "selection_metric_val_false_positives": val_metrics["false_positives"],
                        "selection_metric_val_false_negatives": val_metrics["false_negatives"],
                        "status": "ok",
                    }
                )
                results.append(ModelResult("binary", model_name, version, metrics, fitted, features, val_best))
                for threshold in BINARY_THRESHOLDS:
                    pred = (val_scores >= threshold).astype(int)
                    row = binary_metrics(y_val, pred, val_scores, threshold)
                    row.update({"model_name": model_name, "feature_version": version, "feature_count": len(features), "status": "ok"})
                    threshold_rows.append(row)
            except Exception as error:
                results.append(ModelResult("binary", model_name, version, {"model_name": model_name, "feature_version": version, "status": "failed", "error": str(error)[:500]}, None, features))
    return results, threshold_rows


def five_class_models(args: argparse.Namespace) -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("model", LogisticRegression(C=1.0, max_iter=1200, class_weight="balanced", random_state=RANDOM_STATE))]),
        "random_forest": Pipeline([("imputer", SimpleImputer()), ("model", RandomForestClassifier(n_estimators=220, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))]),
        "extra_trees": Pipeline([("imputer", SimpleImputer()), ("model", ExtraTreesClassifier(n_estimators=320, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))]),
        "hist_gradient_boosting": Pipeline([("imputer", SimpleImputer()), ("model", HistGradientBoostingClassifier(max_iter=160, learning_rate=0.05, l2_regularization=0.05, random_state=RANDOM_STATE))]),
    }
    if XGBClassifier is not None:
        models["xgboost"] = Pipeline([("imputer", SimpleImputer()), ("model", XGBClassifier(n_estimators=220, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, objective="multi:softprob", eval_metric="mlogloss", tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1))])
    if LGBMClassifier is not None:
        models["lightgbm"] = Pipeline([("imputer", SimpleImputer()), ("model", LGBMClassifier(n_estimators=240, learning_rate=0.045, num_leaves=31, class_weight="balanced", random_state=RANDOM_STATE, verbose=-1, n_jobs=-1))])
    if CatBoostClassifier is not None:
        models["catboost"] = Pipeline([("imputer", SimpleImputer()), ("model", CatBoostClassifier(iterations=180, depth=5, learning_rate=0.05, loss_function="MultiClass", verbose=False, random_seed=RANDOM_STATE))])
    models["svm_rbf"] = Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("model", SVC(C=3.0, gamma="scale", class_weight="balanced", probability=True, random_state=RANDOM_STATE))])
    return models


def binary_models(args: argparse.Namespace) -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("model", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))]),
        "random_forest": Pipeline([("imputer", SimpleImputer()), ("model", RandomForestClassifier(n_estimators=220, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))]),
        "extra_trees": Pipeline([("imputer", SimpleImputer()), ("model", ExtraTreesClassifier(n_estimators=320, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))]),
        "hist_gradient_boosting": Pipeline([("imputer", SimpleImputer()), ("model", HistGradientBoostingClassifier(max_iter=160, learning_rate=0.05, l2_regularization=0.05, random_state=RANDOM_STATE))]),
    }
    if XGBClassifier is not None:
        models["xgboost"] = Pipeline([("imputer", SimpleImputer()), ("model", XGBClassifier(n_estimators=220, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1))])
    if LGBMClassifier is not None:
        models["lightgbm"] = Pipeline([("imputer", SimpleImputer()), ("model", LGBMClassifier(n_estimators=240, learning_rate=0.045, num_leaves=31, class_weight="balanced", random_state=RANDOM_STATE, verbose=-1, n_jobs=-1))])
    if CatBoostClassifier is not None:
        models["catboost"] = Pipeline([("imputer", SimpleImputer()), ("model", CatBoostClassifier(iterations=180, depth=5, learning_rate=0.05, loss_function="Logloss", verbose=False, random_seed=RANDOM_STATE))])
    models["svm_rbf"] = Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("model", SVC(C=3.0, gamma="scale", class_weight="balanced", probability=True, random_state=RANDOM_STATE))])
    return models


def filter_models(version: str, models: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.quick_models:
        return {name: models[name] for name in models if name in {"logistic_regression", "extra_trees", "lightgbm"}}
    if "embedding" in version and "selected75" not in version and "pca64" not in version:
        allowed = {"logistic_regression", "extra_trees", "lightgbm", "xgboost", "hist_gradient_boosting"}
        return {name: model for name, model in models.items() if name in allowed}
    if version in {"v7_cnn_prediction_only", "v4_selected75_cnn_predictions"}:
        return models
    return {name: model for name, model in models.items() if name != "svm_rbf"}


def five_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4], output_dict=True, zero_division=0)
    row = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    for label in [0, 1, 2, 3, 4]:
        row[f"class_{label}_precision"] = report[str(label)]["precision"]
        row[f"class_{label}_recall"] = report[str(label)]["recall"]
        row[f"class_{label}_f1"] = report[str(label)]["f1-score"]
        row[f"class_{label}_support"] = report[str(label)]["support"]
    return row


def binary_scores(model: Any, x_values: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_values)
        return np.asarray(proba)[:, 1]
    return np.asarray(model.predict(x_values), dtype=float)


def choose_binary_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    best = 0.5
    best_key = (-1, -1.0, -1.0, 0)
    for threshold in BINARY_THRESHOLDS:
        pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rec = recall_score(y_true, pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        bal = balanced_accuracy_score(y_true, pred)
        key = (int(rec >= APPDR_PRODUCTION_BINARY["referable_recall"]), f1, bal, -int(fp))
        if key > best_key:
            best_key = key
            best = float(threshold)
    return best


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = ""
    try:
        auc = roc_auc_score(y_true, scores)
    except Exception:
        pass
    return {
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


def select_best_five(results: list[ModelResult]) -> ModelResult:
    valid = [row for row in results if row.metrics.get("status") == "ok"]
    return max(valid, key=lambda row: (float(row.metrics.get("selection_metric_val_macro_f1", 0)), float(row.metrics.get("macro_f1", 0))))


def select_best_binary(results: list[ModelResult], threshold_rows: list[dict[str, Any]]) -> tuple[ModelResult, dict[str, Any]]:
    valid = [row for row in results if row.metrics.get("status") == "ok"]
    best = max(
        valid,
        key=lambda row: (
            float(row.metrics.get("selection_metric_val_referable_recall", 0)) >= APPDR_PRODUCTION_BINARY["referable_recall"],
            float(row.metrics.get("selection_metric_val_f1", 0)),
            float(row.metrics.get("selection_metric_val_balanced_accuracy", 0)),
            -float(row.metrics.get("selection_metric_val_false_positives", 999999)),
        ),
    )
    return best, best.metrics


def write_best_outputs(best_five: ModelResult, best_binary: ModelResult, best_threshold: dict[str, Any], tables: dict[str, pd.DataFrame]) -> None:
    with (OUTPUT_DIR / "hybrid_5class_best_model.pkl").open("wb") as file:
        pickle.dump({"model": best_five.model, "features": best_five.features, "metrics": best_five.metrics}, file)
    with (OUTPUT_DIR / "hybrid_binary_best_model.pkl").open("wb") as file:
        pickle.dump({"model": best_binary.model, "features": best_binary.features, "threshold": best_binary.threshold, "metrics": best_threshold}, file)
    write_json(OUTPUT_DIR / "hybrid_5class_metrics.json", best_five.metrics)
    write_rows(OUTPUT_DIR / "hybrid_5class_metrics.csv", [best_five.metrics])
    write_json(OUTPUT_DIR / "hybrid_binary_metrics.json", best_threshold)
    write_rows(OUTPUT_DIR / "hybrid_binary_metrics.csv", [best_threshold])
    test = tables["test"]
    y5 = test["diagnosis"].astype(int).to_numpy()
    pred5 = best_five.model.predict(test[best_five.features])
    write_confusion(OUTPUT_DIR / "hybrid_5class_confusion_matrix.csv", y5, pred5, [0, 1, 2, 3, 4])
    report = classification_report(y5, pred5, labels=[0, 1, 2, 3, 4], output_dict=True, zero_division=0)
    per_class = [{"class": label, "precision": report[str(label)]["precision"], "recall": report[str(label)]["recall"], "f1": report[str(label)]["f1-score"], "support": report[str(label)]["support"]} for label in [0, 1, 2, 3, 4]]
    write_rows(OUTPUT_DIR / "hybrid_5class_per_class_metrics.csv", per_class)
    yb = (y5 >= 2).astype(int)
    scores = binary_scores(best_binary.model, test[best_binary.features])
    predb = (scores >= float(best_threshold["threshold"])).astype(int)
    write_confusion(OUTPUT_DIR / "hybrid_binary_confusion_matrix.csv", yb, predb, [0, 1])


def write_calibration(best_binary: ModelResult, best_threshold: dict[str, Any], tables: dict[str, pd.DataFrame]) -> None:
    val = tables["val"]
    test = tables["test"]
    y_val = (val["diagnosis"].astype(int).to_numpy() >= 2).astype(int)
    y_test = (test["diagnosis"].astype(int).to_numpy() >= 2).astype(int)
    val_scores = binary_scores(best_binary.model, val[best_binary.features])
    test_scores = binary_scores(best_binary.model, test[best_binary.features])
    calibrator = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(val_scores.reshape(-1, 1), y_val)
    calibrated = calibrator.predict_proba(test_scores.reshape(-1, 1))[:, 1]
    rows = [
        {"model": best_binary.model_name, "feature_version": best_binary.feature_version, "calibration": "uncalibrated", "brier": brier_score_loss(y_test, test_scores), "log_loss": log_loss(y_test, np.clip(test_scores, 1e-6, 1 - 1e-6)), "auc": roc_auc_score(y_test, test_scores)},
        {"model": best_binary.model_name, "feature_version": best_binary.feature_version, "calibration": "platt_on_validation_scores", "brier": brier_score_loss(y_test, calibrated), "log_loss": log_loss(y_test, np.clip(calibrated, 1e-6, 1 - 1e-6)), "auc": roc_auc_score(y_test, calibrated)},
    ]
    write_rows(OUTPUT_DIR / "calibration_metrics.csv", rows)
    lines = [
        "# Calibration Report",
        "",
        "Binary Platt calibration was fit on validation-set scores from the selected hybrid binary model and evaluated on the held-out test set.",
        "5-class calibration was not used for selection; the report focuses on binary screening probability quality.",
        "",
        f"Uncalibrated Brier: {rows[0]['brier']:.4f}",
        f"Platt-calibrated Brier: {rows[1]['brier']:.4f}",
    ]
    (OUTPUT_DIR / "calibration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    threshold_rows = read_csv_rows(OUTPUT_DIR / "hybrid_binary_validation_threshold_sweep.csv")
    candidate_rows = [row for row in threshold_rows if row.get("model_name") == best_binary.model_name and row.get("feature_version") == best_binary.feature_version]
    high_recall = max(candidate_rows, key=lambda row: ffloat(row.get("referable_recall")), default={})
    balanced = max(candidate_rows, key=lambda row: ffloat(row.get("balanced_accuracy")), default={})
    low_fp = max(candidate_rows, key=lambda row: (-ffloat(row.get("false_positives")), ffloat(row.get("f1"))), default={})
    appdr_like = min(
        candidate_rows,
        key=lambda row: (
            abs(ffloat(row.get("referable_recall")) - APPDR_PRODUCTION_BINARY["referable_recall"]),
            abs(ffloat(row.get("false_positives")) - APPDR_PRODUCTION_BINARY["false_positives"]),
        ),
        default={},
    )
    op_lines = [
        "# Operating Point Report",
        "",
        "Operating points are chosen from validation-threshold sweeps only. Final metrics for the selected operating point are reported on the held-out test split.",
        "",
        f"Selected test threshold: {best_threshold.get('threshold')}",
        f"Selected test referable recall: {pct(best_threshold.get('referable_recall'))}",
        f"Selected test false negatives / false positives: {best_threshold.get('false_negatives')} / {best_threshold.get('false_positives')}",
        "",
        f"Validation high-recall threshold: {high_recall.get('threshold', 'n/a')}",
        f"Validation balanced threshold: {balanced.get('threshold', 'n/a')}",
        f"Validation low-false-positive threshold: {low_fp.get('threshold', 'n/a')}",
        f"Validation AppDR-comparable threshold: {appdr_like.get('threshold', 'n/a')}",
    ]
    (OUTPUT_DIR / "operating_point_report.md").write_text("\n".join(op_lines) + "\n", encoding="utf-8")


def write_comparison_reports(best_five: ModelResult, best_binary: ModelResult, best_threshold: dict[str, Any]) -> None:
    appdr_5 = read_first(FULL_CNN_DIR / "appdr_current_5class_metrics.csv") or APPDR_PRODUCTION_5
    appdr_bin = read_first(FULL_CNN_DIR / "appdr_current_binary_metrics.csv") or APPDR_PRODUCTION_BINARY
    previous_cnn_5 = read_first(FULL_CNN_DIR / "cnn_5class_metrics.csv")
    previous_cnn_bin = read_first(FULL_CNN_DIR / "cnn_binary_metrics.csv")
    cnn_5 = read_json_or_empty(CNN_SOURCE_DIR / "efficientnet_b3_full_training" / "test_metrics_best_checkpoint.json") or previous_cnn_5
    cnn_bin = load_trained_cnn_binary_metrics() or previous_cnn_bin
    prev_hybrid_5 = read_json_or_empty(PREVIOUS_HYBRID_DIR / "hybrid_5class_metrics.json")
    prev_hybrid_bin = read_json_or_empty(PREVIOUS_HYBRID_DIR / "hybrid_binary_metrics.json")
    comparison_5 = [
        compare_5_row("AppDR production XGBoost", "203 handcrafted", True, False, "", "XGBoost", appdr_5, "same full test split"),
        compare_5_row("Best experimental feature-based AppDR", "expanded handcrafted", True, False, "", "LightGBM top150", BEST_EXPERIMENTAL_5, "reported prior experiment"),
        compare_5_row("Fully trained straight CNN", "image CNN", False, True, "", "EfficientNet-B3 GeM weighted CE", cnn_5, "validation-selected epoch 16; same full test split"),
        compare_5_row("Previous one-epoch straight CNN", "image CNN", False, True, "", "ResNet50 GeM SmoothL1", previous_cnn_5, "previous weak baseline; same full test split"),
        compare_5_row("Previous full hybrid", "handcrafted + CNN", True, True, prev_hybrid_5.get("feature_version", ""), prev_hybrid_5.get("model_name", "LightGBM"), prev_hybrid_5, "previous full split hybrid with one-epoch ResNet source"),
        compare_5_row("Maximized validation-selected hybrid", "handcrafted + CNN", True, True, best_five.feature_version, best_five.model_name, best_five.metrics, "same full test split; selected by validation macro F1"),
    ]
    top_hybrid = sorted(read_csv_rows(OUTPUT_DIR / "hybrid_5class_model_comparison.csv"), key=lambda row: float(row.get("macro_f1", 0) or 0), reverse=True)[:3]
    for row in top_hybrid:
        comparison_5.append(compare_5_row(f"Exploratory maximized hybrid {row.get('model_name')}", "handcrafted + CNN", True, True, row.get("feature_version", ""), row.get("model_name", ""), row, "exploratory top test macro F1 row; not used for final selection"))
    write_rows(OUTPUT_DIR / "comparison_5class_all.csv", comparison_5)
    comparison_bin = [
        compare_bin_row("AppDR production SVM", "203 handcrafted", "", "SVM RBF", appdr_bin, "same full test split"),
        compare_bin_row("Fully trained straight CNN binary", "image CNN", "", "EfficientNet-B3 class >= 2", cnn_bin, "validation-selected epoch 16; same full test split"),
        compare_bin_row("Previous one-epoch straight CNN binary", "image CNN", "", "ResNet50 derived", previous_cnn_bin, "previous weak baseline; same full test split"),
        compare_bin_row("Previous full hybrid binary", "handcrafted + CNN", prev_hybrid_bin.get("feature_version", prev_hybrid_bin.get("feature_set", "")), prev_hybrid_bin.get("model_name", ""), prev_hybrid_bin, "previous full split hybrid"),
        compare_bin_row("Maximized hybrid binary selected operating point", "handcrafted + CNN", best_binary.feature_version, best_binary.model_name, best_threshold, "same full test split; threshold selected on validation"),
    ]
    top_bin = sorted(read_csv_rows(OUTPUT_DIR / "hybrid_binary_model_comparison.csv"), key=lambda row: (float(row.get("referable_recall", 0) or 0), float(row.get("f1", 0) or 0)), reverse=True)[:3]
    for row in top_bin:
        comparison_bin.append(compare_bin_row(f"Full hybrid binary candidate {row.get('model_name')}", "handcrafted + CNN", row.get("feature_version", ""), row.get("model_name", ""), row, "top hybrid binary candidate"))
    write_rows(OUTPUT_DIR / "comparison_binary_all.csv", comparison_bin)
    report = build_report(best_five, best_binary, best_threshold, appdr_5, appdr_bin, cnn_5, cnn_bin)
    (OUTPUT_DIR / "full_training_hybrid_vs_appdr_report.md").write_text(report["markdown"], encoding="utf-8")
    write_json(OUTPUT_DIR / "full_training_hybrid_vs_appdr_report.json", report)
    (OUTPUT_DIR / "final_recommendation.md").write_text(report["recommendation"], encoding="utf-8")


def load_trained_cnn_binary_metrics() -> dict[str, Any]:
    source_id = "efficientnet_b3_full_training"
    features_path = OUTPUT_DIR / "cnn_features_test.csv"
    if not features_path.exists():
        return {}
    class_column = f"cnn_{source_id}__predicted_class"
    score_column = f"cnn_{source_id}__referable_probability"
    frame = pd.read_csv(features_path, usecols=["diagnosis", class_column, score_column])
    y_true = (frame["diagnosis"].to_numpy(dtype=int) >= 2).astype(int)
    y_pred = (frame[class_column].to_numpy(dtype=int) >= 2).astype(int)
    scores = frame[score_column].to_numpy(dtype=float)
    metrics = binary_metrics(y_true, y_pred, scores, 0.5)
    metrics["threshold"] = "class>=2"
    write_json(CNN_SOURCE_DIR / source_id / "test_binary_metrics.json", metrics)
    return metrics


def compare_5_row(model: str, input_type: str, uses_appdr: bool, uses_cnn: bool, version: str, algorithm: str, metrics: dict[str, Any], notes: str) -> dict[str, Any]:
    return {
        "Model": model,
        "Input type": input_type,
        "Uses AppDR handcrafted features?": uses_appdr,
        "Uses CNN?": uses_cnn,
        "Hybrid version": version,
        "Algorithm": algorithm,
        "Accuracy": metrics.get("accuracy", ""),
        "Balanced accuracy": metrics.get("balanced_accuracy", ""),
        "Macro precision": metrics.get("macro_precision", ""),
        "Macro recall": metrics.get("macro_recall", ""),
        "Macro F1": metrics.get("macro_f1", ""),
        "Weighted F1": metrics.get("weighted_f1", ""),
        "Class 0 recall": metrics.get("class_0_recall", ""),
        "Class 1 recall": metrics.get("class_1_recall", ""),
        "Class 2 recall": metrics.get("class_2_recall", ""),
        "Class 3 recall": metrics.get("class_3_recall", ""),
        "Class 4 recall": metrics.get("class_4_recall", ""),
        "Notes": notes,
    }


def compare_bin_row(model: str, input_type: str, version: str, algorithm: str, metrics: dict[str, Any], notes: str) -> dict[str, Any]:
    return {
        "Model": model,
        "Input type": input_type,
        "Hybrid version": version,
        "Algorithm": algorithm,
        "Accuracy": metrics.get("accuracy", ""),
        "Balanced accuracy": metrics.get("balanced_accuracy", ""),
        "Precision": metrics.get("precision", ""),
        "Recall": metrics.get("recall", ""),
        "F1": metrics.get("f1", ""),
        "Referable recall": metrics.get("referable_recall", ""),
        "Non-referable recall": metrics.get("non_referable_recall", ""),
        "False negatives": metrics.get("false_negatives", ""),
        "False positives": metrics.get("false_positives", ""),
        "AUC": metrics.get("auc", ""),
        "Threshold": metrics.get("threshold", ""),
        "Notes": notes,
    }


def build_report(best_five: ModelResult, best_binary: ModelResult, best_threshold: dict[str, Any], appdr_5: dict[str, Any], appdr_bin: dict[str, Any], cnn_5: dict[str, Any], cnn_bin: dict[str, Any]) -> dict[str, Any]:
    hybrid_5 = best_five.metrics
    hybrid_bin = best_threshold
    top_test_5 = max(read_csv_rows(OUTPUT_DIR / "hybrid_5class_model_comparison.csv"), key=lambda row: ffloat(row.get("macro_f1")), default={})
    top_test_bin = max(read_csv_rows(OUTPUT_DIR / "hybrid_binary_model_comparison.csv"), key=lambda row: ffloat(row.get("f1")), default={})
    beat_appdr_5 = ffloat(hybrid_5.get("macro_f1")) > ffloat(appdr_5.get("macro_f1"))
    beat_best_exp = ffloat(hybrid_5.get("macro_f1")) > BEST_EXPERIMENTAL_5["macro_f1"]
    beat_cnn_5 = ffloat(hybrid_5.get("macro_f1")) > ffloat(cnn_5.get("macro_f1"))
    beat_appdr_bin = (
        ffloat(hybrid_bin.get("referable_recall")) >= ffloat(appdr_bin.get("referable_recall"))
        and int(float(hybrid_bin.get("false_negatives", 999999))) <= int(float(appdr_bin.get("false_negatives", 0)))
        and int(float(hybrid_bin.get("false_positives", 999999))) <= int(float(appdr_bin.get("false_positives", 999999))) * 1.25
    )
    future_integration_candidate = (
        beat_appdr_5
        and ffloat(hybrid_5.get("balanced_accuracy")) >= ffloat(appdr_5.get("balanced_accuracy"))
        and ffloat(hybrid_5.get("class_4_recall")) >= 0.60
        and beat_appdr_bin
    )
    production_replacement_now = False
    lines = [
        "# Full Hybrid CNN + AppDR Comparison",
        "",
        "Experiment only. Production was not updated.",
        "",
        "## Main Question",
        "",
        f"Does hybrid CNN + AppDR handcrafted features beat current AppDR production? {future_integration_candidate}",
        "",
        "## 5-Class Grading",
        "",
        f"Best hybrid feature version: `{best_five.feature_version}`",
        f"Best hybrid model: `{best_five.model_name}`",
        "The exported best model is selected by validation macro F1, then evaluated on the held-out test split.",
        f"Top measured test macro-F1 row: `{top_test_5.get('model_name', 'n/a')}` / `{top_test_5.get('feature_version', 'n/a')}` at {pct(top_test_5.get('macro_f1'))}. Treat this as exploratory because it is selected after looking at test metrics.",
        f"Hybrid macro F1: {pct(hybrid_5.get('macro_f1'))}",
        f"AppDR production macro F1 on same split: {pct(appdr_5.get('macro_f1'))}",
        f"Best experimental feature-based macro F1: {pct(BEST_EXPERIMENTAL_5['macro_f1'])}",
        f"Full straight CNN macro F1: {pct(cnn_5.get('macro_f1'))}",
        f"Hybrid beat AppDR production macro F1: {beat_appdr_5}",
        f"Hybrid beat best experimental feature-based macro F1: {beat_best_exp}",
        f"Hybrid beat straight CNN macro F1: {beat_cnn_5}",
        "",
        "| Metric | AppDR production | Full CNN | Best full hybrid |",
        "| --- | ---: | ---: | ---: |",
        f"| Accuracy | {pct(appdr_5.get('accuracy'))} | {pct(cnn_5.get('accuracy'))} | {pct(hybrid_5.get('accuracy'))} |",
        f"| Balanced accuracy | {pct(appdr_5.get('balanced_accuracy'))} | {pct(cnn_5.get('balanced_accuracy'))} | {pct(hybrid_5.get('balanced_accuracy'))} |",
        f"| Macro precision | {pct(appdr_5.get('macro_precision'))} | {pct(cnn_5.get('macro_precision'))} | {pct(hybrid_5.get('macro_precision'))} |",
        f"| Macro recall | {pct(appdr_5.get('macro_recall'))} | {pct(cnn_5.get('macro_recall'))} | {pct(hybrid_5.get('macro_recall'))} |",
        f"| Macro F1 | {pct(appdr_5.get('macro_f1'))} | {pct(cnn_5.get('macro_f1'))} | {pct(hybrid_5.get('macro_f1'))} |",
        f"| Class 1 recall | {pct(appdr_5.get('class_1_recall'))} | {pct(cnn_5.get('class_1_recall'))} | {pct(hybrid_5.get('class_1_recall'))} |",
        f"| Class 3 recall | {pct(appdr_5.get('class_3_recall'))} | {pct(cnn_5.get('class_3_recall'))} | {pct(hybrid_5.get('class_3_recall'))} |",
        f"| Class 4 recall | {pct(appdr_5.get('class_4_recall'))} | {pct(cnn_5.get('class_4_recall'))} | {pct(hybrid_5.get('class_4_recall'))} |",
        "",
        "## Binary Screening",
        "",
        f"Best hybrid binary feature version: `{best_binary.feature_version}`",
        f"Best hybrid binary model: `{best_binary.model_name}`",
        f"Top measured binary F1 row: `{top_test_bin.get('model_name', 'n/a')}` / `{top_test_bin.get('feature_version', 'n/a')}` at {pct(top_test_bin.get('f1'))}.",
        f"Hybrid referable recall: {pct(hybrid_bin.get('referable_recall'))}",
        f"AppDR referable recall on same split: {pct(appdr_bin.get('referable_recall'))}",
        f"Hybrid false negatives / false positives: {hybrid_bin.get('false_negatives')} / {hybrid_bin.get('false_positives')}",
        f"AppDR false negatives / false positives: {appdr_bin.get('false_negatives')} / {appdr_bin.get('false_positives')}",
        f"Hybrid binary tradeoff beats AppDR production: {beat_appdr_bin}",
        "",
        "## Recommendation",
        "",
        f"Future integration candidate: {future_integration_candidate}",
        f"Production replacement now: {production_replacement_now}",
        "Production should remain unchanged.",
        "Hybrid should remain experimental until repeated runs, calibration review, and deployment verification confirm stability.",
    ]
    recommendation = "\n".join(
        [
            "# Final Recommendation",
            "",
            "Do not update production.",
            f"Hybrid future integration candidate: {future_integration_candidate}",
            f"Hybrid should replace production now: {production_replacement_now}",
            f"Hybrid beats AppDR production 5-class macro F1: {beat_appdr_5}",
            f"Hybrid binary tradeoff beats AppDR production: {beat_appdr_bin}",
            "The decision is based on AppDR production as the main baseline, not on straight CNN alone.",
        ]
    ) + "\n"
    return {
        "best_hybrid_5class": best_five.metrics,
        "best_hybrid_binary": best_threshold,
        "beat_appdr_5class_macro_f1": beat_appdr_5,
        "beat_best_experimental_5class_macro_f1": beat_best_exp,
        "beat_straight_cnn_5class_macro_f1": beat_cnn_5,
        "beat_appdr_binary_tradeoff": beat_appdr_bin,
        "future_integration_candidate": future_integration_candidate,
        "production_replacement_now": production_replacement_now,
        "markdown": "\n".join(lines) + "\n",
        "recommendation": recommendation,
    }


def write_confusion(path: Path, y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for idx, values in enumerate(matrix):
        row = {"actual_class": labels[idx]}
        for pred_label, value in zip(labels, values):
            row[f"predicted_{pred_label}"] = int(value)
        rows.append(row)
    write_rows(path, rows)


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


def read_first(path: Path) -> dict[str, Any]:
    rows = read_csv_rows(path)
    return rows[0] if rows else {}


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
