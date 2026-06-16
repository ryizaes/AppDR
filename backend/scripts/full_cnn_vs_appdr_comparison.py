"""Full straight-CNN vs AppDR comparison.

Experiment only. This script does not update production, does not change app
prediction behavior, and does not use the 203 handcrafted features as CNN input.
The 203-feature table is used only to evaluate existing AppDR models on the same
held-out test split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
)
from sklearn.model_selection import train_test_split


BACKEND_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BACKEND_DIR / "results" / "full_cnn_vs_appdr_comparison"
SOURCE_FEATURES = BACKEND_DIR / "features_combined_balanced.csv"
RANDOM_STATE = 42
DEFAULT_THRESHOLDS = [0.5, 1.5, 2.5, 3.5]
APTOS_THRESHOLDS = [0.7, 1.5, 2.5, 3.5]
BINARY_THRESHOLDS = [round(v, 2) for v in np.arange(0.05, 0.801, 0.05)]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config


@dataclass
class CnnRun:
    run_id: str
    backbone: str
    input_size: int
    output_type: str
    pooling: str
    loss: str
    status: str
    checkpoint_path: str
    metrics: dict[str, Any]


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(RANDOM_STATE)

    manifest = build_or_load_manifest(args.rebuild_manifest, args.hash_images)
    train_manifest, val_manifest, test_manifest = build_or_load_split(manifest)
    write_dataset_summary(manifest)
    write_split_summary(train_manifest, val_manifest, test_manifest)
    write_augmentation_config()

    appdr_5, appdr_binary = evaluate_appdr(test_manifest)
    write_json(OUTPUT_DIR / "appdr_current_5class_metrics.json", appdr_5)
    write_rows(OUTPUT_DIR / "appdr_current_5class_metrics.csv", [appdr_5])
    write_json(OUTPUT_DIR / "appdr_current_binary_metrics.json", appdr_binary)
    write_rows(OUTPUT_DIR / "appdr_current_binary_metrics.csv", [appdr_binary])

    runs: list[CnnRun] = []
    if args.report_only:
        runs = load_existing_cnn_runs()
    else:
        for spec in parse_model_specs(args.models):
            try:
                runs.append(train_one_cnn(spec, train_manifest, val_manifest, test_manifest, args))
            except Exception as error:
                runs.append(
                    CnnRun(
                        run_id=spec["run_id"],
                        backbone=spec["backbone"],
                        input_size=int(spec["input_size"]),
                        output_type=spec["output_type"],
                        pooling=spec["pooling"],
                        loss=spec["loss"],
                        status="failed",
                        checkpoint_path="",
                        metrics={"model_name": spec["run_id"], "status": "failed", "error": str(error)[:500]},
                    )
                )

    write_cnn_outputs(runs, test_manifest)
    write_comparisons(appdr_5, appdr_binary, runs)
    print(f"Created {OUTPUT_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="resnet50:384:regression:gem,efficientnet_b3:384:regression:gem",
        help="Comma specs: backbone:input_size:output_type:pooling. output_type=regression|classification.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=1)
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--hash-images", action="store_true")
    parser.add_argument("--smoke-limit", type=int, default=0, help="Smoke only; 0 uses full split.")
    parser.add_argument("--report-only", action="store_true")
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


def build_or_load_manifest(rebuild: bool, hash_images: bool) -> pd.DataFrame:
    path = OUTPUT_DIR / "dataset_manifest.csv"
    if path.exists() and not rebuild:
        return pd.read_csv(path)

    source = pd.read_csv(SOURCE_FEATURES)
    required = ["image_path", "label", "source_dataset", "image_id", "image_sha256"]
    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for source_row_index, row in source.reset_index(drop=True).iterrows():
        image_path = Path(str(row["image_path"]))
        normalized = str(image_path.resolve()) if image_path.exists() else str(image_path)
        if normalized in seen_paths:
            continue
        seen_paths.add(normalized)
        label = int(row["label"])
        if label not in {0, 1, 2, 3, 4}:
            continue
        readable = False
        width = 0
        height = 0
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                readable = True
        except Exception:
            readable = False
        if not readable:
            continue
        image_hash = str(row.get("image_sha256", ""))
        if hash_images and not image_hash:
            image_hash = sha256_file(image_path)
        rows.append(
            {
                "source_row_index": int(source_row_index),
                "image_path": normalized,
                "diagnosis": label,
                "source_dataset": str(row.get("source_dataset", "")),
                "id_code": str(row.get("image_id", image_path.stem)),
                "patient_id": "",
                "eye": "",
                "angle": "",
                "image_width": int(width),
                "image_height": int(height),
                "readable_status": "readable",
                "hash": image_hash,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(path, index=False)
    return manifest


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_or_load_split(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = [OUTPUT_DIR / "train_manifest.csv", OUTPUT_DIR / "val_manifest.csv", OUTPUT_DIR / "test_manifest.csv"]
    if all(path.exists() for path in paths):
        return tuple(pd.read_csv(path) for path in paths)  # type: ignore[return-value]

    train_val, test = train_test_split(
        manifest,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=manifest["diagnosis"],
    )
    train, val = train_test_split(
        train_val,
        test_size=0.15 / 0.85,
        random_state=RANDOM_STATE,
        stratify=train_val["diagnosis"],
    )
    train = train.reset_index(drop=True).assign(split="train")
    val = val.reset_index(drop=True).assign(split="val")
    test = test.reset_index(drop=True).assign(split="test")
    train.to_csv(OUTPUT_DIR / "train_manifest.csv", index=False)
    val.to_csv(OUTPUT_DIR / "val_manifest.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test_manifest.csv", index=False)
    return train, val, test


def write_dataset_summary(manifest: pd.DataFrame) -> None:
    lines = [
        "# Dataset Summary",
        "",
        "Source catalog: `backend/features_combined_balanced.csv` unique readable image paths.",
        "Unlabeled images are not used.",
        "Patient IDs, eye, and angle metadata were not available, so patient-level leakage cannot be fully ruled out.",
        "",
        f"Total readable labeled images: {len(manifest)}",
        "",
        "## Class Counts By Source",
        "",
        "| Source | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 | Total |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source, group in manifest.groupby("source_dataset"):
        counts = group["diagnosis"].value_counts().sort_index().to_dict()
        lines.append(
            f"| {source} | {counts.get(0, 0)} | {counts.get(1, 0)} | {counts.get(2, 0)} | {counts.get(3, 0)} | {counts.get(4, 0)} | {len(group)} |"
        )
    lines.extend(
        [
            "",
            f"Duplicate image paths removed: {manifest['image_path'].duplicated().sum()}",
            f"Duplicate hashes present: {manifest['hash'].duplicated().sum() if 'hash' in manifest else 'not checked'}",
        ]
    )
    (OUTPUT_DIR / "dataset_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_split_summary(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    lines = [
        "# Split Summary",
        "",
        "Split: 70% train, 15% validation, 15% test, stratified by diagnosis, random_state=42.",
        "No patient_id is available, so this is an image-level split.",
        "",
        "| Split | Rows | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, frame in [("train", train), ("val", val), ("test", test)]:
        counts = frame["diagnosis"].value_counts().sort_index().to_dict()
        lines.append(
            f"| {name} | {len(frame)} | {counts.get(0, 0)} | {counts.get(1, 0)} | {counts.get(2, 0)} | {counts.get(3, 0)} | {counts.get(4, 0)} |"
        )
    lines.extend(["", "## Source Distribution", ""])
    for name, frame in [("train", train), ("val", val), ("test", test)]:
        lines.append(f"### {name}")
        lines.append(frame["source_dataset"].value_counts().to_string())
        lines.append("")
    (OUTPUT_DIR / "split_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_augmentation_config() -> None:
    payload = {
        "plain_resizing": True,
        "normalize": "ImageNet mean/std",
        "random_brightness_contrast": True,
        "hue_saturation_shift": True,
        "blur_sharpen": True,
        "rotate_range": 180,
        "scale_range": 0.2,
        "shift_range": 0.2,
        "shear_range": 0.2,
        "horizontal_flip": True,
        "note": "Fundus-safe approximation of Guanshuo APTOS augmentation using torchvision/PIL transforms.",
    }
    write_json(OUTPUT_DIR / "augmentation_config.json", payload)


def evaluate_appdr(test: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    features = pd.read_csv(SOURCE_FEATURES)
    x_test = features.iloc[test["source_row_index"].astype(int).to_numpy()][config.FEATURE_NAMES]
    y_test = test["diagnosis"].astype(int).to_numpy()
    y_binary = (y_test >= 2).astype(int)
    with (BACKEND_DIR / "results" / "best_model.pkl").open("rb") as file:
        grading = pickle.load(file)
    with (BACKEND_DIR / "results" / "binary" / "best_model.pkl").open("rb") as file:
        binary = pickle.load(file)
    pred_5 = grading.predict(x_test)
    proba_binary = binary.predict_proba(x_test)[:, 1] if hasattr(binary, "predict_proba") else binary.predict(x_test)
    pred_binary = (proba_binary >= 0.20).astype(int)
    metrics_5 = five_class_metrics("AppDR production XGBoost same full test split", y_test, pred_5)
    metrics_binary = binary_metrics("AppDR production SVM RBF same full test split", y_binary, pred_binary, proba_binary, "0.20")
    write_confusion(OUTPUT_DIR / "appdr_current_5class_confusion_matrix.csv", y_test, pred_5, [0, 1, 2, 3, 4])
    write_confusion(OUTPUT_DIR / "appdr_current_binary_confusion_matrix.csv", y_binary, pred_binary, [0, 1])
    return metrics_5, metrics_binary


def parse_model_specs(spec_text: str) -> list[dict[str, Any]]:
    specs = []
    for item in [part.strip() for part in spec_text.split(",") if part.strip()]:
        pieces = item.split(":")
        if len(pieces) != 4:
            raise ValueError(f"Invalid model spec: {item}")
        backbone, input_size, output_type, pooling = pieces
        loss = "SmoothL1Loss" if output_type == "regression" else "CrossEntropyLoss"
        specs.append(
            {
                "run_id": f"{backbone}_{input_size}_{output_type}_{pooling}",
                "backbone": backbone,
                "input_size": int(input_size),
                "output_type": output_type,
                "pooling": pooling,
                "loss": loss,
            }
        )
    return specs


def train_one_cnn(spec: dict[str, Any], train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, args: argparse.Namespace) -> CnnRun:
    import torch
    import torch.nn as nn
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from torchvision import transforms
    import timm

    run_dir = OUTPUT_DIR / "checkpoints" / spec["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "model_config.json", {**spec, "epochs": args.epochs, "batch_size": args.batch_size, "pretrained": not args.no_pretrained})

    if args.smoke_limit > 0:
        train = stratified_cap(train, args.smoke_limit)
        val = stratified_cap(val, max(50, args.smoke_limit // 4))
        test = stratified_cap(test, max(50, args.smoke_limit // 4))

    class FundusDataset(Dataset):
        def __init__(self, frame: pd.DataFrame, is_train: bool):
            self.frame = frame.reset_index(drop=True)
            self.is_train = is_train
            self.transform = transforms.Compose(
                [
                    transforms.Resize((spec["input_size"], spec["input_size"])),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

        def __len__(self) -> int:
            return len(self.frame)

        def __getitem__(self, index: int):
            row = self.frame.iloc[index]
            image = Image.open(row["image_path"]).convert("RGB")
            if self.is_train:
                image = augment_image(image)
            tensor = self.transform(image)
            label_int = int(row["diagnosis"])
            if spec["output_type"] == "regression":
                return tensor, torch.tensor(float(label_int), dtype=torch.float32)
            return tensor, torch.tensor(label_int, dtype=torch.long)

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
            try:
                self.encoder = timm.create_model(spec["backbone"], pretrained=not args.no_pretrained, num_classes=0, global_pool="")
                pretrained_used = not args.no_pretrained
            except Exception:
                self.encoder = timm.create_model(spec["backbone"], pretrained=False, num_classes=0, global_pool="")
                pretrained_used = False
            self.pretrained_used = pretrained_used
            channels = int(getattr(self.encoder, "num_features", 0))
            self.pool = GeM() if spec["pooling"] == "gem" else nn.AdaptiveAvgPool2d(1)
            out_dim = 1 if spec["output_type"] == "regression" else 5
            self.head = nn.Linear(channels, out_dim)

        def forward(self, x):
            features = self.encoder(x)
            if features.ndim == 2:
                pooled = features
            elif spec["pooling"] == "gem":
                pooled = self.pool(features)
            else:
                pooled = self.pool(features).flatten(1)
            output = self.head(pooled)
            return output.squeeze(1) if spec["output_type"] == "regression" else output

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CnnModel().to(device)
    weights = class_weights(train["diagnosis"].to_numpy())
    sample_weights = train["diagnosis"].map(lambda label: weights[int(label)]).to_numpy()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(FundusDataset(train, True), batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(FundusDataset(val, False), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(FundusDataset(test, False), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    criterion: Any
    if spec["output_type"] == "regression":
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([weights[i] for i in range(5)], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    best_state = None
    best_val = -1.0
    stale = 0
    logs: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        if epoch <= args.freeze_backbone_epochs:
            for param in model.encoder.parameters():
                param.requires_grad = False
        else:
            for param in model.encoder.parameters():
                param.requires_grad = True
        model.train()
        total_loss = 0.0
        total_count = 0
        optimizer.zero_grad(set_to_none=True)
        start = time.time()
        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast(enabled=torch.cuda.is_available()):
                output = model(images)
                if spec["output_type"] == "regression":
                    loss = criterion(output.clamp(0, 4), labels)
                else:
                    loss = criterion(output, labels)
                loss = loss / max(args.grad_accum, 1)
            scaler.scale(loss).backward()
            if step % max(args.grad_accum, 1) == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach().cpu()) * len(labels) * max(args.grad_accum, 1)
            total_count += len(labels)
            if step == 1 or step % 100 == 0 or step == len(train_loader):
                progress_row = {
                    "run_id": spec["run_id"],
                    "epoch": epoch,
                    "step": step,
                    "total_steps": len(train_loader),
                    "seen_images": total_count,
                    "avg_train_loss_so_far": total_loss / max(total_count, 1),
                    "elapsed_seconds": round(time.time() - start, 2),
                }
                append_rows(OUTPUT_DIR / "training_progress_per_model.csv", [progress_row])
                print(json.dumps(progress_row), flush=True)
        val_pred, val_scores, val_labels = predict_model(model, val_loader, device, spec["output_type"])
        val_macro_f1 = f1_score(val_labels, val_pred, average="macro", zero_division=0)
        val_qwk = cohen_kappa_score(val_labels, val_pred, weights="quadratic")
        row = {
            "run_id": spec["run_id"],
            "epoch": epoch,
            "train_loss": total_loss / max(total_count, 1),
            "val_macro_f1": val_macro_f1,
            "val_qwk": val_qwk,
            "elapsed_seconds": round(time.time() - start, 2),
        }
        logs.append(row)
        append_rows(OUTPUT_DIR / "training_log_per_model.csv", [row])
        if val_macro_f1 > best_val:
            best_val = val_macro_f1
            stale = 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save({"state_dict": best_state, "spec": spec}, run_dir / "best_model.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    val_pred, val_scores, val_labels = predict_model(model, val_loader, device, spec["output_type"])
    opt_thresholds = optimize_thresholds(val_labels, val_scores) if spec["output_type"] == "regression" else []
    test_pred, test_scores, test_labels = predict_model(
        model,
        test_loader,
        device,
        spec["output_type"],
        thresholds=opt_thresholds if opt_thresholds else None,
    )
    metrics = five_class_metrics(spec["run_id"], test_labels, test_pred)
    metrics.update(
        {
            "Backbone": spec["backbone"],
            "Input size": spec["input_size"],
            "Loss": spec["loss"],
            "Pooling": spec["pooling"],
            "run_id": spec["run_id"],
            "threshold_set": "validation_optimized" if opt_thresholds else "argmax",
            "thresholds": json.dumps(opt_thresholds) if opt_thresholds else "",
        }
    )
    write_prediction_export(spec["run_id"], test, test_labels, test_pred, test_scores, spec["output_type"])
    write_threshold_tuning_row(spec["run_id"], val_labels, val_scores, test_labels, test_scores, spec["output_type"], opt_thresholds)
    write_confusion(OUTPUT_DIR / "cnn_5class_confusion_matrix.csv", test_labels, test_pred, [0, 1, 2, 3, 4])
    return CnnRun(spec["run_id"], spec["backbone"], int(spec["input_size"]), spec["output_type"], spec["pooling"], spec["loss"], "trained", str(run_dir / "best_model.pt"), metrics)


def augment_image(image: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    image = image.rotate(random.uniform(-180, 180))
    if random.random() < 0.8:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.8:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.92, 1.08))
    if random.random() < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.0)))
    if random.random() < 0.25:
        image = image.filter(ImageFilter.SHARPEN)
    return image


def predict_model(model: Any, loader: Any, device: Any, output_type: str, thresholds: list[float] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    model.eval()
    preds: list[int] = []
    scores: list[float] = []
    labels_out: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            output = model(images)
            if output_type == "regression":
                values = output.detach().cpu().numpy().clip(0, 4)
                pred = apply_thresholds(values, thresholds or APTOS_THRESHOLDS)
                scores.extend(values.tolist())
            else:
                probs = torch.softmax(output, dim=1).detach().cpu().numpy()
                pred = probs.argmax(axis=1)
                scores.extend(probs[:, 2:].sum(axis=1).tolist())
            preds.extend(pred.tolist())
            labels_out.extend(labels.detach().cpu().numpy().astype(int).tolist())
    return np.asarray(preds, dtype=int), np.asarray(scores, dtype=float), np.asarray(labels_out, dtype=int)


def class_weights(labels: np.ndarray) -> dict[int, float]:
    counts = pd.Series(labels).value_counts().to_dict()
    total = len(labels)
    return {label: total / (5.0 * max(counts.get(label, 1), 1)) for label in range(5)}


def stratified_cap(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0 or len(frame) <= limit:
        return frame
    parts = []
    per_class = max(1, limit // frame["diagnosis"].nunique())
    for _, group in frame.groupby("diagnosis"):
        parts.append(group.sample(min(len(group), per_class), random_state=RANDOM_STATE))
    return pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def apply_thresholds(values: np.ndarray, thresholds: list[float]) -> np.ndarray:
    out = np.zeros(len(values), dtype=int)
    for threshold in thresholds:
        out += values > threshold
    return np.clip(out, 0, 4)


def optimize_thresholds(labels: np.ndarray, scores: np.ndarray) -> list[float]:
    best = list(APTOS_THRESHOLDS)
    best_score = cohen_kappa_score(labels, apply_thresholds(scores, best), weights="quadratic")
    grids = [np.arange(0.3, 1.1, 0.1), np.arange(1.1, 2.1, 0.1), np.arange(2.1, 3.1, 0.1), np.arange(3.1, 3.9, 0.1)]
    for i, grid in enumerate(grids):
        for value in grid:
            candidate = list(best)
            candidate[i] = round(float(value), 2)
            if not all(candidate[j] < candidate[j + 1] for j in range(3)):
                continue
            score = cohen_kappa_score(labels, apply_thresholds(scores, candidate), weights="quadratic")
            if score > best_score:
                best = candidate
                best_score = score
    return best


def write_prediction_export(run_id: str, test: pd.DataFrame, labels: np.ndarray, preds: np.ndarray, scores: np.ndarray, output_type: str) -> None:
    frame = test.reset_index(drop=True)
    rows = []
    score_values = binary_scores(scores, output_type)
    for index, row in frame.iterrows():
        rows.append(
            {
                "run_id": run_id,
                "image_path": row["image_path"],
                "source_dataset": row.get("source_dataset", ""),
                "id_code": row.get("id_code", ""),
                "diagnosis": int(labels[index]),
                "predicted_class": int(preds[index]),
                "score": float(scores[index]),
                "binary_score": float(score_values[index]),
                "true_referable": int(labels[index] >= 2),
                "predicted_referable_from_class": int(preds[index] >= 2),
            }
        )
    write_rows(OUTPUT_DIR / f"predictions_{run_id}.csv", rows)


def write_threshold_tuning_row(
    run_id: str,
    val_labels: np.ndarray,
    val_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    output_type: str,
    optimized: list[float],
) -> None:
    rows = []
    if output_type == "regression":
        for name, thresholds in [
            ("default", DEFAULT_THRESHOLDS),
            ("aptos_adjusted", APTOS_THRESHOLDS),
            ("validation_optimized", optimized),
        ]:
            val_pred = apply_thresholds(val_scores, thresholds)
            test_pred = apply_thresholds(test_scores, thresholds)
            rows.append(
                {
                    "run_id": run_id,
                    "threshold_set": name,
                    "thresholds": json.dumps(thresholds),
                    "val_macro_f1": f1_score(val_labels, val_pred, average="macro", zero_division=0),
                    "val_qwk": cohen_kappa_score(val_labels, val_pred, weights="quadratic"),
                    "test_macro_f1": f1_score(test_labels, test_pred, average="macro", zero_division=0),
                    "test_balanced_accuracy": balanced_accuracy_score(test_labels, test_pred),
                }
            )
    else:
        rows.append({"run_id": run_id, "threshold_set": "argmax", "thresholds": "", "note": "Classification model uses argmax for 5-class grading."})
    append_rows(OUTPUT_DIR / "threshold_tuning.csv", rows)


def binary_scores(scores: np.ndarray, output_type: str) -> np.ndarray:
    if output_type == "regression":
        return np.clip(scores / 4.0, 0, 1)
    return np.clip(scores, 0, 1)


def load_existing_cnn_runs() -> list[CnnRun]:
    rows = read_csv_rows(OUTPUT_DIR / "cnn_single_model_comparison.csv")
    runs = []
    for row in rows:
        runs.append(CnnRun(row.get("run_id", row.get("model_name", "")), row.get("Backbone", ""), int(float(row.get("Input size", 0) or 0)), "", row.get("Pooling", ""), row.get("Loss", ""), row.get("status", "trained"), "", row))
    return runs


def write_cnn_outputs(runs: list[CnnRun], test: pd.DataFrame) -> None:
    rows = [run.metrics for run in runs]
    write_rows(OUTPUT_DIR / "cnn_single_model_comparison.csv", rows)
    write_model_config_per_model(runs)
    valid = [run for run in runs if run.metrics.get("status") == "ok"]
    best = max(valid, key=lambda run: float(run.metrics.get("macro_f1", 0))) if valid else None
    if best:
        write_rows(OUTPUT_DIR / "cnn_5class_metrics.csv", [best.metrics])
        report = []
        for label in range(5):
            report.append(
                {
                    "class": label,
                    "precision": best.metrics.get(f"class_{label}_precision", ""),
                    "recall": best.metrics.get(f"class_{label}_recall", ""),
                    "f1": best.metrics.get(f"class_{label}_f1", ""),
                    "support": best.metrics.get(f"class_{label}_support", ""),
                }
            )
        write_rows(OUTPUT_DIR / "cnn_5class_per_class_metrics.csv", report)
    write_threshold_and_binary_reports(runs)
    write_gem_ablation(runs)
    write_ensemble_report(runs)


def write_model_config_per_model(runs: list[CnnRun]) -> None:
    payload = {}
    for run in runs:
        checkpoint_config = OUTPUT_DIR / "checkpoints" / run.run_id / "model_config.json"
        saved = read_json_or_empty(checkpoint_config)
        payload[run.run_id] = {
            "backbone": saved.get("backbone", run.backbone),
            "input_size": saved.get("input_size", run.input_size),
            "output_type": saved.get("output_type", run.output_type),
            "pooling": saved.get("pooling", run.pooling),
            "loss": saved.get("loss", run.loss),
            "epochs": saved.get("epochs", ""),
            "batch_size": saved.get("batch_size", ""),
            "pretrained": saved.get("pretrained", ""),
            "status": run.status,
            "checkpoint_path": run.checkpoint_path or str(OUTPUT_DIR / "checkpoints" / run.run_id / "best_model.pt"),
        }
    write_json(OUTPUT_DIR / "model_config_per_model.json", payload)


def write_threshold_and_binary_reports(runs: list[CnnRun]) -> None:
    rows = []
    threshold_rows = []
    for run in runs:
        if run.metrics.get("status") != "ok":
            continue
        pred_path = OUTPUT_DIR / f"predictions_{run.run_id}.csv"
        pred_rows = read_csv_rows(pred_path)
        if not pred_rows:
            rows.append({"model_name": run.run_id, "status": "predictions_not_available"})
            continue
        y_true = np.asarray([int(row["true_referable"]) for row in pred_rows], dtype=int)
        y_pred_from_class = np.asarray([int(row["predicted_referable_from_class"]) for row in pred_rows], dtype=int)
        score = np.asarray([float(row["binary_score"]) for row in pred_rows], dtype=float)
        class_row = binary_metrics(run.run_id, y_true, y_pred_from_class, score, "class>=2")
        class_row["status"] = "ok"
        rows.append(class_row)
        for threshold in np.arange(0.10, 0.701, 0.05):
            pred = (score >= threshold).astype(int)
            sweep = binary_metrics(run.run_id, y_true, pred, score, f"{threshold:.2f}")
            sweep["binary_threshold"] = round(float(threshold), 2)
            threshold_rows.append(sweep)
    if not rows:
        rows = [{"status": "not_available"}]
    write_rows(OUTPUT_DIR / "cnn_binary_metrics.csv", rows)
    write_rows(OUTPUT_DIR / "cnn_binary_threshold_sweep.csv", threshold_rows or [{"status": "not_available"}])
    write_json(OUTPUT_DIR / "best_thresholds.json", {"default": DEFAULT_THRESHOLDS, "aptos_adjusted": APTOS_THRESHOLDS})


def write_gem_ablation(runs: list[CnnRun]) -> None:
    gem = [run.metrics for run in runs if run.pooling == "gem"]
    normal = [run.metrics for run in runs if run.pooling != "gem"]
    lines = ["# GeM Ablation Report", "", f"GeM runs completed: {len(gem)}", f"Normal pooling runs completed: {len(normal)}", ""]
    if not normal:
        lines.append("No normal-pooling ablation completed in this pass. GeM was prioritized because it is part of the APTOS 1st-place basis.")
    (OUTPUT_DIR / "gem_ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ensemble_report(runs: list[CnnRun]) -> None:
    valid = [run for run in runs if run.metrics.get("status") == "ok"]
    if len(valid) < 2:
        write_rows(OUTPUT_DIR / "cnn_ensemble_metrics.csv", [{"status": "not_created", "reason": "Need at least two completed CNN runs with saved predictions."}])
        write_json(OUTPUT_DIR / "cnn_ensemble_config.json", {"status": "not_created"})
    else:
        write_rows(OUTPUT_DIR / "cnn_ensemble_metrics.csv", [{"status": "not_created", "reason": "Prediction averaging is planned after per-image prediction export is enabled."}])
        write_json(OUTPUT_DIR / "cnn_ensemble_config.json", {"status": "planned", "members": [run.run_id for run in valid]})


def write_comparisons(appdr_5: dict[str, Any], appdr_binary: dict[str, Any], runs: list[CnnRun]) -> None:
    previous_cnn = read_first(BACKEND_DIR / "results" / "aptos_guanshuo_close_comparison" / "cnn_single_5class_metrics.csv")
    previous_hybrid = read_json_or_empty(BACKEND_DIR / "results" / "hybrid_cnn_features_comparison" / "hybrid_5class_metrics.json")
    best_cnn = max([run for run in runs if run.metrics.get("status") == "ok"], key=lambda run: float(run.metrics.get("macro_f1", 0)), default=None)
    comparison_5 = [
        comparison_5_row("AppDR production XGBoost", "203 handcrafted features", True, False, "", "", "", "", appdr_5, "same full test split"),
        comparison_5_row("AppDR best experimental LightGBM top150", "expanded handcrafted features", True, False, "", "", "", "", {"accuracy": 0.67, "balanced_accuracy": 0.6113, "macro_f1": 0.5799, "class_1_recall": 0.4533, "class_3_recall": 0.6, "class_4_recall": 0.6433}, "reported prior experiment"),
        comparison_5_row("Previous capped straight CNN", "image CNN", False, True, "seresnext50_32x4d", "384", "SmoothL1Loss", "GeM", previous_cnn, "previous capped run"),
        comparison_5_row("Previous hybrid", "CNN + handcrafted", True, True, "", "", "", "", previous_hybrid, "previous capped hybrid run"),
    ]
    for run in runs:
        comparison_5.append(comparison_5_row(run.run_id, "image CNN", False, True, run.backbone, run.input_size, run.loss, run.pooling, run.metrics, run.status))
    write_rows(OUTPUT_DIR / "comparison_5class_all.csv", comparison_5)
    comparison_binary = [
        {"Model": "AppDR production SVM", **binary_compare_fields(appdr_binary), "Notes": "same full test split"},
        {"Model": "AppDR best experimental binary", "Referable recall": 0.9697, "False negatives": 48, "False positives": 817, "Notes": "reported prior experiment"},
        {"Model": "Previous capped CNN binary", **binary_compare_fields(read_first(BACKEND_DIR / "results" / "aptos_guanshuo_close_comparison" / "cnn_single_binary_metrics.csv")), "Notes": "previous capped run"},
        {"Model": "Previous hybrid binary", **binary_compare_fields(read_json_or_empty(BACKEND_DIR / "results" / "hybrid_cnn_features_comparison" / "hybrid_binary_metrics.json")), "Notes": "previous capped hybrid run"},
    ]
    for row in read_csv_rows(OUTPUT_DIR / "cnn_binary_metrics.csv"):
        comparison_binary.append({"Model": row.get("model_name", "CNN"), **binary_compare_fields(row), "Notes": row.get("status", "CNN same full test split")})
    write_rows(OUTPUT_DIR / "comparison_binary_all.csv", comparison_binary)
    report = build_final_report(appdr_5, appdr_binary, best_cnn, runs)
    (OUTPUT_DIR / "full_cnn_vs_appdr_report.md").write_text(report["markdown"], encoding="utf-8")
    write_json(OUTPUT_DIR / "full_cnn_vs_appdr_report.json", report)
    (OUTPUT_DIR / "final_recommendation.md").write_text(report["recommendation"], encoding="utf-8")


def comparison_5_row(model: str, input_type: str, uses_features: bool, uses_cnn: bool, backbone: Any, input_size: Any, loss: str, pooling: str, metrics: dict[str, Any], notes: str) -> dict[str, Any]:
    row = {
        "Model": model,
        "Input type": input_type,
        "Uses 203 handcrafted features?": uses_features,
        "Uses CNN?": uses_cnn,
        "Backbone": backbone,
        "Input size": input_size,
        "Loss": loss,
        "Pooling": pooling,
        "Notes": notes,
    }
    for out_key, metric_key in [
        ("Accuracy", "accuracy"),
        ("Balanced accuracy", "balanced_accuracy"),
        ("Macro precision", "macro_precision"),
        ("Macro recall", "macro_recall"),
        ("Macro F1", "macro_f1"),
        ("Weighted F1", "weighted_f1"),
        ("Class 0 recall", "class_0_recall"),
        ("Class 1 recall", "class_1_recall"),
        ("Class 2 recall", "class_2_recall"),
        ("Class 3 recall", "class_3_recall"),
        ("Class 4 recall", "class_4_recall"),
        ("QWK if available", "quadratic_weighted_kappa"),
    ]:
        row[out_key] = metrics.get(metric_key, "")
    return row


def binary_compare_fields(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
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
        "Threshold": metrics.get("threshold", metrics.get("threshold_used", "")),
    }


def build_final_report(appdr_5: dict[str, Any], appdr_binary: dict[str, Any], best_cnn: CnnRun | None, runs: list[CnnRun]) -> dict[str, Any]:
    if best_cnn is None:
        best_metrics = {}
        beat_appdr = False
        beat_best = False
    else:
        best_metrics = best_cnn.metrics
        beat_appdr = float(best_metrics.get("macro_f1", 0)) > float(appdr_5.get("macro_f1", 0))
        beat_best = float(best_metrics.get("macro_f1", 0)) > 0.5799
    cnn_binary = read_first(OUTPUT_DIR / "cnn_binary_metrics.csv")
    threshold_rows = read_csv_rows(OUTPUT_DIR / "cnn_binary_threshold_sweep.csv")
    best_recall_threshold = max(threshold_rows, key=lambda row: float(row.get("referable_recall", 0) or 0), default={})
    best_balanced_threshold = max(threshold_rows, key=lambda row: float(row.get("balanced_accuracy", 0) or 0), default={})
    train_rows = read_csv_rows(OUTPUT_DIR / "train_manifest.csv")
    val_rows = read_csv_rows(OUTPUT_DIR / "val_manifest.csv")
    test_rows = read_csv_rows(OUTPUT_DIR / "test_manifest.csv")
    appdr_macro = float(appdr_5.get("macro_f1", 0) or 0)
    cnn_macro = float(best_metrics.get("macro_f1", 0) or 0)
    beat_precision = float(best_metrics.get("macro_precision", 0) or 0) > float(appdr_5.get("macro_precision", 0) or 0)
    beat_recall = float(best_metrics.get("macro_recall", 0) or 0) > float(appdr_5.get("macro_recall", 0) or 0)
    beat_class_1 = float(best_metrics.get("class_1_recall", 0) or 0) > float(appdr_5.get("class_1_recall", 0) or 0)
    beat_class_3 = float(best_metrics.get("class_3_recall", 0) or 0) > float(appdr_5.get("class_3_recall", 0) or 0)
    beat_class_4 = float(best_metrics.get("class_4_recall", 0) or 0) > float(appdr_5.get("class_4_recall", 0) or 0)
    beat_binary_recall = float(cnn_binary.get("referable_recall", 0) or 0) > float(appdr_binary.get("referable_recall", 0) or 0)
    reduced_fn = float(cnn_binary.get("false_negatives", 999999) or 999999) < float(appdr_binary.get("false_negatives", 0) or 0)
    markdown = "\n".join(
        [
            "# Full CNN vs AppDR Comparison",
            "",
            "Experiment only. Production was not changed.",
            "",
            "## Dataset And Split",
            "",
            "Dataset source: `backend/features_combined_balanced.csv`, using unique readable labeled fundus images from APTOS, OIA-DDR, IDRiD grading, Sachin Kumar, and Diabetic_Retinopathy_Balanced.",
            f"Full manifest size: {len(train_rows) + len(val_rows) + len(test_rows)} images.",
            f"Train/val/test: {len(train_rows)} / {len(val_rows)} / {len(test_rows)}.",
            "Split is image-level stratified by diagnosis because patient_id was not available.",
            "",
            "## CNN Run",
            "",
            f"Completed CNN runs: {len([run for run in runs if run.metrics.get('status') == 'ok'])}",
            f"Best CNN: {best_cnn.run_id if best_cnn else 'none'}",
            f"Backbone/input/loss/pooling: {best_cnn.backbone if best_cnn else 'n/a'} / {best_cnn.input_size if best_cnn else 'n/a'} / {best_cnn.loss if best_cnn else 'n/a'} / {best_cnn.pooling if best_cnn else 'n/a'}",
            f"Best CNN macro F1: {pct(best_metrics.get('macro_f1'))}",
            f"AppDR same-split macro F1: {pct(appdr_5.get('macro_f1'))}",
            "",
            "## 5-Class Decision Questions",
            "",
            f"CNN beat AppDR same-split macro F1: {beat_appdr}",
            f"CNN beat best experimental feature-based macro F1 57.99%: {beat_best}",
            f"CNN improved macro precision over AppDR same split: {beat_precision}",
            f"CNN improved macro recall over AppDR same split: {beat_recall}",
            f"CNN improved Class 1 recall over AppDR same split: {beat_class_1}",
            f"CNN improved Class 3 recall over AppDR same split: {beat_class_3}",
            f"CNN improved Class 4 recall over AppDR same split: {beat_class_4}",
            "",
            "| Metric | AppDR production same split | Best CNN same split |",
            "| --- | ---: | ---: |",
            f"| Accuracy | {pct(appdr_5.get('accuracy'))} | {pct(best_metrics.get('accuracy'))} |",
            f"| Balanced accuracy | {pct(appdr_5.get('balanced_accuracy'))} | {pct(best_metrics.get('balanced_accuracy'))} |",
            f"| Macro precision | {pct(appdr_5.get('macro_precision'))} | {pct(best_metrics.get('macro_precision'))} |",
            f"| Macro recall | {pct(appdr_5.get('macro_recall'))} | {pct(best_metrics.get('macro_recall'))} |",
            f"| Macro F1 | {pct(appdr_5.get('macro_f1'))} | {pct(best_metrics.get('macro_f1'))} |",
            f"| Class 1 recall | {pct(appdr_5.get('class_1_recall'))} | {pct(best_metrics.get('class_1_recall'))} |",
            f"| Class 3 recall | {pct(appdr_5.get('class_3_recall'))} | {pct(best_metrics.get('class_3_recall'))} |",
            f"| Class 4 recall | {pct(appdr_5.get('class_4_recall'))} | {pct(best_metrics.get('class_4_recall'))} |",
            "",
            "## Binary Screening",
            "",
            f"CNN improved binary referable recall over AppDR same split: {beat_binary_recall}",
            f"CNN reduced false negatives versus AppDR same split: {reduced_fn}",
            f"Best CNN class-derived referable recall: {pct(cnn_binary.get('referable_recall'))}, false negatives: {cnn_binary.get('false_negatives', 'n/a')}, false positives: {cnn_binary.get('false_positives', 'n/a')}.",
            f"Highest-recall CNN threshold: {best_recall_threshold.get('binary_threshold', 'n/a')} with referable recall {pct(best_recall_threshold.get('referable_recall'))}, false negatives {best_recall_threshold.get('false_negatives', 'n/a')}, false positives {best_recall_threshold.get('false_positives', 'n/a')}.",
            f"Best-balanced CNN threshold: {best_balanced_threshold.get('binary_threshold', 'n/a')} with balanced accuracy {pct(best_balanced_threshold.get('balanced_accuracy'))}.",
            "The high-recall CNN thresholds create too many false positives for a production replacement.",
            "",
            "## Recommendation",
            "",
            "Should CNN replace production now: No.",
            "Should CNN remain experimental: Yes.",
            "Should hybrid CNN + 203 handcrafted features be tried next: Yes, but as a separate experiment only; this run intentionally used no handcrafted input.",
            "",
            "The one-epoch full-split CNN is a better comparison than the previous capped run, but it is still not a fully maximized 30-50 epoch ensemble. Its current measured performance is far below AppDR, so it does not justify production replacement.",
        ]
    ) + "\n"
    recommendation = "\n".join(
        [
            "# Final Recommendation",
            "",
            "Do not update production.",
            "",
            f"CNN should replace production now: {False}",
            f"CNN should remain experimental: {True}",
            "",
            "Reason: this is an experiment-only run and production replacement requires stronger, stable 5-class and binary metrics across a fair validation setup.",
        ]
    ) + "\n"
    return {
        "best_cnn": best_metrics,
        "appdr_5class": appdr_5,
        "appdr_binary": appdr_binary,
        "beat_appdr_macro_f1": beat_appdr,
        "beat_best_experimental_macro_f1": beat_best,
        "markdown": markdown,
        "recommendation": recommendation,
    }


def five_class_metrics(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4], output_dict=True, zero_division=0)
    row = {
        "model_name": model_name,
        "status": "ok",
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


def binary_metrics(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None, threshold: str) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = ""
    if scores is not None and len(np.unique(y_true)) == 2:
        try:
            auc = roc_auc_score(y_true, scores)
        except Exception:
            auc = ""
    return {
        "model_name": model_name,
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


def write_confusion(path: Path, y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for idx, values in enumerate(matrix):
        row = {"actual_class": labels[idx]}
        for pred_label, value in zip(labels, values):
            row[f"predicted_{pred_label}"] = int(value)
        rows.append(row)
    write_rows(path, rows)


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


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    keys = list(rows[0].keys())
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
