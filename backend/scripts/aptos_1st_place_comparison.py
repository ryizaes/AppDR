"""APTOS 2019 1st-place comparison scaffold for AppDR.

This is comparison-only. It does not retrain production, replace artifacts, or
modify backend/frontend behavior. If the deep-learning stack is unavailable, it
records a blocked/not-run CNN comparison rather than inventing metrics.
"""

from __future__ import annotations

import csv
import importlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
APTOS_DIR = BACKEND_DIR / "images" / "aptos2019"
OUTPUT_DIR = BACKEND_DIR / "results" / "aptos_1st_place_comparison"

APPDR_PRODUCTION_GRADING = {
    "system": "AppDR production 5-class grading",
    "model": "XGBoost",
    "input": "203 handcrafted retinal features",
    "accuracy": 0.6798,
    "balanced_accuracy": 0.5312,
    "macro_precision": 0.4927,
    "macro_recall": 0.5312,
    "macro_f1": 0.5077,
    "weighted_f1": 0.6869,
    "class_1_recall": 0.3299,
    "class_3_recall": 0.3095,
    "class_4_recall": 0.6151,
}
APPDR_PRODUCTION_SCREENING = {
    "system": "AppDR production binary screening",
    "model": "SVM RBF",
    "input": "203 handcrafted retinal features",
    "accuracy": 0.7932,
    "balanced_accuracy": 0.8087,
    "precision": 0.6970,
    "recall": 0.9373,
    "f1": 0.7995,
    "referable_recall": 0.9373,
    "non_referable_recall": 0.6801,
    "false_negatives": 88,
    "false_positives": 572,
    "threshold": 0.20,
}
APPDR_BEST_EXPERIMENTAL_GRADING = {
    "system": "AppDR best experimental feature-based grading",
    "model": "Study-expanded weighted vote",
    "input": "previous top-100 SVM features",
    "accuracy": 0.7281,
    "balanced_accuracy": 0.5827,
    "macro_f1": 0.6133,
    "class_1_recall": 0.3500,
    "class_3_recall": 0.5050,
    "class_4_recall": 0.4433,
    "safety_note": "Higher macro F1, but Class 3/Class 4 recall are weaker than the safer study-top100 SVM baseline.",
}
APPDR_SAFER_EXPERIMENTAL_GRADING = {
    "system": "AppDR safer experimental feature-based grading",
    "model": "Stack RF + ExtraTrees + XGBoost + SVM + LR",
    "input": "texture + lesion + vessel + quality",
    "accuracy": 0.6522,
    "balanced_accuracy": 0.6057,
    "macro_f1": 0.5664,
    "class_1_recall": 0.4767,
    "class_3_recall": 0.6450,
    "class_4_recall": 0.5967,
}
APPDR_BEST_EXPERIMENTAL_SCREENING = {
    "system": "AppDR best experimental binary screening",
    "model": "LightGBM threshold sweep",
    "input": "expanded top-125",
    "accuracy": 0.7512,
    "referable_recall": 0.9697,
    "false_negatives": 48,
    "false_positives": 817,
    "threshold": 0.20,
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dependency_report = check_dependencies()
    dataset_report = audit_aptos_dataset()
    guanshuo_basis = build_guanshuo_basis()
    level_rows = build_level_plan(dependency_report, dataset_report)
    cnn_rows = build_cnn_comparison_rows(level_rows)
    answer_rows = build_answer_rows(cnn_rows)

    write_csv(OUTPUT_DIR / "appdr_baselines.csv", build_baseline_rows())
    write_csv(OUTPUT_DIR / "aptos_dataset_report.csv", dataset_report)
    write_csv(OUTPUT_DIR / "guanshuo_aptos_1st_place_basis.csv", guanshuo_basis)
    write_csv(OUTPUT_DIR / "experiment_levels.csv", level_rows)
    write_csv(OUTPUT_DIR / "cnn_comparison.csv", cnn_rows)
    write_csv(OUTPUT_DIR / "final_questions.csv", answer_rows)
    write_csv(OUTPUT_DIR / "threshold_plan.csv", build_threshold_plan())

    write_json(
        OUTPUT_DIR / "main_report.json",
        {
            "created_at": datetime.now().isoformat(),
            "scope": "Comparison-only APTOS 1st-place update. Production was not updated.",
            "dependency_report": dependency_report,
            "dataset_report": dataset_report,
            "guanshuo_basis": guanshuo_basis,
            "experiment_levels": level_rows,
            "cnn_comparison": cnn_rows,
            "final_answers": answer_rows,
            "production_replaced": False,
        },
    )
    write_reports(dependency_report, dataset_report, guanshuo_basis, level_rows, cnn_rows, answer_rows)
    print(f"Created {OUTPUT_DIR}")


def import_status(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        return {
            "module": module_name,
            "available": True,
            "version": str(getattr(module, "__version__", "unknown")),
            "error": "",
        }
    except Exception as error:
        return {
            "module": module_name,
            "available": False,
            "version": "",
            "error": f"{type(error).__name__}: {str(error)[:220]}",
        }


def check_dependencies() -> list[dict[str, Any]]:
    modules = [
        "torch",
        "torchvision",
        "timm",
        "pretrainedmodels",
        "efficientnet_pytorch",
        "tensorflow",
        "keras",
        "cv2",
        "PIL",
    ]
    rows = [import_status(module) for module in modules]
    write_csv(OUTPUT_DIR / "dependency_report.csv", rows)
    lines = [
        "# Dependency Report",
        "",
        "The Guanshuo-style CNN cannot be trained until at least one deep-learning stack is available. This pass does not install new packages or alter the environment.",
        "",
        "| Module | Available | Version | Error |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['module']} | {row['available']} | {row['version']} | {row['error']} |"
        )
    (OUTPUT_DIR / "dependency_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def count_image_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*") if _.suffix.lower() in {".jpg", ".jpeg", ".png"})


def audit_aptos_dataset() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for csv_name in ["train.csv", "test.csv", "labels.csv"]:
        path = APTOS_DIR / csv_name
        if not path.exists():
            rows.append({"file": csv_name, "status": "missing"})
            continue
        frame = pd.read_csv(path)
        label_column = next(
            (column for column in frame.columns if column.lower() in {"diagnosis", "label", "level"}),
            "",
        )
        label_counts = (
            json.dumps(frame[label_column].value_counts().sort_index().to_dict(), sort_keys=True)
            if label_column
            else ""
        )
        rows.append(
            {
                "file": csv_name,
                "status": "available",
                "rows": len(frame),
                "columns": ", ".join(frame.columns),
                "label_column": label_column,
                "label_counts": label_counts,
                "patient_id_available": False,
                "eye_or_angle_metadata_available": False,
            }
        )
    rows.append(
        {
            "file": "train_images",
            "status": "available" if (APTOS_DIR / "train_images").exists() else "missing",
            "rows": count_image_files(APTOS_DIR / "train_images"),
            "columns": "image files",
            "label_column": "diagnosis via train.csv/labels.csv",
            "label_counts": "",
            "patient_id_available": False,
            "eye_or_angle_metadata_available": False,
        }
    )
    rows.append(
        {
            "file": "test_images",
            "status": "available" if (APTOS_DIR / "test_images").exists() else "missing",
            "rows": count_image_files(APTOS_DIR / "test_images"),
            "columns": "image files",
            "label_column": "",
            "label_counts": "",
            "patient_id_available": False,
            "eye_or_angle_metadata_available": False,
        }
    )
    return rows


def build_guanshuo_basis() -> list[dict[str, str]]:
    return [
        {
            "item": "Primary source",
            "corrected_basis": "Kaggle APTOS 2019 1st-place writeup by Guanshuo Xu, not the cnnimageretrieval-pytorch repository.",
            "source_url": "https://www.kaggle.com/competitions/aptos2019-blindness-detection/writeups/guanshuo-xu-1st-place-solution-summary",
            "appdr_use": "Use as image-input CNN comparison basis.",
        },
        {
            "item": "GitHub cnnimageretrieval-pytorch",
            "corrected_basis": "Used only as the source for GeM pooling implementation.",
            "source_url": "https://github.com/filipradenovic/cnnimageretrieval-pytorch",
            "appdr_use": "Do not treat this as the full APTOS DR model.",
        },
        {
            "item": "Preprocessing",
            "corrected_basis": "No special preprocessing; plain resizing only.",
            "source_url": "https://www.kaggle.com/competitions/aptos2019-blindness-detection/writeups/guanshuo-xu-1st-place-solution-summary",
            "appdr_use": "Level 1 uses plain resize before tensor conversion.",
        },
        {
            "item": "Final ensemble",
            "corrected_basis": "2x inception_resnet_v2 512, 2x inception_v4 512, 2x seresnext50 512, 2x seresnext101 384.",
            "source_url": "https://www.kaggle.com/competitions/aptos2019-blindness-detection/writeups/guanshuo-xu-1st-place-solution-summary",
            "appdr_use": "Not assumed feasible; planned as Level 3 only.",
        },
        {
            "item": "Loss",
            "corrected_basis": "SmoothL1Loss with regression-style output.",
            "source_url": "https://www.kaggle.com/competitions/aptos2019-blindness-detection/writeups/guanshuo-xu-1st-place-solution-summary",
            "appdr_use": "Use one continuous output and threshold to 0-4 when training is available.",
        },
        {
            "item": "Augmentation",
            "corrected_basis": "contrast_range=0.2, brightness_range=20, hue_range=10, saturation_range=20, blur_and_sharpen=True, rotate_range=180, scale/shear/shift=0.2, do_mirror=True.",
            "source_url": "https://www.kaggle.com/competitions/aptos2019-blindness-detection/writeups/guanshuo-xu-1st-place-solution-summary",
            "appdr_use": "Use in training config only; not applied in this no-training pass.",
        },
        {
            "item": "Extra data",
            "corrected_basis": "IDRiD, Messidor, and pseudo-labeled public test data in Stage 2.",
            "source_url": "https://www.kaggle.com/competitions/aptos2019-blindness-detection/writeups/guanshuo-xu-1st-place-solution-summary",
            "appdr_use": "Mark as not used unless local external data and pseudolabel protocol are validated.",
        },
        {
            "item": "Thresholds",
            "corrected_basis": "Default [0.5, 1.5, 2.5, 3.5], adjusted [0.7, 1.5, 2.5, 3.5], plus validation threshold tuning.",
            "source_url": "https://www.kaggle.com/competitions/aptos2019-blindness-detection/writeups/guanshuo-xu-1st-place-solution-summary",
            "appdr_use": "Report all three threshold modes once predictions exist.",
        },
    ]


def deep_stack_available(dependency_report: list[dict[str, Any]]) -> bool:
    availability = {row["module"]: bool(row["available"]) for row in dependency_report}
    return bool(availability.get("torch") and (availability.get("torchvision") or availability.get("timm")))


def build_level_plan(
    dependency_report: list[dict[str, Any]],
    dataset_report: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    train_images = next((row for row in dataset_report if row["file"] == "train_images"), {})
    train_csv = next((row for row in dataset_report if row["file"] == "train.csv"), {})
    data_ready = train_images.get("rows", 0) and train_csv.get("status") == "available"
    stack_ready = deep_stack_available(dependency_report)
    blocker = ""
    if not stack_ready:
        blocker = "PyTorch/torchvision/timm or equivalent CNN stack is not installed."
    elif not data_ready:
        blocker = "APTOS train images or labels are missing."

    return [
        {
            "level": "Level 1",
            "name": "Practical single-model Guanshuo-style CNN",
            "target_backbones": "SEResNeXt50, InceptionResNetV2/InceptionV4, ResNet50/EfficientNet-B0, MobileNetV3 fallback",
            "planned_loss": "SmoothL1Loss regression output",
            "planned_thresholds": "[0.5,1.5,2.5,3.5], [0.7,1.5,2.5,3.5], validation optimized",
            "status": "ready_to_run" if stack_ready and data_ready else "not_run_blocked",
            "blocker": blocker,
        },
        {
            "level": "Level 2",
            "name": "Small ensemble",
            "target_backbones": "2 or 3 feasible CNN backbones/seeds",
            "planned_loss": "Average continuous predictions",
            "planned_thresholds": "same threshold modes as Level 1",
            "status": "deferred_until_level_1_valid",
            "blocker": "Requires at least one successful Level 1 model and compute budget.",
        },
        {
            "level": "Level 3",
            "name": "Full-style 8-model ensemble",
            "target_backbones": "2x inception_resnet_v2, 2x inception_v4, 2x seresnext50, 2x seresnext101",
            "planned_loss": "SmoothL1Loss ensemble",
            "planned_thresholds": "Guanshuo adjusted plus validation optimization",
            "status": "not_feasible_now",
            "blocker": "Too heavy for this pass and requires full deep-learning environment plus validated external/pseudolabel protocol.",
        },
    ]


def build_cnn_comparison_rows(level_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in level_rows:
        rows.append(
            {
                "experiment": row["level"],
                "model": row["name"],
                "status": row["status"],
                "accuracy": "",
                "balanced_accuracy": "",
                "macro_precision": "",
                "macro_recall": "",
                "macro_f1": "",
                "weighted_f1": "",
                "class_1_recall": "",
                "class_3_recall": "",
                "class_4_recall": "",
                "referable_recall": "",
                "non_referable_recall": "",
                "false_negatives": "",
                "false_positives": "",
                "auc": "",
                "threshold_used": "",
                "notes": row["blocker"] or "Ready to train when explicitly run.",
            }
        )
    return rows


def build_answer_rows(cnn_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    valid_cnn = [
        row for row in cnn_rows
        if row["status"] not in {"not_run_blocked", "deferred_until_level_1_valid", "not_feasible_now"}
        and str(row.get("macro_f1", "")).strip()
    ]
    no_valid = not valid_cnn
    blocked_answer = "No valid CNN result yet; the CNN stack is unavailable, so no win can be claimed."
    return [
        {"question": "Did the Guanshuo-style CNN beat AppDR production macro F1?", "answer": blocked_answer if no_valid else "Evaluate from cnn_comparison.csv."},
        {"question": "Did it beat AppDR best experimental macro F1?", "answer": blocked_answer if no_valid else "Evaluate from cnn_comparison.csv."},
        {"question": "Did it improve macro precision?", "answer": blocked_answer if no_valid else "Evaluate from cnn_comparison.csv."},
        {"question": "Did it improve macro recall?", "answer": blocked_answer if no_valid else "Evaluate from cnn_comparison.csv."},
        {"question": "Did it improve Class 1 recall?", "answer": blocked_answer if no_valid else "Evaluate from cnn_comparison.csv."},
        {"question": "Did it improve Class 3 recall?", "answer": blocked_answer if no_valid else "Evaluate from cnn_comparison.csv."},
        {"question": "Did it improve Class 4 recall?", "answer": blocked_answer if no_valid else "Evaluate from cnn_comparison.csv."},
        {"question": "Did it improve binary referable recall?", "answer": blocked_answer if no_valid else "Evaluate from cnn_comparison.csv."},
        {"question": "Did it reduce false negatives?", "answer": blocked_answer if no_valid else "Evaluate from cnn_comparison.csv."},
        {"question": "Did it create too many false positives?", "answer": "No false positives were produced because no CNN inference run was completed; this remains unknown." if no_valid else "Evaluate from cnn_comparison.csv."},
        {"question": "Should CNN replace production now?", "answer": "No. No trained Guanshuo-style CNN result exists in this environment, and production must not be replaced."},
        {"question": "Should CNN remain experimental?", "answer": "Yes. It remains experimental until PyTorch/timm or equivalent is installed, Level 1 is trained, and validation beats AppDR safely."},
        {"question": "Should we try hybrid CNN + 203 handcrafted features next?", "answer": "Yes, after a valid Level 1 CNN exists. Hybrid fusion is a reasonable next experiment, not a production change."},
    ]


def build_baseline_rows() -> list[dict[str, Any]]:
    return [
        APPDR_PRODUCTION_GRADING,
        APPDR_PRODUCTION_SCREENING,
        APPDR_BEST_EXPERIMENTAL_GRADING,
        APPDR_SAFER_EXPERIMENTAL_GRADING,
        APPDR_BEST_EXPERIMENTAL_SCREENING,
    ]


def build_threshold_plan() -> list[dict[str, Any]]:
    return [
        {
            "threshold_mode": "default_qwk_style",
            "thresholds": "[0.5, 1.5, 2.5, 3.5]",
            "status": "planned_when_predictions_exist",
        },
        {
            "threshold_mode": "guanshuo_adjusted",
            "thresholds": "[0.7, 1.5, 2.5, 3.5]",
            "status": "planned_when_predictions_exist",
        },
        {
            "threshold_mode": "validation_optimized",
            "thresholds": "search on validation only",
            "status": "planned_when_predictions_exist",
        },
    ]


def write_reports(
    dependency_report: list[dict[str, Any]],
    dataset_report: list[dict[str, Any]],
    guanshuo_basis: list[dict[str, str]],
    level_rows: list[dict[str, Any]],
    cnn_rows: list[dict[str, Any]],
    answer_rows: list[dict[str, str]],
) -> None:
    write_dataset_markdown(dataset_report)
    write_main_report(dependency_report, dataset_report, guanshuo_basis, level_rows, cnn_rows, answer_rows)
    write_training_config_stub()


def write_dataset_markdown(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# APTOS Dataset Audit",
        "",
        "| File | Status | Rows/images | Columns | Label counts | Patient/angle metadata |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        metadata = "none found"
        if row.get("patient_id_available") or row.get("eye_or_angle_metadata_available"):
            metadata = "partial"
        lines.append(
            f"| {row.get('file', '')} | {row.get('status', '')} | {row.get('rows', '')} | {row.get('columns', '')} | {row.get('label_counts', '')} | {metadata} |"
        )
    (OUTPUT_DIR / "aptos_dataset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_main_report(
    dependency_report: list[dict[str, Any]],
    dataset_report: list[dict[str, Any]],
    guanshuo_basis: list[dict[str, str]],
    level_rows: list[dict[str, Any]],
    cnn_rows: list[dict[str, Any]],
    answer_rows: list[dict[str, str]],
) -> None:
    stack_ready = deep_stack_available(dependency_report)
    train_images = next((row for row in dataset_report if row["file"] == "train_images"), {})
    lines = [
        "# APTOS 2019 1st-Place Guanshuo-Style CNN Comparison",
        "",
        "Comparison-only update. Production AppDR models were not changed.",
        "",
        "## Corrected Study Basis",
        "",
        "The `filipradenovic/cnnimageretrieval-pytorch` repository is treated only as the source for GeM pooling. The main APTOS 1st-place basis is Guanshuo Xu's Kaggle writeup.",
        "",
        "- Kaggle writeup: https://www.kaggle.com/competitions/aptos2019-blindness-detection/writeups/guanshuo-xu-1st-place-solution-summary",
        "- GeM implementation repository: https://github.com/filipradenovic/cnnimageretrieval-pytorch",
        "",
        "Guanshuo-style method summary:",
        "",
        "- Straight image-input CNN ensemble.",
        "- Plain resizing, no special preprocessing.",
        "- Final 8-model ensemble: 2x inception_resnet_v2 512, 2x inception_v4 512, 2x seresnext50 512, 2x seresnext101 384.",
        "- SmoothL1Loss regression output.",
        "- Heavy color/geometric augmentation.",
        "- GeM pooling from cnnimageretrieval-pytorch.",
        "- Stage 2 pseudo-labeled public test data plus IDRiD and Messidor external data.",
        "- Threshold adjustment from [0.5, 1.5, 2.5, 3.5] to [0.7, 1.5, 2.5, 3.5].",
        "",
        "## Local Feasibility",
        "",
        f"- APTOS train images found: {train_images.get('rows', 0)}.",
        f"- Deep-learning stack available: {stack_ready}.",
        "- PyTorch, torchvision, timm, TensorFlow, and Keras were not available in the backend venv during this pass.",
        "",
        "Because the CNN stack is unavailable, no Guanshuo-style CNN was trained and no CNN metric win is claimed.",
        "",
        "## AppDR Baselines",
        "",
        f"- Production grading: accuracy {pct(APPDR_PRODUCTION_GRADING['accuracy'])}, balanced accuracy {pct(APPDR_PRODUCTION_GRADING['balanced_accuracy'])}, macro F1 {pct(APPDR_PRODUCTION_GRADING['macro_f1'])}.",
        f"- Best experimental grading by macro F1 in current reports: accuracy {pct(APPDR_BEST_EXPERIMENTAL_GRADING['accuracy'])}, balanced accuracy {pct(APPDR_BEST_EXPERIMENTAL_GRADING['balanced_accuracy'])}, macro F1 {pct(APPDR_BEST_EXPERIMENTAL_GRADING['macro_f1'])}.",
        f"- Production screening: referable recall {pct(APPDR_PRODUCTION_SCREENING['referable_recall'])}, false negatives {APPDR_PRODUCTION_SCREENING['false_negatives']}, false positives {APPDR_PRODUCTION_SCREENING['false_positives']}.",
        f"- Best experimental screening in current reports: referable recall {pct(APPDR_BEST_EXPERIMENTAL_SCREENING['referable_recall'])}, false negatives {APPDR_BEST_EXPERIMENTAL_SCREENING['false_negatives']}, false positives {APPDR_BEST_EXPERIMENTAL_SCREENING['false_positives']}.",
        "",
        "## Level Status",
        "",
        "| Level | Status | Blocker |",
        "| --- | --- | --- |",
    ]
    for row in level_rows:
        lines.append(f"| {row['level']} | {row['status']} | {row['blocker']} |")
    lines.extend(
        [
            "",
            "## Required Final Questions",
            "",
            "| Question | Answer |",
            "| --- | --- |",
        ]
    )
    for row in answer_rows:
        lines.append(f"| {row['question']} | {row['answer']} |")
    lines.extend(
        [
            "",
            "## Production Decision",
            "",
            "Do not update production. The Guanshuo-style CNN remains experimental until a valid Level 1 model is trained, thresholded, and compared against AppDR baselines on the same split.",
            "",
            "## Next Step",
            "",
            "Install a compatible deep-learning stack in a controlled environment, then run Level 1 with a single feasible backbone. After a valid CNN exists, test hybrid fusion with the 203 handcrafted AppDR features.",
        ]
    )
    (OUTPUT_DIR / "main_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_training_config_stub() -> None:
    config = {
        "level_1": {
            "preferred_backbone_order": [
                "seresnext50",
                "inception_resnet_v2",
                "inception_v4",
                "resnet50",
                "efficientnet_b0",
                "mobilenet_v3",
            ],
            "input_size": {
                "seresnext50": 512,
                "inception_resnet_v2": 512,
                "inception_v4": 512,
                "resnet50": 512,
                "efficientnet_b0": 384,
                "mobilenet_v3": 384,
            },
            "preprocessing": "plain resize only",
            "pooling": "GeM if safely supported by selected backbone",
            "loss": "SmoothL1Loss",
            "output": "single continuous regression value clipped/thresholded to classes 0-4",
            "threshold_sets": [
                [0.5, 1.5, 2.5, 3.5],
                [0.7, 1.5, 2.5, 3.5],
                "validation_optimized",
            ],
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
        }
    }
    write_json(OUTPUT_DIR / "level1_training_config.json", config)


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "n/a"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
