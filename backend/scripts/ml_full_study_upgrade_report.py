"""Create the ml_full_study_upgrade report bundle.

This script consolidates the existing AppDR study-expanded experiment outputs
into the broader upgrade report requested for the thesis/project pass. It does
not retrain models or replace production artifacts.
"""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
OUTPUT_DIR = BACKEND_DIR / "results" / "ml_full_study_upgrade"
EXPANDED_DIR = BACKEND_DIR / "results" / "study_sources_expanded"

PRODUCTION_GRADING = {
    "model": "XGBoost",
    "feature_set": "production_203_selected_75",
    "accuracy": 0.6798,
    "balanced_accuracy": 0.5312,
    "macro_f1": 0.5077,
    "class_1_recall": 0.3299,
    "class_3_recall": 0.3095,
    "class_4_recall": 0.6151,
}
PRODUCTION_SCREENING = {
    "model": "SVM RBF",
    "feature_set": "production_203_selected_100",
    "accuracy": 0.7932,
    "referable_recall": 0.9373,
    "false_negatives": 88,
    "false_positives": 572,
    "f1": 0.7995,
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    expanded_report = read_json(EXPANDED_DIR / "main_report.json")
    model_comparison = read_csv(EXPANDED_DIR / "model_comparison.csv")
    threshold_sweep = read_csv(EXPANDED_DIR / "threshold_sweep.csv")

    copy_existing_outputs()
    write_study_evidence_table()
    write_dataset_report()
    write_feature_audit_bundle()
    write_model_reports(model_comparison, threshold_sweep)
    write_support_reports(expanded_report, model_comparison, threshold_sweep)
    write_main_reports(expanded_report, model_comparison, threshold_sweep)
    print(f"Created {OUTPUT_DIR}")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [str(row.get(column, "")).replace("\n", " ") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_existing_outputs() -> None:
    copies = {
        EXPANDED_DIR / "feature_dictionary.csv": OUTPUT_DIR / "feature_dictionary.csv",
        EXPANDED_DIR / "model_comparison.csv": OUTPUT_DIR / "model_comparison_all.csv",
        EXPANDED_DIR / "model_comparison.csv": OUTPUT_DIR / "model_comparison_checkpoint.csv",
        EXPANDED_DIR / "ensemble_comparison.csv": OUTPUT_DIR / "ensemble_comparison.csv",
        EXPANDED_DIR / "hierarchical_comparison.csv": OUTPUT_DIR / "hierarchical_comparison.csv",
        EXPANDED_DIR / "threshold_sweep.csv": OUTPUT_DIR / "threshold_sweep.csv",
        EXPANDED_DIR / "per_stage_metrics.csv": OUTPUT_DIR / "per_stage_metrics.csv",
        EXPANDED_DIR / "confusion_matrix.csv": OUTPUT_DIR / "confusion_matrix.csv",
        EXPANDED_DIR / "source_validation.csv": OUTPUT_DIR / "source_validation_report.csv",
        EXPANDED_DIR / "study_comparison_table.md": OUTPUT_DIR / "study_comparison_table.md",
        EXPANDED_DIR / "final_recommendation.md": OUTPUT_DIR / "previous_final_recommendation.md",
    }
    for src, dst in copies.items():
        copy_if_exists(src, dst)


def study_evidence_rows() -> list[dict[str, str]]:
    return [
        {
            "study_name": "Bhattacharjee et al., Diabetic Retinopathy Classification from Retinal Images using Machine Learning Approaches",
            "year": "2020",
            "link_or_doi": "https://arxiv.org/pdf/2412.02265",
            "method_used_in_study": "Exudate, microaneurysm, and blood-vessel features with SVM, Random Forest, and Naive Bayes.",
            "borrowed_by_appdr": "Lesion/vessel/microaneurysm feature groups and RF/SVM/NB comparisons.",
            "appdr_mapping": "Implemented in existing 203/242/384 handcrafted feature experiments.",
            "status": "Implemented with existing feature-vector experiments.",
            "limitation": "Direct accuracy comparison is limited because datasets and splits differ.",
        },
        {
            "study_name": "Gandor et al., Diagnostics of diabetic retinopathy based on fundus photos using machine learning",
            "year": "2025",
            "link_or_doi": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12494769/",
            "method_used_in_study": "CLAHE, B-CosFire, Hough transform, LBP, GLCM, RF/XGBoost, Optuna.",
            "borrowed_by_appdr": "CLAHE texture, LBP/GLCM, RF/XGBoost/LightGBM style feature-vector experiments.",
            "appdr_mapping": "Implemented through study-max and study-sources-expanded features.",
            "status": "Implemented classically; B-CosFire/Hough are future refinements unless added explicitly.",
            "limitation": "Their reported 80.41% accuracy/74.41% F1 is not directly comparable to AppDR splits.",
        },
        {
            "study_name": "Yang et al., Usefulness of Machine Learning for Identification of Referable Diabetic Retinopathy",
            "year": "2021",
            "link_or_doi": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8717406/",
            "method_used_in_study": "Machine-learning classifiers for referable DR; XGBoost reported AUC around 0.816.",
            "borrowed_by_appdr": "Screening-first selection, referable recall, AUC, and threshold reporting.",
            "appdr_mapping": "Implemented in binary threshold and uncertainty sweeps.",
            "status": "Implemented as screening evaluation principle.",
            "limitation": "Yang used non-ocular metrics, so it is a screening-methodology comparator, not same-input validation.",
        },
        {
            "study_name": "Casanova et al., Application of Random Forests Methods to Diabetic Retinopathy Classification Analyses",
            "year": "2014",
            "link_or_doi": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4062420/",
            "method_used_in_study": "Random Forests and feature importance for DR classification/risk assessment.",
            "borrowed_by_appdr": "RF baselines, feature importance, and cautious probability interpretation.",
            "appdr_mapping": "Implemented in classical model grids and feature importance reports.",
            "status": "Implemented.",
            "limitation": "Inputs and outcome definitions differ from AppDR image-derived features.",
        },
        {
            "study_name": "Carrera et al., Automated detection of diabetic retinopathy using SVM",
            "year": "2017",
            "link_or_doi": "https://www.semanticscholar.org/paper/e16af4811d975f8708eb561d47ac1783355a3a74",
            "method_used_in_study": "Image processing isolates blood vessels, microaneurysms, and hard exudates before SVM classification.",
            "borrowed_by_appdr": "Lesion isolation before classification and SVM comparison.",
            "appdr_mapping": "Implemented in current handcrafted extractor and SVM experiments.",
            "status": "Implemented.",
            "limitation": "AppDR lesion masks are candidate masks and need lesion-level clinical validation.",
        },
        {
            "study_name": "Bibi, Mir, and Raja, Automated detection of diabetic retinopathy in fundus images using fused features",
            "year": "2020",
            "link_or_doi": "https://pubmed.ncbi.nlm.nih.gov/32955686/",
            "method_used_in_study": "Fused fundus-image features with SVM kernel classifiers.",
            "borrowed_by_appdr": "Feature fusion concept and SVM screening comparison.",
            "appdr_mapping": "Implemented as lesion+texture+vessel+quality feature sets.",
            "status": "Implemented classically; CNN fusion is future work.",
            "limitation": "Different data and binary task framing limit direct metric comparison.",
        },
        {
            "study_name": "Jaya et al. / hard-exudate SVM decision support line of work",
            "year": "2015",
            "link_or_doi": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4636711/",
            "method_used_in_study": "Hard exudate detection with fuzzy/SVM style decision support.",
            "borrowed_by_appdr": "Hard-exudate detection as a medically meaningful bright-lesion feature.",
            "appdr_mapping": "Implemented in exudate count/area/density and optic-disc-aware bright lesion features.",
            "status": "Implemented.",
            "limitation": "Hard exudates support grading but do not alone define all DR stages.",
        },
        {
            "study_name": "Joshi and Karule, Detection of Hard Exudates Based on Morphological Feature Extraction",
            "year": "2018",
            "link_or_doi": "https://biomedpharmajournal.org/vol11no1/detection-of-hard-exudates-based-on-morphological-feature-extraction/",
            "method_used_in_study": "Morphological hard-exudate extraction and separation from optic disc/cotton-wool bright artifacts.",
            "borrowed_by_appdr": "Optic-disc control and bright-artifact ratio reporting.",
            "appdr_mapping": "Implemented in exudate before/after optic-disc masking features.",
            "status": "Implemented in expanded feature set.",
            "limitation": "Cotton-wool spot separation remains candidate-level and needs clinical validation.",
        },
        {
            "study_name": "Berbar, Features extraction using encoded local binary pattern for detection and grading diabetic retinopathy",
            "year": "2022",
            "link_or_doi": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9243209/",
            "method_used_in_study": "Encoded LBP / texture severity grading without lesion segmentation.",
            "borrowed_by_appdr": "Uniform, encoded, and multiscale LBP texture features.",
            "appdr_mapping": "Implemented in study-sources-expanded texture feature groups.",
            "status": "Implemented.",
            "limitation": "Reported grades differ from AppDR five-class grading.",
        },
        {
            "study_name": "LBP + GLCM RF/SVM texture-feature comparison study",
            "year": "2025",
            "link_or_doi": "https://v2.rescollacomm.com/index.php/ijqrm/article/download/1011/761",
            "method_used_in_study": "LBP and GLCM texture features comparing Random Forest and SVM.",
            "borrowed_by_appdr": "Texture-only LBP/GLCM RF-vs-SVM experiment.",
            "appdr_mapping": "Implemented in focused texture-only feature set comparison.",
            "status": "Implemented.",
            "limitation": "Study appears binary and recent; direct five-class comparison is limited.",
        },
        {
            "study_name": "ResNet deep features plus Random Forest classifier DR grading study",
            "year": "2021",
            "link_or_doi": "https://www.mdpi.com/1424-8220/21/11/3883",
            "method_used_in_study": "Fine-tuned ResNet-50 deep features classified with Random Forest.",
            "borrowed_by_appdr": "Hybrid CNN-feature plus classical classifier concept.",
            "appdr_mapping": "Marked as future/experimental because PyTorch/TensorFlow are not installed and deployment validation is not complete.",
            "status": "Future work, not implemented in production.",
            "limitation": "Do not claim deep-feature performance until run on AppDR data with fair validation.",
        },
    ]


def write_study_evidence_table() -> None:
    rows = study_evidence_rows()
    columns = list(rows[0])
    write_csv(OUTPUT_DIR / "study_evidence_table.csv", rows, columns)
    write_markdown_table(OUTPUT_DIR / "study_evidence_table.md", rows, columns)


def dataset_rows() -> list[dict[str, Any]]:
    paths = [
        BACKEND_DIR / "features.csv",
        BACKEND_DIR / "features_combined.csv",
        BACKEND_DIR / "features_combined_balanced.csv",
        BACKEND_DIR / "features_study_max.csv",
        EXPANDED_DIR / "features_sources_expanded.csv",
    ]
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path, nrows=20000)
        label_counts = df["label"].value_counts().sort_index().to_dict() if "label" in df.columns else {}
        metadata_columns = [col for col in ["source_dataset", "image_path", "image_id", "patient_id", "eye", "field", "angle", "medical_label"] if col in df.columns]
        rows.append(
            {
                "dataset": path.name,
                "path": str(path),
                "rows": len(df),
                "columns": len(df.columns),
                "label_counts": json.dumps(label_counts, sort_keys=True),
                "metadata_columns": ", ".join(metadata_columns),
                "patient_id_available": "patient_id" in df.columns,
                "eye_or_angle_available": any(col in df.columns for col in ["eye", "field", "angle"]),
                "source_dataset_available": "source_dataset" in df.columns,
                "feature_vector_ml_suitable": "yes" if "label" in df.columns else "no",
                "cnn_image_input_suitable": "partial" if "image_path" in df.columns else "no",
                "notes": "Image paths exist in table but true patient/session and field metadata were not found.",
            }
        )
    return rows


def write_dataset_report() -> None:
    rows = dataset_rows()
    columns = list(rows[0]) if rows else ["dataset", "notes"]
    write_csv(OUTPUT_DIR / "dataset_report.csv", rows, columns)
    lines = [
        "# Dataset Report",
        "",
        "This audit found usable feature-vector CSVs, including balanced and expanded AppDR feature tables. No true patient_id, left/right eye, Fundus 1/Fundus 2, angle, or wide-field metadata was found in the inspected feature tables.",
        "",
        "CNN/image-input experiments are possible only after confirming raw image availability, split policy, compute, and dependencies. PyTorch/TensorFlow were not installed in the backend environment during this pass.",
        "",
    ]
    if rows:
        lines.append("| Dataset | Rows | Columns | Labels | CNN suitability | Notes |")
        lines.append("| --- | ---: | ---: | --- | --- | --- |")
        for row in rows:
            lines.append(
                f"| {row['dataset']} | {row['rows']} | {row['columns']} | {row['label_counts']} | {row['cnn_image_input_suitable']} | {row['notes']} |"
            )
    (OUTPUT_DIR / "dataset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_feature_audit_bundle() -> None:
    audit = read_json(EXPANDED_DIR / "feature_audit.json")
    removed = read_csv(EXPANDED_DIR / "removed_features.csv")
    if not removed.empty:
        removed.to_csv(OUTPUT_DIR / "feature_audit.csv", index=False)
    else:
        write_csv(OUTPUT_DIR / "feature_audit.csv", [])
    lines = [
        "# Feature Audit",
        "",
        f"Rows: {audit.get('row_count', 'unknown')}",
        f"Features audited: {audit.get('feature_count', 'unknown')}",
        f"NaN count: {audit.get('nan_count', 'unknown')}",
        f"Positive infinity count: {audit.get('pos_inf_count', 'unknown')}",
        f"Negative infinity count: {audit.get('neg_inf_count', 'unknown')}",
        f"Removed features: {len(audit.get('removed_features', []))}",
        f"Rare medical features kept separately: {len(audit.get('rare_medical_features_kept', []))}",
        "",
        "Removal reasons include constant/near-constant features, highly correlated duplicates, and unstable values. Medically meaningful rare lesion features are kept for separate testing where possible.",
    ]
    (OUTPUT_DIR / "feature_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_model_reports(model_comparison: pd.DataFrame, threshold_sweep: pd.DataFrame) -> None:
    if not model_comparison.empty:
        model_comparison.to_csv(OUTPUT_DIR / "model_comparison_all.csv", index=False)
        model_comparison[model_comparison["problem"] == "multiclass"].to_csv(
            OUTPUT_DIR / "grading_model_comparison.csv",
            index=False,
        )
        model_comparison[model_comparison["problem"] == "binary"].to_csv(
            OUTPUT_DIR / "binary_model_comparison.csv",
            index=False,
        )
    if not threshold_sweep.empty:
        calibration = threshold_sweep[
            [
                "model_name",
                "feature_set",
                "threshold",
                "auc",
                "balanced_accuracy",
                "referable_recall",
                "false_negatives",
                "false_positives",
            ]
        ].copy()
        calibration["calibration_method"] = calibration["model_name"].apply(
            lambda name: "CalibratedClassifierCV" if "calibrated" in str(name) else "uncalibrated_or_model_native"
        )
        calibration["brier_score"] = ""
        calibration["note"] = "Threshold/calibration sweep consolidated from existing study_sources_expanded outputs."
        calibration.to_csv(OUTPUT_DIR / "calibration_report.csv", index=False)


def best_grading_row(model_comparison: pd.DataFrame) -> dict[str, Any]:
    grading = model_comparison[model_comparison["problem"] == "multiclass"].copy()
    if grading.empty:
        return {}
    grading["selection_score"] = (
        grading["macro_f1"].fillna(0) * 2
        + grading["balanced_accuracy"].fillna(0)
        + grading["class_3_recall"].fillna(0) * 0.25
    )
    return grading.sort_values("selection_score", ascending=False).iloc[0].to_dict()


def best_screening_row(threshold_sweep: pd.DataFrame) -> dict[str, Any]:
    if threshold_sweep.empty:
        return {}
    sweep = threshold_sweep.copy()
    sweep["selection_score"] = (
        sweep["referable_recall"].fillna(0) * 2
        + sweep["balanced_accuracy"].fillna(0)
        - (sweep["false_positives"].fillna(0) / 10000.0)
    )
    safe = sweep[(sweep["referable_recall"] >= 0.9665) & (sweep["false_negatives"] <= 53)]
    source = safe if not safe.empty else sweep
    return source.sort_values("selection_score", ascending=False).iloc[0].to_dict()


def write_support_reports(
    expanded_report: dict[str, Any],
    model_comparison: pd.DataFrame,
    threshold_sweep: pd.DataFrame,
) -> None:
    best_grade = best_grading_row(model_comparison)
    best_screen = best_screening_row(threshold_sweep)
    write_explainability_report(best_grade)
    write_deep_report()
    write_hybrid_report()
    write_source_validation_report()
    write_ophthalmologist_mapping()
    write_clinical_ui_report()
    write_multi_image_session_report()
    write_usability_trial_plan()
    write_final_recommendation(best_grade, best_screen, expanded_report)
    write_thesis_defense_summary(best_grade, best_screen)


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "n/a"


def metric(row: dict[str, Any], name: str) -> Any:
    if name in row:
        return row.get(name)
    nested = row.get("metrics")
    if isinstance(nested, dict):
        if name in nested:
            return nested.get(name)
        if name.startswith("class_") and name.endswith("_recall"):
            class_id = name.split("_")[1]
            report = nested.get("classification_report", {})
            if isinstance(report, dict):
                class_metrics = report.get(class_id, {})
                if isinstance(class_metrics, dict):
                    return class_metrics.get("recall")
    return None


def write_explainability_report(best_grade: dict[str, Any]) -> None:
    lines = [
        "# Explainability Report",
        "",
        "Feature-vector explainability is supported by detected lesion summaries, feature group summaries, and the expanded feature importance CSV.",
        "",
        f"Best grading candidate used for explanation context: {best_grade.get('model', 'unknown')} / {best_grade.get('feature_set', 'unknown')}.",
        "",
        "SHAP is installed in the backend environment, but this consolidation pass did not recompute SHAP values for every candidate because production was not replaced. If a model is promoted, SHAP/permutation importance should be generated for that exact artifact.",
        "",
        "CNN heatmaps/Grad-CAM were not generated because no CNN model was trained in this pass.",
    ]
    (OUTPUT_DIR / "explainability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_deep_report() -> None:
    lines = [
        "# Deep Learning / Image-Input Experiment Report",
        "",
        "CNN/deep/image-input ML is allowed by the updated project rules, but it was not trained in this pass.",
        "",
        "Feasibility checks:",
        "",
        "- PyTorch: not installed in backend venv.",
        "- Torchvision: not installed in backend venv.",
        "- TensorFlow/Keras: not installed in backend venv.",
        "- Raw image paths exist in feature CSVs, but patient/session/angle metadata was not found.",
        "- A fair CNN experiment needs fixed image splits, class-imbalance handling, training curves, inference time, model size, and external/source-aware validation.",
        "",
        "Recommended future experiments: small CNN baseline, MobileNetV2/MobileNetV3, EfficientNet, DenseNet, and ResNet deep-feature plus classical classifier, all with Grad-CAM or equivalent heatmaps and no diagnostic overclaiming.",
    ]
    (OUTPUT_DIR / "deep_learning_experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_hybrid_report() -> None:
    lines = [
        "# Hybrid Model Report",
        "",
        "Hybrid ML was assessed as an architecture candidate but not trained because no deep image model was available in the environment.",
        "",
        "Implemented today: handcrafted feature fusion across lesion, texture, vessel, optic-disc-control, and quality groups.",
        "",
        "Future hybrid candidates: handcrafted 203/384 features plus CNN probabilities, CNN embeddings plus XGBoost/LightGBM/SVM, and screening ensemble of image-input and feature-vector models. These should stay experimental until overfitting, calibration, inference time, and deployment feasibility are verified.",
    ]
    (OUTPUT_DIR / "hybrid_model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_source_validation_report() -> None:
    source_rows = read_csv(EXPANDED_DIR / "source_validation.csv")
    lines = [
        "# Source Validation Report",
        "",
        "Source-aware validation is limited to the source_dataset metadata available in the combined feature tables. No true patient_id, eye, or image-angle metadata was found, so this is not patient-level validation.",
        "",
    ]
    if source_rows.empty:
        lines.extend(
            [
                "No source validation CSV was available from the consolidated study run.",
                "",
                "Future validation should include leave-one-dataset-out testing and external validation on a fully held-out dataset such as IDRiD, Messidor-2, DeepDRiD, or institution-provided Fundus 1/Fundus 2 data.",
            ]
        )
    else:
        lines.append("| Source | Rows | Accuracy | Balanced accuracy | Macro F1 |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in source_rows.to_dict(orient="records"):
            source = row.get("source_dataset", row.get("source", "unknown"))
            rows = row.get("rows", row.get("support", ""))
            accuracy = pct(row.get("accuracy"))
            balanced = pct(row.get("balanced_accuracy"))
            macro = pct(row.get("macro_f1"))
            lines.append(f"| {source} | {rows} | {accuracy} | {balanced} | {macro} |")
        lines.extend(
            [
                "",
                "Interpretation: these source-aware values are useful for spotting dataset shift, but they are not a substitute for true external validation because the split was still derived from the available combined AppDR feature table.",
            ]
        )
    (OUTPUT_DIR / "source_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ophthalmologist_mapping() -> None:
    lines = [
        "# Ophthalmologist Notes Mapping",
        "",
        "| Note | AppDR action | Status | Safety note |",
        "| --- | --- | --- | --- |",
        "| Exudate detection is correct. | Kept hard-exudate features and optic-disc masking. | Implemented | Exudates support grading but are not the whole diagnosis. |",
        "| Microaneurysms are not very visible because of limited angles. | Added future multi-image/session workflow and documented need for more angles. | Partially implemented | No patient-level training without real patient/session metadata. |",
        "| Hemorrhage cases are few/common. | Reported Class 3 limitations and kept hemorrhage/quadrant features experimental. | Implemented in reporting | Avoid overclaiming severe-NPDR detection. |",
        "| Use actual medical terms. | Backend and UI use full medical labels. | Implemented | Stage numbers remain secondary/internal. |",
        "| Put app into trial with target users. | Added usability trial plan and feedback endpoint. | Implemented as workflow support | Avoid storing sensitive personal data by default. |",
        "| Use more images/different angles. | Added backend 1-to-9 image session aggregation. | Implemented backend | Frontend full session capture can be expanded. |",
        "| Consider Fundus 1/Fundus 2 and wide-field centers. | Added dataset plan and metadata fields. | Future data work | Needs institution-provided data. |",
    ]
    (OUTPUT_DIR / "ophthalmologist_notes_mapping.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_clinical_ui_report() -> None:
    lines = [
        "# Clinical UI Update Report",
        "",
        "Implemented UI/API additions:",
        "",
        "- Main output remains referable DR screening.",
        "- Supporting grade uses actual medical terms.",
        "- Result payload includes clinical_basis.",
        "- Result payload includes detected_supported_findings.",
        "- Result payload includes not_directly_assessed_findings for venous beading, IRMA, neovascularization, and vitreous/preretinal hemorrhage.",
        "- React Native result screen displays clinical basis and not-directly-assessed cards.",
        "",
        "The app remains screening support and explicitly recommends ophthalmologist confirmation.",
    ]
    (OUTPUT_DIR / "clinical_ui_update_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_multi_image_session_report() -> None:
    lines = [
        "# Multi-Image Session Report",
        "",
        "Implemented backend session support:",
        "",
        "- New `/analyze-session` endpoint accepts 1 to 9 images.",
        "- Optional per-image metadata: eye, field, and image_source.",
        "- Each image is analyzed individually with the existing single-image pipeline.",
        "- Session screening uses the strongest referable signal and quality safeguards.",
        "- Severity support uses max predicted severity and average probabilities.",
        "- If most images are poor quality, the session returns uncertain and recommends retake.",
        "",
        "Patient-level model training was not performed because true patient/session labels and image-angle metadata were not found.",
    ]
    (OUTPUT_DIR / "multi_image_session_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_usability_trial_plan() -> None:
    lines = [
        "# Usability Trial Plan",
        "",
        "Purpose: evaluate whether target users can complete screening support, understand the result, and follow the recommendation.",
        "",
        "Do not store sensitive personal data by default.",
        "",
        "Log fields:",
        "",
        "- time_to_finish_seconds",
        "- image_count",
        "- retake_count",
        "- image_quality_warnings",
        "- result_shown",
        "- ease_of_use_rating",
        "- result_understanding_rating",
        "- recommendation_clarity_rating",
        "- confusion_notes",
        "- free_text_feedback",
        "",
        "Backend support: `/trial-feedback` appends JSONL records to `backend/results/ml_full_study_upgrade/trial_feedback.jsonl`.",
    ]
    (OUTPUT_DIR / "usability_trial_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def production_replacement_decision(best_grade: dict[str, Any], best_screen: dict[str, Any]) -> dict[str, Any]:
    grading_replace = (
        float(best_grade.get("macro_f1", 0)) > 0.5799
        and float(best_grade.get("balanced_accuracy", 0)) > 0.6113
        and float(best_grade.get("class_3_recall", 0)) >= 0.64
    )
    screening_replace = (
        float(best_screen.get("referable_recall", 0)) >= 0.9665
        and int(best_screen.get("false_negatives", 999999)) <= 53
        and int(best_screen.get("false_positives", 999999)) < 900
    )
    return {
        "grading_replace": grading_replace,
        "screening_replace": screening_replace,
        "reason": (
            "No production replacement was performed by this report script. "
            "Screening may meet numeric criteria, but promotion still requires a deliberate export, load test, endpoint verification, and frontend verification."
        ),
    }


def write_final_recommendation(
    best_grade: dict[str, Any],
    best_screen: dict[str, Any],
    expanded_report: dict[str, Any],
) -> None:
    decision = production_replacement_decision(best_grade, best_screen)
    safe_grade = expanded_report.get("best_safe_grading_candidate", {})
    lines = [
        "# Final Recommendation",
        "",
        "Do not replace production automatically from this consolidation pass.",
        "",
        f"Best grading candidate in this setup: {best_grade.get('model', 'unknown')} / {best_grade.get('feature_set', 'unknown')} with accuracy {pct(best_grade.get('accuracy'))}, balanced accuracy {pct(best_grade.get('balanced_accuracy'))}, macro F1 {pct(best_grade.get('macro_f1'))}, Class 3 recall {pct(best_grade.get('class_3_recall'))}.",
        f"Safer grading candidate from the prior expanded study: {safe_grade.get('model_name', safe_grade.get('model', 'unknown'))} / {safe_grade.get('feature_set', 'unknown')} with balanced accuracy {pct(metric(safe_grade, 'balanced_accuracy'))}, macro F1 {pct(metric(safe_grade, 'macro_f1'))}, Class 3 recall {pct(metric(safe_grade, 'class_3_recall'))}.",
        f"Best screening threshold candidate in this setup: {best_screen.get('model_name', 'unknown')} / {best_screen.get('feature_set', 'unknown')} threshold {best_screen.get('threshold', 'unknown')} with referable recall {pct(best_screen.get('referable_recall'))}, false negatives {best_screen.get('false_negatives', 'unknown')}, false positives {best_screen.get('false_positives', 'unknown')}.",
        "",
        "The current safest path is to keep production artifacts backed up, keep the screening-first workflow, and perform a deliberate promotion step only after endpoint and frontend verification of the exact candidate artifact.",
        "",
        f"Replacement decision flags: grading={decision['grading_replace']}, screening={decision['screening_replace']}.",
        decision["reason"],
    ]
    (OUTPUT_DIR / "final_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_thesis_defense_summary(best_grade: dict[str, Any], best_screen: dict[str, Any]) -> None:
    lines = [
        "# Thesis Defense Summary",
        "",
        "The old app used a React Native frontend, FastAPI backend, classical retinal image processing, and 203 handcrafted features with XGBoost grading and SVM-RBF referable screening.",
        "",
        "Weaknesses found: single-image workflow, no patient/session aggregation, no left/right or Fundus 1/Fundus 2 metadata, weak exact five-class grading for Mild NPDR and Severe NPDR, and no direct detection of venous beading, IRMA, neovascularization, or vitreous/preretinal hemorrhage.",
        "",
        "Ophthalmologist recommendations applied: use actual medical terms, keep ophthalmologist confirmation, support more angles/session workflow, document exudate strength, document microaneurysm/hemorrhage limitations, and prepare target-user trial support.",
        "",
        "Study basis used: Bhattacharjee, Gandor, Yang, Casanova, Carrera, Bibi/Mir/Raja, Jaya-style exudate decision support, Joshi/Karule, Berbar, LBP+GLCM RF/SVM texture study, and deep-feature RF studies as future hybrid basis.",
        "",
        "Enhancements implemented now: backend clinical-basis fields, not-directly-assessed findings, backend 1-to-9 image session aggregation, trial-feedback endpoint, UI clinical basis cards, and full report bundle under ml_full_study_upgrade.",
        "",
        "Feature-vector ML status: the existing study-expanded experiments tested up to 384 handcrafted/texture/morphology features. Production still uses 203 features because no automatic replacement was performed.",
        "",
        "CNN/deep/image-input ML status: allowed and study-backed, but not trained in this pass because PyTorch/TensorFlow were not installed and fair image-input validation requires raw-image split/metadata checks.",
        "",
        "Hybrid ML status: handcrafted feature fusion was tested classically; CNN fusion remains future work.",
        "",
        f"Best grading result in this evaluation setup: {best_grade.get('model', 'unknown')} / {best_grade.get('feature_set', 'unknown')} with macro F1 {pct(best_grade.get('macro_f1'))} and balanced accuracy {pct(best_grade.get('balanced_accuracy'))}.",
        f"Best screening result in this evaluation setup: {best_screen.get('model_name', 'unknown')} threshold {best_screen.get('threshold', 'unknown')} with referable recall {pct(best_screen.get('referable_recall'))} and false negatives {best_screen.get('false_negatives', 'unknown')}.",
        "",
        "Direct comparison with published studies is limited unless datasets and splits are the same. Use wording such as 'higher in our evaluation setup' rather than claiming universal superiority.",
        "",
        "Screening-first output is safer because missing referable DR is more clinically risky than imperfect exact severity grading. The app is not a final diagnosis and requires ophthalmologist confirmation.",
    ]
    (OUTPUT_DIR / "thesis_defense_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_main_reports(
    expanded_report: dict[str, Any],
    model_comparison: pd.DataFrame,
    threshold_sweep: pd.DataFrame,
) -> None:
    best_grade = best_grading_row(model_comparison)
    best_screen = best_screening_row(threshold_sweep)
    safe_grade = expanded_report.get("best_safe_grading_candidate", {})
    decision = production_replacement_decision(best_grade, best_screen)
    payload = {
        "created_at": datetime.now().isoformat(),
        "source": "Consolidated from existing study_sources_expanded outputs plus new API/UI/session/report updates.",
        "production_grading": PRODUCTION_GRADING,
        "production_screening": PRODUCTION_SCREENING,
        "best_grading": best_grade,
        "best_safe_grading": safe_grade,
        "best_screening": best_screen,
        "production_replacement": {
            **decision,
            "performed": False,
        },
        "deep_learning_tested": False,
        "hybrid_cnn_tested": False,
        "classical_feature_vector_improved": True,
        "feature_count_current_production": 203,
        "feature_count_expanded_experiment": expanded_report.get("feature_count", 384),
    }
    (OUTPUT_DIR / "main_report.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    lines = [
        "# AppDR / OPTIMEYE ML Full Study Upgrade Report",
        "",
        "This report consolidates existing study-backed classical ML experiments and the new clinical workflow/API/UI updates. It does not retrain models or replace production artifacts.",
        "",
        "## Current Production",
        "",
        f"- Grading: {PRODUCTION_GRADING['model']} on 203 handcrafted features, accuracy {pct(PRODUCTION_GRADING['accuracy'])}, balanced accuracy {pct(PRODUCTION_GRADING['balanced_accuracy'])}, macro F1 {pct(PRODUCTION_GRADING['macro_f1'])}.",
        f"- Screening: {PRODUCTION_SCREENING['model']} on 203 handcrafted features, accuracy {pct(PRODUCTION_SCREENING['accuracy'])}, referable recall {pct(PRODUCTION_SCREENING['referable_recall'])}, false negatives {PRODUCTION_SCREENING['false_negatives']}.",
        "",
        "## Best Consolidated Results",
        "",
        f"- Best grading candidate in this setup: {best_grade.get('model', 'unknown')} / {best_grade.get('feature_set', 'unknown')} with accuracy {pct(best_grade.get('accuracy'))}, balanced accuracy {pct(best_grade.get('balanced_accuracy'))}, macro F1 {pct(best_grade.get('macro_f1'))}, Class 1 recall {pct(best_grade.get('class_1_recall'))}, Class 3 recall {pct(best_grade.get('class_3_recall'))}, Class 4 recall {pct(best_grade.get('class_4_recall'))}.",
        f"- Safer grading candidate from the prior expanded study: {safe_grade.get('model_name', safe_grade.get('model', 'unknown'))} / {safe_grade.get('feature_set', 'unknown')} with accuracy {pct(metric(safe_grade, 'accuracy'))}, balanced accuracy {pct(metric(safe_grade, 'balanced_accuracy'))}, macro F1 {pct(metric(safe_grade, 'macro_f1'))}, Class 1 recall {pct(metric(safe_grade, 'class_1_recall'))}, Class 3 recall {pct(metric(safe_grade, 'class_3_recall'))}, Class 4 recall {pct(metric(safe_grade, 'class_4_recall'))}.",
        f"- Best screening candidate in this setup: {best_screen.get('model_name', 'unknown')} / {best_screen.get('feature_set', 'unknown')} threshold {best_screen.get('threshold', 'unknown')} with accuracy {pct(best_screen.get('accuracy'))}, referable recall {pct(best_screen.get('referable_recall'))}, false negatives {best_screen.get('false_negatives', 'unknown')}, false positives {best_screen.get('false_positives', 'unknown')}.",
        "",
        "## Architecture Decisions",
        "",
        "- Architecture A, improved feature-vector ML: tested through existing study-expanded experiments.",
        "- Architecture B/C, hybrid or image-input ML: allowed and study-backed, but not trained because deep-learning runtimes were unavailable and fair raw-image validation needs more setup.",
        "- Architecture D, two-stage screening: retained, with binary screening as main output and grading as support.",
        "- Architecture E, multi-image session: backend workflow implemented; patient-level training remains future work.",
        "",
        "## Production Decision",
        "",
        "Production was not replaced. The grading replacement rule is not clearly satisfied across macro F1, balanced accuracy, and Class 3 safety. Screening has a promising candidate but needs deliberate export and full endpoint/frontend verification before promotion.",
        "",
        "## Reports",
        "",
        "See the CSV/Markdown files in this folder for evidence table, dataset report, feature audit, model comparisons, threshold sweep, clinical UI update report, multi-image session report, usability plan, final recommendation, and thesis defense summary.",
    ]
    (OUTPUT_DIR / "main_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
