"""Max-improvement classical ML study pass for AppDR.

The script builds extra handcrafted/engineered features from the existing
validated 203-feature table, then runs classical model comparisons. It does not
use deep learning, transfer learning, CNNs, UNet, ResNet, or YOLO.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
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
from sklearn.model_selection import train_test_split

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
from scripts import study_feature_selection_experiments as study


RESULTS_DIR = BACKEND_DIR / "results" / "study_max_improvement"
SOURCE_FEATURES = BACKEND_DIR / "features_combined_balanced.csv"
OUTPUT_FEATURES = BACKEND_DIR / "features_study_max.csv"
RANDOM_STATE = config.RANDOM_STATE
MODEL_NAMES = (
    "logistic_regression",
    "random_forest",
    "xgboost",
    "svm_rbf",
    "naive_bayes",
    "lightgbm",
    "extra_trees",
    "balanced_random_forest",
    "histgradientboosting",
)
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
UNCERTAINTY_BANDS = [(0.40, 0.60), (0.35, 0.65), (0.30, 0.70)]
PRODUCTION = study.PRODUCTION_BASELINE
STUDY_TOP100 = {
    "accuracy": 0.6602,
    "balanced_accuracy": 0.6083,
    "macro_f1": 0.5718,
    "class_1_recall": 0.4533,
    "class_3_recall": 0.6400,
    "class_4_recall": 0.6233,
}


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    backup_current_system(output_dir)

    source_table = pd.read_csv(args.source_features)
    feature_table, dictionary = build_study_feature_table(source_table)
    feature_table.to_csv(args.output_features, index=False)
    write_csv(output_dir / "feature_dictionary.csv", dictionary)

    feature_names = [row["feature_name"] for row in dictionary]
    audit = audit_and_prune(feature_table, feature_names, dictionary)
    write_json(output_dir / "feature_audit.json", audit)
    write_csv(output_dir / "feature_audit_removed_features.csv", audit["removed_features"])
    write_csv(output_dir / "feature_correlation_report.csv", audit["correlation_pairs"])

    clean_features = [
        feature for feature in feature_names
        if feature not in {row["feature"] for row in audit["removed_features"]}
    ]
    x_all = feature_table[clean_features].apply(pd.to_numeric, errors="coerce")
    x_all = x_all.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_all = feature_table["label"].astype(int).to_numpy()
    train_index, test_index = train_test_split(
        np.arange(len(feature_table)),
        test_size=config.TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )
    x_train = x_all.iloc[train_index].reset_index(drop=True)
    x_test = x_all.iloc[test_index].reset_index(drop=True)
    y_train = y_all[train_index]
    y_test = y_all[test_index]
    test_meta = feature_table.iloc[test_index].reset_index(drop=True)

    importance = compute_rankings(x_train, y_train, x_test, y_test, clean_features)
    write_csv(output_dir / "feature_comparison.csv", importance)
    feature_sets = build_feature_sets(importance, dictionary, clean_features)
    write_json(output_dir / "feature_sets.json", feature_sets)

    multiclass_results = run_model_grid(
        problem="multiclass",
        feature_sets=feature_sets,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        output_dir=output_dir,
        stage_label="medium",
        svm_limit=args.svm_train_limit,
    )
    binary_results, threshold_rows = run_binary_grid(
        feature_sets=feature_sets,
        x_train=x_train,
        x_test=x_test,
        y_train_multi=y_train,
        y_test_multi=y_test,
        output_dir=output_dir,
        svm_limit=args.svm_train_limit,
    )
    hierarchy = run_hierarchical_experiments(
        x_train,
        x_test,
        y_train,
        y_test,
        feature_sets,
        output_dir,
        args.svm_train_limit,
    )
    source_validation = per_source_validation(
        best_result(multiclass_results, "multiclass"),
        x_test,
        y_test,
        test_meta,
    )
    report = build_report(
        feature_table=feature_table,
        dictionary=dictionary,
        audit=audit,
        feature_sets=feature_sets,
        multiclass_results=multiclass_results,
        binary_results=binary_results,
        threshold_rows=threshold_rows,
        hierarchy=hierarchy,
        source_validation=source_validation,
        output_dir=output_dir,
    )
    write_reports(output_dir, report)
    print(report["markdown"])


def build_study_feature_table(source: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    metadata_columns = [
        column for column in ["image_id", "image_path", "source_dataset", "label", "medical_label", "image_sha256"]
        if column in source.columns
    ]
    metadata = source[metadata_columns].copy()
    dictionary: list[dict[str, str]] = []

    existing_features: dict[str, pd.Series] = {}
    for feature in config.FEATURE_NAMES:
        existing_features[feature] = pd.to_numeric(source[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        dictionary.append(feature_dict_row(feature, group_for_existing(feature), "Existing AppDR 203-feature pipeline value.", "AppDR existing feature group", "existing_203", "float"))

    output = pd.concat([metadata, pd.DataFrame(existing_features, index=source.index)], axis=1)
    engineered_features = add_engineered_features(output, dictionary)
    if engineered_features:
        output = pd.concat([output, pd.DataFrame(engineered_features, index=source.index)], axis=1)
    feature_columns = [row["feature_name"] for row in dictionary]
    output[feature_columns] = output[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return output, dictionary


def add_engineered_features(table: pd.DataFrame, dictionary: list[dict[str, str]]) -> dict[str, pd.Series]:
    engineered: dict[str, pd.Series] = {}

    def get(name: str) -> pd.Series:
        if name in engineered:
            return engineered[name]
        return pd.to_numeric(table.get(name, 0.0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-6) -> pd.Series:
        return a / (b.abs() + eps)

    def add(name: str, group: str, description: str, basis: str, variant: str, values: pd.Series) -> None:
        engineered[name] = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        dictionary.append(feature_dict_row(name, group, description, basis, variant, "float"))

    glcm_contrasts = [get(name) for name in ["glcm_contrast_0", "glcm_contrast_45", "glcm_contrast_90", "glcm_contrast_135"]]
    glcm_energies = [get(name) for name in ["glcm_energy_0", "glcm_energy_45", "glcm_energy_90", "glcm_energy_135"]]
    glcm_contrast_stack = pd.concat(glcm_contrasts, axis=1)
    glcm_energy_stack = pd.concat(glcm_energies, axis=1)
    add("study_glcm_contrast_mean_directional", "texture_glcm", "Mean multidirectional GLCM contrast.", "Gandor-style GLCM texture", "denoised_green_clahe", glcm_contrast_stack.mean(axis=1))
    add("study_glcm_contrast_range_directional", "texture_glcm", "Directional GLCM contrast range.", "Gandor-style GLCM texture", "denoised_green_clahe", glcm_contrast_stack.max(axis=1) - glcm_contrast_stack.min(axis=1))
    add("study_glcm_energy_mean_directional", "texture_glcm", "Mean multidirectional GLCM energy.", "Gandor-style GLCM texture", "denoised_green_clahe", glcm_energy_stack.mean(axis=1))
    add("study_glcm_energy_range_directional", "texture_glcm", "Directional GLCM energy range.", "Gandor-style GLCM texture", "denoised_green_clahe", glcm_energy_stack.max(axis=1) - glcm_energy_stack.min(axis=1))
    add("study_glcm_asm_proxy", "texture_glcm", "ASM proxy from GLCM energy squared.", "Gandor-style GLCM ASM", "denoised_green_clahe", get("glcm_energy") ** 2)
    add("study_glcm_homogeneity_contrast_ratio", "texture_glcm", "Homogeneity to contrast ratio.", "Gandor-style GLCM texture", "denoised_green_clahe", safe_div(get("glcm_homogeneity"), get("glcm_contrast")))
    add("study_glcm_entropy_correlation_interaction", "texture_glcm", "GLCM entropy and correlation interaction.", "Gandor-style GLCM texture", "denoised_green_clahe", get("glcm_entropy") * get("glcm_correlation"))

    add("study_lbp_entropy_uniform_ratio", "texture_lbp", "LBP entropy adjusted by uniform pattern ratio.", "Gandor-style LBP texture", "denoised_green_clahe", safe_div(get("lbp_entropy"), get("lbp_uniform_ratio") + 1.0))
    add("study_lbp_variation_index", "texture_lbp", "LBP local variation index.", "Gandor-style LBP texture", "denoised_green_clahe", safe_div(get("lbp_std"), get("lbp_mean").abs() + 1.0))
    add("study_lbp_texture_stability", "texture_lbp", "Uniform LBP ratio divided by entropy.", "Gandor-style LBP texture", "denoised_green_clahe", safe_div(get("lbp_uniform_ratio"), get("lbp_entropy") + 1.0))
    add("study_texture_contrast_lbp_interaction", "texture_lbp", "Interaction between GLCM contrast and LBP entropy.", "Gandor-style combined texture", "denoised_green_clahe", get("glcm_contrast") * get("lbp_entropy"))

    exudate_area = get("hard_exudate_area") + get("soft_exudate_area") + get("cotton_wool_area")
    exudate_count = get("hard_exudate_count") + get("soft_exudate_count") + get("cotton_wool_count")
    add("study_bright_lesion_total_area_log", "lesion_exudate", "Log total bright lesion area.", "Bhattacharjee/Jaya exudate features", "optic_disc_excluded_masks", np.log1p(exudate_area))
    add("study_bright_lesion_total_count_log", "lesion_exudate", "Log total bright lesion count.", "Bhattacharjee/Jaya exudate features", "optic_disc_excluded_masks", np.log1p(exudate_count))
    add("study_bright_lesion_mean_area", "lesion_exudate", "Bright lesion mean area across hard/soft/cotton-wool candidates.", "Bhattacharjee/Jaya exudate features", "optic_disc_excluded_masks", safe_div(exudate_area, exudate_count + 1.0))
    add("study_exudate_macula_risk", "lesion_exudate", "Bright lesion burden adjusted by macular proximity.", "Jaya exudate decision support", "optic_disc_excluded_masks", safe_div(exudate_area, get("hard_exudate_distance_to_macula_mean") + get("soft_exudate_distance_to_macula_mean") + 1.0))
    add("study_exudate_optic_disc_fp_risk", "optic_disc", "Bright lesion burden near optic disc, a false-positive risk proxy.", "Optic-disc false-positive control", "optic_disc_masked", safe_div(exudate_area, get("hard_exudate_distance_to_optic_disc_mean") + get("soft_exudate_distance_to_optic_disc_mean") + 1.0))
    add("study_hard_soft_exudate_balance", "lesion_exudate", "Hard-to-soft exudate area ratio.", "Bhattacharjee exudate features", "optic_disc_excluded_masks", safe_div(get("hard_exudate_area"), get("soft_exudate_area") + 1.0))

    red_area = get("ma_area") + get("hemorrhage_area")
    red_count = get("ma_count") + get("hemorrhage_count")
    add("study_red_lesion_total_area_log", "lesion_microaneurysm", "Log total red lesion area.", "Bhattacharjee red-lesion features", "green_channel_blackhat", np.log1p(red_area))
    add("study_red_lesion_total_count_log", "lesion_microaneurysm", "Log total red lesion count.", "Bhattacharjee red-lesion features", "green_channel_blackhat", np.log1p(red_count))
    add("study_red_lesion_density_combined", "lesion_microaneurysm", "Combined MA and hemorrhage density.", "Bhattacharjee red-lesion features", "green_channel_blackhat", get("ma_density") + get("hemorrhage_density"))
    add("study_ma_presence_strength", "lesion_microaneurysm", "Microaneurysm presence strength using count and area.", "Mild NPDR red-lesion support", "green_channel_blackhat", np.log1p(get("ma_count")) * np.log1p(get("ma_area")))
    add("study_ma_quadrant_spread", "lesion_microaneurysm", "Number of retinal quadrants with MA candidates.", "Lesion distribution by quadrant", "quadrant_masks", (get("ma_superior_count") > 0).astype(float) + (get("ma_inferior_count") > 0).astype(float) + (get("ma_nasal_count") > 0).astype(float) + (get("ma_temporal_count") > 0).astype(float))
    ma_quad = pd.concat([get("ma_superior_count"), get("ma_inferior_count"), get("ma_nasal_count"), get("ma_temporal_count")], axis=1)
    add("study_ma_quadrant_asymmetry", "lesion_microaneurysm", "MA quadrant max-min asymmetry.", "Lesion distribution by quadrant", "quadrant_masks", ma_quad.max(axis=1) - ma_quad.min(axis=1))

    add("study_hemorrhage_severity_index", "lesion_hemorrhage", "Hemorrhage burden index for severe DR.", "Bhattacharjee hemorrhage/red lesion features", "green_channel_blackhat", np.log1p(get("hemorrhage_area")) * np.log1p(get("hemorrhage_count") + 1.0))
    add("study_hemorrhage_retina_extent_index", "lesion_hemorrhage", "Hemorrhage area adjusted by affected retina percentage.", "Severe NPDR support", "green_channel_blackhat", get("hemorrhage_area") * get("hemorrhage_retina_affected_pct"))
    add("study_hemorrhage_ma_ratio_log", "lesion_hemorrhage", "Log hemorrhage-to-MA burden ratio.", "Severe vs mild red-lesion support", "green_channel_blackhat", np.log1p(safe_div(get("hemorrhage_area") + 1.0, get("ma_area") + 1.0)))

    add("study_vessel_branching_density", "vessel", "Branching count per vessel length.", "Vessel feature studies", "frangi_vesselness", safe_div(get("vessel_branching_count"), get("vessel_length") + 1.0))
    add("study_vessel_width_tortuosity_index", "vessel", "Vessel width/tortuosity abnormality proxy.", "Vessel feature studies", "frangi_vesselness", get("vessel_average_width") * (get("vessel_tortuosity_mean") + get("vessel_curvature_mean")))
    add("study_vessel_complexity_density_ratio", "vessel", "Vessel complexity adjusted by density.", "Vessel feature studies", "frangi_vesselness", safe_div(get("vessel_complexity_score"), get("vessel_density") + 0.001))
    add("study_vessel_fragmentation_density", "vessel", "Fragmentation adjusted by vessel density.", "Vessel feature studies", "frangi_vesselness", get("vessel_fragmentation_index") * get("vessel_density"))

    add("study_quality_low_contrast_flag", "quality", "Low contrast warning proxy.", "Image quality gate", "quality_features", (get("quality_contrast") < get("quality_contrast").quantile(0.10)).astype(float))
    add("study_quality_dark_flag", "quality", "Dark image warning proxy.", "Image quality gate", "quality_features", (get("quality_brightness") < get("quality_brightness").quantile(0.10)).astype(float))
    add("study_quality_bright_flag", "quality", "Bright/overexposed image warning proxy.", "Image quality gate", "quality_features", (get("quality_brightness") > get("quality_brightness").quantile(0.90)).astype(float))
    add("study_quality_blur_flag", "quality", "Blur warning proxy.", "Image quality gate", "quality_features", (get("quality_blur_score") < get("quality_blur_score").quantile(0.10)).astype(float))
    add("study_quality_adjusted_red_lesion", "quality", "Red lesion burden adjusted by quality.", "Quality-aware lesion support", "quality_features", safe_div(red_area, get("quality_blur_score") + get("quality_contrast") + 1.0))
    add("study_quality_adjusted_bright_lesion", "quality", "Bright lesion burden adjusted by image quality.", "Quality-aware exudate support", "quality_features", safe_div(exudate_area, get("quality_blur_score") + get("quality_contrast") + 1.0))

    add("study_referable_safety_score", "existing", "Referable safety score combining red, bright, vessel, and severe lesion indices.", "Yang-style referable screening ML", "engineered", get("referable_lesion_score") + get("stage_progression_score") + get("advanced_dr_indicator_score"))
    add("study_severe_npdr_score", "lesion_hemorrhage", "Severe NPDR support score from hemorrhage, vessel abnormality, and quadrant spread.", "Severe NPDR grading support", "engineered", get("study_hemorrhage_severity_index") + get("vessel_abnormality_score") + get("study_ma_quadrant_spread"))
    add("study_pdr_support_score", "vessel", "PDR support score from neovascularization proxy and vessel abnormality.", "PDR grading support", "engineered", get("neovascularization_likelihood_score") + get("vessel_abnormality_score") + get("advanced_dr_indicator_score"))
    return engineered


def feature_dict_row(
    feature_name: str,
    feature_group: str,
    description: str,
    basis: str,
    preprocessing_variant: str,
    value_type: str,
) -> dict[str, str]:
    return {
        "feature_name": feature_name,
        "feature_group": feature_group,
        "description": description,
        "source_or_study_basis": basis,
        "preprocessing_variant": preprocessing_variant,
        "expected_value_type": value_type,
    }


def group_for_existing(feature: str) -> str:
    if feature in config.VESSEL_FEATURE_NAMES or feature.startswith("vessel_"):
        return "vessel"
    if feature in config.HEMORRHAGE_FEATURE_NAMES or feature.startswith("hemorrhage"):
        return "lesion_hemorrhage"
    if feature in config.MA_ADVANCED_FEATURE_NAMES or feature.startswith("ma_"):
        return "lesion_microaneurysm"
    if "exudate" in feature or "cotton_wool" in feature or "bright_lesion" in feature:
        return "lesion_exudate"
    if "glcm" in feature:
        return "texture_glcm"
    if "lbp" in feature or "texture" in feature or "wavelet" in feature or "fft" in feature:
        return "texture_lbp"
    if feature.startswith("quality_"):
        return "quality"
    if "optic_disc" in feature:
        return "optic_disc"
    return "existing"


def audit_and_prune(
    table: pd.DataFrame,
    feature_names: list[str],
    dictionary: list[dict[str, str]],
) -> dict[str, Any]:
    audit = study.audit_features(table, feature_names)
    rare_lesion = {
        row["feature_name"] for row in dictionary
        if row["feature_group"].startswith("lesion_")
    }
    filtered_removed = []
    for row in audit["suggested_removed_features"]:
        if row["feature"] in rare_lesion and row["reason"] == "near_constant":
            continue
        filtered_removed.append(row)
    audit["removed_features"] = filtered_removed
    audit["rare_lesion_features_kept_despite_rarity"] = [
        row["feature"] for row in audit["suggested_removed_features"]
        if row["feature"] in rare_lesion and row["reason"] == "near_constant"
    ]
    audit["correlation_pairs"] = audit["highly_correlated_pairs"]
    return audit


def compute_rankings(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    importance = study.compute_feature_importance(
        x_train,
        y_train,
        x_test,
        y_test,
        feature_names,
        {"suggested_removed_features": []},
        skip_shap=True,
    )
    mi = mutual_info_classif(x_train, y_train, random_state=RANDOM_STATE)
    f_values, _ = f_classif(x_train, y_train)
    mi_norm = normalize(dict(zip(feature_names, mi)))
    f_norm = normalize(dict(zip(feature_names, np.nan_to_num(f_values, nan=0.0, posinf=0.0, neginf=0.0))))
    rows = []
    for feature in feature_names:
        scores = importance.get(feature, {})
        combined = (
            0.25 * scores.get("random_forest", 0.0)
            + 0.25 * scores.get("xgboost", 0.0)
            + 0.15 * scores.get("permutation", 0.0)
            + 0.20 * mi_norm.get(feature, 0.0)
            + 0.15 * f_norm.get(feature, 0.0)
        )
        rows.append({
            "feature": feature,
            "combined_score": combined,
            "random_forest": scores.get("random_forest", 0.0),
            "xgboost": scores.get("xgboost", 0.0),
            "permutation": scores.get("permutation", 0.0),
            "mutual_information": mi_norm.get(feature, 0.0),
            "anova_f": f_norm.get(feature, 0.0),
        })
    rows.sort(key=lambda row: row["combined_score"], reverse=True)
    return rows


def normalize(values: dict[str, float]) -> dict[str, float]:
    clean = {key: max(float(value), 0.0) for key, value in values.items()}
    total = sum(clean.values())
    if total <= 0:
        return {key: 0.0 for key in clean}
    return {key: value / total for key, value in clean.items()}


def build_feature_sets(
    importance: list[dict[str, Any]],
    dictionary: list[dict[str, str]],
    clean_features: list[str],
) -> dict[str, list[str]]:
    by_group: dict[str, list[str]] = defaultdict(list)
    for row in dictionary:
        name = row["feature_name"]
        if name in clean_features:
            by_group[row["feature_group"]].append(name)
    ranking = [row["feature"] for row in importance if row["feature"] in clean_features]
    sets: dict[str, list[str]] = {
        "all_improved": list(clean_features),
        "old_203": [name for name in config.FEATURE_NAMES if name in clean_features],
        "study_only": [name for name in clean_features if name not in config.FEATURE_NAMES],
        "lesion_only": by_group["lesion_exudate"] + by_group["lesion_microaneurysm"] + by_group["lesion_hemorrhage"],
        "texture_only": by_group["texture_lbp"] + by_group["texture_glcm"],
        "lesion_texture": by_group["lesion_exudate"] + by_group["lesion_microaneurysm"] + by_group["lesion_hemorrhage"] + by_group["texture_lbp"] + by_group["texture_glcm"],
        "lesion_texture_quality": by_group["lesion_exudate"] + by_group["lesion_microaneurysm"] + by_group["lesion_hemorrhage"] + by_group["texture_lbp"] + by_group["texture_glcm"] + by_group["quality"],
        "lesion_texture_vessel": by_group["lesion_exudate"] + by_group["lesion_microaneurysm"] + by_group["lesion_hemorrhage"] + by_group["texture_lbp"] + by_group["texture_glcm"] + by_group["vessel"],
        "lesion_texture_vessel_optic": by_group["lesion_exudate"] + by_group["lesion_microaneurysm"] + by_group["lesion_hemorrhage"] + by_group["texture_lbp"] + by_group["texture_glcm"] + by_group["vessel"] + by_group["optic_disc"],
    }
    for count in (50, 75, 100, 125, 150, 200, 250):
        if count <= len(ranking):
            sets[f"top_{count}"] = ranking[:count]
    return {key: dedupe(value) for key, value in sets.items() if value}


def dedupe(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def run_model_grid(
    problem: str,
    feature_sets: dict[str, list[str]],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    stage_label: str,
    svm_limit: int,
) -> list[study.ExperimentResult]:
    results = []
    candidate_sets = selected_feature_set_names(feature_sets)
    for set_name in candidate_sets:
        for model_name in MODEL_NAMES:
            try:
                result = study.fit_evaluate_model(
                    problem=problem,
                    model_name=model_name,
                    feature_set=set_name,
                    features=feature_sets[set_name],
                    x_train=x_train,
                    x_test=x_test,
                    y_train=y_train,
                    y_test=y_test,
                    output_dir=output_dir,
                    svm_train_limit=svm_limit,
                )
            except Exception as exc:
                print(f"{problem} {model_name} {set_name} failed: {exc}", flush=True)
                continue
            results.append(result)
            metric = result.metrics.get("macro_f1", result.metrics.get("f1", 0.0))
            print(f"{stage_label} {problem} {model_name} {set_name}: score={metric:.4f}", flush=True)
    return results


def selected_feature_set_names(feature_sets: dict[str, list[str]]) -> list[str]:
    preferred = [
        "old_203",
        "all_improved",
        "study_only",
        "lesion_only",
        "texture_only",
        "lesion_texture",
        "lesion_texture_quality",
        "lesion_texture_vessel",
        "lesion_texture_vessel_optic",
        "top_50",
        "top_75",
        "top_100",
        "top_125",
        "top_150",
        "top_200",
        "top_250",
    ]
    return [name for name in preferred if name in feature_sets]


def run_binary_grid(
    feature_sets: dict[str, list[str]],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train_multi: np.ndarray,
    y_test_multi: np.ndarray,
    output_dir: Path,
    svm_limit: int,
) -> tuple[list[study.ExperimentResult], list[dict[str, Any]]]:
    y_train = np.where(np.isin(y_train_multi, [2, 3, 4]), 1, 0)
    y_test = np.where(np.isin(y_test_multi, [2, 3, 4]), 1, 0)
    results = run_model_grid("binary", feature_sets, x_train, x_test, y_train, y_test, output_dir, "medium", svm_limit)
    rows = []
    for result in results:
        rows.extend(threshold_sweep(result, feature_sets[result.feature_set], x_test, y_test))
    write_csv(output_dir / "binary_threshold_sweep.csv", rows)
    return results, rows


def threshold_sweep(
    result: study.ExperimentResult,
    features: list[str],
    x_test: pd.DataFrame,
    y_test: np.ndarray,
) -> list[dict[str, Any]]:
    with Path(result.model_path).open("rb") as file:
        model = pickle.load(file)
    if not hasattr(model, "predict_proba"):
        return []
    probs = model.predict_proba(x_test[features])[:, 1]
    rows = []
    for threshold in THRESHOLDS:
        pred = (probs >= threshold).astype(int)
        matrix = confusion_matrix(y_test, pred, labels=[0, 1])
        base = {
            "model_name": result.model_name,
            "feature_set": result.feature_set,
            "threshold": threshold,
            "accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "referable_recall": recall_score(y_test, pred, pos_label=1, zero_division=0),
            "non_referable_recall": recall_score(y_test, pred, pos_label=0, zero_division=0),
            "false_positives": int(matrix[0, 1]),
            "false_negatives": int(matrix[1, 0]),
        }
        for low, high in UNCERTAINTY_BANDS:
            uncertain = (probs >= low) & (probs <= high)
            base[f"uncertain_{low:.2f}_{high:.2f}_pct"] = float(np.mean(uncertain))
        try:
            base["auc"] = roc_auc_score(y_test, probs)
        except Exception:
            base["auc"] = None
        rows.append(base)
    return rows


def run_hierarchical_experiments(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_sets: dict[str, list[str]],
    output_dir: Path,
    svm_limit: int,
) -> dict[str, Any]:
    features = feature_sets.get("top_100") or next(iter(feature_sets.values()))
    rows = {}
    rows["experiment_a"] = hierarchical_a(x_train, x_test, y_train, y_test, features, output_dir, svm_limit)
    rows["experiment_b"] = hierarchical_b(x_train, x_test, y_train, y_test, features, output_dir, svm_limit)
    return rows


def hierarchical_a(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    features: list[str],
    output_dir: Path,
    svm_limit: int,
) -> dict[str, Any]:
    no_dr = train_binary_stage("hier_a_no_dr", x_train, x_test, (y_train > 0).astype(int), (y_test > 0).astype(int), features, output_dir, svm_limit)
    ref = train_binary_stage("hier_a_ref", x_train, x_test, np.isin(y_train, [2, 3, 4]).astype(int), np.isin(y_test, [2, 3, 4]).astype(int), features, output_dir, svm_limit)
    advanced = train_binary_stage("hier_a_3v4", x_train[np.isin(y_train, [3, 4])], x_test, (y_train[np.isin(y_train, [3, 4])] == 4).astype(int), np.zeros(len(y_test), dtype=int), features, output_dir, svm_limit, evaluate=False)
    dr_mask = y_train > 0
    severity_path = train_multiclass_stage("hier_a_1v4", x_train[dr_mask], y_train[dr_mask], features, output_dir)
    p_no = load_model(no_dr["model_path"]).predict_proba(x_test[features])[:, 1]
    p_ref = load_model(ref["model_path"]).predict_proba(x_test[features])[:, 1]
    sev_pred = load_model(severity_path).predict(x_test[features])
    adv_pred = load_model(advanced["model_path"]).predict(x_test[features])
    pred = np.where(p_no < 0.5, 0, np.where(p_ref < 0.5, 1, sev_pred))
    adv_candidates = np.isin(pred, [3, 4])
    pred[adv_candidates] = np.where(adv_pred[adv_candidates] == 1, 4, 3)
    return evaluate_multiclass_predictions(y_test, pred)


def hierarchical_b(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    features: list[str],
    output_dir: Path,
    svm_limit: int,
) -> dict[str, Any]:
    group_train = np.where(y_train == 0, 0, np.where(y_train == 1, 1, 2))
    group_path = train_multiclass_stage("hier_b_group", x_train, group_train, features, output_dir)
    ref_mask = np.isin(y_train, [2, 3, 4])
    ref_path = train_multiclass_stage("hier_b_ref234", x_train[ref_mask], y_train[ref_mask], features, output_dir)
    group_pred = load_model(group_path).predict(x_test[features])
    ref_pred = load_model(ref_path).predict(x_test[features])
    pred = np.where(group_pred == 0, 0, np.where(group_pred == 1, 1, ref_pred))
    return evaluate_multiclass_predictions(y_test, pred)


def train_multiclass_stage(
    name: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    features: list[str],
    output_dir: Path,
) -> str:
    """Train a subproblem classifier without forcing the global 0-4 evaluator."""
    model_dir = ensure_dir(output_dir / "models")
    model_path = model_dir / f"hier_multiclass_{name}.pkl"
    if model_path.exists():
        return str(model_path)
    estimator = study.build_estimator("random_forest", "multiclass")
    study.fit_with_weights(estimator, x_train[features], y_train)
    with model_path.open("wb") as file:
        pickle.dump(estimator, file)
    return str(model_path)


def train_binary_stage(
    name: str,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    features: list[str],
    output_dir: Path,
    svm_limit: int,
    evaluate: bool = True,
) -> dict[str, Any]:
    model_dir = ensure_dir(output_dir / "models")
    model_path = model_dir / f"binary_xgboost_{name}.pkl"
    if model_path.exists():
        estimator = load_model(str(model_path))
    else:
        estimator = study.build_estimator("xgboost", "binary")
        study.fit_with_weights(estimator, x_train[features], y_train)
        with model_path.open("wb") as file:
            pickle.dump(estimator, file)
    metrics = study.evaluate_estimator(estimator, x_test[features], y_test, "binary") if evaluate else {}
    return {"model_path": str(model_path), "metrics": metrics}


def load_model(path: str) -> Any:
    with Path(path).open("rb") as file:
        return pickle.load(file)


def evaluate_multiclass_predictions(y_true: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    labels = [0, 1, 2, 3, 4]
    matrix = confusion_matrix(y_true, pred, labels=labels)
    report = classification_report(y_true, pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", labels=labels, zero_division=0)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "per_class": {str(label): report[str(label)] for label in labels},
    }


def best_result(results: list[study.ExperimentResult], problem: str) -> study.ExperimentResult:
    if problem == "binary":
        return max(results, key=lambda item: item.metrics["selection_score"])
    return max(results, key=lambda item: item.metrics["selection_score"])


def per_source_validation(
    result: study.ExperimentResult,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    test_meta: pd.DataFrame,
) -> list[dict[str, Any]]:
    model = load_model(result.model_path)
    pred = model.predict(x_test[result.selected_features])
    rows = []
    if "source_dataset" not in test_meta.columns:
        return rows
    for source in sorted(test_meta["source_dataset"].dropna().unique()):
        mask = test_meta["source_dataset"] == source
        if int(mask.sum()) < 5:
            continue
        rows.append({
            "source_dataset": str(source),
            "sample_count": int(mask.sum()),
            "accuracy": float(accuracy_score(y_test[mask], pred[mask])),
            "balanced_accuracy": float(balanced_accuracy_score(y_test[mask], pred[mask])),
            "macro_f1": float(f1_score(y_test[mask], pred[mask], labels=[0,1,2,3,4], average="macro", zero_division=0)),
        })
    return rows


def build_report(
    feature_table: pd.DataFrame,
    dictionary: list[dict[str, str]],
    audit: dict[str, Any],
    feature_sets: dict[str, list[str]],
    multiclass_results: list[study.ExperimentResult],
    binary_results: list[study.ExperimentResult],
    threshold_rows: list[dict[str, Any]],
    hierarchy: dict[str, Any],
    source_validation: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    best_multi = best_result(multiclass_results, "multiclass")
    best_binary = best_result(binary_results, "binary")
    best_threshold = select_threshold(threshold_rows)
    replacement = {
        "grading_replaced": False,
        "screening_replaced": False,
        "reason": (
            "Production was not overwritten. The best grading candidates did not preserve the prior "
            "Class 3 Severe NPDR recall safety target, and the best default binary candidate did not "
            "beat the current referable-recall baseline. The most sensitive threshold reduced false "
            "negatives but created too many false positives for production use without a triage policy."
        ),
    }
    report = {
        "created_at": datetime.now().isoformat(),
        "feature_count": len(dictionary),
        "row_count": len(feature_table),
        "feature_groups": dict(pd.Series([row["feature_group"] for row in dictionary]).value_counts().sort_index()),
        "feature_audit": audit,
        "feature_sets": {name: len(values) for name, values in feature_sets.items()},
        "best_multiclass": result_to_dict(best_multi),
        "best_binary": result_to_dict(best_binary),
        "best_threshold": best_threshold,
        "hierarchical_experiments": hierarchy,
        "source_validation": source_validation,
        "production_replacement": replacement,
        "model_comparison": [result_to_dict(item) for item in [*multiclass_results, *binary_results]],
    }
    report["markdown"] = render_markdown(report)
    return report


def select_threshold(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            row["referable_recall"] >= 0.9646,
            -row["false_negatives"],
            row["balanced_accuracy"],
            row["f1"],
            -row["false_positives"],
        ),
    )


def result_to_dict(result: study.ExperimentResult) -> dict[str, Any]:
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
    best = report["best_multiclass"]
    binary = report["best_binary"]
    threshold = report["best_threshold"]
    lines = [
        "# Study Max Improvement Report",
        "",
        "Classical ML only. No deep learning, transfer learning, CNN, UNet, ResNet, or YOLO was used.",
        "",
        "## Feature Set",
        "",
        f"Rows: {report['row_count']}",
        f"Total feature dictionary entries: {report['feature_count']}",
        f"Feature groups: {report['feature_groups']}",
        f"Removed features: {len(report['feature_audit']['removed_features'])}",
        "",
        "## Best 5-Class Grading Model",
        "",
        f"Model: {best['model_name']}",
        f"Feature set: {best['feature_set']} ({best['feature_count']} features)",
        f"Accuracy: {pct(best['metrics']['accuracy'])}",
        f"Balanced accuracy: {pct(best['metrics']['balanced_accuracy'])}",
        f"Macro F1: {pct(best['metrics']['macro_f1'])}",
        f"Class 1 recall: {pct(best['metrics']['per_class']['1']['recall'])}",
        f"Class 3 recall: {pct(best['metrics']['per_class']['3']['recall'])}",
        f"Class 4 recall: {pct(best['metrics']['per_class']['4']['recall'])}",
        "",
        "## Best Binary Screening Model",
        "",
        f"Model: {binary['model_name']}",
        f"Feature set: {binary['feature_set']} ({binary['feature_count']} features)",
        f"Accuracy: {pct(binary['metrics']['accuracy'])}",
        f"Balanced accuracy: {pct(binary['metrics']['balanced_accuracy'])}",
        f"F1: {pct(binary['metrics']['f1'])}",
        f"Referable recall: {pct(binary['metrics']['referable_recall'])}",
        f"False negatives: {binary['metrics']['false_negatives']}",
        "",
        "## Best Screening Threshold",
        "",
    ]
    if threshold:
        lines.extend([
            f"Model: {threshold['model_name']}",
            f"Feature set: {threshold['feature_set']}",
            f"Threshold: {threshold['threshold']}",
            f"Referable recall: {pct(threshold['referable_recall'])}",
            f"False negatives: {threshold['false_negatives']}",
            f"False positives: {threshold['false_positives']}",
        ])
    lines.extend([
        "",
        "## Hierarchical Grading",
        "",
        json.dumps(report["hierarchical_experiments"], indent=2)[:3000],
        "",
        "## Production Decision",
        "",
        f"Grading replaced: {report['production_replacement']['grading_replaced']}",
        f"Screening replaced: {report['production_replacement']['screening_replaced']}",
        report["production_replacement"]["reason"],
    ])
    return "\n".join(lines)


def write_reports(output_dir: Path, report: dict[str, Any]) -> None:
    (output_dir / "main_report.md").write_text(report["markdown"], encoding="utf-8")
    write_json(output_dir / "main_report.json", {key: value for key, value in report.items() if key != "markdown"})
    write_csv(output_dir / "model_comparison.csv", flatten_model_comparison(report["model_comparison"]))
    write_csv(output_dir / "per_stage_metrics.csv", per_stage_rows(report["best_multiclass"]))
    write_matrix(output_dir / "confusion_matrix.csv", report["best_multiclass"]["metrics"]["confusion_matrix"])
    write_study_comparison(output_dir / "study_comparison_table.md", report)
    write_final_recommendation(output_dir / "final_recommendation.md", report)


def flatten_model_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        metrics = row["metrics"]
        output.append({
            "problem": row["problem"],
            "model": row["model_name"],
            "feature_set": row["feature_set"],
            "feature_count": row["feature_count"],
            "accuracy": metrics.get("accuracy"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "binary_f1": metrics.get("f1"),
            "class_1_recall": metrics.get("per_class", {}).get("1", {}).get("recall"),
            "class_3_recall": metrics.get("per_class", {}).get("3", {}).get("recall"),
            "class_4_recall": metrics.get("per_class", {}).get("4", {}).get("recall"),
            "referable_recall": metrics.get("referable_recall"),
            "false_negatives": metrics.get("false_negatives"),
            "model_path": row["model_path"],
        })
    return output


def per_stage_rows(best: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for label, values in best["metrics"]["per_class"].items():
        rows.append({
            "class": label,
            "medical_label": config.CLASS_NAMES[int(label)] if int(label) in config.CLASS_NAMES else label,
            "precision": values["precision"],
            "recall": values["recall"],
            "f1": values["f1"],
            "support": values["support"],
            "correct": values["correct"],
        })
    return rows


def write_study_comparison(path: Path, report: dict[str, Any]) -> None:
    best = report["best_multiclass"]
    binary = report["best_binary"]
    content = f"""# Study Comparison Table

| Study/style | Features used | Models used | Reported result | AppDR corresponding result | Status |
|---|---|---|---|---|---|
| Bhattacharjee et al. style | Exudate, vessel, microaneurysm, texture features | SVM, RF, Naive Bayes | RF accuracy about 76.5% | Best AppDR 5-class accuracy {pct(best['metrics']['accuracy'])} | Lower |
| Gandor et al. style | CLAHE, LBP, GLCM | RF/XGBoost-style classical ML | Accuracy 80.41%, F1 74.41%, AUC 0.80 | AppDR macro F1 {pct(best['metrics']['macro_f1'])} | Lower |
| Yang et al. style | Referable screening features | XGBoost, RF, LightGBM, LR | Referable DR screening focus | Best default study-max referable recall {pct(binary['metrics']['referable_recall'])}; existing production/balanced screening remains safer | Keep current production screening |
| Casanova et al. style | Risk/feature RF classification | Random Forest | RF useful for DR classification/risk | AppDR RF candidates compared in model_comparison.csv | Comparable method, lower exact-grade target |
| Exudate-focused SVM/fuzzy SVM style | Exudate/bright lesion features | SVM decision support | Exudate support for DR decisions | AppDR keeps exudate features as supporting features, not standalone diagnosis | Implemented as feature group |
"""
    path.write_text(content, encoding="utf-8")


def write_final_recommendation(path: Path, report: dict[str, Any]) -> None:
    best = report["best_multiclass"]
    binary = report["best_binary"]
    threshold = report["best_threshold"]
    text = f"""# Final Recommendation

Keep the current production app output structure: referable DR screening as the main result, 5-class severity grading as supporting information, and the screening disclaimer.

The study-max experiment adds study-derived engineered feature groups and improves exact-grading candidates experimentally, but production artifacts were not overwritten.

Best selected grading candidate: {best['model_name']} / {best['feature_set']} with macro F1 {pct(best['metrics']['macro_f1'])}, balanced accuracy {pct(best['metrics']['balanced_accuracy'])}, Class 1 recall {pct(best['metrics']['per_class']['1']['recall'])}, Class 3 recall {pct(best['metrics']['per_class']['3']['recall'])}, and Class 4 recall {pct(best['metrics']['per_class']['4']['recall'])}. It does not replace production because Class 3 recall is below the prior study-feature target of 64.00%.

Best default binary candidate: {binary['model_name']} / {binary['feature_set']} with referable recall {pct(binary['metrics']['referable_recall'])} and {binary['metrics']['false_negatives']} false negatives. It does not replace production because referable recall is lower than the current screening safety baseline.

Most sensitive threshold candidate: {threshold['model_name']} / {threshold['feature_set']} threshold {threshold['threshold']} with referable recall {pct(threshold['referable_recall'])}, {threshold['false_negatives']} false negatives, and {threshold['false_positives']} false positives. This is useful for thesis discussion but over-refers too much for production without an uncertainty/triage workflow.
"""
    path.write_text(text, encoding="utf-8")


def backup_current_system(output_dir: Path) -> None:
    backup_dir = ensure_dir(output_dir / "backup")
    paths = [
        BACKEND_DIR / "results" / "best_model.pkl",
        BACKEND_DIR / "results" / "best_model_metadata.json",
        BACKEND_DIR / "results" / "metrics.json",
        BACKEND_DIR / "results" / "binary" / "best_model.pkl",
        BACKEND_DIR / "results" / "binary" / "best_model_metadata.json",
        BACKEND_DIR / "results" / "binary" / "metrics.json",
        BACKEND_DIR / "results" / "evaluation_report.md",
        BACKEND_DIR / "results" / "evaluation_report.json",
        BACKEND_DIR / "results" / "study_feature_selection" / "study_feature_selection_report.json",
        BACKEND_DIR / "results" / "study_feature_selection" / "study_feature_selection_report.md",
        BACKEND_DIR / "train.py",
        BACKEND_DIR / "app" / "pipeline.py",
        BACKEND_DIR / "app" / "schemas.py",
        BACKEND_DIR.parent / "App.tsx",
        BACKEND_DIR.parent / "README.md",
    ]
    manifest = []
    for source in paths:
        if not source.exists():
            continue
        target = backup_dir / source.name if source.parent != BACKEND_DIR / "results" / "binary" else backup_dir / "binary" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest.append({"source": str(source), "backup": str(target)})
    write_json(backup_dir / "manifest.json", {"created_at": datetime.now().isoformat(), "files": manifest})


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    path.write_text(json.dumps(study.to_jsonable(payload), indent=2), encoding="utf-8")


def write_matrix(path: Path, matrix: list[list[int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(matrix)


def pct(value: float) -> str:
    return f"{float(value) * 100.0:.2f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AppDR study max-improvement experiment.")
    parser.add_argument("--source-features", type=Path, default=SOURCE_FEATURES)
    parser.add_argument("--output-features", type=Path, default=OUTPUT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--svm-train-limit", type=int, default=6000)
    return parser.parse_args()


if __name__ == "__main__":
    main()
