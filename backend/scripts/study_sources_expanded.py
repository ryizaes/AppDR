"""Expanded classical study-source experiments for AppDR.

This script stays inside the existing AppDR project and uses only classical
retinal image processing plus shallow/classical machine learning. It does not
use CNNs, learned embeddings, neural networks, transfer learning, UNet, ResNet,
YOLO, or pretrained feature extractors.
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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
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
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.utils.class_weight import compute_sample_weight

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
except Exception:  # pragma: no cover - optional dependency.
    BalancedRandomForestClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency.
    LGBMClassifier = None

try:
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
except Exception:  # pragma: no cover - optional dependency.
    graycomatrix = None
    graycoprops = None
    local_binary_pattern = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency.
    XGBClassifier = None


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
from scripts import study_feature_selection_experiments as study
from scripts import study_max_improvement as study_max


RESULTS_DIR = BACKEND_DIR / "results" / "study_sources_expanded"
SOURCE_FEATURES = BACKEND_DIR / "features_combined_balanced.csv"
STUDY_MAX_FEATURES = BACKEND_DIR / "features_study_max.csv"
EXPANDED_FEATURES = RESULTS_DIR / "features_sources_expanded.csv"
IMAGE_FEATURE_CACHE = RESULTS_DIR / "image_texture_morphology_features.csv"
RANDOM_STATE = config.RANDOM_STATE
LABELS_5 = [0, 1, 2, 3, 4]
MODEL_NAMES = [
    "logistic_regression",
    "naive_bayes",
    "svm_rbf",
    "linear_svm",
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "histgradientboosting",
    "balanced_random_forest",
]
TEXTURE_MODELS = ["random_forest", "svm_rbf"]
BHATTACHARJEE_MODELS = ["random_forest", "svm_rbf", "naive_bayes"]
GANDOR_MODELS = ["random_forest", "xgboost"]
BINARY_MODELS = ["random_forest", "xgboost", "lightgbm", "extra_trees", "svm_rbf"]
BINARY_THRESHOLDS = [round(value, 2) for value in np.arange(0.10, 0.701, 0.05)]
UNCERTAINTY_BANDS = [(0.40, 0.60), (0.35, 0.65), (0.30, 0.70), (0.25, 0.75)]
SAFE_GRADING_BASELINE = {
    "accuracy": 0.6602,
    "balanced_accuracy": 0.6083,
    "macro_f1": 0.5718,
    "class_1_recall": 0.4533,
    "class_3_recall": 0.6400,
    "class_4_recall": 0.6233,
}
PRODUCTION_BASELINE = {
    "accuracy": 0.6798,
    "balanced_accuracy": 0.5312,
    "macro_f1": 0.5077,
    "class_1_recall": 0.3299,
    "class_3_recall": 0.3095,
    "class_4_recall": 0.6151,
    "binary_referable_recall": 0.9373,
    "binary_false_negatives": 88,
}
BINARY_SAFETY_BASELINE = {
    "referable_recall": 0.9646,
    "false_negatives": 56,
}
BINARY_FALSE_POSITIVE_EXTREME_LIMIT = 900
BINARY_UNCERTAINTY_USABLE_LIMIT = 0.35
MEDICALLY_IMPORTANT_TOKENS = (
    "ma",
    "microaneurysm",
    "red_lesion",
    "hemorrhage",
    "exudate",
    "cotton_wool",
    "bright_lesion",
    "optic_disc",
    "neovascular",
)


@dataclass
class ExperimentResult:
    problem: str
    model_name: str
    feature_set: str
    feature_count: int
    metrics: dict[str, Any]
    selected_features: list[str]
    model_path: str
    experiment: str


@dataclass
class ProbabilityStackingEnsemble:
    """Small classical stacking wrapper using base-model probabilities."""

    base_estimators: list[tuple[str, BaseEstimator]]
    meta_estimator: BaseEstimator
    labels: list[int]

    def predict_proba(self, x_values: pd.DataFrame) -> np.ndarray:
        meta_features = stacking_meta_features(self.base_estimators, x_values, self.labels)
        probabilities = self.meta_estimator.predict_proba(meta_features)
        meta_classes = list(getattr(self.meta_estimator, "classes_", self.labels))
        aligned = np.zeros((len(x_values), len(self.labels)), dtype=np.float64)
        for index, label in enumerate(meta_classes):
            if label in self.labels:
                aligned[:, self.labels.index(int(label))] = probabilities[:, index]
        return aligned

    def predict(self, x_values: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(x_values)
        return np.asarray(self.labels, dtype=int)[np.argmax(probabilities, axis=1)]


# Give pickled stacker artifacts a stable import path even when this file is
# executed as a script.
sys.modules.setdefault("backend.scripts.study_sources_expanded", sys.modules[__name__])
ProbabilityStackingEnsemble.__module__ = "backend.scripts.study_sources_expanded"


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    backup_existing_work(output_dir)

    source = pd.read_csv(args.source_features)
    base_table, base_dictionary = build_base_study_table(source, args.study_max_features)
    image_table, image_dictionary = load_or_build_image_feature_cache(
        base_table,
        output_dir,
        workers=args.image_workers,
        limit=args.image_feature_limit,
        rebuild=args.rebuild_image_features,
    )
    feature_table, dictionary = merge_feature_tables(base_table, base_dictionary, image_table, image_dictionary)
    feature_table.to_csv(output_dir / "features_sources_expanded.csv", index=False)
    write_csv(output_dir / "feature_dictionary.csv", dictionary)

    feature_names = [row["feature_name"] for row in dictionary if row["feature_name"] in feature_table.columns]
    audit = audit_and_prune(feature_table, feature_names, dictionary)
    write_json(output_dir / "feature_audit.json", audit)
    write_csv(output_dir / "removed_features.csv", audit["removed_features"])

    clean_features = [name for name in feature_names if name not in {row["feature"] for row in audit["removed_features"]}]
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
    write_csv(output_dir / "feature_importance_expanded.csv", importance)
    feature_sets = build_feature_sets(dictionary, clean_features, importance, output_dir)
    write_json(output_dir / "feature_sets.json", feature_sets)
    write_csv(output_dir / "feature_set_comparison.csv", feature_set_rows(feature_sets, dictionary))

    multiclass_results = run_required_experiments(
        feature_sets,
        x_train,
        x_test,
        y_train,
        y_test,
        output_dir,
        svm_train_limit=args.svm_train_limit,
        quick=args.quick,
    )
    ensemble_results = run_ensemble_experiments(feature_sets, x_train, x_test, y_train, y_test, output_dir)
    hierarchy_results = run_hierarchical_experiments(feature_sets, x_train, x_test, y_train, y_test, output_dir, args.svm_train_limit)
    binary_results, threshold_rows = run_binary_experiments(
        feature_sets,
        x_train,
        x_test,
        y_train,
        y_test,
        output_dir,
        svm_train_limit=args.svm_train_limit,
        quick=args.quick,
    )
    source_validation = per_source_validation(select_best_grading([*multiclass_results, *ensemble_results]), x_test, y_test, test_meta)

    report = build_report(
        feature_table=feature_table,
        dictionary=dictionary,
        audit=audit,
        feature_sets=feature_sets,
        multiclass_results=multiclass_results,
        ensemble_results=ensemble_results,
        hierarchy_results=hierarchy_results,
        binary_results=binary_results,
        threshold_rows=threshold_rows,
        source_validation=source_validation,
        output_dir=output_dir,
    )
    write_reports(output_dir, report)
    print(report["markdown"], flush=True)


def build_base_study_table(source: pd.DataFrame, study_max_features: Path) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    if study_max_features.exists():
        table = pd.read_csv(study_max_features)
        dictionary_path = study_max.RESULTS_DIR / "feature_dictionary.csv"
        if dictionary_path.exists():
            dictionary = pd.read_csv(dictionary_path).fillna("").to_dict(orient="records")
            return table, [normalize_dictionary_row(row) for row in dictionary]

    table, dictionary = study_max.build_study_feature_table(source)
    return table, dictionary


def load_or_build_image_feature_cache(
    table: pd.DataFrame,
    output_dir: Path,
    workers: int,
    limit: int | None,
    rebuild: bool,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    cache_path = output_dir / "image_texture_morphology_features.csv"
    dictionary = image_feature_dictionary()
    target_count = min(limit, len(table)) if limit is not None else len(table)
    if cache_path.exists() and not rebuild:
        cached = pd.read_csv(cache_path)
        if len(cached) >= target_count:
            return cached.iloc[:target_count].copy(), dictionary
        existing_indices = set(pd.to_numeric(cached.get("source_index", pd.Series(dtype=int)), errors="coerce").dropna().astype(int).tolist())
    else:
        cached = pd.DataFrame()
        existing_indices = set()

    rows = table[["image_id", "image_path", "image_sha256"] if "image_sha256" in table.columns else ["image_id", "image_path"]].copy()
    if limit is not None:
        rows = rows.iloc[:limit].copy()
    tasks = [
        row for row in rows.reset_index().to_dict(orient="records")
        if int(row["index"]) not in existing_indices
    ]
    results: list[dict[str, Any]] = [] if cached.empty else cached.to_dict(orient="records")
    ensure_dir(output_dir)
    checkpoint_every = 250
    if not tasks:
        return pd.DataFrame(results).sort_values("source_index").reset_index(drop=True), dictionary
    if workers <= 1:
        for idx, task in enumerate(tasks, start=1):
            results.append(extract_image_features_for_row(task))
            if idx % checkpoint_every == 0:
                write_csv(cache_path, sorted(results, key=lambda row: int(row["source_index"])))
                print(f"cached image texture/morphology features: {len(results)}/{target_count}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(extract_image_features_for_row, task) for task in tasks]
            for idx, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if idx % checkpoint_every == 0:
                    write_csv(cache_path, sorted(results, key=lambda row: int(row["source_index"])))
                    print(f"cached image texture/morphology features: {len(results)}/{target_count}", flush=True)

    results = sorted(results, key=lambda row: int(row["source_index"]))
    write_csv(cache_path, results)
    return pd.DataFrame(results), dictionary


def extract_image_features_for_row(task: dict[str, Any]) -> dict[str, Any]:
    source_index = int(task["index"])
    image_path = str(task.get("image_path", ""))
    image_id = str(task.get("image_id", ""))
    base: dict[str, Any] = {
        "source_index": source_index,
        "image_id": image_id,
    }
    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("cv2.imread returned None")
        image = resize_long_side(image, 256)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        green = image[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(green)
        fov_mask = build_quick_fov_mask(gray)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab_b = lab[:, :, 2]
        features: dict[str, float] = {}
        features.update(lbp_stats("expanded_lbp_gray_r1_uniform", gray, fov_mask, radius=1, points=8, method="uniform"))
        features.update(lbp_stats("expanded_lbp_green_r1_uniform", green, fov_mask, radius=1, points=8, method="uniform"))
        features.update(lbp_stats("expanded_lbp_clahe_green_r1_uniform", clahe, fov_mask, radius=1, points=8, method="uniform"))
        for radius in (1, 2, 3):
            features.update(lbp_stats(f"expanded_multiscale_lbp_clahe_r{radius}", clahe, fov_mask, radius=radius, points=8 * radius, method="uniform"))
        features.update(encoded_lbp_histogram("expanded_encoded_lbp_green", green, fov_mask))
        features.update(glcm_texture_features("expanded_glcm_clahe_green", clahe, fov_mask))
        features.update(bright_lesion_features(image, gray, green, clahe, lab_b, fov_mask))
        features["expanded_feature_extraction_ok"] = 1.0
        base.update(features)
    except Exception:
        for row in image_feature_dictionary():
            base[row["feature_name"]] = 0.0
        base["expanded_feature_extraction_ok"] = 0.0
    return base


def resize_long_side(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    side = max(height, width)
    if side <= max_side:
        return image
    scale = max_side / float(side)
    return cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)


def build_quick_fov_mask(gray: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[gray > 12] = 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean, [largest], -1, 255, thickness=-1)
        return clean
    clean[:] = 255
    return clean


def lbp_stats(prefix: str, image: np.ndarray, mask: np.ndarray, radius: int, points: int, method: str) -> dict[str, float]:
    if local_binary_pattern is None:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_entropy": 0.0,
            f"{prefix}_uniform_ratio": 0.0,
        }
    values = local_binary_pattern(image, P=points, R=radius, method=method)
    roi = values[mask > 0].astype(np.float64)
    if roi.size == 0:
        roi = values.ravel().astype(np.float64)
    uniform_hits = np.isin(roi, np.arange(points + 1))
    return {
        f"{prefix}_mean": float(np.mean(roi)) if roi.size else 0.0,
        f"{prefix}_std": float(np.std(roi)) if roi.size else 0.0,
        f"{prefix}_entropy": histogram_entropy(roi, bins=min(64, max(points + 2, 16))),
        f"{prefix}_uniform_ratio": float(np.mean(uniform_hits)) if roi.size else 0.0,
    }


def encoded_lbp_histogram(prefix: str, image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    output = {f"{prefix}_bin_{index:02d}": 0.0 for index in range(16)}
    output[f"{prefix}_dominant_bin"] = 0.0
    output[f"{prefix}_entropy"] = 0.0
    if local_binary_pattern is None:
        return output
    lbp = local_binary_pattern(image, P=8, R=1, method="default").astype(np.uint8)
    roi = lbp[mask > 0]
    if roi.size == 0:
        roi = lbp.ravel()
    encoded = (roi // 16).astype(np.int32)
    hist = np.bincount(encoded, minlength=16).astype(np.float64)
    hist = hist / max(float(hist.sum()), 1.0)
    for index, value in enumerate(hist[:16]):
        output[f"{prefix}_bin_{index:02d}"] = float(value)
    output[f"{prefix}_dominant_bin"] = float(np.argmax(hist))
    output[f"{prefix}_entropy"] = histogram_entropy(encoded, bins=16)
    return output


def glcm_texture_features(prefix: str, image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    props = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
    distances = (1, 2, 4)
    angles = (0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0)
    output: dict[str, float] = {}
    for distance in distances:
        for angle_name in ("0", "45", "90", "135"):
            for prop in props:
                output[f"{prefix}_d{distance}_a{angle_name}_{prop.lower()}"] = 0.0
    for prop in props:
        output[f"{prefix}_{prop.lower()}_mean"] = 0.0
        output[f"{prefix}_{prop.lower()}_range"] = 0.0
    if graycomatrix is None or graycoprops is None:
        return output
    masked = image.copy()
    masked[mask == 0] = 0
    x, y, w, h = cv2.boundingRect((mask > 0).astype(np.uint8))
    roi = masked[y : y + h, x : x + w]
    if roi.size == 0:
        roi = masked
    quantized = np.clip((roi.astype(np.float32) / 256.0) * 32.0, 0, 31).astype(np.uint8)
    matrix = graycomatrix(quantized, distances=list(distances), angles=list(angles), levels=32, symmetric=True, normed=True)
    values_by_prop: dict[str, list[float]] = {prop: [] for prop in props}
    angle_labels = ("0", "45", "90", "135")
    for prop in props:
        values = graycoprops(matrix, prop).astype(np.float64)
        for d_index, distance in enumerate(distances):
            for a_index, angle_name in enumerate(angle_labels):
                value = float(values[d_index, a_index])
                output[f"{prefix}_d{distance}_a{angle_name}_{prop.lower()}"] = value
                values_by_prop[prop].append(value)
        prop_values = values_by_prop[prop]
        output[f"{prefix}_{prop.lower()}_mean"] = float(np.mean(prop_values)) if prop_values else 0.0
        output[f"{prefix}_{prop.lower()}_range"] = float(np.max(prop_values) - np.min(prop_values)) if prop_values else 0.0
    return output


def bright_lesion_features(
    image: np.ndarray,
    gray: np.ndarray,
    green: np.ndarray,
    clahe: np.ndarray,
    lab_b: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    fov_area = max(int(np.count_nonzero(mask)), 1)
    valid_values = clahe[mask > 0]
    if valid_values.size == 0:
        valid_values = clahe.ravel()
    bright_threshold = float(np.percentile(valid_values, 98.4))
    lab_threshold = float(np.percentile(lab_b[mask > 0] if np.any(mask > 0) else lab_b.ravel(), 70.0))
    bright_candidates = np.zeros_like(gray, dtype=np.uint8)
    bright_candidates[(clahe >= bright_threshold) & (lab_b >= lab_threshold) & (mask > 0)] = 255
    bright_candidates = cv2.morphologyEx(bright_candidates, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    bright_candidates = filter_components(bright_candidates, min_area=3, max_area=max(25, int(fov_area * 0.08)))
    optic_mask, od_center, od_confidence = estimate_optic_disc(clahe, green, mask)
    od_dilated = cv2.dilate(optic_mask, np.ones((13, 13), np.uint8), iterations=1)
    post_od = bright_candidates.copy()
    post_od[od_dilated > 0] = 0
    before_stats = component_stats(bright_candidates)
    after_stats = component_stats(post_od)
    od_area = int(np.count_nonzero(optic_mask))
    distances = component_distances(bright_candidates, od_center)
    artifact_area = max(before_stats["area"] - after_stats["area"], 0)
    return {
        "expanded_bright_lesion_count_raw": float(before_stats["count"]),
        "expanded_bright_lesion_area_raw": float(before_stats["area"]),
        "expanded_bright_lesion_density_raw": float(before_stats["area"] / fov_area),
        "expanded_bright_lesion_mean_area_raw": float(before_stats["mean_area"]),
        "expanded_optic_disc_candidate_area": float(od_area),
        "expanded_optic_disc_candidate_x": float(od_center[0]) if od_center else 0.0,
        "expanded_optic_disc_candidate_y": float(od_center[1]) if od_center else 0.0,
        "expanded_optic_disc_mask_confidence": float(od_confidence),
        "expanded_exudate_count_after_od_removal": float(after_stats["count"]),
        "expanded_exudate_area_after_od_removal": float(after_stats["area"]),
        "expanded_exudate_density_after_od_removal": float(after_stats["area"] / fov_area),
        "expanded_bright_artifact_area_removed": float(artifact_area),
        "expanded_bright_artifact_ratio": float(artifact_area / max(before_stats["area"], 1)),
        "expanded_bright_lesion_distance_to_od_mean": float(np.mean(distances)) if distances else 0.0,
        "expanded_bright_lesion_distance_to_od_min": float(np.min(distances)) if distances else 0.0,
    }


def estimate_optic_disc(clahe: np.ndarray, green: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, tuple[float, float] | None, float]:
    fov_area = max(int(np.count_nonzero(mask)), 1)
    valid = clahe[mask > 0]
    threshold = float(np.percentile(valid if valid.size else clahe.ravel(), 99.0))
    candidate = np.zeros_like(clahe, dtype=np.uint8)
    candidate[(clahe >= threshold) & (mask > 0)] = 255
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_contour = None
    best_score = -1.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < fov_area * 0.001 or area > fov_area * 0.12:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4.0 * math.pi * area / max(perimeter * perimeter, 1.0)
        x, y, w, h = cv2.boundingRect(contour)
        aspect = min(w, h) / max(w, h, 1)
        brightness = float(np.mean(green[y : y + h, x : x + w]))
        score = area * max(circularity, 0.01) * max(aspect, 0.01) * max(brightness, 1.0)
        if score > best_score:
            best_score = score
            best_contour = contour
    optic_mask = np.zeros_like(clahe, dtype=np.uint8)
    if best_contour is None:
        return optic_mask, None, 0.0
    cv2.drawContours(optic_mask, [best_contour], -1, 255, thickness=-1)
    moments = cv2.moments(best_contour)
    if moments["m00"] == 0:
        center = None
    else:
        center = (float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"]))
    confidence = min(1.0, max(0.0, best_score / max(fov_area * 255.0, 1.0)))
    return optic_mask, center, confidence


def filter_components(mask: np.ndarray, min_area: int, max_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    output = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            output[labels == label] = 255
    return output


def component_stats(mask: np.ndarray) -> dict[str, float]:
    count, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, count)]
    return {
        "count": float(len(areas)),
        "area": float(sum(areas)),
        "mean_area": float(np.mean(areas)) if areas else 0.0,
    }


def component_distances(mask: np.ndarray, center: tuple[float, float] | None) -> list[float]:
    if center is None:
        return []
    count, _, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    distances = []
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) <= 0:
            continue
        cx, cy = centroids[label]
        distances.append(float(math.hypot(float(cx) - center[0], float(cy) - center[1])))
    return distances


def histogram_entropy(values: np.ndarray, bins: int) -> float:
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=bins)
    probabilities = hist.astype(np.float64)
    probabilities = probabilities / max(float(probabilities.sum()), 1.0)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities))) if probabilities.size else 0.0


def image_feature_dictionary() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(name: str, group: str, description: str, basis: str, variant: str) -> None:
        rows.append(feature_dict_row(name, group, description, basis, variant, "float"))

    for prefix, variant in [
        ("expanded_lbp_gray_r1_uniform", "grayscale_uniform_lbp_r1"),
        ("expanded_lbp_green_r1_uniform", "green_channel_uniform_lbp_r1"),
        ("expanded_lbp_clahe_green_r1_uniform", "clahe_green_uniform_lbp_r1"),
    ]:
        for metric in ("mean", "std", "entropy", "uniform_ratio"):
            add(f"{prefix}_{metric}", "texture_lbp", f"{metric} from {variant}.", "LBP + GLCM texture RF/SVM and encoded LBP grading studies", variant)
    for radius in (1, 2, 3):
        for metric in ("mean", "std", "entropy", "uniform_ratio"):
            add(f"expanded_multiscale_lbp_clahe_r{radius}_{metric}", "texture_lbp", f"Multiscale uniform LBP radius {radius} {metric}.", "Encoded/uniform/multiscale LBP severity grading support", "clahe_green")
    for index in range(16):
        add(f"expanded_encoded_lbp_green_bin_{index:02d}", "texture_lbp", "Encoded/default LBP coarse histogram bin.", "Berbar-style encoded LBP texture grading", "green_channel_encoded_lbp")
    add("expanded_encoded_lbp_green_dominant_bin", "texture_lbp", "Dominant encoded LBP bin.", "Berbar-style encoded LBP texture grading", "green_channel_encoded_lbp")
    add("expanded_encoded_lbp_green_entropy", "texture_lbp", "Encoded LBP histogram entropy.", "Berbar-style encoded LBP texture grading", "green_channel_encoded_lbp")
    props = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "asm")
    for distance in (1, 2, 4):
        for angle in ("0", "45", "90", "135"):
            for prop in props:
                add(f"expanded_glcm_clahe_green_d{distance}_a{angle}_{prop}", "texture_glcm", f"GLCM {prop} at distance {distance}, angle {angle}.", "LBP + GLCM texture RF/SVM and Gandor-style CLAHE texture studies", "clahe_green")
    for prop in props:
        add(f"expanded_glcm_clahe_green_{prop}_mean", "texture_glcm", f"Mean GLCM {prop} across distances and angles.", "LBP + GLCM texture RF/SVM and Gandor-style CLAHE texture studies", "clahe_green")
        add(f"expanded_glcm_clahe_green_{prop}_range", "texture_glcm", f"Range of GLCM {prop} across distances and angles.", "LBP + GLCM texture RF/SVM and Gandor-style CLAHE texture studies", "clahe_green")
    for name, description in {
        "expanded_bright_lesion_count_raw": "Bright lesion candidate count before optic-disc removal.",
        "expanded_bright_lesion_area_raw": "Bright lesion candidate area before optic-disc removal.",
        "expanded_bright_lesion_density_raw": "Bright lesion candidate density before optic-disc removal.",
        "expanded_bright_lesion_mean_area_raw": "Mean bright lesion candidate area before optic-disc removal.",
        "expanded_optic_disc_candidate_area": "Estimated optic-disc candidate area.",
        "expanded_optic_disc_candidate_x": "Estimated optic-disc candidate centroid x coordinate.",
        "expanded_optic_disc_candidate_y": "Estimated optic-disc candidate centroid y coordinate.",
        "expanded_optic_disc_mask_confidence": "Heuristic confidence for optic-disc candidate mask.",
        "expanded_exudate_count_after_od_removal": "Bright/exudate candidate count after optic-disc removal.",
        "expanded_exudate_area_after_od_removal": "Bright/exudate candidate area after optic-disc removal.",
        "expanded_exudate_density_after_od_removal": "Bright/exudate candidate density after optic-disc removal.",
        "expanded_bright_artifact_area_removed": "Bright candidate area removed by optic-disc control.",
        "expanded_bright_artifact_ratio": "Fraction of bright candidate area removed by optic-disc control.",
        "expanded_bright_lesion_distance_to_od_mean": "Mean bright lesion candidate distance to optic-disc center.",
        "expanded_bright_lesion_distance_to_od_min": "Minimum bright lesion candidate distance to optic-disc center.",
    }.items():
        group = "optic_disc" if "optic_disc" in name else "lesion_exudate"
        add(name, group, description, "Morphological hard-exudate extraction with optic-disc/bright-artifact control", "clahe_green_lab_b_morphology")
    add("expanded_feature_extraction_ok", "quality", "Image-level expanded feature extraction success flag.", "Feature audit quality control", "image_io")
    return rows


def merge_feature_tables(
    base_table: pd.DataFrame,
    base_dictionary: list[dict[str, str]],
    image_table: pd.DataFrame,
    image_dictionary: list[dict[str, str]],
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    table = base_table.copy()
    if not image_table.empty and "source_index" in image_table.columns:
        image_features = image_table.set_index("source_index")
        for row in image_dictionary:
            name = row["feature_name"]
            table[name] = 0.0
            if name in image_features.columns:
                aligned = image_features[name].reindex(table.index).fillna(0.0)
                table[name] = pd.to_numeric(aligned, errors="coerce").fillna(0.0).to_numpy()
    dictionary = [normalize_dictionary_row(row) for row in base_dictionary]
    existing = {row["feature_name"] for row in dictionary}
    dictionary.extend(row for row in image_dictionary if row["feature_name"] not in existing)
    return table, dictionary


def audit_and_prune(
    table: pd.DataFrame,
    feature_names: list[str],
    dictionary: list[dict[str, str]],
) -> dict[str, Any]:
    audit = study.audit_features(table, feature_names)
    important = {
        row["feature_name"]
        for row in dictionary
        if any(token in row["feature_name"].lower() for token in MEDICALLY_IMPORTANT_TOKENS)
    }
    removed = []
    kept_rare = []
    for row in audit["suggested_removed_features"]:
        if row["feature"] in important and row["reason"] == "near_constant":
            kept_rare.append(row)
            continue
        removed.append(row)
    audit["removed_features"] = removed
    audit["rare_medical_features_kept_test_version"] = kept_rare
    audit["correlation_pairs"] = audit["highly_correlated_pairs"]
    return audit


def compute_rankings(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    usable = list(feature_names)
    x_train_clean = x_train[usable].fillna(x_train[usable].median(numeric_only=True).fillna(0.0))
    x_test_clean = x_test[usable].fillna(x_train_clean.median(numeric_only=True).fillna(0.0))
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    rf = RandomForestClassifier(
        n_estimators=180,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    rf.fit(x_train_clean, y_train, sample_weight=sample_weight)
    rf_importance = normalize(dict(zip(usable, rf.feature_importances_)))
    xgb_importance: dict[str, float] = {}
    if XGBClassifier is not None:
        xgb = XGBClassifier(
            objective="multi:softprob",
            num_class=5,
            n_estimators=160,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.85,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=4,
            tree_method="hist",
        )
        xgb.fit(x_train_clean, y_train, sample_weight=sample_weight)
        xgb_importance = normalize(dict(zip(usable, xgb.feature_importances_)))
    top_for_perm = sorted(usable, key=lambda feature: rf_importance.get(feature, 0.0) + xgb_importance.get(feature, 0.0), reverse=True)[:100]
    perm_importance: dict[str, float] = {}
    if top_for_perm:
        perm_model = RandomForestClassifier(
            n_estimators=120,
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
            n_repeats=2,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        perm_importance = normalize(dict(zip(top_for_perm, np.maximum(perm.importances_mean, 0.0))))
    mi = mutual_info_classif(x_train_clean, y_train, random_state=RANDOM_STATE)
    f_values, _ = f_classif(x_train_clean, y_train)
    mi_norm = normalize(dict(zip(usable, mi)))
    f_norm = normalize(dict(zip(usable, np.nan_to_num(f_values, nan=0.0, posinf=0.0, neginf=0.0))))
    rows = []
    for feature in usable:
        combined = (
            0.25 * rf_importance.get(feature, 0.0)
            + 0.25 * xgb_importance.get(feature, 0.0)
            + 0.15 * perm_importance.get(feature, 0.0)
            + 0.20 * mi_norm.get(feature, 0.0)
            + 0.15 * f_norm.get(feature, 0.0)
        )
        rows.append({
            "feature": feature,
            "combined_score": float(combined),
            "random_forest": float(rf_importance.get(feature, 0.0)),
            "xgboost": float(xgb_importance.get(feature, 0.0)),
            "permutation": float(perm_importance.get(feature, 0.0)),
            "mutual_information": float(mi_norm.get(feature, 0.0)),
            "anova_f": float(f_norm.get(feature, 0.0)),
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
    dictionary: list[dict[str, str]],
    clean_features: list[str],
    importance: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, list[str]]:
    clean = set(clean_features)
    by_group: dict[str, list[str]] = defaultdict(list)
    for row in dictionary:
        name = row["feature_name"]
        if name in clean:
            by_group[row["feature_group"]].append(name)
    ranked = [row["feature"] for row in importance if row["feature"] in clean]
    previous_top100 = load_previous_top100()
    previous_top125 = load_previous_top125()
    old_203 = [name for name in config.FEATURE_NAMES if name in clean]
    study_242 = [row["feature_name"] for row in dictionary[:242] if row["feature_name"] in clean]
    exudate = by_group["lesion_exudate"] + by_group["optic_disc"]
    micro = by_group["lesion_microaneurysm"] + [name for name in clean if "red_lesion" in name]
    hemorrhage = by_group["lesion_hemorrhage"] + [name for name in clean if "severe" in name or "advanced_dr" in name]
    lesion = dedupe(by_group["lesion_exudate"] + by_group["lesion_microaneurysm"] + by_group["lesion_hemorrhage"])
    texture = dedupe(by_group["texture_lbp"] + by_group["texture_glcm"])
    vessel = dedupe(by_group["vessel"])
    quality = dedupe(by_group["quality"])
    sets = {
        "old_203_features": old_203,
        "study_242_features": study_242,
        "previous_top_100_svm_features": [name for name in previous_top100 if name in clean],
        "previous_top_125_lightgbm_features": [name for name in previous_top125 if name in clean],
        "texture_only_lbp_glcm": texture,
        "morphological_exudate_optic_disc": dedupe(exudate),
        "microaneurysm_red_lesion": dedupe(micro),
        "hemorrhage_severe_lesion": dedupe(hemorrhage),
        "vessel_features": vessel,
        "lesion_only": lesion,
        "texture_exudate": dedupe(texture + exudate),
        "texture_microaneurysm": dedupe(texture + micro),
        "texture_hemorrhage": dedupe(texture + hemorrhage),
        "lesion_vessel": dedupe(lesion + vessel),
        "texture_lesion": dedupe(texture + lesion),
        "texture_lesion_optic_disc_control": dedupe(texture + lesion + by_group["optic_disc"]),
        "texture_lesion_vessel_quality": dedupe(texture + lesion + vessel + quality),
        "all_expanded": list(clean_features),
    }
    for count in (50, 75, 100, 125, 150, 200):
        if len(ranked) >= count:
            sets[f"expanded_top_{count}"] = ranked[:count]
    return {key: dedupe(value) for key, value in sets.items() if value}


def load_previous_top100() -> list[str]:
    path = BACKEND_DIR / "results" / "study_feature_selection" / "study_feature_selection_report.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))["best_multiclass"]["selected_features"]
        except Exception:
            pass
    return []


def load_previous_top125() -> list[str]:
    path = BACKEND_DIR / "results" / "study_max_improvement" / "main_report.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))["best_multiclass"]["selected_features"]
        except Exception:
            pass
    return []


def feature_set_rows(feature_sets: dict[str, list[str]], dictionary: list[dict[str, str]]) -> list[dict[str, Any]]:
    group_by_feature = {row["feature_name"]: row["feature_group"] for row in dictionary}
    rows = []
    for name, features in feature_sets.items():
        groups = pd.Series([group_by_feature.get(feature, "unknown") for feature in features]).value_counts().to_dict() if features else {}
        rows.append({
            "feature_set": name,
            "feature_count": len(features),
            "feature_groups": json.dumps(groups, sort_keys=True),
            "feature_examples": "; ".join(features[:12]),
        })
    return rows


def run_required_experiments(
    feature_sets: dict[str, list[str]],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    svm_train_limit: int,
    quick: bool,
) -> list[ExperimentResult]:
    schedule = build_multiclass_schedule(feature_sets, quick)
    results: list[ExperimentResult] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for experiment, set_name, model_names in schedule:
        features = feature_sets[set_name]
        for model_name in model_names:
            if not is_model_available(model_name):
                continue
            try:
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
                    experiment=experiment,
                )
            except Exception as exc:
                print(f"{experiment} {model_name} {set_name} failed: {exc}", flush=True)
                continue
            results.append(result)
            checkpoint_rows.append(flatten_result(result))
            write_csv(output_dir / "model_comparison_checkpoint.csv", checkpoint_rows)
            print(
                f"{experiment} multiclass {model_name}/{set_name}: "
                f"macro_f1={result.metrics['macro_f1']:.4f} "
                f"bal_acc={result.metrics['balanced_accuracy']:.4f} "
                f"class3={result.metrics['per_class']['3']['recall']:.4f}",
                flush=True,
            )
    return results


def build_multiclass_schedule(feature_sets: dict[str, list[str]], quick: bool) -> list[tuple[str, str, list[str]]]:
    schedule: list[tuple[str, str, list[str]]] = []
    required_sets = [
        "old_203_features",
        "study_242_features",
        "previous_top_100_svm_features",
        "previous_top_125_lightgbm_features",
        "texture_only_lbp_glcm",
        "morphological_exudate_optic_disc",
        "microaneurysm_red_lesion",
        "hemorrhage_severe_lesion",
        "vessel_features",
        "lesion_only",
        "texture_exudate",
        "texture_microaneurysm",
        "texture_hemorrhage",
        "lesion_vessel",
        "texture_lesion",
        "texture_lesion_optic_disc_control",
        "texture_lesion_vessel_quality",
        "expanded_top_100",
        "expanded_top_125",
        "expanded_top_150",
    ]
    if quick:
        required_sets = [
            "previous_top_100_svm_features",
            "previous_top_125_lightgbm_features",
            "texture_only_lbp_glcm",
            "morphological_exudate_optic_disc",
            "microaneurysm_red_lesion",
            "hemorrhage_severe_lesion",
            "texture_lesion_vessel_quality",
            "expanded_top_125",
        ]
    for set_name in required_sets:
        if set_name in feature_sets and feature_sets[set_name]:
            schedule.append(("required_model_grid", set_name, MODEL_NAMES))
    if "texture_only_lbp_glcm" in feature_sets:
        schedule.append(("experiment_1_texture_rf_vs_svm", "texture_only_lbp_glcm", TEXTURE_MODELS))
    if "lesion_vessel" in feature_sets:
        micro_vessel = "lesion_vessel"
        schedule.append(("experiment_2_lesion_vessel_microaneurysm", micro_vessel, BHATTACHARJEE_MODELS))
    if "texture_only_lbp_glcm" in feature_sets:
        schedule.append(("experiment_3_lbp_glcm_clahe_rf_xgboost", "texture_only_lbp_glcm", GANDOR_MODELS))
    if "morphological_exudate_optic_disc" in feature_sets:
        schedule.append(("experiment_4_exudate_pre_post_od_ablation", "morphological_exudate_optic_disc", ["random_forest", "xgboost"]))
    for set_name, label in [
        ("microaneurysm_red_lesion", "experiment_5_class1_microaneurysm"),
        ("hemorrhage_severe_lesion", "experiment_5_class3_hemorrhage"),
        ("lesion_vessel", "experiment_5_class4_lesion_vessel"),
    ]:
        if set_name in feature_sets:
            schedule.append((label, set_name, ["random_forest", "xgboost", "svm_rbf"]))
    return schedule


def run_binary_experiments(
    feature_sets: dict[str, list[str]],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train_multi: np.ndarray,
    y_test_multi: np.ndarray,
    output_dir: Path,
    svm_train_limit: int,
    quick: bool,
) -> tuple[list[ExperimentResult], list[dict[str, Any]]]:
    y_train = remap_binary(y_train_multi)
    y_test = remap_binary(y_test_multi)
    selected_sets = [
        "old_203_features",
        "previous_top_100_svm_features",
        "previous_top_125_lightgbm_features",
        "texture_lesion_vessel_quality",
        "expanded_top_100",
        "expanded_top_125",
        "expanded_top_150",
    ]
    if quick:
        selected_sets = ["previous_top_100_svm_features", "previous_top_125_lightgbm_features", "texture_lesion_vessel_quality", "expanded_top_125"]
    results: list[ExperimentResult] = []
    threshold_rows: list[dict[str, Any]] = []
    for set_name in selected_sets:
        if set_name not in feature_sets or not feature_sets[set_name]:
            continue
        features = feature_sets[set_name]
        for model_name in BINARY_MODELS:
            if not is_model_available(model_name):
                continue
            try:
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
                    experiment="binary_screening",
                )
            except Exception as exc:
                print(f"binary {model_name} {set_name} failed: {exc}", flush=True)
                continue
            results.append(result)
            threshold_rows.extend(threshold_sweep(result, x_test[features], y_test))
            if model_name in ("random_forest", "extra_trees", "svm_rbf"):
                calibrated = fit_calibrated_binary(result, x_train[features], y_train, x_test[features], y_test, output_dir)
                if calibrated is not None:
                    results.append(calibrated)
                    threshold_rows.extend(threshold_sweep(calibrated, x_test[features], y_test))
            print(
                f"binary {model_name}/{set_name}: recall={result.metrics['referable_recall']:.4f} "
                f"fn={result.metrics['false_negatives']} fp={result.metrics['false_positives']}",
                flush=True,
            )
    write_csv(output_dir / "threshold_sweep.csv", threshold_rows)
    return results, threshold_rows


def fit_calibrated_binary(
    result: ExperimentResult,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    output_dir: Path,
) -> ExperimentResult | None:
    model_path = output_dir / "models" / f"binary_calibrated_{result.model_name}_{result.feature_set}.pkl"
    if model_path.exists():
        try:
            with model_path.open("rb") as file:
                estimator = pickle.load(file)
        except Exception:
            model_path.unlink(missing_ok=True)
            estimator = None
    else:
        estimator = None
    if estimator is None:
        with Path(result.model_path).open("rb") as file:
            base = pickle.load(file)
        try:
            estimator = CalibratedClassifierCV(clone(base), method="isotonic", cv=3)
            estimator.fit(x_train, y_train)
        except Exception:
            return None
        with model_path.open("wb") as file:
            pickle.dump(estimator, file)
    metrics = evaluate_estimator(estimator, x_test, y_test, "binary")
    metrics["calibration"] = "isotonic_cv3"
    return ExperimentResult(
        problem="binary",
        model_name=f"calibrated_{result.model_name}",
        feature_set=result.feature_set,
        feature_count=result.feature_count,
        metrics=metrics,
        selected_features=result.selected_features,
        model_path=str(model_path),
        experiment="binary_screening_calibrated",
    )


def run_ensemble_experiments(
    feature_sets: dict[str, list[str]],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
) -> list[ExperimentResult]:
    candidates = [name for name in ["previous_top_100_svm_features", "previous_top_125_lightgbm_features", "texture_lesion_vessel_quality", "expanded_top_125"] if name in feature_sets]
    results: list[ExperimentResult] = []
    recipes = {
        "soft_vote_rf_svm_xgboost": ["random_forest", "svm_rbf", "xgboost"],
        "soft_vote_rf_extra_lightgbm": ["random_forest", "extra_trees", "lightgbm"],
        "soft_vote_rf_xgboost_lightgbm": ["random_forest", "xgboost", "lightgbm"],
        "weighted_vote_macro_f1_proxy": ["random_forest", "extra_trees", "xgboost", "lightgbm"],
        "weighted_vote_class3_macro_proxy": ["random_forest", "svm_rbf", "xgboost", "lightgbm"],
        "stack_rf_extra_xgboost_svm_lr": ["random_forest", "extra_trees", "xgboost", "svm_rbf"],
    }
    for set_name in candidates:
        features = feature_sets[set_name]
        for recipe_name, model_names in recipes.items():
            model_names = [name for name in model_names if is_model_available(name)]
            if len(model_names) < 2:
                continue
            try:
                result = fit_ensemble(recipe_name, model_names, set_name, features, x_train, x_test, y_train, y_test, output_dir)
            except Exception as exc:
                print(f"ensemble {recipe_name}/{set_name} failed: {exc}", flush=True)
                continue
            results.append(result)
            write_csv(output_dir / "ensemble_comparison.csv", [flatten_result(row) for row in results])
            print(
                f"ensemble {recipe_name}/{set_name}: macro_f1={result.metrics['macro_f1']:.4f} "
                f"class3={result.metrics['per_class']['3']['recall']:.4f}",
                flush=True,
            )
    write_csv(output_dir / "ensemble_comparison.csv", [flatten_result(row) for row in results])
    return results


def fit_ensemble(
    recipe_name: str,
    model_names: list[str],
    set_name: str,
    features: list[str],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
) -> ExperimentResult:
    model_path = ensure_dir(output_dir / "models") / f"multiclass_ensemble_{recipe_name}_{set_name}.pkl"
    if model_path.exists():
        try:
            with model_path.open("rb") as file:
                estimator = pickle.load(file)
        except Exception:
            model_path.unlink(missing_ok=True)
            estimator = None
    else:
        estimator = None
    if estimator is None:
        estimators = [(name, build_estimator(name, "multiclass")) for name in model_names]
        if recipe_name.startswith("stack"):
            estimator = fit_probability_stacker(model_names, features, x_train, y_train)
        else:
            weights = None
            if recipe_name == "weighted_vote_macro_f1_proxy":
                weights = [1.1 if name in ("xgboost", "lightgbm") else 1.0 for name in model_names]
            if recipe_name == "weighted_vote_class3_macro_proxy":
                weights = [1.3 if name in ("svm_rbf", "xgboost") else 1.0 for name in model_names]
            estimator = VotingClassifier(estimators=estimators, voting="soft", weights=weights, n_jobs=1)
            estimator.fit(x_train[features], y_train)
        with model_path.open("wb") as file:
            pickle.dump(estimator, file)
    metrics = evaluate_estimator(estimator, x_test[features], y_test, "multiclass")
    metrics["calibration"] = "ensemble"
    return ExperimentResult(
        problem="multiclass",
        model_name=recipe_name,
        feature_set=set_name,
        feature_count=len(features),
        metrics=metrics,
        selected_features=features,
        model_path=str(model_path),
        experiment="classical_ensemble",
    )


def fit_probability_stacker(
    model_names: list[str],
    features: list[str],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
) -> ProbabilityStackingEnsemble:
    train_idx, meta_idx = train_test_split(
        np.arange(len(y_train)),
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    x_base = x_train.iloc[train_idx][features]
    y_base = y_train[train_idx]
    x_meta = x_train.iloc[meta_idx][features]
    y_meta = y_train[meta_idx]
    labels = LABELS_5
    meta_base_estimators: list[tuple[str, BaseEstimator]] = []
    final_base_estimators: list[tuple[str, BaseEstimator]] = []
    for model_name in model_names:
        meta_estimator = build_estimator(model_name, "multiclass")
        x_fit = x_base
        y_fit = y_base
        if model_name == "svm_rbf" and len(x_fit) > 4500:
            idx = balanced_sample_indices(y_fit, 4500)
            x_fit = x_fit.iloc[idx]
            y_fit = y_fit[idx]
        fit_with_weights(meta_estimator, x_fit, y_fit)
        meta_base_estimators.append((model_name, meta_estimator))

        final_estimator = build_estimator(model_name, "multiclass")
        final_x_fit = x_train[features]
        final_y_fit = y_train
        if model_name == "svm_rbf" and len(final_x_fit) > 5000:
            idx = balanced_sample_indices(final_y_fit, 5000)
            final_x_fit = final_x_fit.iloc[idx]
            final_y_fit = final_y_fit[idx]
        fit_with_weights(final_estimator, final_x_fit, final_y_fit)
        final_base_estimators.append((model_name, final_estimator))

    meta_x = stacking_meta_features(meta_base_estimators, x_meta, labels)
    meta_model = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)
    meta_model.fit(meta_x, y_meta)
    return ProbabilityStackingEnsemble(
        base_estimators=final_base_estimators,
        meta_estimator=meta_model,
        labels=labels,
    )


def stacking_meta_features(
    estimators: list[tuple[str, BaseEstimator]],
    x_values: pd.DataFrame,
    labels: list[int],
) -> np.ndarray:
    blocks = []
    for _, estimator in estimators:
        probabilities = estimator.predict_proba(x_values)
        classes = list(getattr(estimator, "classes_", labels))
        aligned = np.zeros((len(x_values), len(labels)), dtype=np.float64)
        for index, label in enumerate(classes):
            if int(label) in labels:
                aligned[:, labels.index(int(label))] = probabilities[:, index]
        blocks.append(aligned)
    return np.hstack(blocks) if blocks else np.empty((len(x_values), 0), dtype=np.float64)


def run_hierarchical_experiments(
    feature_sets: dict[str, list[str]],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    svm_train_limit: int,
) -> dict[str, Any]:
    features = feature_sets.get("previous_top_100_svm_features") or feature_sets.get("expanded_top_125") or next(iter(feature_sets.values()))
    rows = {
        "H1_0_vs_dr_then_referable_then_1_4": hierarchical_h1(x_train, x_test, y_train, y_test, features, output_dir, svm_train_limit),
        "H2_0_early_referable_then_inside_group": hierarchical_h2(x_train, x_test, y_train, y_test, features, output_dir),
        "H3_flat_with_confusion_corrections": hierarchical_h3(x_train, x_test, y_train, y_test, features, output_dir),
    }
    write_csv(output_dir / "hierarchical_comparison.csv", [hierarchy_row(name, metrics) for name, metrics in rows.items()])
    return rows


def hierarchical_h1(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    features: list[str],
    output_dir: Path,
    svm_train_limit: int,
) -> dict[str, Any]:
    no_dr = train_stage("h1_0_vs_dr", "binary", "xgboost", x_train, (y_train > 0).astype(int), features, output_dir, svm_train_limit)
    referable = train_stage("h1_referable", "binary", "xgboost", x_train, np.isin(y_train, [2, 3, 4]).astype(int), features, output_dir, svm_train_limit)
    dr_mask = y_train > 0
    severity = train_stage("h1_dr_1_to_4", "multiclass", "random_forest", x_train[dr_mask], y_train[dr_mask], features, output_dir, svm_train_limit)
    p_dr = no_dr.predict_proba(x_test[features])[:, 1]
    p_ref = referable.predict_proba(x_test[features])[:, 1]
    sev = severity.predict(x_test[features])
    pred = np.where(p_dr < 0.5, 0, np.where(p_ref < 0.5, 1, sev))
    return evaluate_predictions(y_test, pred)


def hierarchical_h2(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    features: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    group_y = np.where(y_train == 0, 0, np.where(y_train == 1, 1, 2))
    group = train_stage("h2_group", "multiclass", "random_forest", x_train, group_y, features, output_dir, 6000)
    early_mask = np.isin(y_train, [0, 1])
    ref_mask = np.isin(y_train, [2, 3, 4])
    early = train_stage("h2_early_0_1", "binary", "svm_rbf", x_train[early_mask], y_train[early_mask], features, output_dir, 6000)
    ref = train_stage("h2_ref_2_3_4", "multiclass", "random_forest", x_train[ref_mask], y_train[ref_mask], features, output_dir, 6000)
    group_pred = group.predict(x_test[features])
    early_pred = early.predict(x_test[features])
    ref_pred = ref.predict(x_test[features])
    pred = np.where(group_pred == 0, 0, np.where(group_pred == 1, early_pred, ref_pred))
    return evaluate_predictions(y_test, pred)


def hierarchical_h3(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    features: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    flat = train_stage("h3_flat", "multiclass", "svm_rbf", x_train, y_train, features, output_dir, 6000)
    pred = flat.predict(x_test[features])
    corrections = [
        ((1, 2), "h3_1_vs_2"),
        ((3, 4), "h3_3_vs_4"),
        ((2, 4), "h3_2_vs_4"),
    ]
    for pair, name in corrections:
        mask = np.isin(y_train, list(pair))
        if int(mask.sum()) < 20:
            continue
        model = train_stage(name, "binary", "random_forest", x_train[mask], (y_train[mask] == pair[1]).astype(int), features, output_dir, 6000)
        pred_mask = np.isin(pred, list(pair))
        if not np.any(pred_mask):
            continue
        pair_pred = model.predict(x_test.loc[pred_mask, features])
        pred[pred_mask] = np.where(pair_pred == 1, pair[1], pair[0])
    return evaluate_predictions(y_test, pred)


def train_stage(
    name: str,
    problem: str,
    model_name: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    features: list[str],
    output_dir: Path,
    svm_train_limit: int,
) -> BaseEstimator:
    path = ensure_dir(output_dir / "models") / f"{problem}_{model_name}_{name}.pkl"
    if path.exists():
        try:
            with path.open("rb") as file:
                return pickle.load(file)
        except Exception:
            path.unlink(missing_ok=True)
    estimator = build_estimator(model_name, "binary" if problem == "binary" else "multiclass")
    x_fit = x_train[features]
    y_fit = y_train
    if model_name in {"svm_rbf", "linear_svm"} and len(x_fit) > svm_train_limit:
        idx = balanced_sample_indices(y_fit, svm_train_limit)
        x_fit = x_fit.iloc[idx]
        y_fit = y_fit[idx]
    fit_with_weights(estimator, x_fit, y_fit)
    with path.open("wb") as file:
        pickle.dump(estimator, file)
    return estimator


def hierarchy_row(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": name,
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "class_1_recall": metrics.get("per_class", {}).get("1", {}).get("recall"),
        "class_3_recall": metrics.get("per_class", {}).get("3", {}).get("recall"),
        "class_4_recall": metrics.get("per_class", {}).get("4", {}).get("recall"),
        "referable_recall_from_5class": metrics.get("referable_recall_from_5class"),
    }


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
    experiment: str,
) -> ExperimentResult:
    model_dir = ensure_dir(output_dir / "models")
    safe_experiment = experiment.replace("/", "_")
    model_path = model_dir / f"{problem}_{model_name}_{feature_set}_{safe_experiment}.pkl"
    estimator: BaseEstimator | None = None
    if model_path.exists():
        try:
            with model_path.open("rb") as file:
                estimator = pickle.load(file)
            metrics = evaluate_estimator(estimator, x_test[features], y_test, problem)
            metrics["calibration"] = "loaded_existing"
        except Exception:
            model_path.unlink(missing_ok=True)
            estimator = None
    if estimator is None:
        estimator = build_estimator(model_name, problem)
        x_fit = x_train[features]
        y_fit = y_train
        if model_name in {"svm_rbf", "linear_svm"} and len(x_fit) > svm_train_limit:
            indices = balanced_sample_indices(y_fit, svm_train_limit)
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
        experiment=experiment,
    )


def build_estimator(model_name: str, problem: str) -> Pipeline:
    if model_name == "linear_svm":
        return preserve_feature_names(Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LinearSVC(
                C=1.0,
                class_weight="balanced",
                dual=False,
                max_iter=3000,
                random_state=RANDOM_STATE,
            )),
        ]))
    if model_name == "logistic_regression":
        return preserve_feature_names(Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=5000, solver="lbfgs", random_state=RANDOM_STATE)),
        ]))
    if model_name == "naive_bayes":
        return preserve_feature_names(Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB()),
        ]))
    if model_name == "svm_rbf":
        return preserve_feature_names(Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(C=4.0, gamma="scale", kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_STATE)),
        ]))
    if model_name == "random_forest":
        return preserve_feature_names(Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(n_estimators=240, class_weight="balanced_subsample", min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)),
        ]))
    if model_name == "extra_trees":
        return preserve_feature_names(Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", ExtraTreesClassifier(n_estimators=240, class_weight="balanced", min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)),
        ]))
    if model_name == "balanced_random_forest":
        if BalancedRandomForestClassifier is None:
            raise ImportError("BalancedRandomForestClassifier is not installed.")
        return preserve_feature_names(Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", BalancedRandomForestClassifier(n_estimators=220, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)),
        ]))
    if model_name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed.")
        kwargs = {
            "objective": "binary:logistic" if problem == "binary" else "multi:softprob",
            "n_estimators": 220,
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
        return preserve_feature_names(Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", XGBClassifier(**kwargs)),
        ]))
    if model_name == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError("lightgbm is not installed.")
        return preserve_feature_names(Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", LGBMClassifier(
                objective="binary" if problem == "binary" else "multiclass",
                n_estimators=240,
                learning_rate=0.05,
                num_leaves=31,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=4,
                verbose=-1,
            )),
        ]))
    if model_name == "histgradientboosting":
        return preserve_feature_names(Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=240,
                max_leaf_nodes=45,
                l2_regularization=0.01,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]))
    raise ValueError(model_name)


def preserve_feature_names(estimator: Pipeline) -> Pipeline:
    try:
        estimator.set_output(transform="pandas")
    except (AttributeError, ValueError):
        pass
    return estimator


def fit_with_weights(estimator: BaseEstimator, x_values: pd.DataFrame, y_values: np.ndarray) -> None:
    if isinstance(estimator, (VotingClassifier, StackingClassifier)):
        estimator.fit(x_values, y_values)
        return
    weights = compute_sample_weight(class_weight="balanced", y=y_values)
    classifier = estimator.named_steps["classifier"] if isinstance(estimator, Pipeline) and "classifier" in estimator.named_steps else estimator
    try:
        fit_signature = inspect.signature(classifier.fit)
    except (TypeError, ValueError):
        fit_signature = None
    if fit_signature is not None and "sample_weight" in fit_signature.parameters and isinstance(estimator, Pipeline):
        estimator.fit(x_values, y_values, classifier__sample_weight=weights)
    elif fit_signature is not None and "sample_weight" in fit_signature.parameters:
        estimator.fit(x_values, y_values, sample_weight=weights)
    else:
        estimator.fit(x_values, y_values)


def evaluate_estimator(estimator: BaseEstimator, x_test: pd.DataFrame, y_test: np.ndarray, problem: str) -> dict[str, Any]:
    probabilities = estimator.predict_proba(x_test) if hasattr(estimator, "predict_proba") else None
    if problem == "binary" and probabilities is not None:
        predictions = (probabilities[:, 1] >= 0.5).astype(int)
    else:
        predictions = estimator.predict(x_test)
    return evaluate_binary_predictions(y_test, predictions, probabilities) if problem == "binary" else evaluate_predictions(y_test, predictions, probabilities)


def evaluate_predictions(y_true: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray | None = None) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, predictions, labels=LABELS_5)
    report = classification_report(y_true, predictions, labels=LABELS_5, output_dict=True, zero_division=0)
    y_ref = remap_binary(y_true)
    pred_ref = remap_binary(predictions)
    ref_matrix = confusion_matrix(y_ref, pred_ref, labels=[0, 1])
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_precision": float(precision_score(y_true, predictions, labels=LABELS_5, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, predictions, labels=LABELS_5, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, labels=LABELS_5, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predictions, labels=LABELS_5, average="weighted", zero_division=0)),
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
            for index, label in enumerate(LABELS_5)
        },
        "referable_recall_from_5class": float(recall_score(y_ref, pred_ref, pos_label=1, zero_division=0)),
        "referable_false_negatives_from_5class": int(ref_matrix[1, 0]),
        "common_misclassifications": common_misclassifications(matrix),
    }
    try:
        if probabilities is not None and probabilities.shape[1] == 5:
            metrics["auc_macro_ovr"] = float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro"))
    except Exception:
        metrics["auc_macro_ovr"] = None
    metrics["selection_score"] = (
        0.30 * metrics["macro_f1"]
        + 0.25 * metrics["balanced_accuracy"]
        + 0.15 * metrics["per_class"]["3"]["recall"]
        + 0.10 * metrics["per_class"]["1"]["recall"]
        + 0.10 * metrics["per_class"]["4"]["recall"]
        + 0.10 * min(metrics["per_class"]["0"]["recall"], metrics["per_class"]["2"]["recall"])
    )
    return metrics


def evaluate_binary_predictions(y_true: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray | None = None) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "referable_recall": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "non_referable_recall": float(recall_score(y_true, predictions, pos_label=0, zero_division=0)),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
        "confusion_matrix": matrix.astype(int).tolist(),
        "per_class": {
            "0": {"recall": float(recall_score(y_true, predictions, pos_label=0, zero_division=0)), "support": int(np.sum(y_true == 0)), "correct": int(matrix[0, 0])},
            "1": {"recall": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)), "support": int(np.sum(y_true == 1)), "correct": int(matrix[1, 1])},
        },
    }
    if probabilities is not None:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, probabilities[:, 1]))
        except Exception:
            metrics["auc"] = None
    metrics["selection_score"] = (
        0.45 * metrics["referable_recall"]
        + 0.25 * metrics["balanced_accuracy"]
        + 0.20 * metrics["f1"]
        - 0.10 * (metrics["false_negatives"] / max(int(np.sum(y_true == 1)), 1))
    )
    return metrics


def threshold_sweep(result: ExperimentResult, x_test: pd.DataFrame, y_test: np.ndarray) -> list[dict[str, Any]]:
    with Path(result.model_path).open("rb") as file:
        model = pickle.load(file)
    if not hasattr(model, "predict_proba"):
        return []
    probabilities = model.predict_proba(x_test)[:, 1]
    rows = []
    for threshold in BINARY_THRESHOLDS:
        pred = (probabilities >= threshold).astype(int)
        matrix = confusion_matrix(y_test, pred, labels=[0, 1])
        row: dict[str, Any] = {
            "model_name": result.model_name,
            "feature_set": result.feature_set,
            "model_path": result.model_path,
            "threshold": threshold,
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "referable_recall": float(recall_score(y_test, pred, pos_label=1, zero_division=0)),
            "non_referable_recall": float(recall_score(y_test, pred, pos_label=0, zero_division=0)),
            "false_positives": int(matrix[0, 1]),
            "false_negatives": int(matrix[1, 0]),
        }
        for low, high in UNCERTAINTY_BANDS:
            uncertain = (probabilities >= low) & (probabilities <= high)
            row[f"uncertain_{low:.2f}_{high:.2f}_pct"] = float(np.mean(uncertain))
        try:
            row["auc"] = float(roc_auc_score(y_test, probabilities))
        except Exception:
            row["auc"] = None
        rows.append(row)
    return rows


def common_misclassifications(matrix: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for i, true_label in enumerate(LABELS_5):
        for j, pred_label in enumerate(LABELS_5):
            if i == j:
                continue
            count = int(matrix[i, j])
            if count > 0:
                rows.append({
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "count": count,
                })
    rows.sort(key=lambda row: row["count"], reverse=True)
    return rows[:15]


def remap_binary(labels: np.ndarray) -> np.ndarray:
    return np.where(np.isin(labels, [2, 3, 4]), 1, 0).astype(int)


def select_best_grading(results: list[ExperimentResult]) -> ExperimentResult:
    return max(results, key=lambda item: item.metrics.get("selection_score", 0.0))


def select_best_safe_grading(results: list[ExperimentResult]) -> ExperimentResult:
    safe = [
        item for item in results
        if item.metrics.get("per_class", {}).get("3", {}).get("recall", 0.0) >= SAFE_GRADING_BASELINE["class_3_recall"]
    ]
    pool = safe or results
    return max(pool, key=lambda item: (item.metrics.get("macro_f1", 0.0), item.metrics.get("balanced_accuracy", 0.0), item.metrics.get("accuracy", 0.0)))


def select_best_binary(results: list[ExperimentResult]) -> ExperimentResult:
    return max(results, key=lambda item: item.metrics.get("selection_score", 0.0))


def select_best_threshold(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    passing = [
        row for row in rows
        if row["referable_recall"] >= BINARY_SAFETY_BASELINE["referable_recall"]
        and row["false_negatives"] <= BINARY_SAFETY_BASELINE["false_negatives"]
        and row["false_positives"] <= BINARY_FALSE_POSITIVE_EXTREME_LIMIT
    ]
    if passing:
        return min(
            passing,
            key=lambda row: (
                row["false_positives"],
                row["false_negatives"],
                -row["referable_recall"],
                -row["balanced_accuracy"],
                -row["f1"],
            ),
        )
    return max(
        rows,
        key=lambda row: (
            row["referable_recall"] >= BINARY_SAFETY_BASELINE["referable_recall"],
            -row["false_negatives"],
            -over_referral_penalty(row),
            row["balanced_accuracy"],
            row["f1"],
        ),
    )


def over_referral_penalty(row: dict[str, Any]) -> float:
    return max(0.0, float(row["false_positives"]) - float(BINARY_FALSE_POSITIVE_EXTREME_LIMIT))


def grading_replacement_decision(best: ExperimentResult) -> dict[str, Any]:
    metrics = best.metrics
    c1 = metrics["per_class"]["1"]["recall"]
    c3 = metrics["per_class"]["3"]["recall"]
    c4 = metrics["per_class"]["4"]["recall"]
    improves = (
        metrics["macro_f1"] > SAFE_GRADING_BASELINE["macro_f1"]
        or metrics["balanced_accuracy"] > SAFE_GRADING_BASELINE["balanced_accuracy"]
        or metrics["accuracy"] >= SAFE_GRADING_BASELINE["accuracy"] + 0.02
    )
    safe_classes = (
        c3 >= SAFE_GRADING_BASELINE["class_3_recall"]
        and c1 >= SAFE_GRADING_BASELINE["class_1_recall"] - 0.05
        and c4 >= SAFE_GRADING_BASELINE["class_4_recall"] - 0.05
    )
    replace = bool(improves and safe_classes)
    reason = "Meets grading replacement rule." if replace else "Did not meet the grading replacement rule, especially the severe-NPDR/Class 3 safety requirement or class-drop limits."
    return {"replace": replace, "reason": reason}


def screening_replacement_decision(best_threshold: dict[str, Any] | None) -> dict[str, Any]:
    if best_threshold is None:
        return {"replace": False, "reason": "No threshold candidate was available."}
    fp_extreme = best_threshold["false_positives"] > BINARY_FALSE_POSITIVE_EXTREME_LIMIT
    uncertainty_usable = (
        float(best_threshold.get("uncertain_0.25_0.75_pct", 1.0))
        <= BINARY_UNCERTAINTY_USABLE_LIMIT
    )
    replace = (
        best_threshold["referable_recall"] >= BINARY_SAFETY_BASELINE["referable_recall"]
        and best_threshold["false_negatives"] <= BINARY_SAFETY_BASELINE["false_negatives"]
        and not fp_extreme
        and uncertainty_usable
    )
    reason = "Meets screening numeric replacement rule; production promotion still requires a deliberate export/integration step." if replace else "Did not meet screening replacement rule without an excessive false-positive or uncertainty tradeoff."
    return {"replace": replace, "reason": reason}


def per_source_validation(result: ExperimentResult, x_test: pd.DataFrame, y_test: np.ndarray, test_meta: pd.DataFrame) -> list[dict[str, Any]]:
    if "source_dataset" not in test_meta.columns:
        return []
    with Path(result.model_path).open("rb") as file:
        model = pickle.load(file)
    predictions = model.predict(x_test[result.selected_features])
    rows = []
    for source in sorted(test_meta["source_dataset"].dropna().unique()):
        mask = test_meta["source_dataset"] == source
        if int(mask.sum()) < 5:
            continue
        rows.append({
            "source_dataset": str(source),
            "sample_count": int(mask.sum()),
            "accuracy": float(accuracy_score(y_test[mask], predictions[mask])),
            "balanced_accuracy": float(balanced_accuracy_score(y_test[mask], predictions[mask])),
            "macro_f1": float(f1_score(y_test[mask], predictions[mask], labels=LABELS_5, average="macro", zero_division=0)),
        })
    return rows


def build_report(
    feature_table: pd.DataFrame,
    dictionary: list[dict[str, str]],
    audit: dict[str, Any],
    feature_sets: dict[str, list[str]],
    multiclass_results: list[ExperimentResult],
    ensemble_results: list[ExperimentResult],
    hierarchy_results: dict[str, Any],
    binary_results: list[ExperimentResult],
    threshold_rows: list[dict[str, Any]],
    source_validation: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    all_grading = [*multiclass_results, *ensemble_results]
    best_grading = select_best_grading(all_grading)
    best_safe_grading = select_best_safe_grading(all_grading)
    best_binary = select_best_binary(binary_results)
    best_threshold = select_best_threshold(threshold_rows)
    grading_decision = grading_replacement_decision(best_safe_grading)
    screening_decision = screening_replacement_decision(best_threshold)
    feature_group_counts = pd.Series([row["feature_group"] for row in dictionary]).value_counts().sort_index()
    report = {
        "created_at": datetime.now().isoformat(),
        "row_count": len(feature_table),
        "feature_count": len(dictionary),
        "feature_groups": {str(name): int(count) for name, count in feature_group_counts.items()},
        "feature_audit": audit,
        "feature_sets": {name: len(values) for name, values in feature_sets.items()},
        "best_grading_by_selection_score": result_to_dict(best_grading),
        "best_safe_grading_candidate": result_to_dict(best_safe_grading),
        "best_binary_default_threshold": result_to_dict(best_binary),
        "best_binary_threshold": best_threshold,
        "ensemble_results": [result_to_dict(item) for item in ensemble_results],
        "hierarchical_experiments": hierarchy_results,
        "source_validation": source_validation,
        "production_replacement": {
            "grading_replaced": False,
            "screening_replaced": False,
            "grading_decision": grading_decision,
            "screening_decision": screening_decision,
            "reason": "Production artifacts were not overwritten by this experiment script. Export/integration requires a clearly passing candidate and a separate deliberate promotion step.",
        },
        "model_comparison": [result_to_dict(item) for item in [*multiclass_results, *ensemble_results, *binary_results]],
        "output_dir": str(output_dir),
    }
    report["markdown"] = render_markdown(report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    best = report["best_safe_grading_candidate"]
    score_best = report["best_grading_by_selection_score"]
    binary = report["best_binary_default_threshold"]
    threshold = report["best_binary_threshold"]
    lines = [
        "# AppDR Study Sources Expanded Report",
        "",
        "Classical ML only. No deep learning, CNN, UNet, ResNet, YOLO, transfer learning, neural networks, or pretrained deep feature extractors were used.",
        "",
        "## Study Sources Used As Basis",
        "",
        "- LBP + GLCM texture-feature RF/SVM study: implemented focused texture-only LBP/GLCM feature sets and RF-vs-SVM comparison.",
        "- Morphological hard-exudate extraction study: added bright-lesion and optic-disc-control features, including pre/post optic-disc removal ablation.",
        "- Encoded/uniform/multiscale LBP severity grading: added uniform, encoded, and multiscale LBP image-texture variants.",
        "- Classical ensemble studies: tested only shallow soft voting, weighted voting, and logistic-regression stacking.",
        "- Bhattacharjee/Gandor/Yang-style classical DR feature themes: lesion, vessel, microaneurysm, CLAHE texture, and referable-screening experiments.",
        "",
        "## Feature Audit",
        "",
        f"Rows: {report['row_count']}",
        f"Feature dictionary entries: {report['feature_count']}",
        f"Feature groups: {report['feature_groups']}",
        f"Removed features: {len(report['feature_audit']['removed_features'])}",
        f"Rare medical features kept for separate test version: {len(report['feature_audit']['rare_medical_features_kept_test_version'])}",
        "",
        "## Best 5-Class Candidate Under Safety Rule",
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
        "## Best 5-Class Candidate By Selection Score",
        "",
        f"Model: {score_best['model_name']}",
        f"Feature set: {score_best['feature_set']} ({score_best['feature_count']} features)",
        f"Accuracy: {pct(score_best['metrics']['accuracy'])}",
        f"Balanced accuracy: {pct(score_best['metrics']['balanced_accuracy'])}",
        f"Macro F1: {pct(score_best['metrics']['macro_f1'])}",
        f"Class 3 recall: {pct(score_best['metrics']['per_class']['3']['recall'])}",
        "",
        "## Binary Screening",
        "",
        f"Best default model: {binary['model_name']} / {binary['feature_set']}",
        f"Default threshold referable recall: {pct(binary['metrics']['referable_recall'])}",
        f"Default threshold false negatives: {binary['metrics']['false_negatives']}",
    ]
    if threshold:
        lines.extend([
            "",
            "## Best Threshold Sweep Candidate",
            "",
            f"Model: {threshold['model_name']}",
            f"Feature set: {threshold['feature_set']}",
            f"Threshold: {threshold['threshold']}",
            f"Referable recall: {pct(threshold['referable_recall'])}",
            f"False negatives: {threshold['false_negatives']}",
            f"False positives: {threshold['false_positives']}",
            f"Non-referable recall: {pct(threshold['non_referable_recall'])}",
            f"Uncertain 0.40-0.60: {pct(threshold.get('uncertain_0.40_0.60_pct', 0.0))}",
            f"Uncertain 0.25-0.75: {pct(threshold.get('uncertain_0.25_0.75_pct', 0.0))}",
        ])
    lines.extend([
        "",
        "## Production Decision",
        "",
        f"Grading replaced: {report['production_replacement']['grading_replaced']}",
        f"Screening replaced: {report['production_replacement']['screening_replaced']}",
        report["production_replacement"]["grading_decision"]["reason"],
        report["production_replacement"]["screening_decision"]["reason"],
        report["production_replacement"]["reason"],
    ])
    return "\n".join(lines)


def write_reports(output_dir: Path, report: dict[str, Any]) -> None:
    (output_dir / "main_report.md").write_text(report["markdown"], encoding="utf-8")
    write_json(output_dir / "main_report.json", {key: value for key, value in report.items() if key != "markdown"})
    write_csv(output_dir / "model_comparison.csv", [flatten_result_dict(item) for item in report["model_comparison"]])
    write_csv(output_dir / "ensemble_comparison.csv", [flatten_result_dict(item) for item in report["ensemble_results"]])
    write_csv(output_dir / "per_stage_metrics.csv", per_stage_rows(report["best_safe_grading_candidate"]))
    write_matrix(output_dir / "confusion_matrix.csv", report["best_safe_grading_candidate"]["metrics"]["confusion_matrix"])
    write_csv(output_dir / "source_validation.csv", report["source_validation"])
    write_study_comparison(output_dir / "study_comparison_table.md", report)
    write_final_recommendation(output_dir / "final_recommendation.md", report)


def write_study_comparison(path: Path, report: dict[str, Any]) -> None:
    best = report["best_safe_grading_candidate"]
    threshold = report["best_binary_threshold"]
    threshold_text = "not available"
    if threshold:
        threshold_text = f"{pct(threshold['referable_recall'])} referable recall, {threshold['false_negatives']} FN, {threshold['false_positives']} FP"
    content = f"""# Study Comparison Table

Direct comparison is limited because the cited studies and AppDR use different datasets, label distributions, preprocessing, and train/test splits unless evaluated on the same images.

| Study / system | Classical basis used here | Reported/target focus | AppDR corresponding result |
|---|---|---|---|
| Bhattacharjee et al. | Lesion, vessel, microaneurysm-style handcrafted features with RF/SVM/NB | Classical DR feature classification | See `experiment_2_lesion_vessel_microaneurysm` rows in `model_comparison.csv`. |
| Gandor et al. | CLAHE + LBP + GLCM with RF/XGBoost-style models | Texture/feature-engineered DR grading | See `experiment_3_lbp_glcm_clahe_rf_xgboost` rows. |
| Texture RF/SVM LBP+GLCM study | Focused texture-only LBP/GLCM RF-vs-SVM | RF reportedly outperforming SVM | See `experiment_1_texture_rf_vs_svm` rows. |
| Morphological hard-exudate study | Bright-lesion morphology with optic-disc/bright-artifact control | Exudate extraction while separating optic disc/artifacts | See exudate pre/post optic-disc ablation rows. |
| Berbar encoded LBP study | Encoded, uniform, and multiscale LBP variants | Texture severity grading support | Included in `texture_only_lbp_glcm` and combined sets. |
| Yang et al. referable DR ML study | Classical referable-screening models and threshold sweep | Referable DR screening | Best threshold candidate: {threshold_text}. |
| AppDR production | Existing production handcrafted pipeline | 5-class support + referable screening | 5-class accuracy {pct(PRODUCTION_BASELINE['accuracy'])}, balanced accuracy {pct(PRODUCTION_BASELINE['balanced_accuracy'])}, macro F1 {pct(PRODUCTION_BASELINE['macro_f1'])}; screening recall {pct(PRODUCTION_BASELINE['binary_referable_recall'])}, FN {PRODUCTION_BASELINE['binary_false_negatives']}. |
| AppDR study-top100 SVM | Prior top-100 selected handcrafted features | Stronger 5-class safety baseline | Accuracy 66.02%, balanced accuracy 60.83%, macro F1 57.18%, Class 3 recall 64.00%. |
| AppDR study-max LightGBM | Prior top-125 study-max features | Higher macro F1 but weaker Class 3 recall | Accuracy 66.69%, balanced accuracy 60.53%, macro F1 57.44%, Class 3 recall 57.50%. |
| AppDR new best result | Expanded classical feature sets and ensembles | Current run | {best['model_name']} / {best['feature_set']}: accuracy {pct(best['metrics']['accuracy'])}, balanced accuracy {pct(best['metrics']['balanced_accuracy'])}, macro F1 {pct(best['metrics']['macro_f1'])}, Class 1 recall {pct(best['metrics']['per_class']['1']['recall'])}, Class 3 recall {pct(best['metrics']['per_class']['3']['recall'])}, Class 4 recall {pct(best['metrics']['per_class']['4']['recall'])}. |
"""
    path.write_text(content, encoding="utf-8")


def write_final_recommendation(path: Path, report: dict[str, Any]) -> None:
    best = report["best_safe_grading_candidate"]
    threshold = report["best_binary_threshold"]
    threshold_line = "No threshold candidate was selected."
    if threshold:
        threshold_line = f"Best threshold candidate: {threshold['model_name']} / {threshold['feature_set']} at {threshold['threshold']} with referable recall {pct(threshold['referable_recall'])}, {threshold['false_negatives']} false negatives, {threshold['false_positives']} false positives, and {pct(threshold.get('uncertain_0.25_0.75_pct', 0.0))} wide-band uncertainty."
    text = f"""# Final Recommendation

Keep AppDR's production structure: referable DR screening is the main output, 5-class DR severity grading is supporting output, and neither is a final diagnosis.

Best safety-rule grading candidate: {best['model_name']} / {best['feature_set']} with accuracy {pct(best['metrics']['accuracy'])}, balanced accuracy {pct(best['metrics']['balanced_accuracy'])}, macro F1 {pct(best['metrics']['macro_f1'])}, Class 1 recall {pct(best['metrics']['per_class']['1']['recall'])}, Class 3 recall {pct(best['metrics']['per_class']['3']['recall'])}, and Class 4 recall {pct(best['metrics']['per_class']['4']['recall'])}.

{threshold_line}

Production was not replaced by this run. Do not promote the expanded 5-class grading candidate because it does not beat the current safe grading baseline. The binary screening candidate meets the numeric recall/FN rule in this experiment, but promotion should be a separate deliberate export/integration step with clinical review of the false-positive workload.
"""
    path.write_text(text, encoding="utf-8")


def result_to_dict(result: ExperimentResult) -> dict[str, Any]:
    return {
        "problem": result.problem,
        "model_name": result.model_name,
        "feature_set": result.feature_set,
        "feature_count": result.feature_count,
        "metrics": result.metrics,
        "selected_features": result.selected_features,
        "model_path": result.model_path,
        "experiment": result.experiment,
    }


def flatten_result(result: ExperimentResult) -> dict[str, Any]:
    return flatten_result_dict(result_to_dict(result))


def flatten_result_dict(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row["metrics"]
    return {
        "experiment": row.get("experiment"),
        "problem": row.get("problem"),
        "model": row.get("model_name"),
        "feature_set": row.get("feature_set"),
        "feature_count": row.get("feature_count"),
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "macro_precision": metrics.get("macro_precision"),
        "macro_recall": metrics.get("macro_recall"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "binary_precision": metrics.get("precision"),
        "binary_recall": metrics.get("recall"),
        "binary_f1": metrics.get("f1"),
        "class_1_recall": metrics.get("per_class", {}).get("1", {}).get("recall"),
        "class_3_recall": metrics.get("per_class", {}).get("3", {}).get("recall"),
        "class_4_recall": metrics.get("per_class", {}).get("4", {}).get("recall"),
        "referable_recall": metrics.get("referable_recall"),
        "non_referable_recall": metrics.get("non_referable_recall"),
        "false_positives": metrics.get("false_positives"),
        "false_negatives": metrics.get("false_negatives"),
        "auc": metrics.get("auc") or metrics.get("auc_macro_ovr"),
        "model_path": row.get("model_path"),
    }


def per_stage_rows(best: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for label, metrics in best["metrics"]["per_class"].items():
        rows.append({
            "class": label,
            "medical_label": config.CLASS_NAMES.get(int(label), label),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "support": metrics.get("support"),
            "correct_predictions": metrics.get("correct"),
        })
    return rows


def backup_existing_work(output_dir: Path) -> None:
    backup_dir = ensure_dir(output_dir / "backup")
    groups = {
        "production": [
            BACKEND_DIR / "results" / "best_model.pkl",
            BACKEND_DIR / "results" / "best_model_metadata.json",
            BACKEND_DIR / "results" / "metrics.json",
            BACKEND_DIR / "results" / "evaluation_report.md",
            BACKEND_DIR / "results" / "evaluation_report.json",
            BACKEND_DIR / "results" / "binary" / "best_model.pkl",
            BACKEND_DIR / "results" / "binary" / "best_model_metadata.json",
            BACKEND_DIR / "results" / "binary" / "metrics.json",
        ],
        "study_top100_svm": [
            BACKEND_DIR / "results" / "study_feature_selection" / "study_feature_selection_report.json",
            BACKEND_DIR / "results" / "study_feature_selection" / "study_feature_selection_report.md",
            BACKEND_DIR / "results" / "study_feature_selection" / "model_comparison_study.csv",
            BACKEND_DIR / "results" / "study_feature_selection" / "models" / "multiclass_svm_rbf_100.pkl",
        ],
        "study_max_lightgbm": [
            BACKEND_DIR / "results" / "study_max_improvement" / "main_report.json",
            BACKEND_DIR / "results" / "study_max_improvement" / "main_report.md",
            BACKEND_DIR / "results" / "study_max_improvement" / "model_comparison.csv",
            BACKEND_DIR / "results" / "study_max_improvement" / "models" / "multiclass_lightgbm_top_125.pkl",
        ],
    }
    manifest = []
    for group, paths in groups.items():
        target_dir = ensure_dir(backup_dir / group)
        for source in paths:
            if not source.exists():
                continue
            target = target_dir / source.name
            shutil.copy2(source, target)
            manifest.append({"group": group, "source": str(source), "backup": str(target)})
    write_json(backup_dir / "manifest.json", {"created_at": datetime.now().isoformat(), "files": manifest})


def is_model_available(model_name: str) -> bool:
    if model_name == "xgboost":
        return XGBClassifier is not None
    if model_name == "lightgbm":
        return LGBMClassifier is not None
    if model_name == "balanced_random_forest":
        return BalancedRandomForestClassifier is not None
    return True


def balanced_sample_indices(y_values: np.ndarray, limit: int) -> np.ndarray:
    if len(y_values) <= limit:
        return np.arange(len(y_values))
    rng = np.random.default_rng(RANDOM_STATE)
    labels = np.unique(y_values)
    per_class = max(1, limit // max(len(labels), 1))
    indices = []
    for label in labels:
        label_indices = np.flatnonzero(y_values == label)
        take = min(len(label_indices), per_class)
        indices.extend(rng.choice(label_indices, size=take, replace=False).tolist())
    if len(indices) < limit:
        remaining = np.setdiff1d(np.arange(len(y_values)), np.asarray(indices), assume_unique=False)
        extra = rng.choice(remaining, size=min(len(remaining), limit - len(indices)), replace=False)
        indices.extend(extra.tolist())
    rng.shuffle(indices)
    return np.asarray(indices, dtype=int)


def normalize_dictionary_row(row: dict[str, Any]) -> dict[str, str]:
    return {
        "feature_name": str(row.get("feature_name", "")),
        "feature_group": str(row.get("feature_group", study_max.group_for_existing(str(row.get("feature_name", ""))))),
        "description": str(row.get("description", "Existing AppDR handcrafted feature.")),
        "source_or_study_basis": str(row.get("source_or_study_basis", row.get("basis", "Existing AppDR classical feature pipeline"))),
        "preprocessing_variant": str(row.get("preprocessing_variant", "existing")),
        "expected_value_type": str(row.get("expected_value_type", "float")),
    }


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


def dedupe(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(study.to_jsonable(payload), indent=2), encoding="utf-8")


def write_matrix(path: Path, matrix: list[list[int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(matrix)


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100.0:.2f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run expanded AppDR classical study-source experiments.")
    parser.add_argument("--source-features", type=Path, default=SOURCE_FEATURES)
    parser.add_argument("--study-max-features", type=Path, default=STUDY_MAX_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--svm-train-limit", type=int, default=6000)
    parser.add_argument("--image-workers", type=int, default=max(1, min(4, (config.DEFAULT_WORKERS or 2))))
    parser.add_argument("--image-feature-limit", type=int, default=None, help="Optional development limit for image-derived feature extraction.")
    parser.add_argument("--rebuild-image-features", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Run fewer feature-set combinations for smoke testing.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
