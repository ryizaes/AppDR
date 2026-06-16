import os
from pathlib import Path


# Central configuration for the current production supervised handcrafted-feature
# DR pipeline. Future study-backed CNN, image-input, or hybrid experiments should
# use separate artifacts and reports until they pass validation and deployment
# checks.

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MODEL_MODE = os.getenv("MODEL_MODE", "ophthalmologist_demo_hybrid").strip().lower()
DEMO_OPHTHALMOLOGIST_DIR = RESULTS_DIR / "demo_ophthalmologist_update"
DEMO_HYBRID_MODE = "ophthalmologist_demo_hybrid"

FEATURES_CSV = BASE_DIR / "features.csv"
FAILED_SAMPLES_TXT = BASE_DIR / "failed_samples.txt"
BEST_MODEL_PATH = RESULTS_DIR / "best_model.pkl"
METADATA_PATH = RESULTS_DIR / "best_model_metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
DEFAULT_WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))

CLASS_LABELS = [0, 1, 2, 3, 4]
CLASS_NAMES = {
    0: "No apparent diabetic retinopathy",
    1: "Mild non-proliferative diabetic retinopathy",
    2: "Moderate non-proliferative diabetic retinopathy",
    3: "Severe non-proliferative diabetic retinopathy",
    4: "Proliferative diabetic retinopathy",
}

# The dataset uses five DR grades encoded as 0-4. These labels keep the model's
# exact class mapping while avoiding vague UI wording such as "Stage 1".
CLASS_EXPLANATIONS = {
    0: (
        "The model did not find apparent diabetic-retinopathy features in the "
        "extracted retinal measurements."
    ),
    1: (
        "The model found findings compatible with mild non-proliferative "
        "diabetic retinopathy. Mild NPDR is commonly associated with "
        "microaneurysms only when visible and feature-supported. This is still "
        "a screening result and needs professional review."
    ),
    2: (
        "The model found findings compatible with moderate non-proliferative "
        "diabetic retinopathy. This may involve hemorrhages, exudates, cotton "
        "wool spots, or other NPDR signs, and is treated as referable screening "
        "support."
    ),
    3: (
        "The model found findings compatible with severe non-proliferative "
        "diabetic retinopathy. This may indicate an advanced NPDR pattern; "
        "ophthalmologist confirmation is required."
    ),
    4: (
        "The model found findings compatible with proliferative diabetic "
        "retinopathy. This is a serious referable DR category; urgent "
        "ophthalmologist confirmation is recommended."
    ),
}

STAGE_FOLDERS = {
    "Stage_0": 0,
    "Stage_1": 1,
    "Stage_2": 2,
    "Stage_3": 3,
    "Stage_4": 4,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
APTOS_TRAIN_CSV = "train.csv"
APTOS_TEST_CSV = "test.csv"
APTOS_TRAIN_IMAGES_DIR = "train_images"
APTOS_TEST_IMAGES_DIR = "test_images"
APTOS_IMAGE_ID_COLUMN = "id_code"
APTOS_LABEL_COLUMN = "diagnosis"

LEGACY_FEATURE_NAMES = [
    "ma_count",
    "exudate_area",
    "vessel_density",
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
]

CORE_FEATURE_NAMES = [
    "ma_count",
    "ma_area",
    "ma_density",
    "ma_mean_area",
    "exudate_count",
    "exudate_area",
    "exudate_density",
    "exudate_mean_area",
    "vessel_density",
    "vessel_skeleton_length",
    "vessel_endpoint_count",
    "vessel_branchpoint_count",
    "vessel_tortuosity_mean",
    "vessel_tortuosity_max",
    "vessel_tortuosity_std",
    "lab_b_mean",
    "lab_b_std",
    "lab_b_exudate_mean",
    "lab_b_exudate_std",
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_contrast_0",
    "glcm_contrast_45",
    "glcm_contrast_90",
    "glcm_contrast_135",
    "glcm_energy_0",
    "glcm_energy_45",
    "glcm_energy_90",
    "glcm_energy_135",
]

VESSEL_FEATURE_NAMES = [
    "vessel_area_ratio",
    "vessel_length",
    "vessel_branching_count",
    "vessel_average_width",
    "vessel_fragmentation_index",
    "vessel_complexity_score",
    "vessel_curvature_mean",
    "vessel_curvature_std",
    "vessel_curvature_max",
]

HEMORRHAGE_FEATURE_NAMES = [
    "hemorrhage_count",
    "hemorrhage_area",
    "hemorrhage_largest_area",
    "hemorrhage_mean_area",
    "hemorrhage_density",
    "hemorrhage_retina_affected_pct",
]

MA_ADVANCED_FEATURE_NAMES = [
    "ma_max_area",
    "ma_superior_count",
    "ma_inferior_count",
    "ma_nasal_count",
    "ma_temporal_count",
    "ma_distance_to_optic_disc_mean",
    "ma_distance_to_optic_disc_min",
    "ma_density_per_retinal_area",
]

EXUDATE_ADVANCED_FEATURE_NAMES = [
    f"{lesion}_{metric}"
    for lesion in ("hard_exudate", "soft_exudate")
    for metric in (
        "count",
        "area",
        "mean_area",
        "max_area",
        "brightness_mean",
        "brightness_std",
        "texture_mean",
        "texture_std",
        "coverage_pct",
        "distance_to_macula_mean",
        "distance_to_optic_disc_mean",
    )
]

COTTON_WOOL_FEATURE_NAMES = [
    "cotton_wool_count",
    "cotton_wool_area",
    "cotton_wool_mean_area",
    "cotton_wool_circularity_mean",
    "cotton_wool_solidity_mean",
    "cotton_wool_aspect_ratio_mean",
    "cotton_wool_distribution_entropy",
]

TEXTURE_FEATURE_NAMES = [
    "glcm_correlation",
    "glcm_dissimilarity",
    "glcm_entropy",
    "lbp_uniform_ratio",
    "lbp_entropy",
    "lbp_mean",
    "lbp_std",
    "local_texture_variance_mean",
    "local_texture_variance_std",
    "texture_mean",
    "texture_variance",
    "texture_std",
    "texture_skewness",
    "texture_kurtosis",
]

COLOR_SPACES = {
    "rgb": ("r", "g", "b"),
    "hsv": ("h", "s", "v"),
    "lab_color": ("l", "a", "b"),
}
COLOR_FEATURE_NAMES = [
    f"{space}_{channel}_{metric}"
    for space, channels in COLOR_SPACES.items()
    for channel in channels
    for metric in ("mean", "std", "min", "max", "entropy")
]

FREQUENCY_FEATURE_NAMES = [
    "fft_energy",
    "fft_low_frequency_ratio",
    "fft_mid_frequency_ratio",
    "fft_high_frequency_ratio",
    "fft_dominant_radius_mean",
    "fft_dominant_radius_std",
    "wavelet_approx_energy",
    "wavelet_horizontal_energy",
    "wavelet_vertical_energy",
    "wavelet_diagonal_energy",
    "wavelet_detail_energy",
]

LESION_MORPHOLOGY_FEATURE_NAMES = [
    f"all_lesion_{metric}"
    for metric in (
        "area",
        "perimeter_mean",
        "circularity_mean",
        "solidity_mean",
        "eccentricity_mean",
        "compactness_mean",
        "aspect_ratio_mean",
        "convex_hull_area_mean",
    )
]

QUADRANT_FEATURE_NAMES = [
    f"{quadrant}_{metric}"
    for quadrant in ("superior", "inferior", "nasal", "temporal")
    for metric in (
        "lesion_count",
        "lesion_density",
        "vessel_density",
        "texture_mean",
        "texture_std",
    )
]

SEVERITY_FEATURE_NAMES = [
    "total_lesion_count",
    "combined_lesion_burden",
    "lesion_density_score",
    "hemorrhage_to_ma_ratio",
    "vessel_abnormality_score",
    "exudate_burden_score",
    "advanced_dr_indicator_score",
    "neovascularization_likelihood_score",
]

QUALITY_FEATURE_NAMES = [
    "quality_blur_score",
    "quality_sharpness",
    "quality_brightness",
    "quality_contrast",
    "quality_snr",
]

ENGINEERED_FEATURE_NAMES = [
    "ma_to_exudate_ratio",
    "hemorrhage_to_exudate_ratio",
    "lesion_vessel_interaction",
    "texture_lesion_interaction",
    "quality_adjusted_lesion_density",
    "area_adjusted_ma_count",
    "area_adjusted_exudate_count",
    "area_adjusted_hemorrhage_count",
    "referable_lesion_score",
    "stage_progression_score",
]

FEATURE_NAMES = [
    *CORE_FEATURE_NAMES,
    *VESSEL_FEATURE_NAMES,
    *HEMORRHAGE_FEATURE_NAMES,
    *MA_ADVANCED_FEATURE_NAMES,
    *EXUDATE_ADVANCED_FEATURE_NAMES,
    *COTTON_WOOL_FEATURE_NAMES,
    *TEXTURE_FEATURE_NAMES,
    *COLOR_FEATURE_NAMES,
    *FREQUENCY_FEATURE_NAMES,
    *LESION_MORPHOLOGY_FEATURE_NAMES,
    *QUADRANT_FEATURE_NAMES,
    *SEVERITY_FEATURE_NAMES,
    *QUALITY_FEATURE_NAMES,
    *ENGINEERED_FEATURE_NAMES,
]

CSV_COLUMNS = [*FEATURE_NAMES, "label"]

# Image normalization and preprocessing.
MAX_IMAGE_SIZE = 900
FOV_THRESHOLD = 10
FUNDUS_CROP_THRESHOLD = 12
FUNDUS_CROP_MARGIN_RATIO = 0.03
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
ILLUMINATION_SIGMA = 25.0
DENOISE_H = 6

# Classical feature extraction parameters.
FRANGI_SIGMAS = (1.0, 2.0, 4.0)
FRANGI_BETA = 0.5
FRANGI_GAMMA = 15.0
VESSEL_PERCENTILE = 97.0

MA_BLACKHAT_RADIUS = 6
MA_PERCENTILE = 99.7
MA_MIN_AREA = 8
MA_MAX_AREA = 95

EXUDATE_PERCENTILE = 97.5
EXUDATE_LOCAL_PERCENTILE = 98.0
EXUDATE_B_PERCENTILE = 75.0
EXUDATE_MIN_AREA = 12
EXUDATE_MAX_AREA = 5000

HEMORRHAGE_MIN_AREA = 40
HEMORRHAGE_MAX_AREA = 12000
SOFT_EXUDATE_MIN_AREA = 30
SOFT_EXUDATE_MAX_AREA = 9000
COTTON_WOOL_MIN_AREA = 35
COTTON_WOOL_MAX_AREA = 7000

GLCM_LEVELS = 32
GLCM_OFFSETS = ((0, 1), (1, 1), (1, 0), (1, -1))
GLCM_DIRECTION_OFFSETS = {
    "0": (0, 1),
    "45": (-1, 1),
    "90": (-1, 0),
    "135": (-1, -1),
}

IMBALANCE_WARNING_RATIO = 1.5

RF_PARAM_GRID = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [5, 10, None],
    "classifier__min_samples_split": [2, 5],
}

SVC_PARAM_GRID = {
    "classifier__C": [0.1, 1, 10],
    "classifier__kernel": ["linear", "rbf"],
}

HGB_PARAM_GRID = {
    "classifier__learning_rate": [0.05, 0.1],
    "classifier__max_iter": [100, 200],
    "classifier__max_leaf_nodes": [15, 31],
}

SELECTED_FEATURE_COUNT = 80
