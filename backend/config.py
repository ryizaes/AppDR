import os
from pathlib import Path


# Central configuration for the supervised handcrafted-feature DR pipeline.
# The code intentionally avoids TensorFlow, PyTorch, CNNs, and learned image
# embeddings. Only classical image processing features are passed to ML models.

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

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
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
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

FEATURE_NAMES = [
    "ma_count",
    "exudate_area",
    "vessel_density",
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
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
EXUDATE_MIN_AREA = 12
EXUDATE_MAX_AREA = 5000

GLCM_LEVELS = 32
GLCM_OFFSETS = ((0, 1), (1, 1), (1, 0), (1, -1))

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
