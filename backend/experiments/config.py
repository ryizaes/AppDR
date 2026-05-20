from dataclasses import asdict, dataclass
from itertools import product
from typing import Any


@dataclass(frozen=True)
class ImageProcessingParams:
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    gamma: float = 1.0
    illumination_sigma: float = 25.0
    gaussian_sigma: float = 0.0
    frangi_sigmas: tuple[float, ...] = (1.0, 2.0, 4.0)
    frangi_beta: float = 0.5
    frangi_gamma: float = 15.0
    adaptive_block_size: int = 21
    adaptive_c: int = 10
    vessel_percentile: float = 97.0
    morph_close_radius: int = 2
    morph_open_radius: int = 1
    ma_blackhat_radius: int = 6
    ma_percentile: float = 99.7
    ma_min_area: int = 8
    ma_max_area: int = 95
    exudate_percentile: float = 97.5
    exudate_local_percentile: float = 98.0
    exudate_min_area: int = 12
    exudate_max_area: int = 5000


@dataclass(frozen=True)
class FusionParams:
    ma_weight: float = 1.0
    exudate_weight: float = 1.4
    quadrant_weight: float = 5.0
    pai_weight: float = 14.0
    vessel_weight: float = 0.2
    glcm_weight: float = 0.02
    stage1_threshold: float = 1.0
    stage2_threshold: float = 8.0
    stage3_threshold: float = 18.0
    stage4_threshold: float = 42.0
    stage4_vessel_density: float = 0.12
    stage4_glcm_contrast: float = 500.0


def dataclass_to_json_dict(value: Any) -> dict[str, Any]:
    data = asdict(value)

    for key, item in data.items():
        if isinstance(item, tuple):
            data[key] = list(item)

    return data


def build_dataclass_grid(cls: type, grid: dict[str, list[Any]]) -> list[Any]:
    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    entries = []

    for combination in product(*values):
        kwargs = dict(zip(keys, combination))
        entries.append(cls(**kwargs))

    return entries


def quick_processing_grid() -> list[ImageProcessingParams]:
    return build_dataclass_grid(
        ImageProcessingParams,
        {
            "clahe_clip_limit": [1.5, 2.0],
            "gamma": [0.9, 1.0],
            "frangi_sigmas": [(1.0, 2.0, 4.0)],
            "adaptive_block_size": [21],
            "adaptive_c": [8, 10],
            "ma_percentile": [99.5, 99.7],
            "exudate_percentile": [97.5],
        },
    )


def standard_processing_grid() -> list[ImageProcessingParams]:
    return build_dataclass_grid(
        ImageProcessingParams,
        {
            "clahe_clip_limit": [1.5, 2.0, 2.5],
            "gamma": [0.8, 1.0, 1.2],
            "illumination_sigma": [18.0, 25.0, 35.0],
            "gaussian_sigma": [0.0, 0.8],
            "frangi_sigmas": [(1.0, 2.0), (1.0, 2.0, 4.0), (2.0, 4.0)],
            "adaptive_block_size": [15, 21, 31],
            "adaptive_c": [6, 10, 14],
            "morph_close_radius": [1, 2],
            "morph_open_radius": [1, 2],
            "ma_percentile": [99.3, 99.5, 99.7],
            "exudate_percentile": [97.0, 97.5, 98.0],
        },
    )


def quick_fusion_grid() -> list[FusionParams]:
    return build_dataclass_grid(
        FusionParams,
        {
            "ma_weight": [0.8, 1.0],
            "exudate_weight": [1.2, 1.6],
            "quadrant_weight": [4.0, 6.0],
            "pai_weight": [10.0, 14.0],
            "stage2_threshold": [7.0, 9.0],
            "stage3_threshold": [17.0, 21.0],
        },
    )


def standard_fusion_grid() -> list[FusionParams]:
    return build_dataclass_grid(
        FusionParams,
        {
            "ma_weight": [0.6, 0.8, 1.0, 1.2],
            "exudate_weight": [1.0, 1.4, 1.8],
            "quadrant_weight": [3.0, 5.0, 7.0],
            "pai_weight": [8.0, 12.0, 16.0],
            "vessel_weight": [0.0, 0.2, 0.4],
            "glcm_weight": [0.0, 0.02, 0.05],
            "stage2_threshold": [6.0, 8.0, 10.0],
            "stage3_threshold": [16.0, 20.0, 24.0],
        },
    )
