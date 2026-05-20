import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar

import numpy as np

import config


T = TypeVar("T")


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def list_image_files(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.exists():
        return []

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS
    )


def progress_bar(items: Iterable[T], total: int, prefix: str = "Progress") -> Iterator[T]:
    """Minimal dependency-free progress bar.

    This project uses a compact terminal bar instead of an extra progress
    dependency, keeping the stack inside the requested package set.
    """
    if total <= 0:
        for item in items:
            yield item
        return

    width = 32
    start = time.time()
    update_every = max(1, total // 100)

    for index, item in enumerate(items, start=1):
        yield item

        if index == total or index % update_every == 0:
            fraction = index / total
            filled = int(width * fraction)
            bar = "#" * filled + "-" * (width - filled)
            elapsed = time.time() - start
            print(
                f"\r{prefix}: [{bar}] {index}/{total} ({fraction:6.1%}) "
                f"elapsed {elapsed:5.1f}s",
                end="",
                flush=True,
            )

    print()


def class_distribution(labels: Iterable[int]) -> dict[int, int]:
    counts = Counter(int(label) for label in labels)
    return {label: counts.get(label, 0) for label in config.CLASS_LABELS}


def imbalance_report(labels: Iterable[int]) -> dict[str, Any]:
    distribution = class_distribution(labels)
    nonzero_counts = [count for count in distribution.values() if count > 0]

    if not nonzero_counts:
        return {
            "distribution": distribution,
            "max_to_min_ratio": 0.0,
            "is_imbalanced": False,
            "message": "No labeled samples were found.",
        }

    ratio = max(nonzero_counts) / max(min(nonzero_counts), 1)
    is_imbalanced = ratio >= config.IMBALANCE_WARNING_RATIO
    message = (
        "Class imbalance detected. Macro F1, balanced accuracy, stratified "
        "splits, and class_weight='balanced' are used to reduce majority-class bias."
        if is_imbalanced
        else "No severe class imbalance detected by the configured ratio threshold."
    )

    return {
        "distribution": distribution,
        "max_to_min_ratio": float(ratio),
        "is_imbalanced": bool(is_imbalanced),
        "message": message,
    }


def print_class_distribution(labels: Iterable[int]) -> None:
    report = imbalance_report(labels)
    print("Class distribution:")
    for label, count in report["distribution"].items():
        stage_name = config.CLASS_NAMES[label]
        print(f"  Stage {label} ({stage_name}): {count}")
    print(f"Max/min class ratio: {report['max_to_min_ratio']:.2f}")
    print(report["message"])


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def save_text(path: str | Path, text: str) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")


def read_feature_table(path: str | Path):
    import pandas as pd

    table = pd.read_csv(path)
    missing = [column for column in config.CSV_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"Feature CSV is missing required columns: {missing}")
    return table
