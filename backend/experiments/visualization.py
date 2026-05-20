from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.evaluation import CLASS_LABELS


def plot_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted Stage")
    axis.set_ylabel("True Stage")
    axis.set_xticks(CLASS_LABELS)
    axis.set_yticks(CLASS_LABELS)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(
                col,
                row,
                str(int(matrix[row, col])),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > matrix.max() / 2 else "black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_feature_importance(rows: list[dict[str, float | str]], output_path: Path, top_n: int = 12) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = rows[:top_n]
    names = [str(row["feature"]) for row in selected]
    scores = [float(row["fisher_score"]) for row in selected]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(names[::-1], scores[::-1], color="#0E7C7B")
    axis.set_title("Handcrafted Feature Ranking")
    axis.set_xlabel("Fisher score")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_preprocessing_outputs(
    output_dir: Path,
    image_id: str,
    true_label: int,
    predicted_label: int,
    images: dict[str, np.ndarray],
) -> None:
    case_dir = output_dir / f"{image_id}_true{true_label}_pred{predicted_label}"
    case_dir.mkdir(parents=True, exist_ok=True)

    for name, image in images.items():
        cv2.imwrite(str(case_dir / f"{name}.png"), image)
