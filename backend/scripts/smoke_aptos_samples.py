import argparse
import csv
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.pipeline import analyze_image


def smoke(csv_path: Path, samples_per_label: int) -> None:
    rows_by_label = read_rows_by_label(csv_path)
    total = 0
    binary_correct = 0
    stage_correct = 0

    print("label,predicted_stage,dr_probability,quality,classification,image")

    for label in range(5):
        for row in rows_by_label.get(label, [])[:samples_per_label]:
            image_path = resolve_image_path(csv_path.parent, row["image_path"])
            result = analyze_image(image_path.read_bytes(), include_processed_images=False)
            predicted_stage = result.result.stage if result.result.stage is not None else 0
            binary_prediction = 1 if result.result.referable else 0
            binary_label = 1 if label > 0 else 0
            total += 1
            binary_correct += int(binary_prediction == binary_label)
            stage_correct += int(predicted_stage == label)
            print(
                ",".join(
                    [
                        str(label),
                        str(predicted_stage),
                        f"{result.result.dr_probability:.1f}",
                        "ok" if result.quality.is_acceptable else "blocked",
                        quote_csv(result.result.classification),
                        quote_csv(str(image_path)),
                    ],
                ),
            )

    if total == 0:
        print("No rows found.")
        return

    print(f"Binary smoke accuracy: {binary_correct}/{total} = {binary_correct / total:.3f}")
    print(f"Stage smoke accuracy: {stage_correct}/{total} = {stage_correct / total:.3f}")


def read_rows_by_label(csv_path: Path) -> dict[int, list[dict[str, str]]]:
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("CSV has no header.")

        rows = list(reader)

    rows_by_label: dict[int, list[dict[str, str]]] = {}

    for row in rows:
        label_value = row.get("label") or row.get("diagnosis")
        image_path = row.get("image_path")

        if image_path is None and row.get("id_code"):
            image_path = f"train_images/{row['id_code']}.png"
        if label_value is None or image_path is None:
            continue

        label = int(label_value)
        rows_by_label.setdefault(label, []).append({"image_path": image_path})

    return rows_by_label


def resolve_image_path(base_dir: Path, value: str) -> Path:
    image_path = Path(value)

    if not image_path.is_absolute():
        image_path = base_dir / image_path

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return image_path


def quote_csv(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the classical pipeline on labeled APTOS samples.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=BACKEND_ROOT / "images" / "aptos2019" / "labels.csv",
    )
    parser.add_argument("--samples-per-label", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    smoke(args.csv, max(1, args.samples_per_label))
