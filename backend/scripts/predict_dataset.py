import argparse
import csv
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.pipeline import analyze_image


def predict(csv_path: Path, image_dir: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)

        if "id_code" not in reader.fieldnames:
            raise ValueError("CSV must contain an id_code column.")

        rows = list(reader)

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        fieldnames = [
            "id_code",
            "predicted_label",
            "predicted_stage",
            "stage_label",
            "dr_probability",
            "classification",
            "quality_acceptable",
            "reason",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for index, row in enumerate(rows, start=1):
            id_code = row["id_code"]
            image_path = image_dir / f"{id_code}.png"

            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            result = analyze_image(
                image_path.read_bytes(),
                include_processed_images=False,
            )
            writer.writerow(
                {
                    "id_code": id_code,
                    "predicted_label": 1 if result.result.referable else 0,
                    "predicted_stage": result.result.stage,
                    "stage_label": result.result.stage_label,
                    "dr_probability": result.result.dr_probability,
                    "classification": result.result.classification,
                    "quality_acceptable": result.quality.is_acceptable,
                    "reason": result.result.reason,
                },
            )

            if index % 100 == 0:
                print(f"Processed {index}/{len(rows)}")

    print("Saved predictions:", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict DR score for an image CSV.")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args.csv, args.image_dir, args.output)
