import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetItem:
    image_path: Path
    label: int
    image_id: str


def load_dataset(csv_path: Path, samples_per_class: int | None = None) -> list[DatasetItem]:
    rows = read_label_rows(csv_path)
    items: list[DatasetItem] = []
    counts: dict[int, int] = {}

    for row in rows:
        label = int(row["label"])

        if samples_per_class is not None and counts.get(label, 0) >= samples_per_class:
            continue

        image_path = resolve_image_path(csv_path.parent, row["image_path"])
        items.append(
            DatasetItem(
                image_path=image_path,
                label=label,
                image_id=image_path.stem,
            ),
        )
        counts[label] = counts.get(label, 0) + 1

    return items


def read_label_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("CSV has no header.")

        rows = list(reader)

    if "image_path" in reader.fieldnames and "label" in reader.fieldnames:
        return rows
    if "id_code" in reader.fieldnames and "diagnosis" in reader.fieldnames:
        return [
            {
                "image_path": f"train_images/{row['id_code']}.png",
                "label": row["diagnosis"],
            }
            for row in rows
        ]

    raise ValueError("CSV must contain image_path,label or id_code,diagnosis columns.")


def resolve_image_path(base_dir: Path, value: str) -> Path:
    image_path = Path(value)

    if not image_path.is_absolute():
        image_path = base_dir / image_path

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return image_path
