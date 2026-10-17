"""Create a bean-free four-class YOLO dataset without modifying the source."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "detection" / "dataset"
DESTINATION = ROOT / "yolov8_gourd_dataset"
CLASSES = ["bitter_gourd", "bottle_gourd", "ivy_gourd", "snake_gourd"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def image_for(label: Path, image_directory: Path) -> Path:
    for extension in IMAGE_EXTENSIONS:
        candidate = image_directory / f"{label.stem}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No image found for {label}")


def prepare_split(split: str) -> dict[str, int]:
    source_images = SOURCE / split / "images"
    source_labels = SOURCE / split / "labels"
    target_images = DESTINATION / split / "images"
    target_labels = DESTINATION / split / "labels"
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    statistics = {
        "source_images": 0,
        "kept_images": 0,
        "omitted_bean_only_images": 0,
        "removed_bean_boxes": 0,
        **{f"{name}_boxes": 0 for name in CLASSES},
    }

    for label in sorted(source_labels.glob("*.txt")):
        statistics["source_images"] += 1
        kept_lines: list[str] = []
        for raw_line in label.read_text(encoding="utf-8").splitlines():
            fields = raw_line.split()
            if not fields:
                continue
            class_id = int(fields[0])
            if class_id == 0:
                statistics["removed_bean_boxes"] += 1
                continue
            new_class_id = class_id - 1
            if not 0 <= new_class_id < len(CLASSES):
                raise ValueError(f"Unexpected class {class_id} in {label}")
            statistics[f"{CLASSES[new_class_id]}_boxes"] += 1
            kept_lines.append(" ".join([str(new_class_id), *fields[1:]]))

        if not kept_lines:
            statistics["omitted_bean_only_images"] += 1
            continue

        image = image_for(label, source_images)
        shutil.copy2(image, target_images / image.name)
        (target_labels / label.name).write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
        statistics["kept_images"] += 1

    return statistics


def main() -> None:
    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)

    summary = {split: prepare_split(split) for split in ("train", "valid", "test")}
    yaml_text = "\n".join(
        [
            f"path: {DESTINATION.as_posix()}",
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            "names:",
            *[f"  {index}: {name}" for index, name in enumerate(CLASSES)],
            "",
        ]
    )
    (DESTINATION / "data.yaml").write_text(yaml_text, encoding="utf-8")
    (DESTINATION / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
