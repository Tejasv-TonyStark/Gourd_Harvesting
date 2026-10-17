from pathlib import Path
import shutil


SOURCE = Path(r"C:\Reinforcement\yolov8_gourd_dataset\test")
DESTINATION = Path(r"C:\Reinforcement\yolo26s_gourd_test_dataset\test")


def main() -> None:
    images_out = DESTINATION / "images"
    labels_out = DESTINATION / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    image_count = 0
    box_count = 0
    for image_path in sorted((SOURCE / "images").iterdir()):
        if not image_path.is_file():
            continue
        label_path = SOURCE / "labels" / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        output_lines = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            # The four-class evaluation dataset uses 0..3. YOLO26s was trained
            # with beans at class 0, so its gourd class IDs are 1..4.
            fields[0] = str(int(fields[0]) + 1)
            output_lines.append(" ".join(fields))

        if not output_lines:
            continue
        shutil.copy2(image_path, images_out / image_path.name)
        (labels_out / label_path.name).write_text(
            "\n".join(output_lines) + "\n", encoding="utf-8"
        )
        image_count += 1
        box_count += len(output_lines)

    print(f"Prepared {image_count} images and {box_count} gourd boxes")


if __name__ == "__main__":
    main()
