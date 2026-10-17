"""Rebuild and inspect the friend's extracted YOLO checkpoint."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = ROOT / "best.pt" / "best"
OUTPUT = ROOT / "gourd_web_app" / "friend_yolov11_best.pt"


def main() -> None:
    if not EXTRACTED.is_dir():
        raise FileNotFoundError(f"Missing extracted checkpoint: {EXTRACTED}")
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in EXTRACTED.rglob("*"):
            if file.is_file():
                archive.write(file, file.relative_to(EXTRACTED.parent).as_posix())
    model = YOLO(str(OUTPUT))
    info = {
        "checkpoint": str(OUTPUT),
        "size_mb": round(OUTPUT.stat().st_size / 2**20, 2),
        "task": model.task,
        "names": model.names,
        "model_type": model.model.__class__.__name__,
    }
    print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
