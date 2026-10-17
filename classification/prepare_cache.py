"""Create a non-destructive 256-pixel JPEG cache for faster GPU training."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "gourd_classification"
DEST = ROOT / "classification" / "dataset_256"
files = [path for path in SOURCE.rglob("*") if path.is_file()]


def convert(path: Path):
    target = DEST / path.relative_to(SOURCE)
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize((256, 256), Image.Resampling.LANCZOS)
        image.save(target, "JPEG", quality=92, optimize=False)


with ThreadPoolExecutor(max_workers=8) as executor:
    list(executor.map(convert, files))
print(f"Prepared {len(files)} images in {DEST}")
