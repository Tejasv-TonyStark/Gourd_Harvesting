"""Download the recorded qualitative web-image test set."""
import json
from pathlib import Path
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent / "external_test"
sources = json.loads((ROOT / "sources.json").read_text())
headers = {"User-Agent": "Mozilla/5.0 (compatible; research-evaluation/1.0)"}
results = []
for source in sources:
    path = ROOT / source["file"]
    try:
        response = requests.get(source["image_url"], headers=headers, timeout=30)
        response.raise_for_status()
        path.write_bytes(response.content)
        with Image.open(path) as image:
            image.verify()
        results.append({"file": str(path), "downloaded": True, "bytes": path.stat().st_size})
    except Exception as error:
        if path.exists():
            path.unlink()
        results.append({"file": str(path), "downloaded": False, "error": str(error)})
print(json.dumps(results, indent=2))
