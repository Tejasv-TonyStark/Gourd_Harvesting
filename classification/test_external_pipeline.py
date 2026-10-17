"""Qualitative detector + maturity-classifier evaluation on web images."""
import json
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "classification" / "external_test"
OUTPUT_DIR = ROOT / "classification" / "external_results"
DETECTOR = ROOT / "detection" / "models" / "gourd_detector_yolo26s_best.pt"
CLASSIFIER = ROOT / "classification" / "models" / "gourd_maturity_classifier_best.pt"


def main():
    if not CLASSIFIER.exists():
        raise FileNotFoundError("Classification training has not produced the final checkpoint yet.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detector = YOLO(str(DETECTOR))
    checkpoint = torch.load(CLASSIFIER, map_location="cuda", weights_only=False)
    classifier = timm.create_model(checkpoint["model_name"], pretrained=False, num_classes=2)
    classifier.load_state_dict(checkpoint["state_dict"])
    classifier = classifier.to("cuda").eval()
    threshold = checkpoint["mature_probability_threshold"]
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    sources = json.loads((SOURCE_DIR / "sources.json").read_text())
    report = []
    for source in sources:
        path = SOURCE_DIR / source["file"]
        if not path.exists():
            report.append({**source, "error": "download unavailable"})
            continue
        bgr = cv2.imread(str(path))
        detections = detector.predict(bgr, conf=0.20, device=0, verbose=False)[0]
        predictions = []
        for box in detections.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
            crop = bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size == 0:
                continue
            rgb = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            tensor = transform(rgb).unsqueeze(0).to("cuda")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                mature_probability = float(torch.softmax(classifier(tensor), 1)[0, 1])
            maturity = "mature" if mature_probability >= threshold else "immature"
            type_id = int(box.cls[0])
            item = {
                "gourd_type": detector.names[type_id], "maturity": maturity,
                "mature_probability": mature_probability,
                "detection_confidence": float(box.conf[0]), "bbox": [x1, y1, x2, y2],
            }
            predictions.append(item)
            label = f"{item['gourd_type']} | {maturity} | {float(box.conf[0]):.2f}/{mature_probability:.2f}"
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(bgr, label, (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 2, cv2.LINE_AA)
        output = OUTPUT_DIR / f"annotated_{path.stem}.jpg"
        cv2.imwrite(str(output), bgr)
        report.append({**source, "predictions": predictions, "annotated_image": str(output)})
    (OUTPUT_DIR / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
