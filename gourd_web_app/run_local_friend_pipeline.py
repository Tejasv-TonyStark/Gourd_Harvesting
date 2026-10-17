"""Run the best_wb detector and maturity classifier on one local image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import timm
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "gourd_web_app"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    detector = YOLO(str(APP / "best_wb.pt"))
    checkpoint = torch.load(ROOT / "classification" / "models" / "gourd_maturity_classifier_best.pt",
                            map_location="cuda", weights_only=False)
    classifier = timm.create_model(checkpoint["model_name"], pretrained=False, num_classes=2)
    classifier.load_state_dict(checkpoint["state_dict"])
    classifier = classifier.to("cuda").eval()
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image = cv2.imread(str(args.image))
    result = detector.predict(image, conf=args.confidence, device=0, verbose=False)[0]
    predictions = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
        crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            continue
        tensor = transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).unsqueeze(0).to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            mature_probability = float(torch.softmax(classifier(tensor), 1)[0, 1])
        maturity = "mature" if mature_probability >= checkpoint["mature_probability_threshold"] else "immature"
        predictions.append({
            "variety": detector.names[int(box.cls[0])],
            "detection_confidence": round(float(box.conf[0]), 4),
            "maturity": maturity,
            "mature_probability": round(mature_probability, 4),
            "box": [x1, y1, x2, y2],
        })
    print(json.dumps({"image": str(args.image), "confidence_threshold": args.confidence,
                      "predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
