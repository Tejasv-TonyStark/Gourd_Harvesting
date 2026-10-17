"""Run the selected detector and maturity classifier on a local image."""
from pathlib import Path
import argparse
import json

import cv2
import timm
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=ROOT / "two_stage_results")
    parser.add_argument("--detector", type=Path, default=ROOT / "final_report/detection/weights/gourd_detector_yolo26s_best.pt")
    parser.add_argument("--classifier", type=Path, default=ROOT / "final_report/classification/weights/gourd_maturity_classifier_mobilenetv3large_best.pt")
    args = parser.parse_args()
    original = cv2.imread(str(args.image))
    if original is None:
        raise ValueError(f"Cannot read image: {args.image}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = YOLO(str(args.detector))
    checkpoint = torch.load(args.classifier, map_location="cpu", weights_only=False)
    classifier = timm.create_model(checkpoint["model_name"], pretrained=False, num_classes=2)
    classifier.load_state_dict(checkpoint["state_dict"])
    classifier = classifier.to(device).eval()
    threshold = float(checkpoint["mature_probability_threshold"])
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    gourd_ids = [i for i, name in detector.names.items() if "gourd" in name.lower()]
    if not gourd_ids:
        raise ValueError(f"No gourd classes in detector: {detector.names}")
    result = detector.predict(original, conf=args.confidence, classes=gourd_ids,
                              device=device, verbose=False)[0]
    annotated = original.copy()
    height, width = original.shape[:2]
    predictions = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        crop = original[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        tensor = transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        with torch.inference_mode():
            probability = float(classifier(tensor).softmax(1)[0, 1].item())
        maturity = "mature" if probability >= threshold else "immature"
        variety = detector.names[int(box.cls[0].item())]
        confidence = float(box.conf[0].item())
        index = len(predictions) + 1
        predictions.append({"id": index, "variety": variety, "maturity": maturity,
                            "detection_confidence": confidence, "mature_probability": probability,
                            "bbox_xyxy": [x1, y1, x2, y2]})
        color = (70, 210, 70) if maturity == "mature" else (0, 180, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"#{index}"
        font_scale = min(0.5, max(0.28, width / 1000))
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        tx = max(0, min(x1, width - tw - 4))
        ty = min(height - baseline - 3, max(th + 5, y1 - 7))
        cv2.rectangle(annotated, (tx, ty - th - 4), (tx + tw + 4, ty + baseline + 2), (20, 20, 20), -1)
        cv2.putText(annotated, label, (tx + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
    if predictions:
        annotated = cv2.copyMakeBorder(annotated, 0, 26 * len(predictions) + 12, 0,
                                      max(0, 600 - width), cv2.BORDER_CONSTANT, value=(20, 20, 20))
        for row, item in enumerate(predictions):
            color = (70, 210, 70) if item["maturity"] == "mature" else (0, 180, 255)
            label = (f"#{item['id']} {item['variety']} | {item['maturity']} | "
                     f"det {item['detection_confidence']:.1%} | P(mature) {item['mature_probability']:.1%}")
            cv2.putText(annotated, label, (8, height + 24 + row * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
    args.output.mkdir(parents=True, exist_ok=True)
    output_image = args.output / f"{args.image.stem}_two_stage.png"
    if not cv2.imwrite(str(output_image), annotated):
        raise OSError(f"Could not save {output_image}")
    report = {"image": str(args.image.resolve()), "detector": str(args.detector),
              "classifier": str(args.classifier), "device": device,
              "detection_confidence_threshold": args.confidence,
              "mature_probability_threshold": threshold, "predictions": predictions,
              "annotated_image": str(output_image.resolve())}
    report_text = json.dumps(report, indent=2)
    (args.output / f"{args.image.stem}_predictions.json").write_text(report_text + "\n", encoding="utf-8")
    print(report_text)


if __name__ == "__main__":
    main()
