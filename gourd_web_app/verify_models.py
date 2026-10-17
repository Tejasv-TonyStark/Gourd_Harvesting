"""Quick parity and shape checks for the browser ONNX models."""
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import timm
import torch
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(__file__).resolve().parent / "dist" / "models"


def main():
    image_path = ROOT / "classification" / "external_test" / "bottle_green_rainseed.jpg"
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(image_path)

    detector_bytes = b"".join((MODELS / f"gourd_detector_best_wb.part{index}.bin").read_bytes()
                              for index in range(1, 4))
    detector = ort.InferenceSession(detector_bytes, providers=["CPUExecutionProvider"])
    height, width = bgr.shape[:2]
    scale = min(640 / width, 640 / height)
    resized = cv2.resize(bgr, (round(width * scale), round(height * scale)))
    pad_x = (640 - resized.shape[1]) // 2
    pad_y = (640 - resized.shape[0]) // 2
    padded = np.full((640, 640, 3), 114, dtype=np.uint8)
    padded[pad_y:pad_y + resized.shape[0], pad_x:pad_x + resized.shape[1]] = resized
    tensor = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    boxes = detector.run(None, {detector.get_inputs()[0].name: tensor})[0]
    assert boxes.shape == (1, 300, 6), boxes.shape
    assert set(np.unique(boxes[0, :, 5]).astype(int)).issubset(set(range(4)))

    checkpoint = torch.load(ROOT / "classification" / "models" / "gourd_maturity_classifier_best.pt",
                            map_location="cpu", weights_only=False)
    model = timm.create_model(checkpoint["model_name"], pretrained=False, num_classes=2)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    sample = transform(Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        torch_logits = model(sample).numpy()
    classifier = ort.InferenceSession(str(MODELS / "gourd_maturity_mobilenetv3.onnx"),
                                      providers=["CPUExecutionProvider"])
    onnx_logits = classifier.run(None, {classifier.get_inputs()[0].name: sample.numpy()})[0]
    difference = float(np.max(np.abs(torch_logits - onnx_logits)))
    assert difference < 1e-4, difference
    print({"detector_output": list(boxes.shape), "detector_candidates_above_025": int((boxes[0, :, 4] >= .25).sum()),
           "classifier_max_abs_difference": difference, "status": "passed"})


if __name__ == "__main__":
    main()
