"""Export the selected PyTorch models for private, in-browser ONNX inference."""
from pathlib import Path
import shutil

import timm
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "dist" / "models"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    detector = ROOT / "gourd_web_app" / "best_wb.onnx"
    detector_bytes = detector.read_bytes()
    part_size = (len(detector_bytes) + 2) // 3
    detector_parts = []
    for index in range(3):
        part = OUT / f"gourd_detector_best_wb.part{index + 1}.bin"
        part.write_bytes(detector_bytes[index * part_size:(index + 1) * part_size])
        detector_parts.append(str(part))

    checkpoint_path = ROOT / "classification" / "models" / "gourd_maturity_classifier_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = timm.create_model(checkpoint["model_name"], pretrained=False, num_classes=2)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    dummy = torch.zeros(1, 3, 224, 224)
    output = OUT / "gourd_maturity_mobilenetv3.onnx"
    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["images"],
        output_names=["logits"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    print({
        "detector": "YOLO11n best_wb.pt",
        "detector_parts": detector_parts,
        "classifier": str(output),
        "threshold": checkpoint["mature_probability_threshold"],
    })


if __name__ == "__main__":
    main()
