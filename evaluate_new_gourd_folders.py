from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


ROOT = Path(r"C:\Reinforcement")
CHECKPOINT = ROOT / "classification" / "models" / "gourd_maturity_classifier_best.pt"
OUTPUT = ROOT / "classification_new_folders_test"
CLASS_NAMES = ["immature", "mature"]
FOLDERS = [(ROOT / "Immature Gourd", 0), (ROOT / "Mature Gourd", 1)]


class FolderDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label, str(path)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = []
    for folder, label in FOLDERS:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                samples.append((path, label))

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    threshold = float(checkpoint["mature_probability_threshold"])
    model = timm.create_model(checkpoint["model_name"], pretrained=False, num_classes=2)
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(FolderDataset(samples, transform), batch_size=32, shuffle=False,
                        num_workers=4, pin_memory=device.type == "cuda")

    actual, predicted, probabilities, paths = [], [], [], []
    with torch.inference_mode():
        for tensors, labels, batch_paths in loader:
            tensors = tensors.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(tensors)
            mature_probability = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            batch_predictions = (mature_probability >= threshold).astype(int)
            actual.extend(labels.numpy().tolist())
            predicted.extend(batch_predictions.tolist())
            probabilities.extend(mature_probability.tolist())
            paths.extend(batch_paths)

    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    report = classification_report(actual, predicted, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    metrics = {
        "model": checkpoint["model_name"],
        "checkpoint": str(CHECKPOINT),
        "threshold": threshold,
        "device": str(device),
        "images": len(actual),
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
        "macro_f1": report["macro avg"]["f1-score"],
        "roc_auc": roc_auc_score(actual, probabilities),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with (OUTPUT / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "actual", "predicted", "mature_probability", "threshold", "correct"])
        for path, truth, pred, probability in zip(paths, actual, predicted, probabilities):
            writer.writerow([path, CLASS_NAMES[truth], CLASS_NAMES[pred], probability,
                             threshold, truth == pred])

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
