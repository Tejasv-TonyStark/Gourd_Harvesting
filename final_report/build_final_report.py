"""Assemble all final detection and classification artifacts in one folder."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from classification.train_classification import make_loaders, metrics_at_threshold, predict  # noqa: E402

DEST = ROOT / "final_report"
DETECTION = DEST / "detection"
CLASSIFICATION = DEST / "classification"
EXTERNAL = DEST / "external_web_test"


def copy(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def build_detection():
    selected = ROOT / "detection" / "models" / "gourd_detector_yolo26s_best.pt"
    packaged_weight = DETECTION / "weights" / "gourd_detector_yolo26s_best.pt"
    copy(selected, packaged_weight)
    source_run = ROOT / "detection" / "runs" / "20261009_153141" / "yolo26s_b2"
    for name in ["results.csv", "args.yaml"]:
        copy(source_run / name, DETECTION / "training" / name)
    for name in ["results.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
                 "BoxPR_curve.png", "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png",
                 "labels.jpg", "val_batch0_labels.jpg", "val_batch0_pred.jpg"]:
        if (source_run / name).exists():
            copy(source_run / name, DETECTION / "plots" / name)
    copy(ROOT / "detection" / "audit.json", DETECTION / "dataset" / "audit.json")
    copy(ROOT / "detection" / "dataset" / "data.yaml", DETECTION / "dataset" / "data.yaml")

    model = YOLO(str(packaged_weight))
    metrics = model.val(
        data=str(ROOT / "detection" / "dataset" / "data.yaml"), split="test",
        imgsz=640, batch=1, device=0, workers=0, plots=True,
        project=str(DETECTION / "evaluation"), name="test", exist_ok=True,
    )
    overall = {key: float(value) for key, value in metrics.results_dict.items()}
    overall.update({"model": "YOLO26s", "split": "test", "images": 135,
                    "input_size": 640, "weights_mb": packaged_weight.stat().st_size / 2**20,
                    "speed_ms_per_image": metrics.speed})
    per_class = metrics.summary()
    write_json(DETECTION / "metrics" / "test_metrics.json", overall)
    write_json(DETECTION / "metrics" / "test_per_class.json", per_class)
    if per_class:
        with (DETECTION / "metrics" / "test_per_class.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=per_class[0].keys())
            writer.writeheader(); writer.writerows(per_class)
    validation = {
        "model": "YOLO26s", "split": "validation", "best_epoch": 44,
        "precision": 0.7608411552602368, "recall": 0.6396457802542346,
        "mAP50": 0.7248121141464713, "mAP50_95": 0.3198795920865086,
    }
    write_json(DETECTION / "metrics" / "validation_metrics.json", validation)
    return overall, per_class


def plot_classification(labels, probabilities, predictions, history_mobile, history_efficient,
                        official_matrix):
    plots = CLASSIFICATION / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(official_matrix)
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, matrix[row, column], ha="center", va="center", fontsize=14)
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Immature", "Mature"],
             yticklabels=["Immature", "Mature"], xlabel="Predicted", ylabel="Actual",
             title="Maturity Classification - Test Confusion Matrix")
    fig.colorbar(image, ax=axis); fig.tight_layout()
    fig.savefig(plots / "confusion_matrix.png", dpi=180); plt.close(fig)

    fpr, tpr, _ = roc_curve(labels, probabilities)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(fpr, tpr, linewidth=2, label="MobileNetV3Large")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlabel="False positive rate", ylabel="True positive rate",
             title="Test ROC Curve"); axis.grid(alpha=.25); axis.legend()
    fig.tight_layout(); fig.savefig(plots / "roc_curve.png", dpi=180); plt.close(fig)

    precision, recall, _ = precision_recall_curve(labels, probabilities)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(recall, precision, linewidth=2)
    axis.set(xlabel="Recall", ylabel="Precision", title="Test Precision-Recall Curve")
    axis.grid(alpha=.25); fig.tight_layout()
    fig.savefig(plots / "precision_recall_curve.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for history, label in [(history_mobile, "MobileNetV3Large"),
                           (history_efficient, "EfficientNetV2B0")]:
        axes[0].plot([x["epoch"] for x in history], [x["accuracy"] for x in history], label=label)
        axes[1].plot([x["epoch"] for x in history], [x["macro_f1"] for x in history], label=label)
    axes[0].set(title="Validation Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].set(title="Validation Macro-F1", xlabel="Epoch", ylabel="Macro-F1")
    for axis in axes: axis.grid(alpha=.25); axis.legend()
    fig.tight_layout(); fig.savefig(plots / "model_training_comparison.png", dpi=180); plt.close(fig)


def build_classification():
    selected = ROOT / "classification" / "models" / "gourd_maturity_classifier_best.pt"
    packaged = CLASSIFICATION / "weights" / "gourd_maturity_classifier_mobilenetv3large_best.pt"
    copy(selected, packaged)
    run = ROOT / "classification" / "runs" / "20261009_205019"
    copy(run / "tf_efficientnetv2_b0" / "best.pt",
         CLASSIFICATION / "weights" / "candidate_efficientnetv2b0_best.pt")
    for source, name in [
        (run / "final_report.json", "original_final_report.json"),
        (run / "comparison.json", "model_comparison.json"),
        (run / "mobilenetv3_large_100" / "history.json", "mobilenetv3large_history.json"),
        (run / "tf_efficientnetv2_b0" / "history.json", "efficientnetv2b0_history.json"),
    ]:
        copy(source, CLASSIFICATION / "training" / name)

    checkpoint = torch.load(packaged, map_location="cuda", weights_only=False)
    model = timm.create_model(checkpoint["model_name"], pretrained=False, num_classes=2)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to("cuda", memory_format=torch.channels_last)
    _, _, test_loader, _ = make_loaders(32)
    labels, probabilities = predict(model, test_loader, "cuda")
    threshold = float(checkpoint["mature_probability_threshold"])
    predictions = (probabilities >= threshold).astype(int)
    reproduced = metrics_at_threshold(labels, probabilities, threshold)
    reproduced.update({
        "note": "Regenerated CUDA inference differed by one threshold-borderline image from the original recorded evaluation. The checkpoint's original held-out result remains authoritative.",
        "original_accuracy": float(checkpoint["test_metrics"]["accuracy"]),
    })
    overall = dict(checkpoint["test_metrics"])
    overall.update({"model": "MobileNetV3Large", "split": "test", "images": 462,
                    "weights_mb": packaged.stat().st_size / 2**20,
                    "source": "test_metrics embedded in selected training checkpoint"})
    supports = {name: overall["per_class"][name]["support"]
                for name in ["immature", "mature"]}
    total = sum(supports.values())
    report = {name: {
        "precision": overall["per_class"][name]["precision"],
        "recall": overall["per_class"][name]["recall"],
        "f1-score": overall["per_class"][name]["f1"],
        "support": supports[name],
    } for name in ["immature", "mature"]}
    report["accuracy"] = overall["accuracy"]
    report["macro avg"] = {
        key: sum(report[name][key] for name in ["immature", "mature"]) / 2
        for key in ["precision", "recall", "f1-score"]
    } | {"support": total}
    report["weighted avg"] = {
        key: sum(report[name][key] * supports[name] for name in ["immature", "mature"]) / total
        for key in ["precision", "recall", "f1-score"]
    } | {"support": total}
    write_json(CLASSIFICATION / "metrics" / "test_metrics.json", overall)
    write_json(CLASSIFICATION / "metrics" / "reproducibility_check.json", reproduced)
    write_json(CLASSIFICATION / "metrics" / "classification_report.json", report)
    with (CLASSIFICATION / "metrics" / "classification_report.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "precision", "recall", "f1_score", "support"])
        for name in ["immature", "mature", "macro avg", "weighted avg"]:
            row = report[name]
            writer.writerow([name, row["precision"], row["recall"], row["f1-score"], row["support"]])
        writer.writerow(["accuracy", "", "", report["accuracy"], len(labels)])
    matrix = np.asarray(overall["confusion_matrix"])
    with (CLASSIFICATION / "metrics" / "confusion_matrix.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["actual/predicted", "immature", "mature"])
        writer.writerow(["immature", *matrix[0]]); writer.writerow(["mature", *matrix[1]])
    samples = test_loader.dataset.samples
    with (CLASSIFICATION / "metrics" / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "actual", "predicted", "mature_probability", "threshold"])
        for (path, _), actual, predicted, probability in zip(samples, labels, predictions, probabilities):
            writer.writerow([str(Path(path).relative_to(ROOT)), ["immature", "mature"][actual],
                             ["immature", "mature"][predicted], float(probability), threshold])
    mobile_history = json.loads((run / "mobilenetv3_large_100" / "history.json").read_text(encoding="utf-8"))
    efficient_history = json.loads((run / "tf_efficientnetv2_b0" / "history.json").read_text(encoding="utf-8"))
    plot_classification(labels, probabilities, predictions, mobile_history, efficient_history,
                        overall["confusion_matrix"])
    return overall, report


def copy_external():
    source_dir = ROOT / "classification" / "external_results"
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    copy(ROOT / "classification" / "external_test" / "sources.json", EXTERNAL / "sources.json")
    for path in source_dir.glob("*"):
        if path.is_file(): copy(path, EXTERNAL / path.name)


def manifest():
    entries = []
    for path in sorted(DEST.rglob("*")):
        if path.is_file() and path.name != "manifest_sha256.json":
            entries.append({"path": path.relative_to(DEST).as_posix(), "bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    write_json(DEST / "manifest_sha256.json", entries)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    detection, detection_classes = build_detection()
    classification, classification_report_data = build_classification()
    copy_external()
    summary = {
        "selected_pipeline": "YOLO26s detection -> MobileNetV3Large maturity classification",
        "detection_test": detection,
        "detection_test_per_class": detection_classes,
        "classification_test": classification,
        "classification_report": classification_report_data,
        "important_note": "Detection and classification metrics are separate; no combined end-to-end accuracy is claimed. Classification uses the original test result embedded in the selected checkpoint.",
    }
    write_json(DEST / "overall_metrics.json", summary)
    d = detection
    c = classification
    readme = f"""# Final Gourd Detection and Maturity Classification Report

Selected pipeline: **YOLO26s detector -> MobileNetV3Large maturity classifier**.

| Task | Metric | Held-out test result |
|---|---|---:|
| Detection | Precision | {d['metrics/precision(B)']:.2%} |
| Detection | Recall | {d['metrics/recall(B)']:.2%} |
| Detection | mAP50 | {d['metrics/mAP50(B)']:.2%} |
| Detection | mAP50-95 | {d['metrics/mAP50-95(B)']:.2%} |
| Classification | Accuracy | {c['accuracy']:.2%} |
| Classification | Balanced accuracy | {c['balanced_accuracy']:.2%} |
| Classification | Macro-F1 | {c['macro_f1']:.2%} |
| Classification | ROC-AUC | {c['roc_auc']:.2%} |

Detection and classification metrics are reported separately because the datasets
do not provide a single held-out set with all box, type, and maturity ground truth.
External web-image predictions are qualitative and are not included in accuracy.

## Folder layout

- `detection/weights`: selected detector weight.
- `detection/metrics`: validation and held-out test metrics, including per-class AP.
- `detection/plots`: training/validation plots.
- `classification/weights`: selected classifier and EfficientNet comparison candidate.
- `classification/metrics`: official report, confusion matrix, regenerated predictions, and reproducibility check.
- `classification/plots`: ROC, precision-recall, confusion matrix, and training comparison.
- `external_web_test`: source record, predictions, and annotated web images.
- `manifest_sha256.json`: file sizes and SHA-256 hashes for integrity checks.
"""
    (DEST / "README.md").write_text(readme, encoding="utf-8")
    manifest()
    print(json.dumps({"folder": str(DEST), "detection_test": detection,
                      "classification_test": classification}, indent=2, default=str))


if __name__ == "__main__":
    main()
