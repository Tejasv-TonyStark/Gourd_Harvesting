"""GPU-only maturity-classifier comparison with held-out test evaluation."""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "classification" / "dataset_256"
OUT = ROOT / "classification"
RUNS = OUT / "runs"
CLASS_NAMES = ["immature", "mature"]
MODEL_NAMES = ["mobilenetv3_large_100.ra_in1k", "tf_efficientnetv2_b0.in1k"]


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(batch_size: int):
    # Both pretrained backbones use ImageNet mean/std with 224-pixel inputs.
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.82, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.18, contrast=0.15, saturation=0.12, hue=0.02),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    train_set = datasets.ImageFolder(DATA / "train", transform=train_transform)
    valid_set = datasets.ImageFolder(DATA / "valid", transform=eval_transform)
    test_set = datasets.ImageFolder(DATA / "test", transform=eval_transform)
    if train_set.classes != CLASS_NAMES or valid_set.classes != CLASS_NAMES or test_set.classes != CLASS_NAMES:
        raise RuntimeError(f"Unexpected class mapping: {train_set.class_to_idx}")
    # Windows spawn workers each import PyTorch and can exhaust the paging file.
    # The 256-pixel cache makes single-process loading fast enough and stable.
    common = dict(num_workers=0, pin_memory=True)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, **common),
        DataLoader(valid_set, batch_size=batch_size, shuffle=False, **common),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, **common),
        train_set,
    )


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for images, target in loader:
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        logits = model(images)
        probabilities.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy().tolist())
        labels.extend(target.numpy().tolist())
    return np.asarray(labels), np.asarray(probabilities)


def metrics_at_threshold(labels, probabilities, threshold=0.5):
    prediction = (probabilities >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, prediction, labels=[0, 1], zero_division=0
    )
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "confusion_matrix": confusion_matrix(labels, prediction, labels=[0, 1]).tolist(),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
    }


def tune_threshold(labels, probabilities):
    candidates = np.linspace(0.05, 0.95, 181)
    return float(max(candidates, key=lambda value: f1_score(
        labels, (probabilities >= value).astype(int), average="macro"
    )))


def train_candidate(model_name, batch_size, epochs, patience, run_dir, status):
    import timm

    train_loader, valid_loader, _, train_set = make_loaders(batch_size)
    counts = np.bincount(train_set.targets, minlength=2)
    class_weights = len(train_set) / (2.0 * counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, device="cuda", dtype=torch.float32),
        label_smoothing=0.03,
    )
    model = timm.create_model(model_name, pretrained=True, num_classes=2)
    model = model.to("cuda", memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    short_name = model_name.split(".")[0]
    candidate_dir = run_dir / short_name
    candidate_dir.mkdir(parents=True, exist_ok=True)
    best_path = candidate_dir / "best.pt"
    history, best_score, best_epoch, stale = [], -1.0, 0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for images, target in train_loader:
            images = images.to("cuda", non_blocking=True).contiguous(memory_format=torch.channels_last)
            target = target.to("cuda", non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(target)
            seen += len(target)
        scheduler.step()
        labels, probabilities = predict(model, valid_loader, "cuda")
        validation = metrics_at_threshold(labels, probabilities)
        record = {"epoch": epoch, "train_loss": total_loss / seen,
                  "learning_rate": optimizer.param_groups[0]["lr"], **validation}
        history.append(record)
        write_json(candidate_dir / "history.json", history)
        if validation["macro_f1"] > best_score + 1e-4:
            best_score, best_epoch, stale = validation["macro_f1"], epoch, 0
            torch.save({"model_name": model_name, "state_dict": model.state_dict(),
                        "epoch": epoch, "validation": validation,
                        "class_names": CLASS_NAMES, "input_size": 224,
                        "normalization": {"mean": [0.485, 0.456, 0.406],
                                          "std": [0.229, 0.224, 0.225]}}, best_path)
        else:
            stale += 1
        status.update(state="training", model=model_name, epoch=epoch, batch_size=batch_size,
                      latest=record, best_epoch=best_epoch, best_macro_f1=best_score,
                      updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        write_json(OUT / "status.json", status)
        print(json.dumps({"model": model_name, **record, "best_macro_f1": best_score}), flush=True)
        if stale >= patience:
            break
    result = {"model_name": model_name, "batch_size": batch_size, "best_epoch": best_epoch,
              "best_macro_f1": best_score, "checkpoint": str(best_path)}
    del model, optimizer, scaler, train_loader, valid_loader
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing CPU fallback.")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    set_seed(42)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True)
    status = {
        "state": "starting", "pid": os.getpid(), "run_id": run_id,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "python": platform.python_version(), "models": MODEL_NAMES,
        "completed": [], "selection_metric": "validation macro-F1",
    }
    write_json(OUT / "status.json", status)
    results = []
    for model_name in MODEL_NAMES:
        completed = None
        for batch_size in dict.fromkeys([args.batch, 24, 16, 8]):
            try:
                completed = train_candidate(model_name, batch_size, args.epochs, args.patience,
                                            run_dir, status)
                break
            except torch.cuda.OutOfMemoryError:
                print(f"CUDA OOM for {model_name} at batch {batch_size}; retrying smaller batch", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
        if completed is None:
            status.setdefault("skipped", []).append({"model": model_name, "reason": "CUDA OOM"})
            write_json(OUT / "status.json", status)
            continue
        results.append(completed)
        status["completed"].append(completed)
        write_json(run_dir / "comparison.json", results)
        write_json(OUT / "status.json", status)
    if not results:
        raise RuntimeError("No candidate completed training.")

    winner = max(results, key=lambda item: item["best_macro_f1"])
    status.update(state="final_test", winner=winner)
    write_json(OUT / "status.json", status)
    import timm
    checkpoint = torch.load(winner["checkpoint"], map_location="cuda", weights_only=False)
    model = timm.create_model(checkpoint["model_name"], pretrained=False, num_classes=2)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to("cuda", memory_format=torch.channels_last)
    _, valid_loader, test_loader, _ = make_loaders(winner["batch_size"])
    valid_labels, valid_probabilities = predict(model, valid_loader, "cuda")
    threshold = tune_threshold(valid_labels, valid_probabilities)
    test_labels, test_probabilities = predict(model, test_loader, "cuda")
    test_metrics = metrics_at_threshold(test_labels, test_probabilities, threshold)
    checkpoint["mature_probability_threshold"] = threshold
    checkpoint["test_metrics"] = test_metrics
    final_path = OUT / "models" / "gourd_maturity_classifier_best.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, final_path)
    final = {
        "winner": winner, "threshold_tuned_on_validation": threshold,
        "test_metrics": test_metrics, "final_checkpoint": str(final_path),
        "dataset_counts": {"train": 4897, "valid": 545, "test": 462},
        "selection_note": "Candidates selected only on validation macro-F1; test used once for winner.",
    }
    write_json(run_dir / "final_report.json", final)
    status.update(state="complete", final_report=str(run_dir / "final_report.json"),
                  final_checkpoint=str(final_path))
    write_json(OUT / "status.json", status)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        path = OUT / "status.json"
        previous = json.loads(path.read_text()) if path.exists() else {}
        previous.update(state="failed", error=error)
        write_json(path, previous)
        raise
