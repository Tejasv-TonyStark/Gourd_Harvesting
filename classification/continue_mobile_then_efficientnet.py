"""Finish MobileNet to epoch 40, then train EfficientNetV2B0 in two phases."""
from __future__ import annotations

import gc
import json
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import timm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from classification.train_classification import (
    CLASS_NAMES,
    OUT,
    RUNS,
    make_loaders,
    metrics_at_threshold,
    predict,
    set_seed,
    tune_threshold,
    write_json,
)

MOBILE_SOURCE_RUN = ROOT / "classification" / "runs" / "20261009_195630" / "mobilenetv3_large_100"
MOBILE_SOURCE_CHECKPOINT = MOBILE_SOURCE_RUN / "best.pt"
MOBILE_SOURCE_HISTORY = MOBILE_SOURCE_RUN / "history.json"
MOBILE_NAME = "mobilenetv3_large_100.ra_in1k"
EFFICIENT_NAME = "tf_efficientnetv2_b0.in1k"
BATCH_SIZE = 32


def class_weighted_loss(train_set):
    counts = np.bincount(train_set.targets, minlength=2)
    weights = len(train_set) / (2.0 * counts)
    return nn.CrossEntropyLoss(
        weight=torch.tensor(weights, device="cuda", dtype=torch.float32),
        label_smoothing=0.03,
    )


def train_range(model, model_name, train_loader, valid_loader, criterion,
                optimizer, scheduler, epoch_numbers, candidate_dir, history,
                best_score, best_epoch, best_path, status, phase,
                patience=None):
    scaler = torch.amp.GradScaler("cuda")
    stale = 0
    for epoch in epoch_numbers:
        model.train()
        total_loss, seen = 0.0, 0
        for images, target in train_loader:
            images = images.to("cuda", non_blocking=True).contiguous(memory_format=torch.channels_last)
            target = target.to("cuda", non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = criterion(model(images), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(target)
            seen += len(target)
        scheduler.step()
        labels, probabilities = predict(model, valid_loader, "cuda")
        validation = metrics_at_threshold(labels, probabilities)
        record = {"epoch": int(epoch), "phase": phase,
                  "train_loss": total_loss / seen,
                  "learning_rate": optimizer.param_groups[0]["lr"], **validation}
        history.append(record)
        if validation["macro_f1"] > best_score + 1e-4:
            best_score, best_epoch, stale = validation["macro_f1"], int(epoch), 0
            torch.save({"model_name": model_name, "state_dict": model.state_dict(),
                        "epoch": int(epoch), "validation": validation,
                        "class_names": CLASS_NAMES, "input_size": 224,
                        "normalization": {"mean": [0.485, 0.456, 0.406],
                                          "std": [0.229, 0.224, 0.225]}}, best_path)
        else:
            stale += 1
        write_json(candidate_dir / "history.json", history)
        status.update(state="training", model=model_name, phase=phase, epoch=int(epoch),
                      batch_size=BATCH_SIZE, latest=record, best_epoch=best_epoch,
                      best_macro_f1=best_score,
                      updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        write_json(OUT / "status.json", status)
        print(json.dumps({"model": model_name, **record,
                          "best_macro_f1": best_score}), flush=True)
        if patience is not None and stale >= patience:
            break
    return best_score, best_epoch


def finish_mobile(run_dir, status):
    if not MOBILE_SOURCE_CHECKPOINT.exists():
        raise FileNotFoundError(MOBILE_SOURCE_CHECKPOINT)
    candidate_dir = run_dir / "mobilenetv3_large_100"
    candidate_dir.mkdir(parents=True)
    best_path = candidate_dir / "best.pt"
    shutil.copy2(MOBILE_SOURCE_CHECKPOINT, best_path)
    history = json.loads(MOBILE_SOURCE_HISTORY.read_text())
    write_json(candidate_dir / "history.json", history)
    checkpoint = torch.load(best_path, map_location="cuda", weights_only=False)
    model = timm.create_model(MOBILE_NAME, pretrained=False, num_classes=2)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to("cuda", memory_format=torch.channels_last)
    train_loader, valid_loader, _, train_set = make_loaders(BATCH_SIZE)
    criterion = class_weighted_loss(train_set)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=1e-6)
    best_score = float(checkpoint["validation"]["macro_f1"])
    best_epoch = int(checkpoint["epoch"])
    best_score, best_epoch = train_range(
        model, MOBILE_NAME, train_loader, valid_loader, criterion,
        optimizer, scheduler, range(29, 41), candidate_dir, history,
        best_score, best_epoch, best_path, status, "continued_full_finetune"
    )
    result = {"model_name": MOBILE_NAME, "batch_size": BATCH_SIZE,
              "best_epoch": best_epoch, "best_macro_f1": best_score,
              "checkpoint": str(best_path), "training": "epochs 1-40 full fine-tuning"}
    del model, train_loader, valid_loader, optimizer
    gc.collect(); torch.cuda.empty_cache()
    return result


def train_efficientnet(run_dir, status):
    candidate_dir = run_dir / "tf_efficientnetv2_b0"
    candidate_dir.mkdir(parents=True)
    best_path = candidate_dir / "best.pt"
    train_loader, valid_loader, _, train_set = make_loaders(BATCH_SIZE)
    criterion = class_weighted_loss(train_set)
    model = timm.create_model(EFFICIENT_NAME, pretrained=True, num_classes=2)
    model = model.to("cuda", memory_format=torch.channels_last)
    history, best_score, best_epoch = [], -1.0, 0

    # Phase 1: train only the new binary classification head.
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.get_classifier().parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(model.get_classifier().parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-4)
    best_score, best_epoch = train_range(
        model, EFFICIENT_NAME, train_loader, valid_loader, criterion,
        optimizer, scheduler, range(1, 6), candidate_dir, history,
        best_score, best_epoch, best_path, status, "frozen_backbone"
    )

    # Phase 2: unfreeze the entire backbone and fine-tune gently.
    for parameter in model.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=5e-7)
    best_score, best_epoch = train_range(
        model, EFFICIENT_NAME, train_loader, valid_loader, criterion,
        optimizer, scheduler, range(6, 36), candidate_dir, history,
        best_score, best_epoch, best_path, status, "unfrozen_finetune", patience=8
    )
    result = {"model_name": EFFICIENT_NAME, "batch_size": BATCH_SIZE,
              "best_epoch": best_epoch, "best_macro_f1": best_score,
              "checkpoint": str(best_path),
              "training": "5 frozen-backbone epochs plus up to 30 unfrozen fine-tuning epochs"}
    del model, train_loader, valid_loader, optimizer
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    set_seed(42)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True)
    status = {"state": "starting", "pid": os.getpid(), "run_id": run_id,
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "python": platform.python_version(),
              "plan": ["MobileNet epochs 29-40", "EfficientNet 5 frozen + 30 fine-tune"],
              "completed": [], "selection_metric": "validation macro-F1"}
    write_json(OUT / "status.json", status)

    mobile = finish_mobile(run_dir, status)
    status["completed"].append(mobile); write_json(OUT / "status.json", status)
    efficient = train_efficientnet(run_dir, status)
    status["completed"].append(efficient); write_json(OUT / "status.json", status)
    results = [mobile, efficient]
    write_json(run_dir / "comparison.json", results)
    winner = max(results, key=lambda item: item["best_macro_f1"])
    status.update(state="final_test", winner=winner); write_json(OUT / "status.json", status)

    checkpoint = torch.load(winner["checkpoint"], map_location="cuda", weights_only=False)
    model = timm.create_model(checkpoint["model_name"], pretrained=False, num_classes=2)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to("cuda", memory_format=torch.channels_last)
    _, valid_loader, test_loader, _ = make_loaders(BATCH_SIZE)
    valid_labels, valid_probabilities = predict(model, valid_loader, "cuda")
    threshold = tune_threshold(valid_labels, valid_probabilities)
    test_labels, test_probabilities = predict(model, test_loader, "cuda")
    test_metrics = metrics_at_threshold(test_labels, test_probabilities, threshold)
    checkpoint["mature_probability_threshold"] = threshold
    checkpoint["test_metrics"] = test_metrics
    final_path = OUT / "models" / "gourd_maturity_classifier_best.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, final_path)
    final = {"winner": winner, "candidates": results,
             "threshold_tuned_on_validation": threshold,
             "test_metrics": test_metrics, "final_checkpoint": str(final_path),
             "dataset_counts": {"train": 4897, "valid": 545, "test": 462},
             "selection_note": "Selected on validation macro-F1; test evaluated once for winner."}
    write_json(run_dir / "final_report.json", final)
    status.update(state="complete", final_report=str(run_dir / "final_report.json"),
                  final_checkpoint=str(final_path)); write_json(OUT / "status.json", status)
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
