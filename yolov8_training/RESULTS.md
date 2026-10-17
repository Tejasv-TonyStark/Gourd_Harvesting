# YOLOv8s Gourd Detection Results (Beans Excluded)

## Dataset preparation

- Original dataset was not modified.
- Bean annotations were removed.
- Images with no remaining gourd annotations were omitted.
- Remaining classes were reindexed to:
  - 0: `bitter_gourd`
  - 1: `bottle_gourd`
  - 2: `ivy_gourd`
  - 3: `snake_gourd`
- Train: 522 images
- Validation: 107 images
- Test: 108 images
- Removed: 993 bean boxes

## Training configuration

- Model: YOLOv8s pretrained on COCO (`yolov8s.pt`)
- Image size: 640
- Batch size: 8
- Maximum epochs: 100
- Early stopping patience: 20
- Completed epochs: 60 (stopped early)
- Device: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB)
- Seed: 42; deterministic mode enabled
- Best checkpoint selected at CSV epoch 48 (the 49th training epoch)

## Best validation checkpoint

| Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---:|---:|---:|---:|
| 0.7824 | 0.7435 | 0.8150 | 0.3488 |

## Held-out test results

| Class | Images | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|---:|
| All | 108 | 518 | 0.758 | 0.694 | 0.756 | 0.314 |
| bitter_gourd | 42 | 154 | 0.798 | 0.786 | 0.850 | 0.394 |
| bottle_gourd | 38 | 125 | 0.744 | 0.792 | 0.834 | 0.346 |
| ivy_gourd | 16 | 185 | 0.754 | 0.524 | 0.667 | 0.243 |
| snake_gourd | 13 | 54 | 0.737 | 0.675 | 0.674 | 0.272 |

## Test speed

- Preprocess: 1.2 ms/image
- Inference: 11.0 ms/image
- Postprocess: 1.2 ms/image

The test split was used only after training and checkpoint selection. The YOLO26 results retained elsewhere were obtained on a different five-class setup that included beans, so they are not a strict apples-to-apples comparison with this four-class run.
