# Final Gourd Detection and Maturity Classification Report

Selected pipeline: **YOLO26s detector -> MobileNetV3Large maturity classifier**.

| Task | Metric | Held-out test result |
|---|---|---:|
| Detection | Precision | 71.65% |
| Detection | Recall | 61.89% |
| Detection | mAP50 | 67.08% |
| Detection | mAP50-95 | 28.81% |
| Classification | Accuracy | 84.63% |
| Classification | Balanced accuracy | 77.89% |
| Classification | Macro-F1 | 77.59% |
| Classification | ROC-AUC | 85.88% |

Detection and classification metrics are reported separately because the datasets
do not provide a single held-out set with all box, type, and maturity ground truth.
External web-image predictions are qualitative and are not included in accuracy.

## Folder layout

- `detection/weights`: selected detector weight.
- `detection/metrics`: validation and held-out test metrics, including per-class AP.
- `detection/plots`: training/validation plots.
- `classification/weights`: selected classifier and EfficientNet comparison candidate.
- `classification/metrics`: official report, confusion matrix, summary metrics, and reproducibility check.
- `classification/plots`: ROC, precision-recall, confusion matrix, and training comparison.

The private dataset, sample images, and per-image prediction records are excluded from this repository.
