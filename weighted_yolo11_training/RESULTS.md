# Class-Weighted YOLO11n Experiment

## Setup

- Base model: pretrained `yolo11n.pt`
- Beans excluded; four gourd classes
- Class-weight power: `cls_pw=0.5`
- Applied weights: bitter 0.654, bottle 0.866, ivy 0.512, snake 1.967
- Training images: 522
- Validation images: 107
- Epochs: 50
- Batch size: 8
- Image size: 640
- Best CSV epoch: 46 (47th training epoch)
- Held-out test: 108 images and 518 instances

## Weighted-model test results

| Class | Images | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|---:|
| All | 108 | 518 | 0.758 | 0.707 | 0.779 | 0.344 |
| bitter_gourd | 42 | 154 | 0.738 | 0.822 | 0.876 | 0.421 |
| bottle_gourd | 38 | 125 | 0.768 | 0.736 | 0.804 | 0.351 |
| ivy_gourd | 16 | 185 | 0.745 | 0.601 | 0.704 | 0.259 |
| snake_gourd | 13 | 54 | 0.780 | 0.667 | 0.731 | 0.344 |

## Comparison with existing unweighted YOLO11n

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| Existing unweighted YOLO11n | 0.818 | 0.683 | 0.810 | 0.385 |
| New weighted YOLO11n | 0.758 | 0.707 | 0.779 | 0.344 |
| Change (weighted - existing) | -0.060 | +0.024 | -0.031 | -0.041 |

Snake-gourd recall increased from 0.598 to 0.667, but snake precision decreased from 0.866 to 0.780, snake mAP@0.5 decreased from 0.778 to 0.731, and snake mAP@0.5:0.95 decreased from 0.406 to 0.344. The existing unweighted `best_wb.pt` remains the recommended production checkpoint.
