# YOLO26s Held-Out Test Results (Beans Excluded)

This evaluation used the same 108 bean-free test images and 518 gourd boxes used for the YOLO8s and YOLO11n tests. YOLO26s was trained with five classes, including beans at class ID 0, so the gourd-only test annotations retain the model's original class IDs 1 through 4. Predictions and evaluation were restricted to those four gourd classes.

| Class | Images | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|---:|
| All | 108 | 518 | 0.821 | 0.673 | 0.755 | 0.334 |
| bitter_gourd | 42 | 154 | 0.780 | 0.806 | 0.867 | 0.419 |
| bottle_gourd | 38 | 125 | 0.871 | 0.720 | 0.802 | 0.346 |
| ivy_gourd | 16 | 185 | 0.725 | 0.611 | 0.696 | 0.256 |
| snake_gourd | 13 | 54 | 0.907 | 0.556 | 0.655 | 0.314 |

## Evaluation configuration

- Model: YOLO26s
- Checkpoint: `C:\Reinforcement\detection\runs\20261009_153141\yolo26s_b2\weights\best.pt`
- Dataset: `C:\Reinforcement\yolo26s_gourd_test_dataset\data.yaml`
- Split: test
- Image size: 640
- Batch size: 8
- Classes evaluated: 1, 2, 3, 4 (all gourds; beans excluded)
- Device: NVIDIA GeForce RTX 3050 Laptop GPU
- Test speed: 1.1 ms preprocessing, 11.3 ms inference, and 0.9 ms postprocessing per image
