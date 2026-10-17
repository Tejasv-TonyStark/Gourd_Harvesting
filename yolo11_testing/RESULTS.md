# YOLO11n Held-Out Test Results

This is an independent test evaluation of `gourd_web_app/best_wb.pt` on the same bean-free 108-image test split used to evaluate YOLOv8s.

| Class | Images | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|---:|
| All | 108 | 518 | 0.818 | 0.683 | 0.810 | 0.385 |
| bitter_gourd | 42 | 154 | 0.728 | 0.831 | 0.886 | 0.448 |
| bottle_gourd | 38 | 125 | 0.903 | 0.712 | 0.853 | 0.394 |
| ivy_gourd | 16 | 185 | 0.775 | 0.589 | 0.724 | 0.292 |
| snake_gourd | 13 | 54 | 0.866 | 0.598 | 0.778 | 0.406 |

## Evaluation configuration

- Checkpoint: `C:\Reinforcement\gourd_web_app\best_wb.pt`
- Dataset: `C:\Reinforcement\yolov8_gourd_dataset\data.yaml`
- Split: test
- Image size: 640
- Batch size: 8
- Device: NVIDIA GeForce RTX 3050 Laptop GPU
- Test speed: 1.9 ms preprocessing, 6.0 ms inference, and 1.9 ms postprocessing per image
