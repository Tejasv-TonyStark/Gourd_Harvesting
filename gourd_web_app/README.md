# GourdVision website

Static, privacy-preserving research web app for the selected two-stage gourd pipeline.

- YOLO11n `best_wb.pt` detects four gourd varieties: bitter, bottle, ivy, and snake gourd.
- The exact YOLO11n checkpoint was evaluated on 108 held-out images with 518 instances: precision 81.8%, recall 68.3%, mAP@0.5 81.0% (`../yolo11_testing/RESULTS.md`). These are detection metrics, not classification accuracy.
- MobileNetV3Large estimates mature or immature for every detected crop.
- Its held-out crop classification accuracy is 84.63% on 462 images (`../final_report/classification/metrics/test_metrics.json`). The two separate evaluations do not establish end-to-end scan accuracy.
- ONNX Runtime Web executes both models in the visitor's browser.
- The detector is byte-split into three static assets and reassembled unchanged in browser memory.
- Upload and rear-camera capture are supported.

Serve `dist/` over HTTP(S). Camera capture requires HTTPS or localhost.
