# Maturity classification

This compares two ImageNet-pretrained classifiers on the existing mature/immature
splits: MobileNetV3Large and EfficientNetV2B0. It uses GPU-only mixed-precision
training, class-weighted loss, mild image augmentation and early stopping.

Model selection uses validation macro-F1 because the dataset contains about three
times as many mature images as immature images. The test split is evaluated once,
after model selection. The mature-probability threshold is tuned only on validation.

Live status is written to `status.json`. Each run stores histories and checkpoints
under `runs/<run-id>`. The selected checkpoint is copied to
`models/gourd_maturity_classifier_best.pt`.

External web images are qualitative out-of-distribution checks. They are not part
of the reported test accuracy unless their maturity labels are independently
verified by an agricultural expert.
