# TODO - Diagnostics Fixes (Model Collapse + Confidence Inflation)

## Step 1 — app.py (confidence + validation + debugging)
- [ ] Fix ML confidence logic to correctly handle softmax/probabilities.
- [ ] Add debug logging: raw model outputs, softmax outputs, top-3 predictions/probabilities.
- [ ] Add runtime validation: assert sum(probabilities) ~= 1 and probs in [0,1].
- [ ] Ensure UI confidence percent = probs[best] * 100 exactly once.
- [ ] Add class-index consistency checks against class_names.json.
- [ ] Prevent fake high-confidence via probability/logit detection + entropy/margin checks.

## Step 2 — model/train.py (improve training)
- [ ] Replace MobileNetV2(alpha=0.35) with EfficientNetB0 or ResNet50.
- [ ] Increase epochs (20+), set batch size (8 or 16).
- [ ] Add class balancing using compute_class_weight.
- [ ] Add label smoothing (CategoricalCrossentropy label_smoothing=0.1) or equivalent.
- [ ] Add stronger augmentation: brightness, contrast, rotation, zoom, translation.
- [ ] Add staged fine-tuning: head training then unfreeze top backbone layers.
- [ ] Add confusion matrix + per-class accuracy reporting after training.

## Step 3 — training/self_learning.py (incremental + replay safety)
- [ ] Add replay balancing validation: ensure roughly equal replay samples per class.
- [ ] Validate class-label mapping before/after model expansion and training.
- [ ] Add confusion matrix generation after every retraining job.
- [ ] Ensure replay learning preserves old disease knowledge and reduces collapse.

## Step 4 — Verification
- [ ] Train model: python model/train.py
- [ ] Run diagnostics: python analytics/model_diagnostics.py
- [ ] Verify app confidence UI: no inflated 75–90% unless model supports it.
- [ ] Verify no single-class collapse: confusion matrix should show multiple predicted classes.

## Implementation Progress
- app.py: done ✅ (confidence fix + debug + runtime validation)
- model/train.py: pending
- training/self_learning.py: pending







