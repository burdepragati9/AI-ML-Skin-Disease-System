# TODO - Skin model improvements

- [x] Implement hard 60% confidence threshold + TTA only in model/predict.py.
- [x] Save wrongly predicted validation images into analytics/misclassified/<True>_as_<Pred>/.

- [ ] Add Grad-CAM heatmap visualization for predictions.
- [ ] Improve Tinea vs Psoriasis separation: more diverse Tinea images + sharper lesion-focused crops.
- [ ] Reduce overconfidence: add label_smoothing=0.1.
- [ ] Unfreeze last 20 MobileNetV2 layers instead of 10 for final fine-tuning.
- [ ] Lower Stage 2 learning rate to 1e-5.
- [ ] Add early stopping patience=4.
- [ ] Generate classification report and per-class F1 score after training.

