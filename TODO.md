# ML Prediction Pipeline Audit — TODO

## Objective
Find the exact Acne image that produces the incorrect ML prediction
(Vitiligo @ ~79.38% soft-vote confidence) and pinpoint the root cause.
READ-ONLY — no production code is modified during investigation.

## Steps
- [x] 1. Understand pipeline (read model_manager, prediction_service, prediction_comparison, config, routes, frontend)
- [x] 2. Verify class mapping files (class_names.json, class_indices.json, class_mapping.json, active_classes.json)
- [x] 3. Confirm AI is correctly skipped (source=ML, should_call_ai=False at 79.38% >= 70.0)
- [x] 4. Sweep CroppedData/Acne images — found 1 Vitiligo hit at 36.83% (not matching 79.38%)
- [x] 5. Identify the exact image producing Vitiligo @ ~79.38% = `history/uploads/Acne/search_Acne_129fa90406.jpg` (image_name="Acne 8.jpg", DB id=589, confidence=79.38, source=ML)
- [x] 6. Deep-audit the failed image (raw vectors, soft/majority vote, final) — reproduced Vitiligo @ 79.18%
- [x] 7. Pinpoint exact root cause — all 3 ML models independently predict Vitiligo (MobileNetV2 88.80%, EfficientNetB0 49.94%, DenseNet121 98.79%); soft vote averages to Vitiligo 79.18%
- [x] 8. Verify frontend reads prediction.predicted_disease / prediction.confidence / prediction.prediction_source directly (no mapping)
- [x] 9. Produce final root-cause report (raw probs, class mapping, voting, root cause, files)

## Audit scripts created (diagnostic only, no production code changed)
- deep_audit_129fa90406.py — full 6-step audit of the exact failing image

## Root cause
The ML ensemble itself is wrong. All three trained models (mobilenet_model.keras,
efficientnet_model.keras, densenet_model.keras) independently classify the Acne image
"Acne 8.jpg" as Vitiligo with high confidence. The soft-vote correctly averages them to
Vitiligo 79.18% (production log 79.38%). Class mapping, preprocessing, voting, and final
assignment are all correct. Gemini is NOT involved. The fix must be in the ML models/training,
not the pipeline.
