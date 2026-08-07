# End-to-End Image Trace — Face Privacy Consent & Prediction Pipeline

**Date:** 2026-08-05
**Method:** `trace_e2e_image.py` — replicates the exact production flow (frontend → `/predict/detect-face` → `/predict/predict`).
**Test image:** `analytics/misclassified/true_Acne_pred_Vitiligo/acne-cystic-142.jpeg` (ground truth = **Acne**)
**No code was modified.** This is a pure diagnostic trace.

---

## 1. Filename / Image identity

| Step | Value |
|---|---|
| Uploaded filename | `acne-cystic-142.jpeg` |
| File bytes | identical original bytes read by both endpoints |
| Image received by React | `File` object = original upload bytes |
| Sent to `/predict/detect-face` | `file.file` (original bytes) |
| Sent to `/predict/predict` | `file.file` (original bytes; opened fresh) |
| **Same file to both endpoints?** | **YES** |

## 2. Intermediate images (debug, saved to `trace_debug/`)

| Image | Size | Mean | Min | Max | MAE vs original |
|---|---|---|---|---|---|
| original (resized 160) | (160,160,3) | 70.9 | 0 | 214 | — |
| after face detection (160) | (160,160,3) | 70.9 | 0 | 214 | **0.0000** |
| passed to prediction (160) | (160,160,3) | 70.9 | 0 | 214 | **0.0000** |

**Prediction uses the ORIGINAL uploaded skin image → TRUE.**
The face-detection step only **reads** the image to compute a consent boolean; it never crops,
mutates, or returns an image. `/predict/predict` independently re-opens `file.file` and passes
the original to `predict_disease()`.

## 3. Preprocessing details (unchanged)

| Property | Value |
|---|---|
| target size | 160×160 |
| color conversion | RGB (no BGR) |
| normalization | none in preprocess (0–255); model has internal Rescaling layer |
| dtype | float32 |
| preprocess shape | (1, 160, 160, 3), range [0.0, 214.0] |

## 4. Model weight files loaded (correct)

| Model | .keras file | Size | Modified |
|---|---|---|---|
| mobilenetv2 | mobilenet_model.keras | 10,956,031 | 2026-06-04 16:25 |
| efficientnetb0 | efficientnet_model.keras | 18,376,840 | 2026-06-04 16:51 |
| densenet121 | densenet_model.keras | 30,730,224 | 2026-06-04 18:14 |

All three are the intended production weights (each has internal Rescaling layer, 4-class output).

## 5. Raw probability vectors (this Acne image)

| Model | Acne | Psoriasis | Tinea | Vitiligo | Pred |
|---|---|---|---|---|---|
| mobilenetv2 | 13.3% | 10.3% | 21.5% | **54.9%** | Vitiligo ❌ |
| efficientnetb0 | 22.9% | 15.0% | 18.4% | **43.7%** | Vitiligo ❌ |
| densenet121 | **87.7%** | 0.1% | 12.2% | 0.0% | Acne ✅ |

## 6. Majority vote result

```
vote_counts    : {'Vitiligo': 2, 'Acne': 1}
selected_class : Vitiligo (supporting: ['mobilenetv2', 'efficientnetb0'])
```

## 7. Final production prediction

```
predict_disease -> Vitiligo (49.3%) source=majority_voting
```

---

## Root Cause — where predictions first become incorrect

**The prediction never becomes incorrect due to the image, preprocessing, weights, or face-consent
workflow. The error originates entirely at the ML model/ensemble output level.**

Concretely, for this Acne image:
1. **Individual model bias (ROOT):** MobileNetV2 (54.9%) and EfficientNetB0 (43.7%) both
   independently predict **Vitiligo** on this Acne image. Only DenseNet121 (87.7%) is correct.
2. **Majority voting (amplifier):** The hard class-count vote selects **Vitiligo 2/3**, overriding
   DenseNet121's high-confidence correct Acne (87.7%).
3. Soft voting would also pick Vitiligo here (weights dominated by the two biased models).

This is the **same root cause** as the Tinea→Psoriasis issue: MobileNetV2 and EfficientNetB0 carry
per-class biases (they rarely emit Tinea — EfficientNetB0 predicted Tinea on 0/113 Tinea images),
and the equal-weight hard majority vote lets a low-confidence wrong majority override a
high-confidence correct model.

---

## Final Answers

| Question | Answer |
|---|---|
| **Did the prediction image change?** | **NO.** MAE = 0.0; prediction uses the original upload. |
| **Did preprocessing change?** | **NO.** RGB, 0–255, float32, internal Rescaling — unchanged. |
| **Are wrong model weights loaded?** | **NO.** All three production `.keras` files verified. |
| **Did the face consent implementation affect the prediction pipeline?** | **NO.** `/predict/detect-face` only returns a consent boolean; `/predict/predict` re-opens the original file independently. |
| **Exact files responsible?** | The **model weights themselves** (`mobilenet_model.keras`, `efficientnet_model.keras`) carry the per-class bias, and the **ensemble decision logic** in `utils/prediction_comparison.py` (`majority_vote`) + `backend/services/prediction_service.py` (`predict_disease`) amplify it via equal-weight hard majority voting. |

## Recommended fixes (NOT yet applied — pending approval)
1. **Use confidence-weighted voting** (weight each model by softmax confidence) in `majority_vote`
   so a high-confidence correct model (DenseNet121 @ 87.7%) can outweigh two low-confidence wrong
   models (54.9% / 43.7%).
2. **Prefer soft voting** as the default (it is more robust than hard majority).
3. **Retrain or reweight MobileNetV2 / EfficientNetB0** to correct the Vitiligo and Psoriasis
   per-class biases (the ultimate source of the errors).
4. Face-consent workflow needs **no change** — it does not affect predictions.
