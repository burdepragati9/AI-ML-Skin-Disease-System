# Regression Root-Cause Analysis — Prediction Pipeline

**Date:** 2026-08-05
**Scope:** Diagnostic only. **No code, model, threshold, dataset, class-mapping, or frontend changes were made.**
**Branch under analysis:** `multi-model-enhancement` @ `b2aa6ba2` ("Update prediction and Gemini configuration")
**Known-good baseline (original):** `main` @ `f6f2e0b7` ("Stable working version before multi-model and AI enhancements")

---

## 1. What the original (working) logic did

From `git show f6f2e0b7:...` the original pipeline was **single-model**:

```
Upload → model/predict.py (skin_model.keras, MobileNetV2)
   → Test-Time Augmentation (TTA, 8 samples averaged)
   → best class + confidence
   → IF confidence < 60%  →  call Gemini (recognize_with_ai)  [AI = FALLBACK ONLY]
   → ELSE                 →  use ML result
```

- **One model** (`skin_model.keras`).
- **TTA** (8 augmented inferences averaged) stabilised probabilities.
- **AI was a fallback** — only invoked when the ML confidence dropped below the hard 60% threshold.
- Final disease = ML result unless ML was low-confidence, in which case AI answered.

## 2. What the current (broken) logic does

From the working tree `backend/services/prediction_service.py`, `utils/model_manager.py`,
`utils/prediction_comparison.py`:

```
predict_disease(image)
   manager.predict_with_all_models(image)      # 3 models, NO TTA (single forward pass each)
   compare_predictions(...)                    # consensus
   majority_vote(...)                          # equal-weight HARD vote
   soft_vote(...)                              # average probabilities
   verify_multi_model_predictions_with_ai(...) # called UNCONDITIONALLY (every image)
   build_ai_verification_summary(...)          # final_class = majority | soft | AI-override
   predicted_disease = ai_verification_summary["final_class"] or soft["selected_class"]
   return predicted_disease, prediction_source
```

Key differences introduced:
| Aspect | Original (f6f2e0b7) | Current (b2aa6ba2) |
|---|---|---|
| Models | 1 (MobileNetV2) | 3 (MobileNetV2, EfficientNetB0, DenseNet121) |
| TTA | Yes (8 samples) | **No** (single pass) |
| Final decision | ML result, or AI if ML < 60% | **Hard majority vote / soft vote, possibly AI-overridden** |
| AI call | Only when ML confidence < 60% | **Unconditionally on every image** |
| Threshold gate | Hard 60% in `predict.py` | `AI_VERIFICATION_THRESHOLD` defined but **never enforced** |

---

## Answers to the 8 diagnostic questions

### 1. Is AI now running for every prediction? — **YES (once a Gemini key exists)**
`prediction_service.predict_disease()` calls `verify_multi_model_predictions_with_ai(...)`
**unconditionally** for every image. The only gate is `GEMINI_API_KEY` presence. The config flags
`ENABLE_AI_VERIFICATION` and `AI_VERIFICATION_THRESHOLD` are **defined but never read** anywhere in
the prediction path. The original "only if confidence < threshold" fallback condition is **gone**.

### 2. Is the final disease being overwritten? — **YES**
```
predicted_disease = ai_verification_summary.get("final_class") or soft.get("selected_class")
```
`final_class` is produced by `build_ai_verification_summary()` which can return:
- consensus (unanimous), **majority**, **soft vote**, or **AI override** (if AI conf ≥ final conf + 10).
In addition, the "safety validation gate" can reassign `predicted_disease` to the soft-vote class,
and force `prediction_source = "majority_voting"` even when it was actually soft/AI. So the final
disease is **not simply the majority vote** — it is repeatedly overwritten.

### 3. Is the confidence threshold still active? — **NO**
The original 60% threshold gate in `model/predict.py` is bypassed because the route now calls
`predict_disease()` (the multi-model path), not `predict_image()`. `AI_VERIFICATION_THRESHOLD = 70.0`
exists in `utils/config.py` but is **never referenced** in the code. The threshold-based AI fallback
has been effectively removed.

### 4. Is majority voting still the final decision? — **NO**
Majority voting is only one input. The final class can be soft-voting or AI-overridden, and the
"safety validation gate" can forcibly replace it. Moreover, the majority itself uses **equal-weight
hard voting**, which is the mechanism that lets low-confidence wrong models override a
high-confidence correct model.

### 5. Why are Acne images becoming Vitiligo?
Equal-weight hard majority voting amplifies individual-model bias. From the end-to-end trace of a
real Acne image (`acne-cystic-142.jpeg`):
| Model | Acne | Psoriasis | Tinea | Vitiligo | Pred |
|---|---|---|---|---|---|
| mobilenetv2 | 13.3 | 10.3 | 21.5 | **54.9** | Vitiligo ❌ |
| efficientnetb0 | 22.9 | 15.0 | 18.4 | **43.7** | Vitiligo ❌ |
| densenet121 | **87.7** | 0.1 | 12.2 | 0.0 | Acne ✅ |
```
vote_counts    = {'Vitiligo': 2, 'Acne': 1}
selected_class = Vitiligo   (majority)
FINAL          = Vitiligo (49.3%)
```
Two low-confidence biased models (54.9% + 43.7%) override DenseNet121's high-confidence correct
Acne (87.7%). The removal of TTA also changed per-model probabilities vs. the original single-model path.

### 6. Why are Tinea images becoming Psoriasis?
Same mechanism. Full-set diagnosis (113 true-Tinea images):
- **EfficientNetB0 predicted Tinea on 0/113 images** (never emits Tinea).
- MobileNetV2 predicted Psoriasis on 67/113; DenseNet121 predicted Psoriasis on 40/113.
- Final `predict_disease` → Tinea 55 (48.7%), Psoriasis 46 (40.7%).
When MobileNetV2 and EfficientNetB0 both vote Psoriasis on a Tinea image, the equal-weight hard
majority selects Psoriasis even though DenseNet121 may be correct. The broken EfficientNetB0
(0/113 Tinea) is a heavy drag on the ensemble.

### 7. Why does face consent now appear for almost every image?
`backend/services/detect_face.py` is too permissive:
- **RetinaFace threshold lowered to 0.3** (default is 0.5) — detects weak/partial matches.
- **`MIN_FACE_AREA_RATIO` lowered from 0.01 → 0.005** (comment: "to catch smaller valid faces") —
  lets tiny detections (skin texture/noise) pass.
- **Aggressive MediaPipe fallback** (threshold 0.5) runs whenever RetinaFace returns no face and can
  false-positive on skin-lesion images.
Failures are NOT treated as `face_detected=true` (error handling returns False / HTTP 500), so this
is not an error-handling bug — it is over-sensitivity in the detector thresholds.

### 8. Which files introduced the regression?
| File | Status | Role in regression |
|---|---|---|
| `backend/services/prediction_service.py` | **NEW** | Replaced single-model path; unconditional AI call; final-class overwrite logic |
| `utils/model_manager.py` | **NEW** | Multi-model ensemble; equal-weight hard majority; **TTA removed** |
| `utils/prediction_comparison.py` | **NEW** | Majority/soft/AI decision + override rules |
| `utils/config.py` | MODIFIED | Added `ENABLE_AI_VERIFICATION`/`AI_VERIFICATION_THRESHOLD` but **not enforced** |
| `utils/ai_recognition.py` | MODIFIED | Added `verify_multi_model_predictions_with_ai` (called every time) |
| `backend/services/detect_face.py` | MODIFIED | Over-sensitive face thresholds (0.3 + 0.005 area + MediaPipe fallback) |
| `backend/routes/prediction.py` | NEW path | Now calls `predict_disease()` instead of `model.predict_image()` |

### 9. Which specific code block changed the original working logic?
**This block in `backend/services/prediction_service.py` replaced the original threshold-gated AI fallback:**
```python
# AI verification (enabled via GEMINI_API_KEY/config inside ai_recognition)
ai_result = verify_multi_model_predictions_with_ai(            # <-- no threshold check
    image=image,
    model_predictions=model_predictions,
    comparison_summary=...,
)
ai_verification_summary = build_ai_verification_summary(..., ai_result=ai_result)
predicted_disease = ai_verification_summary.get("final_class") or soft.get("selected_class")
prediction_source = ai_verification_summary.get("source")
```
Combined with `utils/model_manager.predict_with_all_models()` (equal-weight hard majority, no TTA),
this is the exact code that changed the previously-correct behavior.

---

## Summary of the regression chain
1. **`multi-model-enhancement` branch** replaced the single-model + low-confidence-AI-fallback
   pipeline with a 3-model **equal-weight hard-majority ensemble**.
2. **TTA was removed**, changing every per-model probability.
3. **EfficientNetB0 is broken for Tinea** (0/113) and both MobileNetV2/EfficientNetB0 carry
   Vitiligo/Psoriasis biases → the hard majority overrides the one correct high-confidence model.
4. **AI is now called unconditionally** and the final class can be overwritten by soft-vote or AI,
   so the confidence threshold (60%/70%) is no longer active.
5. **Face detection was made over-sensitive** (threshold 0.3, area ratio 0.005, MediaPipe fallback),
   so consent appears for almost every image.

## Recommended fixes (NOT applied — pending your approval)
1. **Restore threshold-gated AI fallback**: only call `verify_multi_model_predictions_with_ai` when
   the ensemble confidence is below `AI_VERIFICATION_THRESHOLD` (and honor `ENABLE_AI_VERIFICATION`).
2. **Use confidence-weighted voting (or soft voting)** instead of equal-weight hard majority so a
   high-confidence correct model (DenseNet121 87.7% Acne) is not overridden by two sub-50% votes.
3. **Restore TTA** in `predict_with_model()` (matches the original stable behavior).
4. **Fix or down-weight EfficientNetB0** for Tinea (it never emits Tinea).
5. **Tighten face detection**: raise RetinaFace threshold toward 0.5, restore `MIN_FACE_AREA_RATIO`
   to 0.01, and make the MediaPipe fallback stricter (or disable for non-face skin images).
6. Optionally **revert to the `f6f2e0b7` single-model path** as the safest immediate fix.

No application code was modified during this analysis.
</content>
