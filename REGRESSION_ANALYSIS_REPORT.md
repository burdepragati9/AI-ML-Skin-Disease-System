# Prediction Pipeline Regression Analysis Report

## Executive Summary

The prediction pipeline regression was caused by **two independent changes** introduced during the Streamlit → FastAPI migration and subsequent "Face Privacy Consent" workflow implementation:

1. **AI Fallback Logic Regression**: The original threshold-based AI fallback (`confidence > 70` → use ML, else → call AI) was completely removed. Gemini AI now runs for **every** prediction and can override correct ML predictions.
2. **Face Detection Regression**: RetinaFace threshold was lowered from 0.5 to 0.3, minimum face area was lowered from 5% to 0.5%, and a MediaPipe fallback was added — causing false-positive face detections on almost every skin lesion image.

---

## Step 1 — Current Pipeline vs Original Logic

### Original Workflow (app.py @ commit f6f2e0b7)

```
Upload Image
↓
Run ML Model (single model)
↓
Calculate confidence + entropy
↓
should_use_ml = not (confidence < 80.0 OR entropy > ML_ENTROPY_FALLBACK_MAX)
↓
IF should_use_ml AND confidence > CONFIDENCE_THRESHOLD (70.0)
    → Use ML result, prediction_source = "ML"
ELSE
    → Call Gemini AI (recognize_with_ai)
    → If AI succeeds: Use AI result, prediction_source = "AI"
    → If AI fails: Fall back to ML result
```

### Current Workflow (prediction_service.py @ HEAD)

```
Upload Image
↓
RetinaFace Detection (separate /detect-face endpoint)
↓
Run ML Models (multi-model ensemble)
↓
Majority vote + Soft vote computed
↓
verify_multi_model_predictions_with_ai() called UNCONDITIONALLY  ← REGRESSION
↓
build_ai_verification_summary():
    IF ai_confidence >= final_confidence + 10.0:
        → AI OVERRIDES ML prediction  ← REGRESSION
↓
predicted_disease = ai_verification_summary.get("final_class")
```

### Verdict: The original threshold-based AI fallback workflow is NO LONGER implemented.

---

## Step 2 — AI Fallback Logic Verification

### Where Gemini AI is called:
- **File**: `backend/services/prediction_service.py`
- **Lines**: 64-74
- **Function**: `verify_multi_model_predictions_with_ai(image, model_predictions, comparison_summary)`

### Is AI called only when confidence < threshold?
**NO.** AI is called for **every** prediction unconditionally. There is no `if confidence < threshold:` check anywhere in `predict_disease()`.

### Printed values for every prediction:
| Variable | Value |
|---|---|
| `confidence` | From soft vote (e.g., 85.0%) |
| `threshold` | **NOT CHECKED** — config has `LOW_CONFIDENCE_THRESHOLD=70.0` and `AI_VERIFICATION_THRESHOLD=70.0` but neither is used |
| `should_use_ai` | Always `True` (no conditional gate exists) |
| `prediction_source` | `"ai_verification"` when AI overrides, else `"majority_voting"` / `"soft_voting"` / `"unanimous_voting"` |

### Verdict: AI is now running for every prediction. The threshold-based fallback has been bypassed.

---

## Step 3 — Final Prediction Assignment Verification

### Trace of every assignment:

1. **ML prediction** (`prediction_service.py` line 58):
   ```python
   model_predictions = manager.predict_with_all_models(image_array)
   ```
   → Correct. Each model returns `predicted_class` and `confidence`.

2. **Majority vote** (`prediction_service.py` line 60):
   ```python
   majority = majority_vote(model_predictions)
   ```
   → Correct. Returns `selected_class` with vote counts.

3. **Soft vote** (`prediction_service.py` line 61):
   ```python
   soft = manager.soft_vote(model_predictions)
   ```
   → Correct. Returns `selected_class` from averaged probabilities.

4. **Gemini prediction** (`prediction_service.py` lines 64-74):
   ```python
   ai_result = verify_multi_model_predictions_with_ai(image, model_predictions, comparison_summary)
   ```
   → Returns `ai_prediction`, `ai_confidence`, `verification_source`, `explanation`.

5. **Final response** (`prediction_service.py` lines 85-87):
   ```python
   predicted_disease = ai_verification_summary.get("final_class") or soft.get("selected_class")
   confidence = ai_verification_summary.get("final_confidence", soft.get("selected_confidence"))
   ```
   → **PROBLEM**: `final_class` comes from `build_ai_verification_summary()` which may have been overridden by AI.

### AI Override Logic (`prediction_comparison.py` lines 219-222):
```python
if ai_prediction_valid and ai_confidence >= final_confidence + 10.0:
    final_class = str(ai_prediction_raw)   # AI REPLACES ML
    final_confidence = ai_confidence
    source = "ai_verification"
```

### Verdict: YES, the final disease IS being overwritten after majority voting. If Gemini returns a confidence 10+ points higher than the ML confidence, the ML prediction is discarded.

---

## Step 4 — Majority Voting Verification

### Model predictions (from `predict_with_all_models`):
- Model 1 (mobilenetv2): `predicted_class`, `confidence` ✓
- Model 2 (efficientnetb0): `predicted_class`, `confidence` ✓
- Model 3 (densenet121): `predicted_class`, `confidence` ✓

### Vote counts (from `majority_vote`):
```python
vote_counts = Counter(p["predicted_class"] for p in model_predictions)
selected_class, _ = vote_counts.most_common(1)[0]
```
→ Correct. Returns the most common class with supporting models.

### Soft vote result (from `soft_vote`):
```python
avg_probs = np.mean(np.stack(probability_rows, axis=0), axis=0)
best_index = int(np.argmax(avg_probs))
```
→ Correct. Averages probabilities across models.

### Final selected result:
- Majority result: **Correct** but can be overridden
- Soft vote result: **Correct** but can be overridden
- Final selected result: **OVERWRITTEN** by AI in `build_ai_verification_summary()`

### Verdict: The majority result IS being replaced unexpectedly by the AI override logic.

---

## Step 5 — RetinaFace Verification

### RetinaFace response:
- `app.get(image_bgr)` returns a list of face objects
- Each face has `bbox` (bounding box) and `det_score` (confidence)

### face_detected / confidence / bounding boxes:
- `face_detected`: `True` if any valid face found
- `confidence`: `det_score` from RetinaFace (0.0-1.0)
- `bounding boxes`: `[x1, y1, x2, y2]` from `face_data.bbox`

### Why consent now appears for almost every image:

| Setting | Original | Current | Impact |
|---|---|---|---|
| `RETINAFACE_THRESHOLD` | 0.5 (InsightFace default) | **0.3** | More false-positive detections |
| `MIN_FACE_AREA_RATIO` | 0.05 (5%) | **0.005 (0.5%)** | Tiny false positives accepted |
| MediaPipe fallback | None | **Added** | Catches even more false positives |
| `is_face_complete` | Strict (5% min, 98% max, 3-boundary limit) | **Permissive** (0.5% min, 99% max, area-only) | Almost any detection passes |

### Failure handling verification:
- No face → `face_detected = false` ✓
- Service failure → HTTP 500 ✓ (in `routes/prediction.py` lines 127-130)
- Visible face → `face_detected = true` ✓
- Failures are NOT treated as `face_detected=true` ✓

### Verdict: The face detection pipeline correctly handles failures, but the thresholds are too low and the MediaPipe fallback causes excessive false positives on skin lesion images.

---

## Step 6 — Git History Comparison

### Commit History:
| Commit | Message | Significance |
|---|---|---|
| `f6f2e0b7` | Stable working version before multi-model and AI enhancements | **Last known working version** (Streamlit) |
| `b44517ac` | Working Streamlit version before React FastAPI migration | Streamlit version |
| `2b104c9a` | Updated Admin UI, Prediction History, Reports and System Monitoring | **FastAPI migration — regression introduced here** |
| `b2aa6ba2` | Update prediction and Gemini configuration | Added SAFETY VALIDATION GATE (latest HEAD) |

### Key findings from git diff:

**`prediction_service.py`** (created in `2b104c9a`):
- The original `app.py` had `ML_CONFIDENCE_FALLBACK_MIN = 80.0`, `should_use_ml`, `CONFIDENCE_THRESHOLD = 70.0`, and conditional AI call
- The new `prediction_service.py` calls `verify_multi_model_predictions_with_ai()` **unconditionally** — the threshold check was NOT carried over

**`prediction_comparison.py`** (created in `2b104c9a`):
- Added `build_ai_verification_summary()` with AI override at `ai_confidence >= final_confidence + 10.0`
- This override did not exist in the original `app.py`

**`detect_face.py`** (created in `2b104c9a`, modified in `b2aa6ba2`):
- Original version (at `b44517ac`): 5% min face size, 98% max, 2px margin, 3-boundary limit, no MediaPipe, no threshold setting (default 0.5)
- Current version: 0.3 threshold, 0.5% min area, MediaPipe fallback, permissive validation

### Verdict: The regression was introduced in commit `2b104c9a` during the Streamlit → FastAPI migration. The original threshold-based AI fallback logic was not carried over to the new `prediction_service.py`.

---

## Step 7 — Original Threshold Verification

### Search results for threshold constants:

| Constant | Location | Value | Used in prediction pipeline? |
|---|---|---|---|
| `LOW_CONFIDENCE_THRESHOLD` | `utils/config.py` line 61 | 70.0 | **NO** — not imported in `prediction_service.py` |
| `AI_VERIFICATION_THRESHOLD` | `utils/config.py` line 88 | 70.0 | **NO** — not imported in `prediction_service.py` |
| `ENABLE_AI_VERIFICATION` | `utils/config.py` line 86 | true | **NO** — not imported in `prediction_service.py` |
| `ML_CONFIDENCE_FALLBACK_MIN` | Original `app.py` line 2004 | 80.0 | **REMOVED** — not present in current code |
| `ML_ENTROPY_FALLBACK_MAX` | Original `app.py` | (0.9) | **REMOVED** — not present in current code |
| `CONFIDENCE_THRESHOLD` | Original `app.py` line 71 | `max(70.0, LOW_CONFIDENCE_THRESHOLD)` | **REMOVED** — not present in current code |
| `should_use_ml` | Original `app.py` line 2007 | Conditional gate | **REMOVED** — not present in current code |

### Verdict: The original threshold-based AI fallback has been completely removed/bypassed. The config values exist but are dead code — nothing in the prediction pipeline references them.

---

## Step 8 — Root Cause

### 1. Is AI now running for every prediction?
**YES.** `verify_multi_model_predictions_with_ai()` is called unconditionally in `backend/services/prediction_service.py` lines 64-74. There is no `if confidence < threshold:` check. The original conditional gate (`should_use_ml and confidence > CONFIDENCE_THRESHOLD`) was removed during the Streamlit → FastAPI migration.

### 2. Is the final disease being overwritten?
**YES.** In `utils/prediction_comparison.py` lines 219-222, `build_ai_verification_summary()` overrides the ML prediction if `ai_confidence >= final_confidence + 10.0`. Then in `prediction_service.py` line 85, `predicted_disease` is assigned from `ai_verification_summary.get("final_class")`, which may contain the AI-overridden value.

### 3. Is the confidence threshold still active?
**NO.** The config values `LOW_CONFIDENCE_THRESHOLD=70.0` and `AI_VERIFICATION_THRESHOLD=70.0` exist in `utils/config.py` but are **not imported or used** anywhere in `prediction_service.py` or `prediction_comparison.py`. The original `ML_CONFIDENCE_FALLBACK_MIN=80.0`, `ML_ENTROPY_FALLBACK_MAX`, `CONFIDENCE_THRESHOLD`, and `should_use_ml` logic was completely removed.

### 4. Is majority voting still the final decision?
**NO.** The majority vote and soft vote are computed correctly, but the final `predicted_disease` comes from `ai_verification_summary.get("final_class")`, which can be overridden by Gemini AI. The AI override condition (`ai_confidence >= final_confidence + 10.0`) replaces the ML majority result.

### 5. Why are Acne images becoming Vitiligo?
Gemini AI is called for **every** Acne image (no threshold check). Gemini returns "Vitiligo" with a confidence 10+ points higher than the ML confidence. The AI override condition is met, so the ML's correct "Acne" prediction is replaced with Gemini's incorrect "Vitiligo" prediction.

### 6. Why are Tinea images becoming Psoriasis?
Same mechanism. Gemini AI is called for every Tinea image. Gemini returns "Psoriasis" with a confidence 10+ points higher than the ML confidence. The AI override replaces the ML's correct "Tinea" prediction with Gemini's incorrect "Psoriasis" prediction.

### 7. Why does face consent now appear for almost every image?
Three changes to `backend/services/detect_face.py`:
1. **`RETINAFACE_THRESHOLD = 0.3`** (line 54) — lowered from InsightFace default of 0.5, causing more false-positive face detections on skin lesion textures
2. **`MIN_FACE_AREA_RATIO = 0.005`** (line 69) — lowered from 5% to 0.5%, accepting tiny false-positive detections as valid faces
3. **MediaPipe fallback added** (lines 452-466) — catches additional false positives that RetinaFace misses
4. **`is_face_complete` simplified** — "intentionally permissive", only rejects faces <0.5% or >99% of image area

### 8. Which files introduced the regression?
| File | Regression |
|---|---|
| `backend/services/prediction_service.py` | Removed threshold check; calls AI unconditionally (lines 64-74); assigns `predicted_disease` from AI summary (line 85) |
| `utils/prediction_comparison.py` | Added AI override logic: `ai_confidence >= final_confidence + 10.0` (lines 219-222) |
| `backend/services/detect_face.py` | Lowered `RETINAFACE_THRESHOLD` to 0.3 (line 54); lowered `MIN_FACE_AREA_RATIO` to 0.005 (line 69); added MediaPipe fallback (lines 452-466); made `is_face_complete` permissive |
| `utils/config.py` | Added `AI_VERIFICATION_THRESHOLD` and `ENABLE_AI_VERIFICATION` but they are dead code (not used) |

### 9. Which specific code block changed the original working logic?

**Original working logic** (`app.py` @ `f6f2e0b7`, lines 2002-2029):
```python
# If confidence < 80 OR entropy > 0.9 => do NOT trust ML (use Gemini AI)
ML_CONFIDENCE_FALLBACK_MIN = 80.0
should_use_ml = not (
    confidence < ML_CONFIDENCE_FALLBACK_MIN
    or probs_entropy > ML_ENTROPY_FALLBACK_MAX
)
if should_use_ml and confidence > CONFIDENCE_THRESHOLD:  # 70.0
    final_class = predicted_class
    final_confidence = confidence
    prediction_source = "ML"
else:
    prediction_source = "AI"
    ai_result = recognize_with_ai(image)
```

**Current broken logic** (`prediction_service.py` @ HEAD, lines 64-87):
```python
# AI verification — called UNCONDITIONALLY (no threshold check)
ai_result = verify_multi_model_predictions_with_ai(
    image=image,
    model_predictions=model_predictions,
    comparison_summary=build_ai_verification_summary(...).get("summary", ""),
)

ai_verification_summary = build_ai_verification_summary(
    model_predictions=model_predictions,
    comparison_result=comparison,
    majority_result=majority,
    soft_vote_result=soft,
    ai_result=ai_result,  # AI can override ML here
)

# Final selection uses ai_verification_summary which already applies override rules
predicted_disease = ai_verification_summary.get("final_class") or soft.get("selected_class")
```

**AI override block** (`prediction_comparison.py` @ HEAD, lines 219-222):
```python
if ai_prediction_valid and ai_confidence >= final_confidence + 10.0:
    final_class = str(ai_prediction_raw)  # AI REPLACES ML prediction
    final_confidence = ai_confidence
    source = "ai_verification"
```

---

## Required Fixes (to restore original behavior)

### Fix 1: Restore threshold-based AI fallback in `prediction_service.py`
- Import `LOW_CONFIDENCE_THRESHOLD` from `utils/config.py`
- Only call `verify_multi_model_predictions_with_ai()` when `soft.get("selected_confidence") < LOW_CONFIDENCE_THRESHOLD`
- When confidence >= threshold, use ML result directly (no AI call)

### Fix 2: Remove or disable AI override in `prediction_comparison.py`
- Remove the `ai_confidence >= final_confidence + 10.0` override block
- OR only allow AI override when ML confidence is below threshold (which Fix 1 already gates)

### Fix 3: Restore face detection thresholds in `detect_face.py`
- Restore `RETINAFACE_THRESHOLD` to 0.5 (InsightFace default)
- Restore `MIN_FACE_AREA_RATIO` to 0.05 (5%)
- Remove or disable MediaPipe fallback
- Restore stricter `is_face_complete` validation