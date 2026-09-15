# TODO: MediaPipe Fallback Landmark Validation Improvement

## Goal
Improve **only** the MediaPipe fallback so it no longer produces false positives
on chin-only / forehead-only images while preserving all currently working valid
face detection scenarios.

## IMPORTANT
- Do NOT modify RetinaFace or any RetinaFace validation.
- Do NOT modify consent workflow, frontend, API contract, or prediction pipeline.
- Modify ONLY `_detect_mediapipe(...)` and its landmark validation.

## Steps
- [x] 1. Add named constants for mouth-below-nose validation threshold.
- [x] 2. Enhance `_mediapipe_landmarks_complete()` to validate mouth-below-nose.
- [x] 3. Add temporary debug log block in `_detect_mediapipe()`.
- [x] 4. Run `verify_mediapipe_landmark_completeness.py` to verify scenarios.
- [x] 5. Confirm Full/Half/Selfie/slightly-cropped still PASS; Chin/Forehead/lesion FAIL.
- [x] 6. Confirm RetinaFace code unchanged.

## Verification Results (MediaPipe fallback only)
| Scenario | Expected | Actual | Notes |
| -------- | -------- | ------ | ----- |
| Full face | PASS | True | mouth_nose_gap=0.069, misalign=0.047 |
| Half face | PASS | True | mouth_nose_gap=0.128, eyes cropped out (nose+mouth valid) |
| Chin image | FAIL | False | eyes misaligned ratio=0.636 (> 0.6) |
| Forehead image | FAIL | False | face too large (bbox validation) |
| Skin lesion (Tinea) | FAIL | False | landmark validation rejects false positive |
