# TODO: MediaPipe Landmark-Based Face Completeness Validation

## Goal
Improve ONLY the MediaPipe fallback detector so it accepts only recognizable faces
(full, half, slightly-cropped selfie) while rejecting partial-face crops (chin-only,
forehead-only, nose-only, mouth-only, tiny fragments).

Do NOT modify RetinaFace, `_detect_retinaface()`, `is_face_complete()`, consent
workflow, prediction workflow, API contract, or frontend.

## Steps
- [ ] Add named constants for landmark completeness thresholds.
- [ ] Add helper `_mediapipe_landmarks_complete()` that validates keypoint presence
      and spatial relationships (eye alignment, nose-to-eye gap).
- [ ] Modify `_detect_mediapipe()` to require BOTH `is_face_complete` AND landmark
      completeness before accepting a face.
- [ ] Add temporary debug logging (Left/Right eye, Nose, completeness PASS/FAIL).
- [ ] Verify scenarios: Full face PASS, Half face PASS, Selfie PASS, Chin FAIL, Forehead FAIL.
- [ ] Confirm RetinaFace behavior unchanged.
