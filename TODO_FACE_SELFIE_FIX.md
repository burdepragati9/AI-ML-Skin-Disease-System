# TODO — Face Selfie Edge-Case Fix

Allow close-up selfie images (large face occupying nearly the whole frame)
while keeping all other validation unchanged.

## Steps
- [x] 1. Add `touches_image_boundary()` helper (tolerance ~5-10 px)
- [x] 2. Add `SELFIE_MIN_CONFIDENCE = 0.75` constant
- [x] 3. Add `confidence` parameter to `is_face_complete()`
- [x] 4. Add MediaPipe selfie exception inside the over-large rejection check
- [x] 5. Pass confidence at both call sites (MediaPipe, RetinaFace)
- [x] 6. Update function docstrings
- [x] 7. Validate: `python -m py_compile backend/services/detect_face.py` (COMPILE_OK)
- [x] 8. Run `verify_face_selfie_fix.py` (OVERALL: PASS) — confirms selfie exception, boundary logic, and unchanged normal-face behaviour
