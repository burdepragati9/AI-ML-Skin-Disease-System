# TODO — Face Detection + Consent Flow Fix

## Steps
- [x] 1. Verify MediaPipe Tasks API availability and model asset in environment
       (Downloaded official blaze_face_short_range.tflite to mediapipe/modules/face_detection/)
- [x] 2. Modify `backend/services/detect_face.py`:
       - [x] Download official MediaPipe model
       - [x] Add MediaPipe Tasks FaceDetector as fallback
       - [x] Relax `is_face_complete` validation for cropped/upper-body/full-body faces
       - [x] Add debug logging for each pipeline stage
       - [x] Ensure boolean-only returns (never None)
       - [x] Exception handling chaining RetinaFace → MediaPipe → False
- [ ] 3. Fix JPEG P-mode issue in `utils/security.py` (explicit RGB conversion on working copy)
- [ ] 4. Remove debug `alert()`/blocking `console.log` in `ImagePrediction.jsx` (keep consent flow)
- [ ] 5. Validate: `python -m py_compile backend/services/detect_face.py`
- [ ] 6. Run scenario tests (selfie, front, L/R profile, upper-body, full-body, cropped face, lesion-only, lips-only, chin-only, P-mode PNG, RGBA PNG)
- [ ] 7. Final report
