# Face Detection Improvement Plan

## Problem
RetinaFace/InsightFace sometimes returns `face_detected: false` even when a human face is clearly visible (e.g., Image42.jpeg). This causes the consent popup to be skipped incorrectly.

## Solution
Add **MediaPipe Face Detection** as a lightweight CPU-optimized fallback detector when RetinaFace fails.

### Why MediaPipe?
1. Optimized for CPU inference (runs fast without GPU)
2. Excellent at detecting faces in difficult conditions: partial faces, side-profiles, small faces, low resolution, unusual lighting, boundary faces, occlusion
3. Single lightweight model - no "multiple unnecessary detectors"
4. Works natively with OpenCV/numpy format already used in the codebase
5. Widely used in production for real-time face detection

## Files to Modify

### 1. `backend/services/detect_face.py` (Primary file)
- Keep RetinaFace/InsightFace as the **primary** detector with threshold 0.3
- Add MediaPipe Face Detection as the **fallback** detector
- Add detailed logging for all steps:
  - Filename
  - Primary detector result and confidence
  - Primary detector bounding box
  - Whether fallback was triggered
  - Fallback detector result and confidence
  - Final `face_detected` result
  - Which detector detected the face
- Logic flow:
  ```
  Primary (RetinaFace) detects face → face_detected = true
  Primary fails → run Fallback (MediaPipe)
  Fallback detects face → face_detected = true
  Both fail → face_detected = false
  ```

### 2. `backend/requirements.txt`
- Add `mediapipe` dependency

### 3. `backend/routes/prediction.py`
- No changes needed (the `/detect-face` endpoint calls `detect_face()` which will now use the improved pipeline internally)

### 4. `frontend/src/pages/prediction/ImagePrediction.jsx`
- No changes needed (just consumes the `face_detected` boolean from the API)

## Files NOT Modified
- Disease prediction logic ✗
- ML models ✗
- AI/Gemini prediction ✗
- Voting logic ✗
- API endpoints ✗
- React routing ✗
- UI layout ✗

## Testing Plan
After implementation, test with:
1. Clear frontal face → RetinaFace detects → true
2. Side-profile face → RetinaFace may fail → MediaPipe catches → true
3. Partial face → RetinaFace may fail → MediaPipe catches → true
4. Small face → RetinaFace may fail → MediaPipe catches → true
5. Low-resolution face → RetinaFace may fail → MediaPipe catches → true
6. Skin lesion image (no face) → Both fail → false

