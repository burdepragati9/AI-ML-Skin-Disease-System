"""
Face Detection Pipeline: RetinaFace (via InsightFace) → MediaPipe Tasks fallback.

Detects recognizable human faces in uploaded images to determine whether
the Face Privacy Consent popup should be shown to the user.

Pipeline:
  1. RetinaFace (InsightFace) — the PRIMARY detector. It detects front faces,
     profiles, selfies, and clearly visible faces (including upper-body /
     full-body images where the face is visible).
  2. If RetinaFace does not detect a valid face (or throws), MediaPipe Tasks
     FaceDetector runs as the FALLBACK detector.
  3. If at least one detector finds a valid face → face_detected = true.
     Otherwise → face_detected = false.

Validation (is_face_complete) is intentionally permissive enough to accept
slightly cropped faces (a cropped chin, forehead, or cheeks) as long as the
face is still clearly recognizable, while still rejecting pure skin lesions,
individual lips/chin/ear/nose/forehead crops, backgrounds, objects, hands,
legs, and close-up disease regions.

Every execution path explicitly returns True or False (never None).
"""
import os
import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional
import cv2

# ---------------------------------------------------------------------------
# Singleton detector instance
# ---------------------------------------------------------------------------
_face_detector: Optional['FaceAnalysis'] = None
_detector_initialized = False
_detector_init_error = None


# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------

# RetinaFace detection threshold.
# InsightFace default is 0.5. We intentionally use the default (strict) to
# avoid false-positive detections on skin-lesion textures. The previous
# permissive value of 0.3 caused almost every lesion image to trigger consent.
RETINAFACE_THRESHOLD = 0.5

# Detection size for RetinaFace model.
DETECTION_SIZE = (640, 640)

# Minimum face area ratio relative to image area to be considered valid.
# Set to a small value (0.5%) so upper-body / full-body images with a clearly
# visible face are still accepted. The minimum dimension checks and the
# MediumPipe fallback still guard against pure-lesion false positives.
MIN_FACE_AREA_RATIO = 0.005  # 0.5% of image area

# Boundary-crop tolerance (pixels) for is_face_complete.
# Relaxed so a slightly cropped face (chin/forehead/cheek) is still accepted.
BOUNDARY_MARGIN = 2

# Maximum face dimension ratio (fraction of image dimension) allowed.
# A face covering more than 98% of the image is likely a false positive.
MAX_FACE_DIM_RATIO = 0.98

# Minimum face dimension ratio (fraction of image dimension) required.
# Lowered so a clearly-visible face in an upper-body / full-body image is
# still accepted, while pure-lesion noise (much smaller) is rejected.
MIN_FACE_DIM_RATIO = 0.02  # 2% of image dimension

# --- Selfie edge-case exception (MediaPipe only) ---
# A close-up selfie fills nearly the entire frame (area_ratio > MAX_FACE_AREA_RATIO
# or a face dimension > MAX_FACE_DIM_RATIO). Such a detection is rejected by the
# normal over-large check, even though it is a genuine human face.
#
# We only relax this for genuine selfies: the detector must be MediaPipe, the
# MediaPipe confidence must be high, and the bounding box must touch at least one
# image border (typical selfie behaviour where the face fills / slightly overflows
# the frame). This keeps pure-lesion, object, and background false positives rejected.
SELFIE_MIN_CONFIDENCE = 0.75  # MediaPipe confidence threshold for the selfie exception

# Boundary-crop tolerance (pixels) for touches_image_boundary().
# A slightly cropped selfie (face overflowing 5-10 px past an edge) is still
# accepted as a selfie.
SELFIE_BOUNDARY_TOLERANCE = 10


# ---------------------------------------------------------------------------
# MediaPipe Tasks Face Detector (fallback) configuration
# ---------------------------------------------------------------------------

# Path to the official MediaPipe face detection model.
# The model asset is bundled into mediapipe/modules/face_detection/.
MEDIAPIPE_MODEL_FILENAME = "face_detection.tflite"

# Minimum detection confidence for the MediaPipe fallback. We use a moderate
# value (0.3) so it can catch profile / slightly-cropped faces that RetinaFace
# misses, while still rejecting pure-lesion textures.
MEDIAPIPE_MIN_CONFIDENCE = 0.3

# Minimum face area ratio (image area) for the MediaPipe fallback.
# Mirrors the RetinaFace area check to keep behavior consistent.
MEDIAPIPE_MIN_FACE_AREA_RATIO = 0.005

# Minimum face dimension ratio (fraction of image dimension) for the fallback.
MEDIAPIPE_MIN_FACE_DIM_RATIO = 0.02


# ---------------------------------------------------------------------------
# MediaPipe detector singleton
# ---------------------------------------------------------------------------
_mediapipe_detector: Optional[object] = None
_mediapipe_initialized = False
_mediapipe_init_error = None

# Official MediaPipe model download URL (used only if the bundled model is
# missing). blaze_face_short_range is the CPU-optimized face detector.
MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)


# ---------------------------------------------------------------------------
# Face validation
# ---------------------------------------------------------------------------
def touches_image_boundary(face_bbox: Tuple[int, int, int, int],
                           image_width: int,
                           image_height: int,
                           tolerance: int = SELFIE_BOUNDARY_TOLERANCE) -> bool:
    """
    Return True if the face bounding box touches (or slightly overflows) at
    least one image edge within the given pixel tolerance.

    A genuine close-up selfie fills (or slightly overflows) the frame, so its
    bounding box reaches the image border. This distinguishes a real selfie
    from a large-but-centred false positive (e.g. a lesion or object that
    fills the frame but leaves a margin on all sides).

    Args:
        face_bbox: Tuple of (x1, y1, x2, y2) bounding box coordinates
        image_width: Image width in pixels
        image_height: Image height in pixels
        tolerance: Pixel tolerance for 'touching' an edge (default 10 px)

    Returns:
        bool: True if the box touches at least one image edge, else False
    """
    x1, y1, x2, y2 = face_bbox
    # Box touches (or overflows) the left edge
    touches_left = x1 <= tolerance
    # Box touches (or overflows) the right edge
    touches_right = x2 >= (image_width - 1) - tolerance
    # Box touches (or overflows) the top edge
    touches_top = y1 <= tolerance
    # Box touches (or overflows) the bottom edge
    touches_bottom = y2 >= (image_height - 1) - tolerance

    return touches_left or touches_right or touches_top or touches_bottom


def is_face_complete(face_bbox: Tuple[int, int, int, int],
                     image_width: int,
                     image_height: int,
                     detector_name: str = "RetinaFace",
                     confidence: float = 0.0) -> bool:
    """
    Validate if the detected face is complete and sufficiently visible for
    consent purposes.

    A face is considered valid ONLY if ALL of the following hold:
        1. The face does NOT cover the ENTIRE image (whole-image-as-face is a
           false positive). Slightly-cropped faces (a cropped chin, forehead, or
           cheeks) are accepted as long as the face is still recognizable.
        2. The face is at least MIN_FACE_DIM_RATIO (2%) wide AND tall relative to
           the image (guards against tiny pure-lesion noise).
        3. The face does NOT cover more than MAX_FACE_DIM_RATIO (98%) of the image
           (likely a false positive).
        4. The face area is at least MIN_FACE_AREA_RATIO (0.5%) of the image area.

    SELFIE EXCEPTION (MediaPipe only): a close-up selfie often fills nearly the
    entire frame, so it would fail check 1 above. When the detector is
    MediaPipe, the confidence is high (>= SELFIE_MIN_CONFIDENCE), and the box
    touches at least one image border (typical selfie behaviour), the face is
    accepted as a valid selfie instead of being rejected. This exception does
    NOT apply to RetinaFace, preserving the existing strict over-large
    rejection for the primary detector.

    This validation is intentionally permissive enough to accept slightly
    cropped faces and upper-body / full-body faces, while still filtering out
    tiny lesion/background textures mistaken as faces.

    Args:
        face_bbox: Tuple of (x1, y1, x2, y2) bounding box coordinates
        image_width: Width of the image
        image_height: Height of the image
        detector_name: Name of the detector for logging
        confidence: Detection confidence (used for the MediaPipe selfie check)

    Returns:
        bool: True if face is sufficiently visible, False otherwise
    """
    x1, y1, x2, y2 = face_bbox
    face_width = x2 - x1
    face_height = y2 - y1
    face_area = face_width * face_height
    image_area = image_width * image_height
    area_ratio = face_area / image_area if image_area > 0 else 0

    print(f"[{detector_name}][is_face_complete] face bbox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    print(f"[{detector_name}][is_face_complete] face size: {face_width} x {face_height}")
    print(f"[{detector_name}][is_face_complete] image size: {image_width} x {image_height}")
    print(f"[{detector_name}][is_face_complete] area_ratio={area_ratio:.4f} (min={MIN_FACE_AREA_RATIO})")

    # --- Check 1: Whole-image-as-face (false positive) ---
    # A slightly cropped face (touching one or two boundaries) is accepted.
    # Only a face that essentially covers the entire image is rejected.
    max_w = image_width * MAX_FACE_DIM_RATIO
    max_h = image_height * MAX_FACE_DIM_RATIO
    if face_width > max_w or face_height > max_h:
        # Selfie edge-case exception: only for MediaPipe with high confidence
        # and a box that touches an image border (genuine close-up selfie).
        if (
            detector_name == "MediaPipe"
            and confidence >= SELFIE_MIN_CONFIDENCE
            and touches_image_boundary(face_bbox, image_width, image_height)
        ):
            print(f"[{detector_name}][is_face_complete] ACCEPTED: large face as valid selfie "
                  f"(conf={confidence:.2f}, touches boundary)")
            return True
        print(f"[{detector_name}][is_face_complete] REJECTED: face too large (possible false positive)")
        return False

    # --- Check 2: Minimum face size ---
    min_face_width = image_width * MIN_FACE_DIM_RATIO
    min_face_height = image_height * MIN_FACE_DIM_RATIO
    print(f"[{detector_name}][is_face_complete] min required size: {min_face_width:.1f} x {min_face_height:.1f}")

    if face_width < min_face_width or face_height < min_face_height:
        print(f"[{detector_name}][is_face_complete] REJECTED: face too small relative to image")
        return False

    # --- Check 3: Minimum area ratio ---
    if area_ratio < MIN_FACE_AREA_RATIO:
        print(f"[{detector_name}][is_face_complete] REJECTED: area_ratio={area_ratio:.4f} < {MIN_FACE_AREA_RATIO}")
        return False

    print(f"[{detector_name}][is_face_complete] ACCEPTED: recognizable face sufficiently visible")
    return True


# ---------------------------------------------------------------------------
# Detector initialization
# ---------------------------------------------------------------------------
def _get_detector():
    """
    Get or initialize the singleton RetinaFace detector.

    Returns:
        FaceAnalysis: The initialized detector instance

    Raises:
        RuntimeError: If detector initialization fails
    """
    global _face_detector, _detector_initialized, _detector_init_error

    if _detector_initialized:
        if _detector_init_error:
            raise RuntimeError(f"RetinaFace detector initialization failed previously: {_detector_init_error}")
        if _face_detector is not None:
            return _face_detector

    print("[RetinaFace Detection] Initializing detector (singleton)")

    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=DETECTION_SIZE)

        # Set detection threshold to the strict default (0.5).
        if hasattr(app, 'det_model') and hasattr(app.det_model, 'det_thresh'):
            app.det_model.det_thresh = RETINAFACE_THRESHOLD
        elif hasattr(app, 'models'):
            for model in app.models:
                if hasattr(model, 'det_thresh'):
                    model.det_thresh = RETINAFACE_THRESHOLD

        _face_detector = app
        _detector_initialized = True
        _detector_init_error = None

        print(f"[RetinaFace Detection] Model loaded successfully (threshold={RETINAFACE_THRESHOLD})")
        return app
    except Exception as e:
        _detector_initialized = True
        _detector_init_error = str(e)
        print(f"[RetinaFace Detection] Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"RetinaFace detector initialization failed: {str(e)}") from e


# ---------------------------------------------------------------------------
# MediaPipe Tasks Face Detector (fallback) initialization
# ---------------------------------------------------------------------------
def _resolve_mediapipe_model_path() -> Optional[str]:
    """
    Resolve the path to the official MediaPipe face detection model.

    The model is bundled into mediapipe/modules/face_detection/. If it is
    missing, attempt to download the official blaze_face_short_range model.

    Returns:
        str: Absolute path to the model file, or None if unavailable.
    """
    try:
        import mediapipe as mp
        base = os.path.dirname(mp.__file__)
        model_dir = os.path.join(base, "modules", "face_detection")
        model_path = os.path.join(model_dir, MEDIAPIPE_MODEL_FILENAME)

        if os.path.exists(model_path):
            return model_path

        # Attempt to download the official model (only if missing).
        try:
            import urllib.request
            os.makedirs(model_dir, exist_ok=True)
            print("[MediaPipe] Downloading official face detection model...")
            urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, model_path)
            if os.path.exists(model_path):
                print(f"[MediaPipe] Model downloaded: {model_path}")
                return model_path
        except Exception as dle:
            print(f"[MediaPipe] Model download failed: {dle}")

        return None
    except Exception as e:
        print(f"[MediaPipe] Model path resolution failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _get_mediapipe_detector():
    """
    Get or initialize the singleton MediaPipe Tasks FaceDetector (fallback).

    Returns:
        The created FaceDetector instance, or None if unavailable/failed.
    """
    global _mediapipe_detector, _mediapipe_initialized, _mediapipe_init_error

    if _mediapipe_initialized:
        if _mediapipe_init_error:
            print(f"[MediaPipe] Previously failed: {_mediapipe_init_error}")
            return None
        return _mediapipe_detector

    print("[MediaPipe] Initializing MediaPipe Tasks FaceDetector (fallback)")

    try:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        model_path = _resolve_mediapipe_model_path()
        if not model_path:
            _mediapipe_initialized = True
            _mediapipe_init_error = "MediaPipe face detection model not available"
            print("[MediaPipe] Model not available; fallback unavailable")
            return None

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=MEDIAPIPE_MIN_CONFIDENCE,
        )
        detector = vision.FaceDetector.create_from_options(options)

        _mediapipe_detector = detector
        _mediapipe_initialized = True
        _mediapipe_init_error = None

        print("[MediaPipe] FaceDetector loaded successfully")
        return detector
    except Exception as e:
        _mediapipe_initialized = True
        _mediapipe_init_error = str(e)
        print(f"[MediaPipe] FaceDetector initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Fallback detection: MediaPipe Tasks FaceDetector
# ---------------------------------------------------------------------------
def _detect_mediapipe(image_rgb: np.ndarray,
                      filename: str,
                      image_width: int,
                      image_height: int) -> dict:
    """
    Run MediaPipe Tasks FaceDetector on an RGB image (fallback detector).

    Args:
        image_rgb: RGB image as numpy array
        filename: Original filename for logging
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        dict with keys: face_detected (bool), confidence (float), bbox (list)
    """
    result = {
        "face_detected": False,
        "confidence": 0.0,
        "bbox": [],
        "detector_ok": False,
    }

    try:
        detector = _get_mediapipe_detector()
        if detector is None:
            print(f"[MediaPipe] filename={filename} detector unavailable -> face_detected=false")
            return result

        result["detector_ok"] = True

        # In mediapipe 0.10.x, the Image class lives at mediapipe.Image.
        from mediapipe.tasks.python import vision
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=image_rgb)
        detection_result = detector.detect(mp_image)

        if detection_result is None:
            print(f"[MediaPipe] filename={filename} detection_result=None -> face_detected=false")
            return result

        detections = getattr(detection_result, "detections", []) or []
        print(f"[MediaPipe] filename={filename} {len(detections)} raw detection(s)")
        print(f"[MediaPipe] Raw detections = {len(detections)}")

        best_face = None
        for d in detections:
            score = 0.0
            if hasattr(d, "categories") and d.categories:
                try:
                    score = float(d.categories[0].score)
                except Exception:
                    score = 0.0

            bbox = None
            if hasattr(d, "bounding_box") and d.bounding_box is not None:
                bb = d.bounding_box
                x1 = int(getattr(bb, "origin_x", 0))
                y1 = int(getattr(bb, "origin_y", 0))
                w = int(getattr(bb, "width", 0))
                h = int(getattr(bb, "height", 0))
                bbox = (x1, y1, x1 + w, y1 + h)

            print(f"[MediaPipe] detection score={score:.4f} bbox={bbox}")

            if bbox is None:
                continue

            # Validate the face with the permissive is_face_complete check.
            if is_face_complete(bbox, image_width, image_height,
                                detector_name="MediaPipe",
                                confidence=score):
                if best_face is None or score > best_face[0]:
                    best_face = (score, list(bbox))

        if best_face is not None and best_face[0] >= MEDIAPIPE_MIN_CONFIDENCE:
            result["face_detected"] = True
            result["confidence"] = best_face[0]
            result["bbox"] = best_face[1]
            print(f"[MediaPipe] filename={filename} valid face found -> "
                  f"face_detected=true conf={best_face[0]:.4f}")
        else:
            print(f"[MediaPipe] filename={filename} no valid face -> face_detected=false")

        return result
    except Exception as e:
        print(f"[MediaPipe] Detection error: {e}")
        import traceback
        traceback.print_exc()
        return result


# ---------------------------------------------------------------------------
# Primary detection: RetinaFace
# ---------------------------------------------------------------------------
def _detect_retinaface(image_bgr: np.ndarray,
                       filename: str) -> dict:
    """
    Run RetinaFace face detection on a BGR image with strict validation.

    Args:
        image_bgr: BGR image as numpy array (OpenCV format)
        filename: Original filename for logging

    Returns:
        dict with keys:
            face_detected (bool): Whether a valid face was found
            confidence (float): Detection confidence (0.0 if no face)
            bbox (list): [x1, y1, x2, y2] or empty list
            num_faces (int): Number of valid faces found
    """
    result = {
        "face_detected": False,
        "confidence": 0.0,
        "bbox": [],
        "num_faces": 0,
        "all_faces": [],  # list of (confidence, bbox) for all valid detections
    }

    height, width = image_bgr.shape[:2]

    try:
        app = _get_detector()

        faces = app.get(image_bgr)

        print(f"[RetinaFace] Raw faces found = {0 if faces is None else len(faces)}")

        if faces is None or len(faces) == 0:
            print(f"[RetinaFace] filename={filename} detections=0 -> face_detected=false")
            return result

        print(f"[RetinaFace] filename={filename} {len(faces)} raw face(s) found")

        valid_faces = []

        for face_idx, face_data in enumerate(faces):
            bbox = face_data.bbox
            box_int = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))

            det_score = 0.0
            if hasattr(face_data, 'det_score'):
                det_score = float(face_data.det_score)

            print(f"[RetinaFace] face {face_idx}: confidence={det_score:.4f} bbox={list(bbox[:4])}")

            # Strict validation of the face
            if is_face_complete(box_int, width, height,
                                detector_name="RetinaFace",
                                confidence=det_score):
                valid_faces.append((det_score, list(bbox[:4])))
            else:
                print(f"[RetinaFace] face {face_idx}: INVALID (rejected)")

        if valid_faces:
            # Pick the highest confidence valid face
            best_face = max(valid_faces, key=lambda x: x[0])
            result["face_detected"] = True
            result["confidence"] = best_face[0]
            result["bbox"] = best_face[1]
            result["num_faces"] = len(valid_faces)
            result["all_faces"] = valid_faces
            print(f"[RetinaFace] filename={filename} valid_faces={len(valid_faces)} "
                  f"-> face_detected=true conf={best_face[0]:.4f} bbox={best_face[1]}")
        else:
            print(f"[RetinaFace] filename={filename} no valid faces after strict validation -> face_detected=false")

        return result

    except Exception as e:
        print(f"[RetinaFace] Detection error: {e}")
        import traceback
        traceback.print_exc()
        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _detect_face_impl(image: Union[Image.Image, np.ndarray],
                      filename: str = "<unknown>") -> dict:
    """
    Detect if a recognizable human face is present in the image.

    Pipeline:
      1. RetinaFace (InsightFace) — the PRIMARY detector.
      2. If RetinaFace detects a valid face → face_detected = true.
      3. If RetinaFace fails / detects nothing / throws → MediaPipe Tasks
         FaceDetector runs as the FALLBACK.
      4. If MediaPipe detects a valid face → face_detected = true.
         Otherwise → face_detected = false.

    Every execution path explicitly returns True or False (never None).

    Args:
        image: PIL Image or numpy array
        filename: Original filename for debug logging

    Returns:
        dict with keys: face_detected (bool), detector_used (str),
        consent_required (bool)
    """
    print(f"[FaceDetection] filename={filename} Starting RetinaFace detection...")

    # Convert PIL Image to numpy array (work on a copy; never mutate input).
    if isinstance(image, Image.Image):
        image_for_detection = image if image.mode == "RGB" else image.convert("RGB")
        image_array = np.array(image_for_detection)
    else:
        image_array = np.array(image)

    # Ensure RGB format and correct channel count (working copy only).
    if len(image_array.shape) == 2:
        # Grayscale → RGB
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 4:
        # RGBA → RGB
        image_array = image_array[:, :, :3]

    height, width = image_array.shape[:2]

    # --- RetinaFace (primary detector) ---
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    rf_result = _detect_retinaface(image_bgr, filename) or {"face_detected": False}
    retinaface_detected = bool(rf_result.get("face_detected", False))

    conf = float(rf_result.get("confidence", 0.0))
    bbox = rf_result.get("bbox", [])

    print(f"[FaceDetection] filename={filename} RetinaFace result = {retinaface_detected} "
          f"(confidence={conf:.4f} bbox={bbox})")

    if retinaface_detected:
        print(f"[FaceDetection] filename={filename} FINAL RESULT = True (detector=retinaface)")
        return {
            "face_detected": True,
            "detector_used": "retinaface",
            "consent_required": True,
        }

    # --- MediaPipe fallback ---
    print(f"[FaceDetection] filename={filename} RetinaFace failed. Starting MediaPipe fallback...")
    mp_result = _detect_mediapipe(image_array, filename, width, height) or {"face_detected": False}
    mediapipe_detected = bool(mp_result.get("face_detected", False))

    print(f"[FaceDetection] filename={filename} MediaPipe result = {mediapipe_detected}")

    if mediapipe_detected:
        print(f"[FaceDetection] filename={filename} FINAL RESULT = True (detector=mediapipe)")
        return {
            "face_detected": True,
            "detector_used": "mediapipe",
            "consent_required": True,
        }

    print(f"[FaceDetection] filename={filename} FINAL RESULT = False (no face detected)")
    return {
        "face_detected": False,
        "detector_used": "none",
        "consent_required": False,
    }


def detect_face_details(image: Union[Image.Image, np.ndarray],
                        filename: str = "<unknown>") -> dict:
    """Return the full face detection contract with boolean-safe fields."""
    try:
        result = _detect_face_impl(image, filename=filename)
        face_detected = bool(result.get("face_detected", False))
        detector_used = result.get("detector_used") or ("unknown" if face_detected else "none")
        final_result = {
            "face_detected": face_detected,
            "detector_used": detector_used,
            "consent_required": bool(result.get("consent_required", face_detected)),
        }
        return final_result
    except Exception as e:
        print(f"[FaceDetection] pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "face_detected": False,
            "detector_used": "error",
            "consent_required": False,
        }


def detect_face(image: Union[Image.Image, np.ndarray],
                filename: str = "<unknown>") -> bool:
    """Public face detection API that always returns a boolean."""
    return bool(detect_face_details(image, filename=filename)["face_detected"])
