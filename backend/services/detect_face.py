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
import math
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
MEDIAPIPE_MIN_CONFIDENCE = 0.50

# Minimum face area ratio (image area) for the MediaPipe fallback.
# Mirrors the RetinaFace area check to keep behavior consistent.
MEDIAPIPE_MIN_FACE_AREA_RATIO = 0.005

# Minimum face dimension ratio (fraction of image dimension) for the fallback.
MEDIAPIPE_MIN_FACE_DIM_RATIO = 0.02

# --- MediaPipe landmark-based completeness thresholds (fallback only) ---
# blaze_face_short_range keypoint indices (normalized 0-1 coordinates).
MEDIAPIPE_KP_RIGHT_EYE = 0
MEDIAPIPE_KP_LEFT_EYE = 1
MEDIAPIPE_KP_NOSE_TIP = 2
MEDIAPIPE_KP_MOUTH_CENTER = 3

# Margin (normalized) required for a keypoint to be considered "in-bounds".
# A keypoint that lands outside this margin is treated as not visible/present.
MEDIAPIPE_IN_BOUNDS_MARGIN = 0.2

# Minimum normalized vertical gap between the eye line and the nose tip.
# Ensures the nose is structurally below the eyes (face-like arrangement).
MEDIAPIPE_MIN_NOSE_EYE_GAP = 0.03

# Maximum allowed ratio of the vertical eye-difference to the eye-line→nose gap.
# A genuine face shows closely-leveled eyes relative to the eye-to-nose spacing.
# Partial-face crops (e.g. chin-only) produce strongly misaligned "eyes" with a
# large ratio. Calibrated: genuine faces < 0.15, chin-only ~0.64. We use 0.6.
MEDIAPIPE_MAX_EYE_MISALIGN_RATIO = 0.6

# Minimum normalized vertical gap between the mouth center and the nose tip.
# On a genuine face the mouth lies clearly below the nose. Forehead-only crops
# may place the mouth above/at the nose or out of bounds, so this check helps
# reject such false positives while remaining permissive for real faces.
MEDIAPIPE_MIN_MOUTH_NOSE_GAP = 0.02

# Maximum allowed horizontal deviation of the nose tip from the eye-center
# line, expressed as a ratio of the eye-to-eye distance. A genuine face has its
# nose almost exactly on the vertical line between the eyes (|offset|/e2e ~ 0.06
# in calibration). Hallucinated landmark layouts on forehead-only crops place
# the "nose" far off that line. We use a permissive bound (0.6) so tilted and
# profile faces are still accepted while impossible layouts are rejected.
MEDIAPIPE_MAX_NOSE_X_OFFSET_RATIO = 0.6

# Minimum normalized eye-to-eye distance (resolution-independent). A genuine
# face always has a meaningful separation between the eyes. Hallucinated
# keypoint clusters on partial-face crops frequently collapse the two eyes into
# a single degenerate point (e2e ~ 0), which is an impossible face layout.
MEDIAPIPE_MIN_EYE_TO_EYE_NORM = 0.04

# Minimum / maximum ratio of the eye-center-to-mouth distance to the eye-to-eye
# distance. On a genuine face the mouth is a meaningful distance below the eyes
# (calibration: e2m/e2e ~ 0.9 full face, ~2.3 half face). Layouts where the
# mouth is nearly on top of the eyes (ratio -> 0) or absurdly far away indicate
# an impossible/abnormal landmark arrangement.
MEDIAPIPE_MIN_EYE_MOUTH_RATIO = 0.4
MEDIAPIPE_MAX_EYE_MOUTH_RATIO = 3.5

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


# ---------------------------------------------------------------------------
# Face bounding-box validation
# ---------------------------------------------------------------------------

def is_face_complete(
    face_bbox: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    detector_name: str = "RetinaFace",
    confidence: float = 0.0,
) -> bool:
    """
    Validate detected face bounding box.

    Rules:
        - Invalid/empty bbox -> reject
        - Very small bbox -> reject
        - Very large false-positive regions -> reject
        - RetinaFace boundary-touching bbox -> reject
        - MediaPipe boundary-touching bbox -> allow for landmark validation

    IMPORTANT:
        For MediaPipe, bbox validation is only the first stage.
        Landmark validation must decide whether the detected region is
        actually a complete recognizable face.
    """

    try:
        x1, y1, x2, y2 = face_bbox

        if image_width <= 0 or image_height <= 0:
            print(
                f"[{detector_name}][is_face_complete] "
                f"REJECTED: invalid image dimensions"
            )
            return False

        face_width = x2 - x1
        face_height = y2 - y1

        if face_width <= 0 or face_height <= 0:
            print(
                f"[{detector_name}][is_face_complete] "
                f"REJECTED: invalid bbox dimensions"
            )
            return False

        image_area = image_width * image_height
        face_area = face_width * face_height

        area_ratio = (
            face_area / image_area
            if image_area > 0
            else 0.0
        )

        width_ratio = face_width / image_width
        height_ratio = face_height / image_height

        print(
            f"[{detector_name}][is_face_complete] "
            f"face bbox: "
            f"x1={x1}, y1={y1}, x2={x2}, y2={y2}"
        )

        print(
            f"[{detector_name}][is_face_complete] "
            f"face size: "
            f"{face_width} x {face_height}"
        )

        print(
            f"[{detector_name}][is_face_complete] "
            f"image size: "
            f"{image_width} x {image_height}"
        )

        print(
            f"[{detector_name}][is_face_complete] "
            f"width_ratio={width_ratio:.4f} "
            f"height_ratio={height_ratio:.4f} "
            f"area_ratio={area_ratio:.4f}"
        )

        # -------------------------------------------------------------
        # Completely invalid overflow
        # -------------------------------------------------------------

        if (
            x1 < -5
            or y1 < -5
            or x2 > image_width + 5
            or y2 > image_height + 5
        ):
            print(
                f"[{detector_name}][is_face_complete] "
                f"REJECTED: bbox extends too far outside image"
            )
            return False

        # -------------------------------------------------------------
        # Minimum size
        # -------------------------------------------------------------

        min_face_width = (
            image_width *
            MIN_FACE_DIM_RATIO
        )

        min_face_height = (
            image_height *
            MIN_FACE_DIM_RATIO
        )

        if (
            face_width < min_face_width
            or
            face_height < min_face_height
        ):
            print(
                f"[{detector_name}][is_face_complete] "
                f"REJECTED: face too small"
            )
            return False

        # -------------------------------------------------------------
        # Minimum area
        # -------------------------------------------------------------

        if area_ratio < MIN_FACE_AREA_RATIO:
            print(
                f"[{detector_name}][is_face_complete] "
                f"REJECTED: area_ratio={area_ratio:.4f} "
                f"< {MIN_FACE_AREA_RATIO}"
            )
            return False

        # -------------------------------------------------------------
        # Boundary detection
        # -------------------------------------------------------------

        touches_left = x1 <= 0
        touches_top = y1 <= 0
        touches_right = x2 >= image_width
        touches_bottom = y2 >= image_height

        touches_boundary = (
            touches_left
            or touches_top
            or touches_right
            or touches_bottom
        )

        if touches_boundary:
            boundaries = []

            if touches_left:
                boundaries.append("left")

            if touches_top:
                boundaries.append("top")

            if touches_right:
                boundaries.append("right")

            if touches_bottom:
                boundaries.append("bottom")

            boundary_text = ", ".join(boundaries)

            print(
                f"[{detector_name}][is_face_complete] "
                f"Boundary touched: {boundary_text}"
            )

            # ---------------------------------------------------------
            # RetinaFace:
            # Keep primary detector strict.
            # ---------------------------------------------------------

            if detector_name == "RetinaFace":
                print(
                    f"[{detector_name}][is_face_complete] "
                    f"REJECTED: RetinaFace bbox touches image boundary"
                )
                return False

            # ---------------------------------------------------------
            # MediaPipe:
            # DO NOT reject here.
            #
            # Landmark validation will decide whether the region
            # actually represents a complete face.
            # ---------------------------------------------------------

            if detector_name == "MediaPipe":
                print(
                    f"[{detector_name}][is_face_complete] "
                    f"Boundary touched -> "
                    f"allowing landmark validation to decide"
                )

        # -------------------------------------------------------------
        # Maximum size protection
        #
        # Do not use 95% here for MediaPipe, because a legitimate
        # close-up face may occupy almost the complete image.
        # -------------------------------------------------------------

        if (
            width_ratio > 1.10
            or
            height_ratio > 1.10
        ):
            print(
                f"[{detector_name}][is_face_complete] "
                f"REJECTED: bbox dimensions invalid"
            )
            return False

        print(
            f"[{detector_name}][is_face_complete] "
            f"BBOX ACCEPTED for further validation "
            f"(confidence={confidence:.4f})"
        )

        return True

    except Exception as e:

        print(
            f"[{detector_name}][is_face_complete] "
            f"VALIDATION ERROR: {e}"
        )

        traceback.print_exc()

        return False

def _retinaface_landmarks_complete(
    kps,
    face_bbox: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> Tuple[bool, dict]:
    """
    Validate RetinaFace's 5 facial landmarks.

    InsightFace RetinaFace normally returns:
        0 = left eye
        1 = right eye
        2 = nose
        3 = left mouth corner
        4 = right mouth corner

    The validation is intentionally permissive for slightly cropped faces,
    but rejects geometrically impossible detections that are commonly caused
    by skin lesions / forehead-only / chin-only regions.
    """

    detail = {
        "eyes": False,
        "nose": False,
        "mouth": False,
        "eye_to_eye": None,
        "nose_eye_gap": None,
        "mouth_nose_gap": None,
        "eye_mouth_ratio": None,
        "nose_x_offset_ratio": None,
        "complete": False,
        "reason": "",
    }

    try:
        if kps is None:
            detail["reason"] = "no landmarks"
            return False, detail

        points = np.asarray(kps, dtype=np.float32)

        if points.ndim != 2 or points.shape[0] < 5 or points.shape[1] < 2:
            detail["reason"] = f"invalid landmark shape: {points.shape}"
            return False, detail

        left_eye = points[0][:2]
        right_eye = points[1][:2]
        nose = points[2][:2]
        left_mouth = points[3][:2]
        right_mouth = points[4][:2]

        x1, y1, x2, y2 = face_bbox

        face_width = max(float(x2 - x1), 1.0)
        face_height = max(float(y2 - y1), 1.0)

        # ---------------------------------------------------------------
        # Convert coordinates to face-bbox-relative coordinates.
        # This makes the checks independent of image resolution.
        # ---------------------------------------------------------------
        le = np.array([
            (left_eye[0] - x1) / face_width,
            (left_eye[1] - y1) / face_height,
        ])

        re = np.array([
            (right_eye[0] - x1) / face_width,
            (right_eye[1] - y1) / face_height,
        ])

        no = np.array([
            (nose[0] - x1) / face_width,
            (nose[1] - y1) / face_height,
        ])

        lm = np.array([
            (left_mouth[0] - x1) / face_width,
            (left_mouth[1] - y1) / face_height,
        ])

        rm = np.array([
            (right_mouth[0] - x1) / face_width,
            (right_mouth[1] - y1) / face_height,
        ])

        # ---------------------------------------------------------------
        # Check for finite values.
        # ---------------------------------------------------------------
        all_points = np.vstack([le, re, no, lm, rm])

        if not np.all(np.isfinite(all_points)):
            detail["reason"] = "non-finite landmark coordinates"
            return False, detail

        # ---------------------------------------------------------------
        # Allow a small amount of landmark overflow for cropped faces.
        # Do NOT require every point to be inside the image.
        # ---------------------------------------------------------------
        landmark_margin = 0.35

        def point_reasonable(p):
            return (
                -landmark_margin <= p[0] <= 1.0 + landmark_margin
                and
                -landmark_margin <= p[1] <= 1.0 + landmark_margin
            )

        reasonable_points = [
            point_reasonable(le),
            point_reasonable(re),
            point_reasonable(no),
            point_reasonable(lm),
            point_reasonable(rm),
        ]

        if sum(reasonable_points) < 3:
            detail["reason"] = "too few reasonable facial landmarks"
            return False, detail

        # ---------------------------------------------------------------
        # Eye geometry
        # ---------------------------------------------------------------
        eye_center = (le + re) / 2.0

        eye_to_eye = float(np.linalg.norm(le - re))
        detail["eye_to_eye"] = eye_to_eye

        # RetinaFace landmarks are bbox-relative.
        # A real face should have meaningful eye separation.
        if eye_to_eye < 0.12:
            detail["reason"] = (
                f"eyes too close together ({eye_to_eye:.3f})"
            )
            return False, detail

        detail["eyes"] = True

        # ---------------------------------------------------------------
        # Nose geometry
        # ---------------------------------------------------------------
        nose_eye_gap = float(no[1] - eye_center[1])
        detail["nose_eye_gap"] = nose_eye_gap

        # Nose should normally be below the eye line.
        # Small tolerance allows tilted/profile/cropped faces.
        if nose_eye_gap < 0.05:
            detail["reason"] = (
                f"nose not sufficiently below eyes "
                f"(gap={nose_eye_gap:.3f})"
            )
            return False, detail

        detail["nose"] = True

        # Nose should remain reasonably close to the eye-center vertical line.
        nose_x_offset_ratio = (
            abs(float(no[0] - eye_center[0])) / eye_to_eye
        )

        detail["nose_x_offset_ratio"] = nose_x_offset_ratio

        if nose_x_offset_ratio > 1.25:
            detail["reason"] = (
                f"nose too far from eye center "
                f"(ratio={nose_x_offset_ratio:.2f})"
            )
            return False, detail

        # ---------------------------------------------------------------
        # Mouth geometry
        # ---------------------------------------------------------------
        mouth_center = (lm + rm) / 2.0

        mouth_nose_gap = float(mouth_center[1] - no[1])
        detail["mouth_nose_gap"] = mouth_nose_gap

        # For a recognizable face, mouth should be below nose.
        if mouth_nose_gap < 0.04:
            detail["reason"] = (
                f"mouth not sufficiently below nose "
                f"(gap={mouth_nose_gap:.3f})"
            )
            return False, detail

        detail["mouth"] = True

        # ---------------------------------------------------------------
        # Eye-to-mouth proportional relationship.
        # ---------------------------------------------------------------
        eye_to_mouth = float(
            np.linalg.norm(mouth_center - eye_center)
        )

        eye_mouth_ratio = eye_to_mouth / eye_to_eye
        detail["eye_mouth_ratio"] = eye_mouth_ratio

        if eye_mouth_ratio < 0.45:
            detail["reason"] = (
                f"mouth too close to eyes "
                f"(ratio={eye_mouth_ratio:.2f})"
            )
            return False, detail

        if eye_mouth_ratio > 3.5:
            detail["reason"] = (
                f"mouth too far from eyes "
                f"(ratio={eye_mouth_ratio:.2f})"
            )
            return False, detail

        # ---------------------------------------------------------------
        # Mouth corners should have meaningful separation.
        # ---------------------------------------------------------------
        mouth_width = float(np.linalg.norm(lm - rm))

        if mouth_width < eye_to_eye * 0.20:
            detail["reason"] = (
                f"mouth landmarks too close "
                f"(mouth_width={mouth_width:.3f})"
            )
            return False, detail

        detail["complete"] = True
        detail["reason"] = "ok"

        return True, detail

    except Exception as e:
        detail["reason"] = f"landmark validation error: {e}"
        return False, detail


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
# MediaPipe landmark-based face completeness validation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# MediaPipe landmark-based face completeness validation
# ---------------------------------------------------------------------------

def _mediapipe_landmarks_complete(
    keypoints: list,
    image_width: int,
    image_height: int,
) -> Tuple[bool, dict]:
    """
    Validate MediaPipe face landmarks for privacy consent.

    Required landmarks:
        0 = left eye
        1 = right eye
        2 = nose
        3 = mouth

    A face is accepted only if:
        - both eyes exist
        - nose exists
        - mouth exists
        - eye distance is reasonable
        - nose is below eyes
        - mouth is below nose
        - eyes are reasonably aligned
        - nose is reasonably centered between eyes
        - facial landmarks are vertically distributed like a real face
        - landmarks are not clustered in one small region

    This helps reject:
        - cheek-only detections
        - chin-only detections
        - forehead-only detections
        - lesion false positives
        - partial-face false positives
    """

    detail = {
        "left_eye": False,
        "right_eye": False,
        "nose": False,
        "mouth": False,
        "eye_to_eye": None,
        "nose_eye_gap": None,
        "nose_x_offset_ratio": None,
        "eye_mouth_ratio": None,
        "eye_delta_y": None,
        "misalign_ratio": None,
        "mouth_nose_gap": None,
        "eye_center_y": None,
        "face_vertical_span": None,
        "complete": False,
        "reason": "",
    }

    try:

        # ==============================================================
        # Validate keypoints
        # ==============================================================

        if not keypoints:
            detail["reason"] = "no keypoints"
            return False, detail

        if len(keypoints) < 4:
            detail["reason"] = (
                f"insufficient keypoints: {len(keypoints)}"
            )
            return False, detail

        # ==============================================================
        # Safe coordinate extraction
        # ==============================================================

        def _xy(kp):

            if kp is None:
                return None, None

            try:

                x = float(
                    getattr(
                        kp,
                        "x",
                        float("nan"),
                    )
                )

                y = float(
                    getattr(
                        kp,
                        "y",
                        float("nan"),
                    )
                )

                if not math.isfinite(x):
                    return None, None

                if not math.isfinite(y):
                    return None, None

                return x, y

            except Exception:

                return None, None

        # ==============================================================
        # IMPORTANT:
        # MediaPipe documented keypoint order
        # ==============================================================

        left_eye = keypoints[0]
        right_eye = keypoints[1]
        nose = keypoints[2]
        mouth = keypoints[3]

        left_eye_x, left_eye_y = _xy(left_eye)
        right_eye_x, right_eye_y = _xy(right_eye)
        nose_x, nose_y = _xy(nose)
        mouth_x, mouth_y = _xy(mouth)

        # ==============================================================
        # Check coordinates
        # ==============================================================

        if (
            left_eye_x is None
            or left_eye_y is None
            or right_eye_x is None
            or right_eye_y is None
            or nose_x is None
            or nose_y is None
            or mouth_x is None
            or mouth_y is None
        ):
            detail["reason"] = (
                "one or more facial landmarks "
                "have invalid coordinates"
            )
            return False, detail

        # ==============================================================
        # Normalized MediaPipe coordinates
        # ==============================================================

        def _valid_normalized_coordinate(x, y):

            return (
                -0.02 <= x <= 1.02
                and
                -0.02 <= y <= 1.02
            )

        if not _valid_normalized_coordinate(
            left_eye_x,
            left_eye_y,
        ):
            detail["reason"] = "left eye out of range"
            return False, detail

        if not _valid_normalized_coordinate(
            right_eye_x,
            right_eye_y,
        ):
            detail["reason"] = "right eye out of range"
            return False, detail

        if not _valid_normalized_coordinate(
            nose_x,
            nose_y,
        ):
            detail["reason"] = "nose out of range"
            return False, detail

        if not _valid_normalized_coordinate(
            mouth_x,
            mouth_y,
        ):
            detail["reason"] = "mouth out of range"
            return False, detail

        detail["left_eye"] = True
        detail["right_eye"] = True
        detail["nose"] = True
        detail["mouth"] = True

        # ==============================================================
        # Eye center
        # ==============================================================

        eye_center_x = (
            left_eye_x +
            right_eye_x
        ) / 2.0

        eye_center_y = (
            left_eye_y +
            right_eye_y
        ) / 2.0

        detail["eye_center_y"] = eye_center_y

        # ==============================================================
        # Eye-to-eye distance
        # ==============================================================

        eye_to_eye = math.hypot(
            left_eye_x - right_eye_x,
            left_eye_y - right_eye_y,
        )

        detail["eye_to_eye"] = eye_to_eye

        min_eye_distance = float(
            globals().get(
                "MEDIAPIPE_MIN_EYE_TO_EYE_NORM",
                0.08,
            )
        )

        if eye_to_eye < min_eye_distance:

            detail["reason"] = (
                f"eyes too close together "
                f"({eye_to_eye:.3f})"
            )

            return False, detail

        # ==============================================================
        # Nose horizontal alignment
        # ==============================================================

        nose_x_offset_ratio = (
            abs(
                nose_x -
                eye_center_x
            )
            /
            max(
                eye_to_eye,
                1e-6,
            )
        )

        detail[
            "nose_x_offset_ratio"
        ] = nose_x_offset_ratio

        max_nose_offset = float(
            globals().get(
                "MEDIAPIPE_MAX_NOSE_X_OFFSET_RATIO",
                0.75,
            )
        )

        if nose_x_offset_ratio > max_nose_offset:

            detail["reason"] = (
                "nose is too far horizontally "
                "from eye center"
            )

            return False, detail

        # ==============================================================
        # Nose must be below eyes
        # ==============================================================

        nose_eye_gap = (
            nose_y -
            eye_center_y
        )

        detail["nose_eye_gap"] = nose_eye_gap

        min_nose_eye_gap = float(
            globals().get(
                "MEDIAPIPE_MIN_NOSE_EYE_GAP",
                0.05,
            )
        )

        if nose_eye_gap < min_nose_eye_gap:

            detail["reason"] = (
                "nose is too close to eye line"
            )

            return False, detail

        # ==============================================================
        # Eye alignment
        # ==============================================================

        eye_delta_y = abs(
            left_eye_y -
            right_eye_y
        )

        detail["eye_delta_y"] = eye_delta_y

        misalign_ratio = (
            eye_delta_y /
            max(
                nose_eye_gap,
                1e-6,
            )
        )

        detail["misalign_ratio"] = (
            misalign_ratio
        )

        max_misalign = float(
            globals().get(
                "MEDIAPIPE_MAX_EYE_MISALIGN_RATIO",
                0.60,
            )
        )

        if misalign_ratio > max_misalign:

            detail["reason"] = (
                f"eyes excessively misaligned "
                f"(ratio={misalign_ratio:.3f})"
            )

            return False, detail

        # ==============================================================
        # Mouth must be below nose
        # ==============================================================

        mouth_nose_gap = (
            mouth_y -
            nose_y
        )

        detail[
            "mouth_nose_gap"
        ] = mouth_nose_gap

        min_mouth_nose_gap = float(
            globals().get(
                "MEDIAPIPE_MIN_MOUTH_NOSE_GAP",
                0.05,
            )
        )

        if mouth_nose_gap < min_mouth_nose_gap:

            detail["reason"] = (
                "mouth is too close to nose "
                "or above nose"
            )

            return False, detail

        # ==============================================================
        # Mouth horizontal plausibility
        # ==============================================================

        mouth_x_offset = (
            abs(
                mouth_x -
                eye_center_x
            )
            /
            max(
                eye_to_eye,
                1e-6,
            )
        )

        if mouth_x_offset > 1.50:

            detail["reason"] = (
                "mouth is too far from "
                "facial center"
            )

            return False, detail

        # ==============================================================
        # Eye-to-mouth distance
        # ==============================================================

        eye_to_mouth = math.hypot(
            mouth_x - eye_center_x,
            mouth_y - eye_center_y,
        )

        eye_mouth_ratio = (
            eye_to_mouth /
            max(
                eye_to_eye,
                1e-6,
            )
        )

        detail[
            "eye_mouth_ratio"
        ] = eye_mouth_ratio

        # ==============================================================
        # CRITICAL:
        # Facial landmarks must occupy a reasonable vertical span.
        #
        # This rejects detections where MediaPipe finds a face-like
        # structure near one small/partial area of the image.
        # ==============================================================

        face_vertical_span = (
            mouth_y -
            left_eye_y
        )

        detail[
            "face_vertical_span"
        ] = face_vertical_span

        if face_vertical_span < 0.20:

            detail["reason"] = (
                f"facial landmark vertical span too small "
                f"({face_vertical_span:.3f})"
            )

            return False, detail

        # ==============================================================
        # CRITICAL:
        # Eyes should normally be in the upper portion of the image.
        #
        # This helps reject lower-cheek/chin detections.
        # ==============================================================

        if eye_center_y > 0.62:

            detail["reason"] = (
                f"eyes positioned too low in image "
                f"(eye_center_y={eye_center_y:.3f})"
            )

            return False, detail

        # ==============================================================
        # Nose should not be extremely low
        # ==============================================================

        if nose_y > 0.82:

            detail["reason"] = (
                f"nose positioned too low "
                f"(nose_y={nose_y:.3f})"
            )

            return False, detail

        # ==============================================================
        # Mouth should not be outside the expected facial area
        # ==============================================================

        if mouth_y > 0.98:

            detail["reason"] = (
                f"mouth positioned too close to bottom "
                f"(mouth_y={mouth_y:.3f})"
            )

            return False, detail

        # ==============================================================
        # FINAL SUCCESS
        # ==============================================================

        detail["complete"] = True

        detail["reason"] = (
            "complete recognizable face: "
            "both eyes + nose + mouth + "
            "coherent geometry"
        )

        print(
            "[MediaPipe][LandmarkValidation] "
            "PASS: complete recognizable face"
        )

        return True, detail

    except Exception as e:

        detail["complete"] = False
        detail["reason"] = (
            f"landmark validation error: {e}"
        )

        print(
            f"[MediaPipe][LandmarkValidation] "
            f"ERROR: {e}"
        )

        traceback.print_exc()

        return False, detail
# ---------------------------------------------------------------------------
# Fallback detection: MediaPipe Tasks FaceDetector
# ---------------------------------------------------------------------------

def _detect_mediapipe(
    image_rgb: np.ndarray,
    filename: str,
    image_width: int,
    image_height: int,
) -> dict:
    """
    Run MediaPipe Tasks FaceDetector as the fallback face detector.

    Privacy-consent rules:

        1. MediaPipe must detect a face.
        2. Bounding-box validation must pass.
        3. Detection confidence must be >= MEDIAPIPE_MIN_CONFIDENCE.
        4. Both eyes must be detected.
        5. Nose must be detected.
        6. Mouth must be detected.
        7. Landmark geometry must be valid.
        8. Only then is face_detected=True.

    MediaPipe documented keypoint order:

        0 = left eye
        1 = right eye
        2 = nose tip
        3 = mouth
        4 = left eye tragion
        5 = right eye tragion

    Important:
        Landmark validation is a privacy-consent gate.
        Therefore, an invalid/incomplete face must NOT trigger consent.

    Returns:
        dict:
            face_detected
            confidence
            bbox
            detector_ok
            num_faces
            all_faces
            detector_used
    """

    # =======================================================================
    # RESULT DEFAULT
    # =======================================================================

    result = {
        "face_detected": False,
        "confidence": 0.0,
        "bbox": [],
        "detector_ok": False,
        "num_faces": 0,
        "all_faces": [],
        "detector_used": "mediapipe",
    }

    try:

        # ===================================================================
        # 1. GET / INITIALIZE MEDIAPIPE DETECTOR
        # ===================================================================

        detector = _get_mediapipe_detector()

        if detector is None:

            print(
                f"[MediaPipe] filename={filename} "
                f"detector unavailable"
            )

            return result

        result["detector_ok"] = True

        # ===================================================================
        # 2. VALIDATE IMAGE
        # ===================================================================

        if image_rgb is None:

            print(
                f"[MediaPipe] filename={filename} "
                f"image_rgb=None"
            )

            return result

        if not isinstance(
            image_rgb,
            np.ndarray,
        ):

            print(
                f"[MediaPipe] filename={filename} "
                f"image is not numpy array"
            )

            return result

        if image_rgb.size == 0:

            print(
                f"[MediaPipe] filename={filename} "
                f"image is empty"
            )

            return result

        # ===================================================================
        # 3. NORMALIZE IMAGE
        # ===================================================================

        try:

            if image_rgb.ndim == 2:

                image_rgb = cv2.cvtColor(
                    image_rgb,
                    cv2.COLOR_GRAY2RGB,
                )

            elif (
                image_rgb.ndim == 3
                and image_rgb.shape[2] == 4
            ):

                image_rgb = image_rgb[:, :, :3]

            elif (
                image_rgb.ndim != 3
                or image_rgb.shape[2] != 3
            ):

                print(
                    f"[MediaPipe] filename={filename} "
                    f"unsupported image shape="
                    f"{image_rgb.shape}"
                )

                return result

        except Exception as image_shape_error:

            print(
                f"[MediaPipe] filename={filename} "
                f"image shape conversion error: "
                f"{image_shape_error}"
            )

            traceback.print_exc()

            return result

        # -------------------------------------------------------------------
        # Ensure uint8
        # -------------------------------------------------------------------

        if image_rgb.dtype != np.uint8:

            image_rgb = np.clip(
                image_rgb,
                0,
                255,
            ).astype(
                np.uint8
            )

        image_rgb = np.ascontiguousarray(
            image_rgb
        )

        # ===================================================================
        # 4. CREATE MEDIAPIPE IMAGE
        # ===================================================================

        try:

            import mediapipe as mp

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image_rgb,
            )

        except Exception as mp_image_error:

            print(
                f"[MediaPipe] filename={filename} "
                f"MediaPipe image creation error: "
                f"{mp_image_error}"
            )

            traceback.print_exc()

            return result

        # ===================================================================
        # 5. RUN FACE DETECTION
        # ===================================================================

        try:

            detection_result = detector.detect(
                mp_image
            )

        except Exception as detection_error:

            print(
                f"[MediaPipe] filename={filename} "
                f"detector.detect() error: "
                f"{detection_error}"
            )

            traceback.print_exc()

            return result

        if detection_result is None:

            print(
                f"[MediaPipe] filename={filename} "
                f"detection_result=None"
            )

            return result

        # ===================================================================
        # 6. READ DETECTIONS
        # ===================================================================

        detections = (
            getattr(
                detection_result,
                "detections",
                None,
            )
            or []
        )

        print(
            f"[MediaPipe] filename={filename} "
            f"raw detections={len(detections)}"
        )

        if not detections:

            print(
                f"[MediaPipe] filename={filename} "
                f"no raw detections"
            )

            return result

        valid_faces = []

        # ===================================================================
        # 7. PROCESS EACH DETECTION
        # ===================================================================

        for face_idx, detection in enumerate(
            detections
        ):

            print(
                "=================================================="
            )

            print(
                f"[MediaPipe] Processing detection={face_idx}"
            )

            # =================================================================
            # 7A. EXTRACT CONFIDENCE
            # =================================================================

            score = 0.0

            try:

                categories = (
                    getattr(
                        detection,
                        "categories",
                        None,
                    )
                    or []
                )

                if categories:

                    score = float(
                        getattr(
                            categories[0],
                            "score",
                            0.0,
                        )
                        or 0.0
                    )

            except Exception as confidence_error:

                print(
                    f"[MediaPipe] detection={face_idx} "
                    f"confidence extraction error: "
                    f"{confidence_error}"
                )

                traceback.print_exc()

                score = 0.0

            print(
                f"[MediaPipe] detection={face_idx} "
                f"confidence={score:.4f}"
            )

            # =================================================================
            # 7B. CONFIDENCE GATE
            # =================================================================

            try:

                min_confidence = float(
                    globals().get(
                        "MEDIAPIPE_MIN_CONFIDENCE",
                        0.50,
                    )
                )

            except Exception:

                min_confidence = 0.50

            if score < min_confidence:

                print(
                    f"[MediaPipe] detection={face_idx} "
                    f"CONFIDENCE TOO LOW -> rejected "
                    f"({score:.4f} < "
                    f"{min_confidence:.4f})"
                )

                continue

            print(
                f"[MediaPipe] detection={face_idx} "
                f"CONFIDENCE PASSED "
                f"({score:.4f} >= "
                f"{min_confidence:.4f})"
            )

            # =================================================================
            # 7C. EXTRACT BOUNDING BOX
            # =================================================================

            bbox = None

            try:

                bounding_box = getattr(
                    detection,
                    "bounding_box",
                    None,
                )

                if bounding_box is not None:

                    x1 = int(
                        getattr(
                            bounding_box,
                            "origin_x",
                            0,
                        )
                    )

                    y1 = int(
                        getattr(
                            bounding_box,
                            "origin_y",
                            0,
                        )
                    )

                    box_width = int(
                        getattr(
                            bounding_box,
                            "width",
                            0,
                        )
                    )

                    box_height = int(
                        getattr(
                            bounding_box,
                            "height",
                            0,
                        )
                    )

                    x2 = (
                        x1
                        +
                        box_width
                    )

                    y2 = (
                        y1
                        +
                        box_height
                    )

                    bbox = [
                        x1,
                        y1,
                        x2,
                        y2,
                    ]

            except Exception as bbox_error:

                print(
                    f"[MediaPipe] detection={face_idx} "
                    f"bbox extraction error: "
                    f"{bbox_error}"
                )

                traceback.print_exc()

                bbox = None

            print(
                f"[MediaPipe] detection={face_idx} "
                f"bbox={bbox}"
            )

            # =================================================================
            # 7D. BBOX VALIDATION
            # =================================================================

            if (
                bbox is None
                or len(bbox) != 4
            ):

                print(
                    f"[MediaPipe] detection={face_idx} "
                    f"INVALID BBOX -> rejected"
                )

                continue

            try:

                bbox_valid = is_face_complete(
                    tuple(bbox),
                    image_width,
                    image_height,
                    detector_name="MediaPipe",
                    confidence=score,
                )

            except Exception as bbox_validation_error:

                print(
                    f"[MediaPipe] detection={face_idx} "
                    f"BBOX VALIDATION ERROR: "
                    f"{bbox_validation_error}"
                )

                traceback.print_exc()

                continue

            if not bbox_valid:

                print(
                    f"[MediaPipe] detection={face_idx} "
                    f"BBOX VALIDATION FAILED -> rejected"
                )

                continue

            print(
                f"[MediaPipe] detection={face_idx} "
                f"BBOX VALIDATION PASSED"
            )

            # =================================================================
            # 8. GET RAW MEDIAPIPE KEYPOINTS
            # =================================================================

            keypoints = (
                getattr(
                    detection,
                    "keypoints",
                    None,
                )
                or []
            )

            # =================================================================
            # 8A. RAW KEYPOINT DEBUG
            #
            # IMPORTANT:
            # This verifies that our landmark indexes match MediaPipe.
            # =================================================================

            print(
                "---------- MEDIAPIPE RAW KEYPOINT DEBUG ----------"
            )

            print(
                f"filename: {filename}"
            )

            print(
                f"detection index: {face_idx}"
            )

            print(
                f"keypoint count: {len(keypoints)}"
            )

            for kp_idx, kp in enumerate(
                keypoints
            ):

                try:

                    kp_x = float(
                        getattr(
                            kp,
                            "x",
                            float("nan"),
                        )
                    )

                    kp_y = float(
                        getattr(
                            kp,
                            "y",
                            float("nan"),
                        )
                    )

                    kp_label = getattr(
                        kp,
                        "label",
                        None,
                    )

                    kp_score = getattr(
                        kp,
                        "score",
                        None,
                    )

                    print(
                        f"KP[{kp_idx}] "
                        f"x={kp_x:.4f} "
                        f"y={kp_y:.4f} "
                        f"label={kp_label} "
                        f"score={kp_score}"
                    )

                except Exception as kp_error:

                    print(
                        f"KP[{kp_idx}] "
                        f"ERROR={kp_error}"
                    )

            print(
                "=================================================="
            )

            # =================================================================
            # 8B. IMPORTANT:
            #
            # MediaPipe Face Detector keypoint order:
            #
            # 0 = left eye
            # 1 = right eye
            # 2 = nose tip
            # 3 = mouth
            # 4 = left eye tragion
            # 5 = right eye tragion
            #
            # _mediapipe_landmarks_complete() should use these exact
            # positions.
            # =================================================================

            if len(keypoints) < 4:

                print(
                    f"[MediaPipe] detection={face_idx} "
                    f"INSUFFICIENT KEYPOINTS "
                    f"({len(keypoints)} < 4) "
                    f"-> rejected"
                )

                continue

            # =================================================================
            # 9. LANDMARK VALIDATION
            # =================================================================

            lm_complete = False

            lm_detail = {
                "left_eye": False,
                "right_eye": False,
                "nose": False,
                "mouth": False,

                "eye_to_eye": None,
                "nose_eye_gap": None,
                "nose_x_offset_ratio": None,
                "eye_mouth_ratio": None,
                "eye_delta_y": None,
                "misalign_ratio": None,
                "mouth_nose_gap": None,

                "complete": False,
                "reason": "not evaluated",
            }

            try:

                landmark_output = (
                    _mediapipe_landmarks_complete(
                        keypoints=[
                            keypoints[0],  # left eye
                            keypoints[1],  # right eye
                            keypoints[2],  # nose
                            keypoints[3],  # mouth
                        ],
                        image_width=image_width,
                        image_height=image_height,
                    )
                )

                if (
                    isinstance(
                        landmark_output,
                        tuple,
                    )
                    and
                    len(landmark_output) == 2
                ):

                    lm_complete = bool(
                        landmark_output[0]
                    )

                    returned_detail = (
                        landmark_output[1]
                    )

                    if isinstance(
                        returned_detail,
                        dict,
                    ):

                        # Safe merge.
                        for key in lm_detail.keys():

                            if key in returned_detail:

                                lm_detail[key] = (
                                    returned_detail[key]
                                )

            except Exception as landmark_error:

                print(
                    f"[MediaPipe] detection={face_idx} "
                    f"LANDMARK VALIDATION ERROR: "
                    f"{landmark_error}"
                )

                traceback.print_exc()

                lm_complete = False

                lm_detail["reason"] = (
                    "landmark validation exception"
                )

            # =================================================================
            # 10. READ LANDMARK FLAGS SAFELY
            # =================================================================

            has_left_eye = bool(
                lm_detail.get(
                    "left_eye",
                    False,
                )
            )

            has_right_eye = bool(
                lm_detail.get(
                    "right_eye",
                    False,
                )
            )

            has_nose = bool(
                lm_detail.get(
                    "nose",
                    False,
                )
            )

            has_mouth = bool(
                lm_detail.get(
                    "mouth",
                    False,
                )
            )

            landmark_complete_flag = bool(
                lm_detail.get(
                    "complete",
                    False,
                )
            )

            reason = lm_detail.get(
                "reason",
                "unknown",
            )

            # =================================================================
            # 11. SAFE DEBUG METRICS
            # =================================================================

            def _format_metric(value):

                if value is None:
                    return "n/a"

                try:

                    number = float(value)

                    if not math.isfinite(
                        number
                    ):
                        return "n/a"

                    return f"{number:.3f}"

                except (
                    TypeError,
                    ValueError,
                ):

                    return "n/a"

            print(
                "---------- MEDIAPIPE LANDMARK RESULT ----------"
            )

            print(
                f"filename: {filename}"
            )

            print(
                f"detection index: {face_idx}"
            )

            print(
                f"confidence: {score:.4f}"
            )

            print(
                f"bbox: {bbox}"
            )

            print(
                f"left_eye={has_left_eye}"
            )

            print(
                f"right_eye={has_right_eye}"
            )

            print(
                f"nose={has_nose}"
            )

            print(
                f"mouth={has_mouth}"
            )

            print(
                f"complete={landmark_complete_flag}"
            )

            print(
                f"validation_result={lm_complete}"
            )

            print(
                f"reason={reason}"
            )

            print(
                "metrics: "
                f"eye2eye="
                f"{_format_metric(lm_detail.get('eye_to_eye'))} | "
                f"nose_eye_gap="
                f"{_format_metric(lm_detail.get('nose_eye_gap'))} | "
                f"misalign_ratio="
                f"{_format_metric(lm_detail.get('misalign_ratio'))} | "
                f"mouth_nose_gap="
                f"{_format_metric(lm_detail.get('mouth_nose_gap'))} | "
                f"nose_x_offset="
                f"{_format_metric(lm_detail.get('nose_x_offset_ratio'))} | "
                f"eye_mouth_ratio="
                f"{_format_metric(lm_detail.get('eye_mouth_ratio'))}"
            )

            print(
                "=================================================="
            )

            # =================================================================
            # 12. FINAL PRIVACY CONSENT GATE
            # =================================================================
            #
            # ALL FOUR required landmarks must exist.
            #
            # No:
            #   one-eye exception
            #   upper-face exception
            #   missing-mouth exception
            #   bbox-only acceptance
            # =================================================================

            consent_landmarks_valid = (
                lm_complete
                and landmark_complete_flag
                and has_left_eye
                and has_right_eye
                and has_nose
                and has_mouth
            )

            if not consent_landmarks_valid:

                print(
                    f"[MediaPipe] detection={face_idx} "
                    f"REJECTED FOR CONSENT "
                    f"(confidence={score:.4f}, "
                    f"left_eye={has_left_eye}, "
                    f"right_eye={has_right_eye}, "
                    f"nose={has_nose}, "
                    f"mouth={has_mouth}, "
                    f"complete={landmark_complete_flag}, "
                    f"reason={reason})"
                )

                continue

            # =================================================================
            # 13. ACCEPT FACE FOR CONSENT
            # =================================================================

            valid_faces.append(
                {
                    "confidence": float(score),
                    "bbox": list(bbox),
                    "landmarks_valid": True,
                    "landmark_detail": lm_detail,
                }
            )

            print(
                f"[MediaPipe] detection={face_idx} "
                f"LANDMARK CONSENT GATE PASSED"
            )

            print(
                f"[MediaPipe] detection={face_idx} "
                f"ACCEPTED FOR CONSENT "
                f"(confidence={score:.4f})"
            )

        # ===================================================================
        # 14. NO VALID FACE
        # ===================================================================

        if not valid_faces:

            print(
                f"[MediaPipe] filename={filename} "
                f"FINAL RESULT = NO VALID FACE"
            )

            result["face_detected"] = False
            result["confidence"] = 0.0
            result["bbox"] = []
            result["num_faces"] = 0
            result["all_faces"] = []

            return result

        # ===================================================================
        # 15. SELECT HIGHEST-CONFIDENCE VALID FACE
        # ===================================================================

        best_face = max(
            valid_faces,
            key=lambda face: float(
                face.get(
                    "confidence",
                    0.0,
                )
            ),
        )

        # ===================================================================
        # 16. FINAL RESULT
        # ===================================================================

        result["face_detected"] = True

        result["confidence"] = float(
            best_face.get(
                "confidence",
                0.0,
            )
        )

        result["bbox"] = list(
            best_face.get(
                "bbox",
                [],
            )
        )

        result["num_faces"] = len(
            valid_faces
        )

        result["all_faces"] = valid_faces

        result["detector_used"] = "mediapipe"

        print(
            f"[MediaPipe] filename={filename} "
            f"FINAL ACCEPTED "
            f"confidence={result['confidence']:.4f} "
            f"bbox={result['bbox']} "
            f"num_faces={result['num_faces']}"
        )

        return result

    # =======================================================================
    # FATAL ERROR
    # =======================================================================

    except Exception as e:

        print(
            f"[MediaPipe] FATAL detection error "
            f"filename={filename}: {e}"
        )

        traceback.print_exc()

        # Never convert a detector failure into a positive detection.
        result["face_detected"] = False
        result["confidence"] = 0.0
        result["bbox"] = []
        result["num_faces"] = 0
        result["all_faces"] = []

        return result
# ---------------------------------------------------------------------------
# Primary detection: RetinaFace
# ---------------------------------------------------------------------------

def _detect_retinaface(
    image_bgr: np.ndarray,
    filename: str,
) -> dict:

    result = {
        "face_detected": False,
        "confidence": 0.0,
        "bbox": [],
        "num_faces": 0,
        "all_faces": [],
        "detector_used": "retinaface",
    }

    try:

        if image_bgr is None:
            print(
                f"[RetinaFace] filename={filename} "
                f"image=None"
            )
            return result

        if not isinstance(
            image_bgr,
            np.ndarray,
        ):
            print(
                f"[RetinaFace] filename={filename} "
                f"image is not numpy array"
            )
            return result

        if image_bgr.size == 0:
            print(
                f"[RetinaFace] filename={filename} "
                f"image is empty"
            )
            return result

        height, width = (
            image_bgr.shape[:2]
        )

        app = _get_detector()

        if app is None:

            print(
                f"[RetinaFace] filename={filename} "
                f"detector unavailable"
            )

            return result

        faces = app.get(
            image_bgr
        )

        print(
            f"[RetinaFace] Raw faces found = "
            f"{0 if faces is None else len(faces)}"
        )

        if not faces:

            print(
                f"[RetinaFace] filename={filename} "
                f"detections=0 -> face_detected=false"
            )

            return result

        valid_faces = []

        for face_idx, face_data in enumerate(
            faces
        ):

            # ----------------------------------------------------------
            # Bounding box
            # ----------------------------------------------------------

            bbox = getattr(
                face_data,
                "bbox",
                None,
            )

            if bbox is None or len(bbox) < 4:

                print(
                    f"[RetinaFace] face={face_idx} "
                    f"missing bbox -> rejected"
                )

                continue

            try:

                box_int = [
                    int(
                        float(
                            bbox[0]
                        )
                    ),
                    int(
                        float(
                            bbox[1]
                        )
                    ),
                    int(
                        float(
                            bbox[2]
                        )
                    ),
                    int(
                        float(
                            bbox[3]
                        )
                    ),
                ]

            except Exception as bbox_error:

                print(
                    f"[RetinaFace] face={face_idx} "
                    f"bbox conversion error: "
                    f"{bbox_error}"
                )

                continue

            # ----------------------------------------------------------
            # Confidence
            # ----------------------------------------------------------

            try:

                det_score = float(
                    getattr(
                        face_data,
                        "det_score",
                        0.0,
                    )
                    or 0.0
                )

            except Exception:

                det_score = 0.0

            print(
                f"[RetinaFace] face={face_idx} "
                f"confidence={det_score:.4f} "
                f"bbox={box_int}"
            )

            # ----------------------------------------------------------
            # Bounding box validation
            # ----------------------------------------------------------

            try:

                bbox_valid = is_face_complete(
                    tuple(box_int),
                    width,
                    height,
                    detector_name="RetinaFace",
                    confidence=det_score,
                )

            except Exception as bbox_error:

                print(
                    f"[RetinaFace] face={face_idx} "
                    f"bbox validation ERROR: "
                    f"{bbox_error}"
                )

                traceback.print_exc()

                continue

            if not bbox_valid:

                print(
                    f"[RetinaFace] face={face_idx} "
                    f"INVALID bbox -> rejected"
                )

                continue

            # ----------------------------------------------------------
            # Confidence
            # ----------------------------------------------------------

            if det_score < 0.50:

                print(
                    f"[RetinaFace] face={face_idx} "
                    f"confidence={det_score:.4f} "
                    f"< 0.50 -> rejected"
                )

                continue

            # ----------------------------------------------------------
            # Landmark validation
            #
            # INFORMATIONAL ONLY.
            # ----------------------------------------------------------

            kps = getattr(
                face_data,
                "kps",
                None,
            )

            landmarks_valid = False

            landmark_detail = {
                "reason": "not evaluated",
            }

            try:

                landmark_output = (
                    _retinaface_landmarks_complete(
                        kps=kps,
                        face_bbox=box_int,
                        image_width=width,
                        image_height=height,
                    )
                )

                if (
                    isinstance(
                        landmark_output,
                        tuple,
                    )
                    and
                    len(landmark_output) == 2
                ):

                    landmarks_valid = bool(
                        landmark_output[0]
                    )

                    if isinstance(
                        landmark_output[1],
                        dict,
                    ):

                        landmark_detail = (
                            landmark_output[1]
                        )

            except Exception as landmark_error:

                landmarks_valid = False

                landmark_detail = {
                    "reason": (
                        "landmark validation error: "
                        f"{landmark_error}"
                    )
                }

                print(
                    f"[RetinaFace] face={face_idx} "
                    f"landmark validation ERROR: "
                    f"{landmark_error}"
                )

                traceback.print_exc()

            print(
                f"[RetinaFace] face={face_idx} "
                f"landmark validation="
                f"{'PASS' if landmarks_valid else 'FAIL'}"
            )

            print(
                f"[RetinaFace] face={face_idx} "
                f"landmark reason="
                f"{landmark_detail.get('reason', 'n/a')}"
            )

            # ----------------------------------------------------------
            # ACCEPT
            #
            # Landmark failure does NOT reject the face.
            # ----------------------------------------------------------

            valid_faces.append(
                {
                    "confidence": float(
                        det_score
                    ),
                    "bbox": list(
                        box_int
                    ),
                    "landmarks_valid": bool(
                        landmarks_valid
                    ),
                    "landmark_detail": (
                        landmark_detail
                    ),
                }
            )

            print(
                f"[RetinaFace] face={face_idx} "
                f"ACCEPTED for privacy consent"
            )

        # --------------------------------------------------------------
        # Select best face
        # --------------------------------------------------------------

        if not valid_faces:

            print(
                f"[RetinaFace] filename={filename} "
                f"no valid faces after validation "
                f"-> face_detected=false"
            )

            return result

        best_face = max(
            valid_faces,
            key=lambda face: float(
                face.get(
                    "confidence",
                    0.0,
                )
            ),
        )

        result["face_detected"] = True

        result["confidence"] = float(
            best_face.get(
                "confidence",
                0.0,
            )
        )

        result["bbox"] = list(
            best_face.get(
                "bbox",
                [],
            )
        )

        result["num_faces"] = len(
            valid_faces
        )

        result["all_faces"] = valid_faces

        print(
            f"[RetinaFace] filename={filename} "
            f"FINAL ACCEPTED "
            f"confidence={result['confidence']:.4f} "
            f"bbox={result['bbox']}"
        )

        return result

    except Exception as e:

        print(
            f"[RetinaFace] FATAL detection error "
            f"filename={filename}: {e}"
        )

        traceback.print_exc()

        return result

def detect_face_details(image, filename="<unknown>"):
    """
    Public face-detection entry point.

    RetinaFace is the primary detector.
    MediaPipe is used only when RetinaFace does not detect a valid face.
    """

    print(
        f"[FaceDetection] filename={filename} "
        f"Starting face detection..."
    )

    # ---------------------------------------------------------
    # Validate image
    # ---------------------------------------------------------

    if image is None:
        raise ValueError("Image is None")

    # PIL -> RGB NumPy
    if isinstance(image, Image.Image):
        image_rgb = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        image_rgb = image
        if image_rgb.ndim == 2:
            image_rgb = cv2.cvtColor(
                image_rgb,
                cv2.COLOR_GRAY2RGB
            )
    else:
        raise TypeError(
            f"Unsupported image type: {type(image)}"
        )

    if image_rgb.size == 0:
        raise ValueError("Image is empty")

    image_rgb = np.ascontiguousarray(image_rgb)

    height, width = image_rgb.shape[:2]

    # RetinaFace expects BGR
    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    # ---------------------------------------------------------
    # 1. RetinaFace PRIMARY
    # ---------------------------------------------------------

    print(
        f"[FaceDetection] filename={filename} "
        f"Starting RetinaFace detection..."
    )

    retina_result = _detect_retinaface(
        image_bgr,
        filename,
    )

    if retina_result.get("face_detected", False):

        print(
            f"[FaceDetection] filename={filename} "
            f"RetinaFace result = True "
            f"(confidence="
            f"{retina_result.get('confidence', 0.0):.4f})"
        )

        return {
            "face_detected": True,
            "confidence": float(
                retina_result.get("confidence", 0.0)
            ),
            "bbox": retina_result.get("bbox", []),
            "detector_used": "retinaface",
            "consent_required": True,
            "num_faces": int(
                retina_result.get("num_faces", 0)
            ),
            "all_faces": retina_result.get(
                "all_faces",
                []
            ),
        }

    print(
        f"[FaceDetection] filename={filename} "
        f"RetinaFace result = False"
    )

    # ---------------------------------------------------------
    # 2. MediaPipe FALLBACK
    # ---------------------------------------------------------

    print(
        f"[FaceDetection] filename={filename} "
        f"RetinaFace failed. "
        f"Starting MediaPipe fallback..."
    )

    mediapipe_result = _detect_mediapipe(
        image_rgb,
        filename,
        width,
        height,
    )

    if mediapipe_result.get("face_detected", False):

        print(
            f"[FaceDetection] filename={filename} "
            f"MediaPipe result = True "
            f"(confidence="
            f"{mediapipe_result.get('confidence', 0.0):.4f})"
        )

        return {
            "face_detected": True,
            "confidence": float(
                mediapipe_result.get("confidence", 0.0)
            ),
            "bbox": mediapipe_result.get("bbox", []),
            "detector_used": "mediapipe",
            "consent_required": True,
            "num_faces": int(
                mediapipe_result.get("num_faces", 0)
            ),
            "all_faces": mediapipe_result.get(
                "all_faces",
                []
            ),
        }

    # ---------------------------------------------------------
    # 3. No face detected
    # ---------------------------------------------------------

    print(
        f"[FaceDetection] filename={filename} "
        f"FINAL RESULT = False (no face detected)"
    )

    return {
        "face_detected": False,
        "confidence": 0.0,
        "bbox": [],
        "detector_used": "none",
        "consent_required": False,
        "num_faces": 0,
        "all_faces": [],
    }