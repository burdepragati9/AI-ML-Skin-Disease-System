"""Verify the MediaPipe selfie edge-case exception in is_face_complete."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.services.detect_face import (
    is_face_complete,
    touches_image_boundary,
    SELFIE_MIN_CONFIDENCE,
)

# Image dimensions
W, H = 640, 480
max_w = int(W * 0.98)
max_h = int(H * 0.98)

def check(name, got, expected):
    status = "PASS" if got == expected else "FAIL"
    print(f"[{status}] {name}: got={got}, expected={expected}")
    return got == expected

ok = True

# --- touches_image_boundary tests ---
print("--- touches_image_boundary ---")
ok &= check("touches left edge", touches_image_boundary((0, 100, 300, 400), W, H), True)
ok &= check("touches right edge", touches_image_boundary((300, 100, 639, 400), W, H), True)
ok &= check("touches top edge", touches_image_boundary((100, 0, 400, 300), W, H), True)
ok &= check("touches bottom edge", touches_image_boundary((100, 200, 400, 479), W, H), True)
ok &= check("centered (no boundary)", touches_image_boundary((100, 100, 400, 300), W, H), False)
ok &= check("within tolerance left", touches_image_boundary((8, 100, 400, 300), W, H), True)
ok &= check("just outside tolerance", touches_image_boundary((11, 100, 400, 300), W, H), False)

# --- is_face_complete: MediaPipe selfie exception ---
print("--- is_face_complete: MediaPipe large-face selfie ---")
# Large face filling near-full frame, touches boundary, high confidence -> valid selfie
selfie_bbox = (0, 0, 639, 479)
ok &= check("MediaPipe large selfie accepted",
            is_face_complete(selfie_bbox, W, H, detector_name="MediaPipe",
                             confidence=0.95),
            True)
# Large face but low confidence -> rejected
ok &= check("MediaPipe large face low conf rejected",
            is_face_complete((0, 0, 639, 479), W, H, detector_name="MediaPipe",
                             confidence=0.5),
            False)
# Large face but centred (no boundary) high conf -> rejected.
# Use a larger canvas so a face can exceed 98% of a dimension while keeping
# margins larger than the boundary tolerance on all sides.
W2, H2 = 2000, 2000  # 0.98*2000 = 1960
# width = 1981-20 = 1961 > 1960; margins 20px (>10 tolerance) on all sides
ok &= check("MediaPipe large centred high-conf rejected",
            is_face_complete((20, 20, 1981, 1981), W2, H2, detector_name="MediaPipe",
                             confidence=0.95),
            False)
# On the larger canvas, a large face touching boundary + high conf -> accepted
ok &= check("MediaPipe large touching boundary accepted",
            is_face_complete((0, 0, 1999, 1999), W2, H2, detector_name="MediaPipe",
                             confidence=0.95),
            True)
# Large face but RetinaFace -> rejected (exception only for MediaPipe)
ok &= check("RetinaFace large face rejected",
            is_face_complete((0, 0, 639, 479), W, H, detector_name="RetinaFace",
                             confidence=0.95),
            False)

# --- is_face_complete: normal faces unchanged ---
print("--- is_face_complete: normal faces unchanged ---")
# Normal-sized face -> accepted
ok &= check("normal face accepted",
            is_face_complete((200, 100, 440, 300), W, H, detector_name="MediaPipe",
                             confidence=0.9),
            True)
# Tiny face -> rejected (too small)
ok &= check("tiny face rejected",
            is_face_complete((300, 200, 310, 210), W, H, detector_name="MediaPipe",
                             confidence=0.9),
            False)

print("\nOVERALL:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
