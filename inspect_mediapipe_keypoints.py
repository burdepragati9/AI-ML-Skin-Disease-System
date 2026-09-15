"""
Temporary inspection script (READ-ONLY).

Inspect the MediaPipe Tasks FaceDetector detection structure, specifically the
`keypoints` field, on a real face image. This confirms how to access facial
landmarks (left eye, right eye, nose, mouth) for the landmark-based
completeness validation of the MediaPipe fallback.

No existing source files are modified.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from PIL import Image, ImageOps
import numpy as np
import mediapipe as mp

from backend.services.detect_face import _get_mediapipe_detector


def _load(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def main():
    # Real face image
    path = os.path.join("history", "profile_photos", "doctor_2_a242688c10.jpg")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        return

    detector = _get_mediapipe_detector()
    if detector is None:
        print("ERROR: MediaPipe detector unavailable")
        return

    img = _load(path)
    arr = np.array(img)
    arr = np.ascontiguousarray(arr)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
    result = detector.detect(mp_image)

    if result is None or not result.detections:
        print("No detections found.")
        return

    print(f"Found {len(result.detections)} detection(s).\n")

    for di, detection in enumerate(result.detections):
        print(f"--- Detection {di} ---")
        print(f"  score: {detection.categories[0].score}")
        bbox = detection.bounding_box
        print(f"  bbox: origin=({bbox.origin_x},{bbox.origin_y}) "
              f"size=({bbox.width}x{bbox.height})")

        print("  keypoints:")
        kps = getattr(detection, "keypoints", None)
        if kps is None:
            print("    (no keypoints field)")
            continue
        for ki, kp in enumerate(kps):
            label = getattr(kp, "label", None)
            x = getattr(kp, "x", None)
            y = getattr(kp, "y", None)
            score = getattr(kp, "score", None)
            print(f"    [{ki}] label={label} x={x} y={y} score={score}")
        print("  keypoints type:", type(kps))
        if kps:
            print("  single kp type:", type(kps[0]))
            print("  single kp dir:", [a for a in dir(kps[0]) if not a.startswith('__')])


if __name__ == "__main__":
    main()
