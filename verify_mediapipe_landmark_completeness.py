"""
Temporary verification for MediaPipe landmark-based completeness (fallback only).

Purpose:
    Confirm that the new landmark completeness check inside _detect_mediapipe()
    accepts recognizable faces (Full face, Half face, slightly-cropped selfie)
    while rejecting partial-face crops (Chin-only, Forehead-only, Mouth-only,
    Nose-only) when RetinaFace does NOT detect a valid face.

No existing detection/consent logic is changed by this script; it only calls
the existing _detect_mediapipe() with generated scenarios and prints results.
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from backend.services.detect_face import _detect_mediapipe
from PIL import Image, ImageOps
import numpy as np


PROFILE_PHOTOS = [
    os.path.join("history", "profile_photos", "doctor_2_a242688c10.jpg"),
    os.path.join("history", "profile_photos", "doctor_3_19fa8ba62d.jpg"),
    os.path.join("history", "profile_photos", "doctor_3_aceb258bce.jpg"),
]


def _load(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _crop(img, box):
    w, h = img.size
    l, u, r, d = box
    l = max(0, int(l)); u = max(0, int(u))
    r = min(w, int(r)); d = min(h, int(d))
    if r <= l or d <= u:
        return img.copy()
    return img.crop((l, u, r, d))


def build_scenarios(base_img, base_name):
    w, h = base_img.size
    scenarios = [
        ("Full face", base_img.copy(), f"{base_name}_full"),
        ("Half face", _crop(base_img, (0, int(h * 0.25), w, int(h * 0.75))), f"{base_name}_half"),
        ("Chin image", _crop(base_img, (0, int(h * 0.65), w, h)), f"{base_name}_chin"),
        ("Forehead image", _crop(base_img, (0, 0, w, int(h * 0.35))), f"{base_name}_forehead"),
    ]
    return scenarios


def main():
    print("=" * 90)
    print("MEDIAPIPE LANDMARK COMPLETENESS VERIFICATION (fallback only)")
    print("=" * 90)

    sources = [p for p in PROFILE_PHOTOS if os.path.exists(p)]
    if not sources:
        print("ERROR: no real face images found")
        return

    base_path = sources[0]
    base_img = _load(base_path)
    base_name = os.path.basename(base_path)

    print(f"Using source: {base_path} ({base_img.size})")

    summary = []
    for label, crop_img, fname in build_scenarios(base_img, base_name):
        print("\n" + "#" * 90)
        print(f"# SCENARIO: {label}  (filename={fname})  image_size={crop_img.size}")
        print("#" * 90)
        arr = np.array(crop_img)
        try:
            res = _detect_mediapipe(arr, fname, crop_img.size[0], crop_img.size[1])
            print(f"--- RESULT: {label} -> face_detected={res.get('face_detected')} "
                  f"conf={res.get('confidence'):.4f} bbox={res.get('bbox')}")
            summary.append((label, res.get("face_detected"), res.get("confidence"), res.get("bbox")))
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            summary.append((label, "ERROR", 0.0, []))

    print("\n" + "=" * 90)
    print("SUMMARY (MediaPipe fallback only)")
    print("=" * 90)
    print(f"{'Scenario':<16} {'MediaPipe Detected':<20} {'Conf':<8} {'BBox'}")
    for label, detected, conf, bbox in summary:
        print(f"{label:<16} {str(detected):<20} {conf:<8.4f} {bbox}")

    print("\nExpected success table:")
    print("  Full face / Half face -> MediaPipe True (accepted)")
    print("  Chin / Forehead       -> MediaPipe False (rejected)")


if __name__ == "__main__":
    main()

