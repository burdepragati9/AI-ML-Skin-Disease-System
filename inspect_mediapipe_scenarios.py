"""
Temporary calibration script (READ-ONLY).

Runs the MediaPipe Tasks FaceDetector on four face scenarios (Full, Half,
Chin, Forehead) and prints the 6 normalized keypoints for each detection.
This lets us calibrate the landmark-based completeness validation for the
MediaPipe fallback (which required landmarks are visible within image bounds).

No existing source files are modified.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from PIL import Image, ImageOps
import numpy as np
import mediapipe as mp

from backend.services.detect_face import _get_mediapipe_detector

# Known blaze_face_short_range keypoint order
KP_LABELS = {
    0: "right_eye",
    1: "left_eye",
    2: "nose_tip",
    3: "mouth_center",
    4: "right_ear",
    5: "left_ear",
}
REQUIRED = ["right_eye", "left_eye", "nose_tip"]


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
    return [
        ("Full face", base_img.copy(), f"{base_name}_full"),
        ("Half face", _crop(base_img, (0, int(h * 0.25), w, int(h * 0.75))), f"{base_name}_half"),
        ("Chin image", _crop(base_img, (0, int(h * 0.65), w, h)), f"{base_name}_chin"),
        ("Forehead image", _crop(base_img, (0, 0, w, int(h * 0.35))), f"{base_name}_forehead"),
    ]


def main():
    detector = _get_mediapipe_detector()
    if detector is None:
        print("ERROR: MediaPipe detector unavailable")
        return

    path = os.path.join("history", "profile_photos", "doctor_2_a242688c10.jpg")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        return

    base_img = _load(path)
    base_name = os.path.basename(path)

    print("=" * 80)
    print("MEDIAPIPE KEYPOINT CALIBRATION ACROSS SCENARIOS")
    print("=" * 80)

    for label, crop_img, fname in build_scenarios(base_img, base_name):
        print("\n" + "#" * 80)
        print(f"# {label}  ({fname})  image_size={crop_img.size}")
        print("#" * 80)

        arr = np.ascontiguousarray(np.array(crop_img))
        ih, iw = arr.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
        result = detector.detect(mp_image)

        if result is None or not result.detections:
            print("  NO DETECTION")
            continue

        for di, detection in enumerate(result.detections):
            score = detection.categories[0].score
            bbox = detection.bounding_box
            print(f"  Detection {di}: score={score:.4f} "
                  f"bbox=({bbox.origin_x},{bbox.origin_y}) {bbox.width}x{bbox.height}")

            kps = getattr(detection, "keypoints", None) or []
            if not kps:
                print("  No keypoints available.")
                continue

            print(f"  {'idx':<4}{'label':<14}{'nx':<10}{'ny':<10}{'px':<8}{'py':<8}{'in_bounds(0.05)'}")
            present = {}
            for ki, kp in enumerate(kps):
                label_name = KP_LABELS.get(ki, f"kp{ki}")
                x = float(getattr(kp, "x", -1))
                y = float(getattr(kp, "y", -1))
                px = x * iw
                py = y * ih
                margin = 0.05
                in_bounds = (margin <= x <= 1 - margin) and (margin <= y <= 1 - margin)
                present[label_name] = in_bounds
                print(f"  {ki:<4}{label_name:<14}{x:<10.4f}{y:<10.4f}"
                      f"{px:<8.1f}{py:<8.1f}{str(in_bounds)}")

            missing = [lab for lab in REQUIRED if not present.get(lab, False)]
            complete = len(missing) == 0
            print(f"  REQUIRED present: "
                  f"right_eye={present.get('right_eye')} left_eye={present.get('left_eye')} "
                  f"nose_tip={present.get('nose_tip')}")
            print(f"  MISSING: {missing}")
            print(f"  COMPLETE (margin=0.05): {complete}")


if __name__ == "__main__":
    main()

