"""
Temporary verification script (READ-ONLY).

Purpose:
    Verify whether InsightFace RetinaFace returns facial landmarks (`kps`)
    for every detected face across four scenarios:
      1. Full face
      2. Half face (centre vertical crop of the face region)
      3. Chin image  (bottom portion of the image)
      4. Forehead image (top portion of the image)

This script does NOT modify any detection / consent logic. It only calls the
existing `_detect_retinaface()` function and captures its stdout, so the
temporary FACE DEBUG logging already present in backend/services/detect_face.py
can be inspected.

No existing source files are changed.
"""
import sys
import io
import os
import time
import traceback
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from backend.services.detect_face import _detect_retinaface

from PIL import Image, ImageOps
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Source face images (guaranteed to contain human faces)
# ---------------------------------------------------------------------------
PROFILE_PHOTOS = [
    os.path.join("history", "profile_photos", "doctor_2_a242688c10.jpg"),
    os.path.join("history", "profile_photos", "doctor_3_19fa8ba62d.jpg"),
    os.path.join("history", "profile_photos", "doctor_3_aceb258bce.jpg"),
]

FACE_CROPS = [
    os.path.join("CroppedData", "Acne", "acne-face-1-1__ProtectWyJQcm90ZWN0Il0_FocusFillWzI5NCwyMjIsInkiLDg1XQ.jpeg"),
    os.path.join("CroppedData", "Acne", "acne-face-1-30__ProtectWyJQcm90ZWN0Il0_FocusFillWzI5NCwyMjIsIngiLDFd.jpeg"),
    os.path.join("CroppedData", "Acne", "acne-face-1-46__ProtectWyJQcm90ZWN0Il0_FocusFillWzI5NCwyMjIsIngiLDFd.jpeg"),
]


def _find_face_images():
    """Return real face image paths from CroppedData/Acne."""
    base = os.path.join("CroppedData", "Acne")
    out = []
    if os.path.isdir(base):
        for f in sorted(os.listdir(base)):
            if f.lower().startswith("acne-face") and f.lower().endswith((".jpeg", ".jpg", ".png")):
                out.append(os.path.join(base, f))
    return out


def _load(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _crop(img, box):
    """Crop with a PIL box (left, upper, right, lower), clamped."""
    w, h = img.size
    l, u, r, d = box
    l = max(0, int(l)); u = max(0, int(u))
    r = min(w, int(r)); d = min(h, int(d))
    if r <= l or d <= u:
        return img.copy()
    return img.crop((l, u, r, d))


def build_scenarios(base_img, base_name):
    """
    Build four scenarios from a real face image:
      1. Full face      -> original image
      2. Half face      -> centre vertical crop (middle 50% height)
      3. Chin image     -> bottom 35% of height
      4. Forehead image -> top 35% of height
    """
    w, h = base_img.size
    scenarios = [
        ("Full face", base_img.copy(), f"{base_name}_full"),
        ("Half face", _crop(base_img, (0, int(h * 0.25), w, int(h * 0.75))), f"{base_name}_half"),
        ("Chin image", _crop(base_img, (0, int(h * 0.65), w, h)), f"{base_name}_chin"),
        ("Forehead image", _crop(base_img, (0, 0, w, int(h * 0.35))), f"{base_name}_forehead"),
    ]
    return scenarios


def run_retinaface_capture(img, filename):
    """
    Run _detect_retinaface on a PIL image, capturing all stdout so the
    temporary FACE DEBUG block can be inspected in the returned string.
    """
    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    buf = io.StringIO()
    result = {}
    with redirect_stdout(buf):
        try:
            result = _detect_retinaface(bgr, filename)
        except Exception as e:
            result = {"face_detected": False, "error": str(e)}
            traceback.print_exc()
    output = buf.getvalue()
    return result, output


def warmup_detector(base_img, base_name):
    """
    Initialize the RetinaFace singleton detector before running the scenario
    matrix. The buffalo_l pack loads 5 ONNX models, and the very first init
    can transiently hit a MemoryError on some machines. Retry a few times so
    the scenario runs use the already-initialized singleton.

    Does NOT modify any detection/consent logic.
    """
    print("[Warm-up] Initializing RetinaFace detector singleton (may retry)...")
    for attempt in range(1, 6):
        try:
            result, output = run_retinaface_capture(base_img, base_name + "_warmup")
            if "Model loaded successfully" in output:
                print(f"[Warm-up] Detector initialized on attempt {attempt}.")
                return True
            print(f"[Warm-up] Attempt {attempt}: model not ready yet "
                  f"(detected={result.get('face_detected')}). Retrying...")
        except Exception as e:
            print(f"[Warm-up] Attempt {attempt} raised: {e}. Retrying...")
        time.sleep(2)
    print("[Warm-up] WARNING: could not confirm detector initialization after retries.")
    return False


def main():
    print("=" * 90)
    print("RETINAFACE kps VERIFICATION (temporary debug, read-only)")
    print("=" * 90)

    # Gather real face sources
    face_imgs = _find_face_images()
    sources = []
    for p in PROFILE_PHOTOS:
        if os.path.exists(p):
            sources.append(("profile", p))
    for p in face_imgs:
        if len(sources) >= 2:
            break
        sources.append(("face", p))
    for p in FACE_CROPS:
        if os.path.exists(p) and len(sources) < 3:
            sources.append(("face", p))

    if not sources:
        print("ERROR: No real face source images found. Cannot run scenarios.")
        return

    print(f"Found {len(sources)} real face source(s).")
    for kind, p in sources:
        print(f"  [{kind}] {p}")

    # Track summary info
    summary = []

    # Pick one source for the detailed scenario matrix (full/half/chin/forehead)
    base_kind, base_path = sources[0]
    base_img = _load(base_path)
    base_name = os.path.basename(base_path)

    # Warm up the RetinaFace singleton so the scenario runs don't hit a
    # transient MemoryError during the first (buffalo_l) model load.
    warmup_detector(base_img, base_name)

    print("\n" + "=" * 90)
    print(f"SCENARIO MATRIX using source: {base_path}")
    print("=" * 90)

    for label, crop_img, fname in build_scenarios(base_img, base_name):
        print("\n" + "#" * 90)
        print(f"# SCENARIO: {label}  (filename={fname})  image_size={crop_img.size}")
        print("#" * 90)
        result, output = run_retinaface_capture(crop_img, fname)
        # Print the captured stdout (includes the temporary FACE DEBUG block)
        print(output, end="")
        if output.strip() == "":
            print("(no debug output captured)")
        print(f"--- SCENARIO RESULT: {label} -> face_detected={result.get('face_detected')} "
              f"conf={result.get('confidence')} bbox={result.get('bbox')} num_faces={result.get('num_faces')}")
        summary.append((label, fname, result.get('face_detected'),
                        result.get('confidence'), result.get('bbox'),
                        result.get('num_faces')))

    # Also run the full public pipeline once to show end-to-end result for the base image
    print("\n" + "=" * 90)
    print("END-TO-END PUBLIC PIPELINE (base source, full face)")
    print("=" * 90)
    from backend.services.detect_face import detect_face_details
    try:
        details = detect_face_details(base_img, filename=base_name)
        print(f"detect_face_details -> {details}")
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'Scenario':<16} {'Filename':<46} {'Detected':<10} {'Conf':<8} {'#Faces'}")
    for label, fname, detected, conf, bbox, num_faces in summary:
        print(f"{label:<16} {fname[:45]:<46} {str(detected):<10} {str(conf):<8} {str(num_faces)}")

    print("\nNote: Full FACE DEBUG details (det_score, bbox, kps, landmark count and")
    print("coordinates) are visible in the captured output above for each scenario.")


if __name__ == "__main__":
    main()

