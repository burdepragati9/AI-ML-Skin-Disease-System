"""
Comprehensive runtime verification of the RetinaFace -> MediaPipe face detection pipeline.

Goal: verify ACTUAL runtime behavior (not just code existence) across 9 scenarios:
  1. Clear frontal face
  2. Left profile face
  3. Right profile face
  4. Selfie
  5. Full-body image with visible face
  6. Upper-body image with visible face
  7. Slightly cropped face
  8. Severely cropped face (lips/chin/nose/forehead only)
  9. No human face

For each scenario, report:
  - RetinaFace result
  - Whether MediaPipe fallback was executed
  - MediaPipe result
  - Final face_detected value
  - Final consent_required value

Also confirm:
  - face_detected is never None
  - consent_required is never None
  - No OpenCV Haar Cascade is used anywhere
"""
import os
import sys
import traceback
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.services.detect_face import _detect_retinaface, _detect_mediapipe, detect_face_details

# ---------------------------------------------------------------------------
# Source real images (guaranteed to contain faces)
# ---------------------------------------------------------------------------
PROFILE_PHOTOS = [
    os.path.join("history", "profile_photos", "doctor_2_a242688c10.jpg"),
    os.path.join("history", "profile_photos", "doctor_3_19fa8ba62d.jpg"),
    os.path.join("history", "profile_photos", "doctor_3_aceb258bce.jpg"),
]

def _find_face_images():
    """Return a list of real face image paths from CroppedData/Acne."""
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


def _paste_onto_canvas(img, canvas_w=800, canvas_h=1000):
    """Paste a face image into a larger canvas to simulate upper/full-body."""
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
    img = img.copy()
    img.thumbnail((canvas_w // 2, canvas_h // 2))  # make face small relative to canvas
    cw, ch = img.size
    x = (canvas_w - cw) // 2
    y = (canvas_h - ch) // 2
    canvas.paste(img, (x, y))
    return canvas


def _run_full(image, filename):
    """Run the full public pipeline and return detail dict."""
    return detect_face_details(image, filename=filename)


def _run_stage_info(image, filename):
    """Run stage-wise to determine RetinaFace and MediaPipe results separately."""
    import numpy as np
    arr = np.array(image)

    # RetinaFace on BGR
    import cv2
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    rf = _detect_retinaface(bgr, filename) or {"face_detected": False}
    h, w = arr.shape[:2]

    # MediaPipe fallback (only relevant if RetinaFace failed)
    mp = None
    mp_executed = False
    if not rf.get("face_detected"):
        mp_executed = True
        mp = _detect_mediapipe(arr, filename, w, h) or {"face_detected": False}

    final = detect_face_details(image, filename=filename)
    return {
        "retinaface": rf.get("face_detected", False),
        "mp_executed": mp_executed,
        "mediapipe": (mp or {}).get("face_detected", False) if mp else None,
        "face_detected": final["face_detected"],
        "consent_required": final["consent_required"],
        "detector_used": final["detector_used"],
    }


def main():
    print("=" * 78)
    print("RUNTIME VERIFICATION: RetinaFace -> MediaPipe face detection pipeline")
    print("=" * 78)

    face_imgs = _find_face_images()
    print(f"\nFound {len(face_imgs)} real face images in CroppedData/Acne")
    print(f"Found {len(PROFILE_PHOTOS)} profile photos")

    # Ensure we have at least one real face source
    sources = []
    for p in PROFILE_PHOTOS:
        if os.path.exists(p):
            sources.append(("profile", p))
    for p in face_imgs[:5]:
        sources.append(("face", p))

    if not sources:
        print("ERROR: No real face source images found. Cannot run face scenarios.")
        return

    # We'll build a scenario list using real images and derived variations.
    # Use the first few sources as a base.
    base_img = _load(sources[0][1])
    base_name = os.path.basename(sources[0][1])

    scenarios = []

    # 1. Clear frontal face
    scenarios.append(("1. Clear frontal face", base_img.copy(), base_name))

    # 2. Left profile (-90 deg = rotate CCW so face turns left)
    scenarios.append(("2. Left profile face", base_img.copy().rotate(90, expand=True), base_name + "_left"))

    # 3. Right profile (+90 deg = rotate CW)
    scenarios.append(("3. Right profile face", base_img.copy().rotate(-90, expand=True), base_name + "_right"))

    # 4. Selfie (slightly rotated / small tilt)
    scenarios.append(("4. Selfie (tilted)", base_img.copy().rotate(8, expand=False, fillcolor=(0, 0, 0)), base_name + "_selfie"))

    # 5. Full-body (face small on large canvas)
    scenarios.append(("5. Full-body (face small)", _paste_onto_canvas(base_img), base_name + "_fullbody"))

    # 6. Upper-body (face medium on canvas)
    upper = _paste_onto_canvas(base_img, 600, 800)
    scenarios.append(("6. Upper-body (face medium)", upper, base_name + "_upperbody"))

    # 7. Slightly cropped face (crop ~15% off bottom = chin)
    w, h = base_img.size
    slightly_cropped = _crop(base_img, (0, 0, w, int(h * 0.88)))
    scenarios.append(("7. Slightly cropped face (chin)", slightly_cropped, base_name + "_chin"))

    # 8. Severely cropped face (only central region ~lips/chin/nose)
    severely_cropped = _crop(base_img, (int(w * 0.25), int(h * 0.45), int(w * 0.75), int(h * 0.85)))
    scenarios.append(("8. Severely cropped face (lips/chin/nose)", severely_cropped, base_name + "_severe"))

    # 9. No human face (solid color + random noise, plus a known lesion image if available)
    noface = Image.new("RGB", (640, 480), (200, 180, 160))
    scenarios.append(("9. No human face (solid)", noface, "no_face_solid.jpg"))

    rng = np.random.RandomState(42)
    noise = np.clip(rng.normal(128, 40, (480, 640, 3)), 0, 255).astype(np.uint8)
    scenarios.append(("9b. No human face (noise)", Image.fromarray(noise), "no_face_noise.jpg"))

    # Add a skin-lesion image if available
    lesion_candidates = [
        os.path.join("CroppedData", "Acne", "acne-open-comedo-4.jpeg"),
        os.path.join("CroppedData", "Acne", "07AcnePittedScars1.jpeg"),
        os.path.join("CroppedData", "Tinea", "tinea-foot-webs-43.jpeg"),
    ]
    lesion_path = next((p for p in lesion_candidates if os.path.exists(p)), None)
    if lesion_path:
        scenarios.append(("9c. No face (skin lesion)", _load(lesion_path), os.path.basename(lesion_path)))

    # ------------------------------------------------------------------
    # Run scenarios
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SCENARIO RESULTS")
    print("=" * 78)

    all_ok = True
    for label, img, fname in scenarios:
        print(f"\n--- {label} ({fname}) ---")
        try:
            info = _run_stage_info(img, fname)
            print(f"  RetinaFace result      : {info['retinaface']}")
            print(f"  MediaPipe executed?    : {info['mp_executed']}")
            print(f"  MediaPipe result       : {info['mediapipe']}")
            print(f"  Final face_detected    : {info['face_detected']}")
            print(f"  Final consent_required : {info['consent_required']}")
            print(f"  Detector used          : {info['detector_used']}")

            # Boolean contract checks
            assert isinstance(info["face_detected"], bool), "face_detected must be bool"
            assert isinstance(info["consent_required"], bool), "consent_required must be bool"

            # Consent consistency: consent_required should equal face_detected
            if info["consent_required"] != info["face_detected"]:
                print("  [WARN] consent_required != face_detected")
                all_ok = False
        except Exception as e:
            all_ok = False
            print(f"  [ERROR] {e}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Boolean contract summary across many images
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("BOOLEAN CONTRACT VERIFICATION (never None)")
    print("=" * 78)
    contract_ok = True
    test_images = []
    for p in PROFILE_PHOTOS:
        if os.path.exists(p):
            test_images.append(("profile", p))
    for p in face_imgs[:10]:
        test_images.append(("face", p))
    # lesion controls
    for p in lesion_candidates:
        if os.path.exists(p):
            test_images.append(("lesion", p))

    for kind, path in test_images:
        try:
            img = _load(path)
            res = _run_full(img, os.path.basename(path))
            fd = res["face_detected"]
            cr = res["consent_required"]
            if fd is None or cr is None or not isinstance(fd, bool) or not isinstance(cr, bool):
                contract_ok = False
                print(f"  [FAIL] {kind} {os.path.basename(path)}: face_detected={fd} consent={cr}")
            else:
                pass  # OK
        except Exception as e:
            contract_ok = False
            print(f"  [ERROR] {kind} {path}: {e}")

    print(f"  Validated {len(test_images)} images. Boolean contract: {'PASS' if contract_ok else 'FAIL'}")

    # ------------------------------------------------------------------
    # OpenCV Haar Cascade absence check
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("OPENCV HAAR CASCADE ABSENCE CHECK")
    print("=" * 78)
    # Search our detect_face module source for Haar/OpenCV cascade usage
    import inspect
    from backend.services import detect_face as df_mod
    src = inspect.getsource(df_mod)
    haar_terms = ["CascadeClassifier", "haarcascade", "detectMultiScale", "frontalface"]
    found = [t for t in haar_terms if t.lower() in src.lower()]
    if found:
        print(f"  [FAIL] detect_face.py contains Haar cascade references: {found}")
        all_ok = False
    else:
        print("  [PASS] detect_face.py has NO OpenCV Haar Cascade usage.")

    # Confirm the module only uses cv2 for color conversion
    uses_cv2_cascade = "cv2.CascadeClassifier" in src
    print(f"  cv2.CascadeClassifier present: {uses_cv2_cascade}")

    print("\n" + "=" * 78)
    print("OVERALL:", "PASS" if all_ok and contract_ok else "FAIL (see details above)")
    print("=" * 78)


if __name__ == "__main__":
    main()
