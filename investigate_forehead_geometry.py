"""
Temporary investigation script for forehead-only false positives.

Prints raw MediaPipe landmark coordinates + geometric metrics for a range of
top-crops (forehead) so we can calibrate geometric validation that rejects
hallucinated keypoint layouts on forehead-only images while accepting real faces.
"""
import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from PIL import Image, ImageOps
import numpy as np
import mediapipe as mp

from backend.services.detect_face import _get_mediapipe_detector

KP_LABELS = {
    0: "right_eye", 1: "left_eye", 2: "nose_tip",
    3: "mouth_center", 4: "right_ear", 5: "left_ear",
}


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


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def main():
    detector = _get_mediapipe_detector()
    if detector is None:
        print("ERROR: MediaPipe detector unavailable")
        return

    path = os.path.join("history", "profile_photos", "doctor_2_a242688c10.jpg")
    base_img = _load(path)
    w, h = base_img.size
    base = os.path.basename(path)
    print(f"source {path} size=({w},{h})")

    # scenario fractions of top of the image
    scenarios = [
        ("Full face", (0, 0, w, h)),
        ("Half face", (0, int(h*0.20), w, int(h*0.60))),
        # foreground-ish top crops
        ("Forehead 30%", (0, 0, w, int(h*0.30))),
        ("Forehead 35%", (0, 0, w, int(h*0.35))),
        ("Forehead 40%", (0, 0, w, int(h*0.40))),
        ("Forehead 45%", (0, 0, w, int(h*0.45))),
        ("Chin 65%-end", (0, int(h*0.65), w, h)),
    ]

    for label, box in scenarios:
        crop = _crop(base_img, box)
        iw, ih = crop.size
        arr = np.ascontiguousarray(np.array(crop))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
        res = detector.detect(mp_image)
        print("\n" + "=" * 70)
        print(f"# {label} crop=({iw},{ih})")
        if res is None or not res.detections:
            print("  NO DETECTION")
            continue
        for di, det in enumerate(res.detections):
            score = det.categories[0].score
            bb = det.bounding_box
            print(f"  Detection {di}: score={score:.4f} "
                  f"bbox=({bb.origin_x},{bb.origin_y}) {bb.width}x{bb.height}")
            kps = getattr(det, "keypoints", None) or []
            pts = {}
            for ki, kp in enumerate(kps):
                x = float(getattr(kp, "x", -1))
                y = float(getattr(kp, "y", -1))
                pts[KP_LABELS.get(ki, f"kp{ki}")] = (x, y)
                print(f"    {KP_LABELS.get(ki, 'kp%02d' % ki):<12} nx={x:.4f} ny={y:.4f} "
                      f"px={x*iw:.1f} py={y*ih:.1f}")
            # geometry
            re_ = pts.get("right_eye"); le_ = pts.get("left_eye")
            ns = pts.get("nose_tip"); mo = pts.get("mouth_center")
            if re_ and le_ and ns:
                eye_cx = (re_[0]+le_[0])/2.0  # scalar x
                eye_cy = (re_[1]+le_[1])/2.0  # scalar y
                eye2eye = dist(re_, le_)
                eye2nose = dist((eye_cx, eye_cy), ns)
                nosemout = dist(ns, mo) if mo else float('nan')
                eyemout = dist((eye_cx, eye_cy), mo) if mo else float('nan')
                nose_center_off = ns[0] - eye_cx
                print(f"    eye2eye(e2e)={eye2eye:.4f} eye2nose={eye2nose:.4f}")
                print(f"    nose_center_x_off={nose_center_off:.4f}")
                if mo:
                    print(f"    nose2mouth={nosemout:.4f} eye_center2mouth={eyemout:.4f}")
                    print(f"    e2m/e2e={eyemout/eye2eye:.3f}  n2m/e2e={nosemout/eye2eye:.3f}")
                # aspect ratio of eye+mouth vertical span vs horizontal
                vspan = mo[1]-min(re_[1], le_[1]) if mo else (ns[1]-min(re_[1], le_[1]))
                print(f"    vert_span(eye->mouth)/e2e={(vspan/eye2eye) if eye2eye else float('nan'):.3f}")
            else:
                print("    (eyes/nose not all present)")


if __name__ == "__main__":
    main()

