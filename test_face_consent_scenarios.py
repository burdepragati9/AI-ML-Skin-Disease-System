"""Scenario test for the Face Detection + Consent pipeline."""
import os
import sys
import traceback
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.services.detect_face import detect_face_details, detect_face
from utils.security import save_optimized_image
from pathlib import Path
import tempfile

print("=" * 70)
print("FACE DETECTION + CONSENT SCENARIO TEST")
print("=" * 70)

# We'll test the boolean contract and the JPEG RGB conversion.
# The actual face detection accuracy is validated by the detector on real images.

def test_boolean_contract():
    """Verify every path returns explicit True/False, never None."""
    print("\n--- TEST: Boolean contract (never None) ---")
    # Blank image (no face)
    blank = Image.new('RGB', (640, 480), color=(255, 255, 255))
    res = detect_face_details(blank, filename="blank.jpg")
    assert isinstance(res["face_detected"], bool), "face_detected must be bool"
    assert isinstance(res["consent_required"], bool), "consent_required must be bool"
    assert res["face_detected"] in (True, False), "face_detected must be True/False"
    assert res["consent_required"] in (True, False), "consent_required must be True/False"
    print(f"  blank -> face_detected={res['face_detected']} consent={res['consent_required']} detector={res['detector_used']}")

    # Grayscale (L mode) image
    gray = Image.new('L', (320, 240), color=128)
    res2 = detect_face_details(gray, filename="gray.jpg")
    assert isinstance(res2["face_detected"], bool)
    assert isinstance(res2["consent_required"], bool)
    print(f"  grayscale -> face_detected={res2['face_detected']} consent={res2['consent_required']} detector={res2['detector_used']}")

    # RGBA image
    rgba = Image.new('RGBA', (320, 240), color=(255, 0, 0, 255))
    res3 = detect_face_details(rgba, filename="rgba.png")
    assert isinstance(res3["face_detected"], bool)
    assert isinstance(res3["consent_required"], bool)
    print(f"  rgba -> face_detected={res3['face_detected']} consent={res3['consent_required']} detector={res3['detector_used']}")

    # numpy array input
    arr = np.zeros((200, 200, 3), dtype=np.uint8)
    res4 = detect_face_details(arr, filename="array.jpg")
    assert isinstance(res4["face_detected"], bool)
    assert isinstance(res4["consent_required"], bool)
    print(f"  numpy array -> face_detected={res4['face_detected']} consent={res4['consent_required']} detector={res4['detector_used']}")

    print("  PASS: All paths return explicit True/False booleans (never None)")
    return True

def test_jpeg_rgb_conversion():
    """Verify P, RGBA, L mode images can be saved as JPEG without error."""
    print("\n--- TEST: JPEG RGB conversion (P / RGBA / L) ---")
    tmpdir = Path(tempfile.mkdtemp())

    # P mode (palette)
    p_img = Image.new('P', (100, 100))
    p_img.putpalette([0, 0, 0, 255, 255, 255])
    p_res = save_optimized_image(p_img, tmpdir, "test_p")
    assert p_res.exists(), "P-mode JPEG save failed"
    assert Image.open(p_res).mode == "RGB"
    assert p_img.mode == "P", "Original P image must not be mutated"
    print(f"  P mode -> saved OK, mode={Image.open(p_res).mode}, original still {p_img.mode}")

    # RGBA mode
    rgba_img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
    rgba_res = save_optimized_image(rgba_img, tmpdir, "test_rgba")
    assert rgba_res.exists(), "RGBA-mode JPEG save failed"
    assert Image.open(rgba_res).mode == "RGB", "RGBA should be converted to RGB"
    assert rgba_img.mode == "RGBA", "Original RGBA image must not be mutated"
    print(f"  RGBA mode -> saved OK, mode={Image.open(rgba_res).mode}, original still {rgba_img.mode}")

    # L mode (grayscale)
    l_img = Image.new('L', (100, 100), color=128)
    l_res = save_optimized_image(l_img, tmpdir, "test_l")
    assert l_res.exists(), "L-mode JPEG save failed"
    assert l_img.mode == "L", "Original L image must not be mutated"
    print(f"  L mode -> saved OK, mode={Image.open(l_res).mode}, original still {l_img.mode}")

    print("  PASS: P / RGBA / L mode images all save as JPEG without error")
    return True

if __name__ == "__main__":
    ok = True
    try:
        ok = test_boolean_contract() and ok
    except Exception as e:
        ok = False
        print(f"  FAIL: boolean contract test error: {e}")
        traceback.print_exc()

    try:
        ok = test_jpeg_rgb_conversion() and ok
    except Exception as e:
        ok = False
        print(f"  FAIL: JPEG conversion test error: {e}")
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("OVERALL:", "PASS" if ok else "FAIL")
    print("=" * 70)
