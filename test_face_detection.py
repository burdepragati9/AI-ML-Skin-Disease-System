"""
Test script for the improved face detection pipeline (RetinaFace + MediaPipe fallback).

Tests various image categories to ensure the detector correctly identifies faces
in difficult cases while avoiding false positives on skin lesion images.
"""
import os
import sys
import traceback
from PIL import Image, ImageOps
import numpy as np

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.services.detect_face import detect_face

# Test image paths (adjust as needed)
TEST_IMAGES = {
    "clear_frontal_face": [],       # We'll find these
    "side_profile_face": [],
    "partial_face": [],
    "small_face": [],
    "low_res_face": [],
    "boundary_face": [],
    "no_face_skin_lesion": [],
}

# Directories to search for test images
SEARCH_DIRS = [
    "CroppedData/Acne",      # Has face images (acne-face-*)
    "CroppedData/Psoriasis",
    "CroppedData/Tinea",
    "CroppedData/Vitiligo",
    "CroppedData_bad",
]

def find_test_images():
    """Scan directories for relevant test images."""
    all_jpegs = []
    for d in SEARCH_DIRS:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_jpegs.append(os.path.join(d, f))

    print(f"\nFound {len(all_jpegs)} total images to scan\n")

    # Categorize by filename patterns
    for path in all_jpegs:
        fname = os.path.basename(path).lower()
        # Skin lesion / no face images
        if any(kw in fname for kw in ['open-comedo', 'closed-comedo', 'pitted', 'lesion',
                                       'plaque', 'guttate', 'inversus', 'digit',
                                       'hand', 'foot', 'ankle', 'body', 'webs',
                                       'dorsum', 'versicolor']):
            TEST_IMAGES["no_face_skin_lesion"].append(path)
        elif 'face' in fname:
            # Face images - could be various types
            if 'side' in fname or 'profile' in fname:
                TEST_IMAGES["side_profile_face"].append(path)
            elif 'partial' in fname or 'crop' in fname:
                TEST_IMAGES["partial_face"].append(path)
            elif 'small' in fname:
                TEST_IMAGES["small_face"].append(path)
            elif 'low' in fname or 'blurry' in fname:
                TEST_IMAGES["low_res_face"].append(path)
            elif 'boundary' in fname or 'edge' in fname:
                TEST_IMAGES["boundary_face"].append(path)
            else:
                TEST_IMAGES["clear_frontal_face"].append(path)

    # Print summary
    for category, paths in TEST_IMAGES.items():
        print(f"  {category}: {len(paths)} images")
        for p in paths[:3]:  # Show first 3
            print(f"    - {p}")

    if not any(TEST_IMAGES.values()):
        print("\nWARNING: No images found in expected locations.")
        print("Using images from 'acne-face-*' pattern as frontal faces and")
        print("skin lesion images as no-face controls.")

def main():
    print("=" * 70)
    print("FACE DETECTION PIPELINE TEST")
    print("=" * 70)

    find_test_images()

    # --- Test clear frontal face images ---
    print("\n" + "=" * 70)
    print("TEST 1: Clear Frontal Face")
    print("=" * 70)
    test_category("clear_frontal_face", expect_face=True)

    # --- Test no-face skin lesion ---
    print("\n" + "=" * 70)
    print("TEST 2: Skin Lesion (No Face)")
    print("=" * 70)
    test_category("no_face_skin_lesion", expect_face=False)

    # --- Test any available face images ---
    print("\n" + "=" * 70)
    print("TEST 3: Side Profile / Partial Face (if available)")
    print("=" * 70)
    test_category("side_profile_face", expect_face=True)
    test_category("partial_face", expect_face=True)

    # --- Test more categories ---
    print("\n" + "=" * 70)
    print("TEST 4: Challenging Cases (if available)")
    print("=" * 70)
    test_category("small_face", expect_face=True)
    test_category("low_res_face", expect_face=True)
    test_category("boundary_face", expect_face=True)

    # --- If no categorized face images, use all face-containing images ---
    total_face_images = sum(len(v) for k, v in TEST_IMAGES.items() if k != "no_face_skin_lesion")
    if total_face_images == 0:
        print("\n" + "=" * 70)
        print("No categorized face images found. Scanning for any face-containing images...")
        print("=" * 70)
        scan_for_faces()

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

def test_category(category, expect_face=None):
    """Run detection on all images in a category."""
    images = TEST_IMAGES.get(category, [])
    if not images:
        print(f"  No images in category '{category}', skipping.")
        return

    passed = 0
    failed = 0

    # Limit to 5 images per category to keep output manageable
    for img_path in images[:5]:
        fname = os.path.basename(img_path)
        try:
            print(f"\n  Testing: {img_path}")
            image = Image.open(img_path)
            image = ImageOps.exif_transpose(image)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Call detect_face - the improved pipeline with RetinaFace + MediaPipe
            result = detect_face(image, filename=fname)

            status = "✓" if result else "✗"
            expected = " (expected face)" if expect_face else " (expected no face)"

            if expect_face is not None and result == expect_face:
                passed += 1
                print(f"  [{status}] {fname} -> face_detected={result}{expected}")
            elif expect_face is None:
                print(f"  [{status}] {fname} -> face_detected={result}")
            else:
                failed += 1
                print(f"  [{status}] {fname} -> face_detected={result}{expected} " +
                      ("← MISMATCH!" if expect_face is not None else ""))

        except Exception as e:
            failed += 1
            print(f"  [ERROR] {fname}: {e}")
            traceback.print_exc()

    if expect_face is not None:
        print(f"\n  Category '{category}': {passed} passed, {failed} failed out of {len(images[:5])} tested")
    else:
        print(f"\n  Category '{category}': {failed} errors")


def scan_for_faces():
    """Try to detect faces in all available images."""
    import cv2

    all_images = []
    for d in SEARCH_DIRS:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_images.append(os.path.join(d, f))

    print(f"  Scanning {len(all_images)} images for faces...\n")

    face_found_count = 0
    no_face_count = 0
    face_images = []

    for img_path in all_images[:30]:  # Limit to 30
        fname = os.path.basename(img_path)
        try:
            image = Image.open(img_path)
            image = ImageOps.exif_transpose(image)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            result = detect_face(image, filename=fname)
            if result:
                face_found_count += 1
                face_images.append(img_path)
                print(f"  [FACE] {fname}")
            else:
                no_face_count += 1
        except Exception as e:
            print(f"  [ERROR] {fname}: {e}")

    print(f"\n  Scan results: {face_found_count} images WITH face, {no_face_count} without face")
    if face_images:
        print(f"\n  Images with detected faces:")
        for p in face_images:
            print(f"    - {p}")


if __name__ == "__main__":
    main()

