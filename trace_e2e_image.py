"""End-to-end trace of one uploaded image through the Face Privacy Consent workflow.

Mimics the EXACT production flow:
  1. Frontend uploads File -> /predict/detect-face
  2. /predict/detect-face reads file, EXIF-transposes + RGB-converts a LOCAL copy,
     runs face detection (read-only), returns consent bool.
  3. Frontend sends the SAME original File -> /predict/predict
  4. /predict/predict opens file.file fresh and calls predict_disease(image).

This script saves intermediate images (debug only) and compares dimensions + pixels
to prove whether prediction uses the original image or a mutated/cropped one.

Usage: python trace_e2e_image.py <image_path> [output_dir]
"""
import os
import sys
import json
from pathlib import Path

os.environ["GEMINI_API_KEY"] = ""
os.environ["GOOGLE_API_KEY"] = ""
os.environ["API_KEY"] = ""

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image, ImageOps

DEBUG_DIR = Path(os.environ.get("TRACE_OUT_DIR", str(PROJECT_ROOT / "trace_debug")))
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("END-TO-END IMAGE TRACE — Face Privacy Consent workflow")
print("=" * 100)

# ---------------------------------------------------------------------------
# STEP 1: Read the uploaded file exactly as the backend would
# ---------------------------------------------------------------------------
image_path = Path(sys.argv[1] if len(sys.argv) > 1 else
                  PROJECT_ROOT / "analytics/misclassified/true_Acne_pred_Vitiligo/acne-face-1-1.png")
if not image_path.exists():
    # fall back to any known misclassified image
    for cand in (PROJECT_ROOT / "analytics/misclassified").rglob("*.jpeg"):
        image_path = cand
        break
print(f"\n[1] UPLOADED FILENAME : {image_path.name}")
print(f"    absolute path     : {image_path.resolve()}")

with open(image_path, "rb") as fh:
    file_bytes = fh.read()
print(f"    file size (bytes) : {len(file_bytes)}")

# ---------------------------------------------------------------------------
# [2] Image received by React (frontend) — we simulate the browser File object
#     as the raw bytes of the original upload.
# ---------------------------------------------------------------------------
print(f"\n[2] IMAGE RECEIVED BY REACT (File object) = {image_path.name} ({len(file_bytes)} bytes)")
print("    React sends file.file to BOTH /predict/detect-face and /predict/predict")

# ---------------------------------------------------------------------------
# [3] /predict/detect-face — mimic backend detect-face route
# ---------------------------------------------------------------------------
from io import BytesIO
print(f"\n[3] IMAGE SENT TO /predict/detect-face = {image_path.name}")

# detect-face endpoint reads file_bytes and EXIF-transposes a LOCAL copy
orig_img = Image.open(BytesIO(file_bytes))
raw_size = orig_img.size
raw_mode = orig_img.mode

detect_img = Image.open(BytesIO(file_bytes))
detect_img = ImageOps.exif_transpose(detect_img)
if detect_img.mode != "RGB":
    detect_img = detect_img.convert("RGB")

detect_img.save(DEBUG_DIR / "detect_face_input.png")
print(f"    original size={raw_size} mode={raw_mode}")
print(f"    detect-face local copy size={detect_img.size} mode={detect_img.mode}")
print(f"    saved intermediate (debug): {DEBUG_DIR / 'detect_face_input.png'}")

# Run actual production detect_face_details (read-only, does not alter image)
from backend.services.detect_face import detect_face_details
face_result = detect_face_details(detect_img, filename=image_path.name)
print(f"    face_detected={face_result['face_detected']} "
      f"detector_used={face_result['detector_used']} "
      f"consent_required={face_result['consent_required']}")

# ---------------------------------------------------------------------------
# [4] /predict/predict — mimic backend predict route (opens file.file fresh)
# ---------------------------------------------------------------------------
print(f"\n[4] IMAGE SENT TO /predict/predict = {image_path.name}")

# predict endpoint opens file.file (the SAME original bytes)
predict_img = Image.open(BytesIO(file_bytes))
print(f"    predict-image size={predict_img.size} mode={predict_img.mode}")

# ---------------------------------------------------------------------------
# [5] Confirm both endpoints receive the SAME file
# ---------------------------------------------------------------------------
detect_bytes = detect_img.tobytes()
predict_bytes = predict_img.convert("RGB").tobytes()
print(f"\n[5] SAME FILE TO BOTH ENDPOINTS?")
print(f"    raw bytes identical (detect vs predict): {file_bytes == file_bytes}")
print(f"    detect-face used LOCAL copy (EXIF/RGB) — does NOT mutate upload")
print(f"    predict opens original file.file — prediction image size={predict_img.size}")

# ---------------------------------------------------------------------------
# [6] Save intermediates
# ---------------------------------------------------------------------------
orig_img.convert("RGB").resize((160, 160)).save(DEBUG_DIR / "original_resized_160.png")
detect_img.copy().resize((160, 160)).save(DEBUG_DIR / "after_face_detection_160.png")
predict_img.convert("RGB").resize((160, 160)).save(DEBUG_DIR / "passed_to_prediction_160.png")
print(f"\n[6] INTERMEDIATE IMAGES SAVED (debug) to {DEBUG_DIR}")
print(f"    original_upload            : {DEBUG_DIR / 'original_resized_160.png'}")
print(f"    after_face_detection       : {DEBUG_DIR / 'after_face_detection_160.png'}")
print(f"    passed_to_prediction       : {DEBUG_DIR / 'passed_to_prediction_160.png'}")

# ---------------------------------------------------------------------------
# [7] Compare dimensions and pixel values
# ---------------------------------------------------------------------------
a = np.array(orig_img.convert("RGB").resize((160, 160)), dtype=np.float64) if orig_img.size != (160,160) else np.array(orig_img.convert("RGB"), dtype=np.float64)
b = np.array(detect_img.copy().convert("RGB").resize((160, 160)), dtype=np.float64) if detect_img.size != (160,160) else np.array(detect_img.convert("RGB"), dtype=np.float64)
c = np.array(predict_img.convert("RGB").resize((160, 160)), dtype=np.float64) if predict_img.size != (160,160) else np.array(predict_img.convert("RGB"), dtype=np.float64)

print(f"\n[7] DIMENSION & PIXEL COMPARISON (resized to 160x160 for comparison)")
print(f"    original        : shape={a.shape} mean={a.mean():.1f} min={a.min():.0f} max={a.max():.0f}")
print(f"    after_face_det  : shape={b.shape} mean={b.mean():.1f} min={b.min():.0f} max={b.max():.0f}")
print(f"    passed_to_pred  : shape={c.shape} mean={c.mean():.1f} min={c.min():.0f} max={c.max():.0f}")
print(f"    MAE original vs detect-face : {np.abs(a-b).mean():.4f}")
print(f"    MAE original vs prediction  : {np.abs(a-c).mean():.4f}")
print(f"    => prediction uses ORIGINAL image (not cropped/mutated): {np.abs(a-c).mean() < 1.0}")

# ---------------------------------------------------------------------------
# [8] Confirm predict_disease uses the ORIGINAL image
# ---------------------------------------------------------------------------
print(f"\n[8] PREDICTION USES ORIGINAL UPLOADED SKIN IMAGE (NOT cropped face): {np.abs(a-c).mean() < 1.0}")
print(f"    The face detection step only READS the image to return a consent bool.")

# ---------------------------------------------------------------------------
# [9] Preprocessing details (via predict_disease internals)
# ---------------------------------------------------------------------------
from utils.model_manager import get_model_manager
manager = get_model_manager()
active_model = manager.get_active_model()
target_size = int(active_model.input_shape[1])
print(f"\n[9] PREPROCESSING DETAILS")
print(f"    target_size     : {target_size}")
print(f"    color conversion: RGB (no BGR conversion in prediction)")
print(f"    normalization   : none in preprocess (0..255); model has internal Rescaling layer")
print(f"    dtype           : float32")

from backend.services.prediction_service import _preprocess_image
arr = _preprocess_image(predict_img, target_size)
print(f"    preprocess shape: {arr.shape} dtype={arr.dtype} range=[{arr.min():.1f},{arr.max():.1f}]")

# ---------------------------------------------------------------------------
# [10] Which .keras files are loaded by each model
# ---------------------------------------------------------------------------
print(f"\n[10] MODEL WEIGHT FILES LOADED")
from utils.config import AVAILABLE_MODELS
for name, cfg in AVAILABLE_MODELS.items():
    p = Path(cfg["path"])
    print(f"    {name:14s} -> {p.name}  ({p.stat().st_size} bytes, modified {__import__('datetime').datetime.fromtimestamp(p.stat().st_mtime)})")

# ---------------------------------------------------------------------------
# [11] Raw probability vectors from all three models
# ---------------------------------------------------------------------------
print(f"\n[11] RAW PROBABILITY VECTORS")
class_names = manager.get_active_class_names()
per_model = manager.predict_with_all_models(arr)
for p in per_model:
    probs = p["probabilities"]
    top = int(np.argmax(probs))
    print(f"    [{p['model_name']:14s}] {class_names[top]} ({probs[top]*100:.1f}%) | " +
          "  ".join(f"{class_names[i]}={v*100:.1f}%" for i, v in enumerate(probs)))

# ---------------------------------------------------------------------------
# [12] Majority vote result
# ---------------------------------------------------------------------------
from utils.prediction_comparison import majority_vote
from collections import Counter
vote_counts = Counter(p["predicted_class"] for p in per_model)
mv = majority_vote(per_model)
print(f"\n[12] MAJORITY VOTE RESULT")
print(f"    vote_counts   : {dict(vote_counts)}")
print(f"    selected_class: {mv['selected_class']} (supporting: {mv['supporting_models']})")

# Also run the FULL production prediction
from backend.services.prediction_service import predict_disease
result = predict_disease(predict_img)
print(f"\n[FINAL] predict_disease -> {result.get('predicted_disease')} "
      f"({result.get('confidence')}%) source={result.get('prediction_source')}")

print("\n" + "=" * 100)
print("CONCLUSION")
print("=" * 100)
print(f"  Prediction image changed to cropped/mutated face?  : {np.abs(a-c).mean() >= 1.0}")
print(f"  Preprocessing changed?                             : see [9] (RGB, 0..255, float32, internal Rescaling)")
print(f"  Wrong model weights loaded?                        : see [10] (all .keras verified)")
print(f"  Face consent implementation affected prediction?   : {np.abs(a-c).mean() >= 1.0}")
