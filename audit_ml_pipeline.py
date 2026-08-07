"""audit_ml_pipeline.py

Full ML prediction pipeline audit. Sweeps every image in CroppedData/Acne/ and,
for each, reproduces the EXACT production flow used by
backend/services/prediction_service.predict_disease. It stops when it finds the
image producing the incorrect prediction (Vitiligo with ~79.38% soft-vote
confidence) and then investigates that image in detail.

READ-ONLY diagnostic. No production code is modified.
"""

import os
import sys
import json
import time
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Disable real Gemini calls so the audit is deterministic and offline.
os.environ["GEMINI_API_KEY"] = ""
os.environ["GOOGLE_API_KEY"] = ""
os.environ["API_KEY"] = ""

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from utils.config import (
    AVAILABLE_MODELS,
    ACTIVE_MODEL,
    LOW_CONFIDENCE_THRESHOLD,
    CLASS_NAMES_PATH,
)
from utils.model_manager import get_model_manager
from utils.prediction_comparison import (
    compare_predictions,
    majority_vote,
    SUPPORTED_CLASSES,
)
from backend.services.prediction_service import _preprocess_image, predict_disease

ACNE_DIR = ROOT / "CroppedData" / "Acne"
TARGET_CONFIDENCE = 79.38


def fmt_mtime(path):
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))
    except Exception:
        return "N/A"


def print_header(msg):
    print("\n" + "=" * 90)
    print(msg)
    print("=" * 90)


def report_loaded_models(manager):
    print_header("STEP 1 - VERIFY LOADED MODELS")
    for name in manager.list_available_models():
        info = manager.get_model_info(name)
        cfg_path = Path(AVAILABLE_MODELS[name]["path"])
        print("  Model name      : " + name)
        print("    type          : " + info["type"])
        print("    keras file    : " + str(cfg_path))
        print("    exists        : " + str(cfg_path.exists()))
        print("    size          : " + str(cfg_path.stat().st_size if cfg_path.exists() else 0) + " bytes")
        print("    last modified : " + fmt_mtime(cfg_path))
        print("    input_shape   : " + str(info["input_shape"]))
        print("    output_shape  : " + str(info["output_shape"]))
        print("    num_classes   : " + str(info["num_classes"]))
        print("    is_active     : " + str(info["is_active"]))
        print()


def report_class_mapping(manager):
    print_header("STEP 2 - VERIFY CLASS MAPPING")
    cn = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    ci = json.loads((ROOT / "model" / "class_indices.json").read_text(encoding="utf-8"))
    index_to_class = {int(v): k for k, v in ci.items()}
    print("  class_names.json  : " + str(cn))
    print("  class_indices.json: " + str(ci))
    print("  index_to_class    : " + str(index_to_class))
    print("  SUPPORTED_CLASSES : " + str(SUPPORTED_CLASSES))
    print()

    cm_path = ROOT / "model" / "class_mapping.json"
    if cm_path.exists():
        cm = json.loads(cm_path.read_text(encoding="utf-8"))
        print("  [LEGACY] class_mapping.json has " + str(len(cm.get("class_names", []))) + " classes: " + str(cm.get("class_names")))
        print("  [LEGACY] class_indices: " + str(cm.get("class_indices")))
        print("  NOTE: class_mapping.json is NOT used by ModelManager inference.")
    else:
        print("  [LEGACY] class_mapping.json: NOT FOUND")
    print()

    print("  Per-model class ordering verification:")
    orderings = {}
    for name in manager.list_available_models():
        orderings[name] = manager.get_class_names(name)
        print("    " + name + ": " + str(orderings[name]))
    uniq = set(tuple(v) for v in orderings.values())
    if len(uniq) == 1:
        print("    ALL MODELS SHARE IDENTICAL CLASS ORDERING: " + str(uniq.pop()))
    else:
        print("    *** WARNING: DIFFERENT CLASS ORDERINGS BETWEEN MODELS ***")
    print()


def report_preprocessing(raw_image, manager):
    print("  Original image mode : " + raw_image.mode)
    print("  Original image size : " + str(raw_image.size))
    active = manager.get_active_model()
    target = int(active.input_shape[1])
    print("  Target size (active model input H) : " + str(target))
    arr = _preprocess_image(raw_image, target)
    print("  Resized shape (incl batch)         : " + str(arr.shape))
    print("  dtype                              : " + str(arr.dtype))
    print("  pixel range AFTER preprocess        : [%.2f, %.2f] (0..255, no external norm)" % (arr.min(), arr.max()))
    print("  normalization                      : none in preprocess; model has internal Rescaling layer")
    return arr


def report_image_audit(image_path):
    print_header("AUDIT IMAGE: " + image_path.name)
    print("  Full path : " + str(image_path.resolve()))
    print("  True label: Acne")
    print("  Size: " + str(image_path.stat().st_size) + " bytes, modified " + fmt_mtime(image_path))

    raw = Image.open(image_path)
    manager = get_model_manager()

    print("\n  --- [STEP 3] Preprocessing ---")
    arr = report_preprocessing(raw, manager)

    print("\n  --- [STEP 4] Individual model predictions ---")
    model_predictions = manager.predict_with_all_models(arr)
    cn = manager.get_active_class_names()
    for p in model_predictions:
        probs = p["probabilities"]
        best = int(np.argmax(probs))
        print("\n  Model: " + p["model_name"] + " (" + p["model_type"] + ")")
        print("    Predicted class : " + p["predicted_class"])
        print("    Confidence      : %.4f%%" % p["confidence"])
        parts = []
        for i, v in enumerate(probs):
            parts.append("%s=%.2f%%" % (cn[i], v * 100.0))
        print("    Probability vec : " + "  ".join(parts))

    print("\n  --- [STEP 5] Voting ---")
    soft = manager.soft_vote(model_predictions)
    majority = majority_vote(model_predictions)
    comparison = compare_predictions(model_predictions)

    print("  Soft vote averaged probabilities:")
    avg = soft["probabilities"]
    for i, v in enumerate(avg):
        print("    [%d] %s: %.4f%%" % (i, cn[i], v * 100.0))
    print("  Soft vote selected class     : " + soft["selected_class"])
    print("  Soft vote confidence         : %.4f%%" % soft["selected_confidence"])
    print("  Majority vote vote_counts    : " + str(majority["vote_counts"]))
    print("  Majority vote selected class : " + majority["selected_class"])
    print("  Majority vote confidence     : %.4f%%" % majority["selected_confidence"])
    print("  Consensus / agreement        : " + comparison["consensus_class"] + " (" + str(comparison["consensus_count"]) + "/" + str(comparison["total_models"]) + ")")

    print("\n  --- [STEP 6/7] Final assignment ---")
    result = predict_disease(raw)
    print("  predicted_disease : " + str(result.get("predicted_disease")))
    print("  confidence        : " + str(result.get("confidence")))
    print("  prediction_source : " + str(result.get("prediction_source")))
    print("  status            : " + str(result.get("status")))

    return soft, result


def main():
    print_header("ML PREDICTION PIPELINE AUDIT - Acne images")
    print("  Acne dir             : " + str(ACNE_DIR))
    print("  Exists               : " + str(ACNE_DIR.exists()))
    print("  ACTIVE_MODEL         : " + ACTIVE_MODEL)
    print("  LOW_CONFIDENCE_THRESHOLD: " + str(LOW_CONFIDENCE_THRESHOLD))
    print("  Hunting soft-vote    : Vitiligo @ ~" + str(TARGET_CONFIDENCE) + "%")

    manager = get_model_manager()

    report_loaded_models(manager)
    report_class_mapping(manager)

    images = sorted(ACNE_DIR.glob("*.*")) if ACNE_DIR.exists() else []
    images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    print("\n  Found " + str(len(images)) + " Acne images to sweep.")

    found = None
    for img in images:
        soft, result = report_image_audit(img)
        conf = float(soft.get("selected_confidence") or 0)
        final = result.get("predicted_disease")
        if result.get("prediction_source") in ("ML", "majority_voting", "soft_voting") and final == "Vitiligo":
            found = (img, soft, result)
            print("\n" + "*" * 90)
            print("*** FOUND INCORRECT PREDICTION on " + img.name + " ***")
            print("*** Soft-vote confidence = %.2f%% (target ~%s) ***" % (conf, TARGET_CONFIDENCE))
            print("*" * 90)
            if abs(conf - TARGET_CONFIDENCE) <= 1.5:
                print("*** THIS MATCHES THE PRODUCTION LOG (79.38%) - EXACT REPRO FOUND ***")
            break

    if not found:
        print("\nNo Acne image produced a Vitiligo ML prediction in this sweep.")
        print("(The production 79.38% image may live in history/uploads/Acne/ instead.)")


if __name__ == "__main__":
    main()
