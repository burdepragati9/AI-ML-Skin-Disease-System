"""deep_audit_129fa90406.py - Full detailed audit of the exact failing image."""
import os, sys, json, time
from pathlib import Path
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["GEMINI_API_KEY"]=""; os.environ["GOOGLE_API_KEY"]=""; os.environ["API_KEY"]=""
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from utils.config import AVAILABLE_MODELS, ACTIVE_MODEL, LOW_CONFIDENCE_THRESHOLD, CLASS_NAMES_PATH
from utils.model_manager import get_model_manager
from utils.prediction_comparison import compare_predictions, majority_vote, SUPPORTED_CLASSES
from backend.services.prediction_service import _preprocess_image, predict_disease

IMG = ROOT / "history/uploads/Acne/search_Acne_129fa90406.jpg"
print("IMAGE:", IMG, "exists=", IMG.exists())
print("="*90)

# STEP 1 models
manager = get_model_manager()
print("STEP1 LOADED MODELS")
for name in manager.list_available_models():
    info = manager.get_model_info(name)
    p = Path(AVAILABLE_MODELS[name]["path"])
    print("  %-14s file=%s mtime=%s in=%s out=%s" % (name, p.name, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)), info['input_shape'], info['output_shape']))

print("="*90)
print("STEP2 CLASS MAPPING")
cn = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
ci = json.loads((ROOT/"model/class_indices.json").read_text(encoding="utf-8"))
print("  class_names:", cn)
print("  class_indices:", ci)
print("  index_to_class:", {int(v):k for k,v in ci.items()})
print("  all models class order:", {n: manager.get_class_names(n) for n in manager.list_available_models()})

print("="*90)
raw = Image.open(IMG)
print("STEP3 PREPROCESSING")
print("  original size:", raw.size, "mode:", raw.mode)
target = int(manager.get_active_model().input_shape[1])
arr = _preprocess_image(raw, target)
print("  resized shape:", arr.shape, "dtype:", arr.dtype)
print("  pixel range before norm: [%.2f, %.2f]" % (raw.convert("RGB").getextrema()[0], raw.convert("RGB").getextrema()[1]) if False else "  (0..255, no external normalization; model has internal Rescaling)")
print("  pixel range after preprocess: [%.2f, %.2f]" % (arr.min(), arr.max()))

print("="*90)
print("STEP4 INDIVIDUAL RAW PROBABILITIES")
mps = manager.predict_with_all_models(arr)
for p in mps:
    probs = p["probabilities"]
    best = int(np.argmax(probs))
    print("  %-14s %s (%.4f%%) | %s" % (p["model_name"], cn[best], p["confidence"], "  ".join("%s=%.2f%%"% (cn[i], v*100) for i,v in enumerate(probs))))

print("="*90)
print("STEP5 VOTING")
soft = manager.soft_vote(mps)
maj = majority_vote(mps)
comp = compare_predictions(mps)
print("  soft averaged probs:", "  ".join("%s=%.4f%%"% (cn[i], v*100) for i,v in enumerate(soft['probabilities'])))
print("  soft selected class:", soft['selected_class'], "conf:", round(soft['selected_confidence'],2))
print("  majority vote_counts:", maj['vote_counts'], "selected:", maj['selected_class'], "conf:", round(maj['selected_confidence'],2))
print("  consensus:", comp['consensus_class'], comp['consensus_count'], "/", comp['total_models'])

print("="*90)
print("STEP6 FINAL ASSIGNMENT")
result = predict_disease(raw)
print("  predicted_disease:", result.get('predicted_disease'))
print("  confidence:", result.get('confidence'))
print("  prediction_source:", result.get('prediction_source'))
print("  status:", result.get('status'))
