"""scan_acne_vitiligo.py - Fast summary scan of all Acne images.
Only records final disease + soft confidence per image to a file, and prints
any image whose ML final prediction is Vitiligo. READ-ONLY.
"""
import os, sys, json, time
from pathlib import Path
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["GEMINI_API_KEY"]=""; os.environ["GOOGLE_API_KEY"]=""; os.environ["API_KEY"]=""
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from utils.model_manager import get_model_manager
from utils.prediction_comparison import majority_vote
from backend.services.prediction_service import _preprocess_image, predict_disease

ACNE_DIR = ROOT / "CroppedData" / "Acne"
manager = get_model_manager()
active = manager.get_active_model()
target = int(active.input_shape[1])
cn = manager.get_active_class_names()

images = sorted([p for p in ACNE_DIR.glob("*.*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])
print("Total Acne images:", len(images))
out = []
vitiligo_hits = []
for img in images:
    try:
        raw = Image.open(img)
        arr = _preprocess_image(raw, target)
        mps = manager.predict_with_all_models(arr)
        soft = manager.soft_vote(mps)
        result = predict_disease(raw)
        conf = round(float(soft.get("selected_confidence") or 0), 2)
        final = result.get("predicted_disease")
        src = result.get("prediction_source")
        out.append((img.name, final, conf, src))
        if src in ("ML","majority_voting","soft_voting") and final == "Vitiligo":
            vitiligo_hits.append((img.name, conf))
            print("HIT Vitiligo:", img.name, "conf=", conf)
    except Exception as e:
        print("ERR", img.name, e)

# write summary file
with open(ROOT / "acne_summary.tsv", "w", encoding="utf-8") as f:
    for name, final, conf, src in out:
        f.write("%s\t%s\t%s\t%s\n" % (name, final, conf, src))

print("DONE. Total=%d, Vitiligo_ML_hits=%d" % (len(out), len(vitiligo_hits)))
for name, conf in vitiligo_hits:
    print("  ", name, conf)
