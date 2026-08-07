"""scan_extended.py - Scan additional Acne image dirs for the 79.38% Vitiligo hit."""
import os, sys
from pathlib import Path
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["GEMINI_API_KEY"]=""; os.environ["GOOGLE_API_KEY"]=""; os.environ["API_KEY"]=""
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from PIL import Image
from utils.model_manager import get_model_manager
from backend.services.prediction_service import _preprocess_image, predict_disease

manager = get_model_manager()
active = manager.get_active_model()
target = int(active.input_shape[1])

dirs = [
    ROOT / "analytics/misclassified/true_Acne_pred_Vitiligo",
    ROOT / "history/uploads/Acne",
    ROOT / "analytics/misclassified/true_Acne_pred_Psoriasis",
    ROOT / "analytics/misclassified/true_Acne_pred_Tinea",
]
best = None
for d in dirs:
    if not d.exists():
        print("missing dir:", d); continue
    for img in sorted([p for p in d.glob("*.*") if p.suffix.lower() in {".jpg",".jpeg",".png"}]):
        try:
            raw = Image.open(img)
            arr = _preprocess_image(raw, target)
            mps = manager.predict_with_all_models(arr)
            soft = manager.soft_vote(mps)
            conf = round(float(soft.get("selected_confidence") or 0), 2)
            cls = soft.get("selected_class")
            result = predict_disease(raw)
            final = result.get("predicted_disease")
            src = result.get("prediction_source")
            print("%-55s soft=%s(%.2f) final=%s src=%s" % (img.name, cls, conf, final, src))
            if final == "Vitiligo" and src in ("ML","majority_voting","soft_voting"):
                if best is None or abs(conf-79.38)<abs(best[1]-79.38):
                    best=(img.name, conf)
        except Exception as e:
            print("ERR", img.name, e)
print("BEST MATCH to 79.38:", best)
