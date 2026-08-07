"""Reproduce the Tinea -> Psoriasis misclassification through the production pipeline.

Feeds the known-misclassified Tinea images (analytics/misclassified/Tinea_as_Psoriasis/)
through the exact production `predict_disease` path and prints per-model raw
probabilities + final selection to pinpoint the root cause.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from backend.services.prediction_service import predict_disease, _preprocess_image
from utils.model_manager import get_model_manager

MISCLASS_DIR = PROJECT_ROOT / "analytics" / "misclassified" / "true_Tinea_pred_Psoriasis"

images = sorted(MISCLASS_DIR.glob("*.jpeg")) + sorted(MISCLASS_DIR.glob("*.jpg"))
if not images:
    print("No images found in", MISCLASS_DIR)
    sys.exit(1)

manager = get_model_manager()
active_model = manager.get_active_model()
target_size = int(active_model.input_shape[1])
class_names = manager.get_active_class_names()
print(f"Active model target_size={target_size}, class_names={class_names}")
print("=" * 90)

for img_path in images:
    print(f"\nIMAGE: {img_path.name}")
    image = Image.open(img_path).convert("RGB")
    arr = _preprocess_image(image, target_size)
    print(f"  preprocessed shape={arr.shape} dtype={arr.dtype} range=[{arr.min():.1f},{arr.max():.1f}]")

    # Per-model raw probabilities
    per_model = manager.predict_with_all_models(arr)
    for p in per_model:
        probs = p["probabilities"]
        top = max(range(len(probs)), key=lambda i: probs[i])
        print(f"  [{p['model_type']}] probs={[f'{v*100:.1f}%' for v in probs]} "
              f"-> {class_names[top]} ({probs[top]*100:.1f}%)")

    # Full production pipeline final decision
    result = predict_disease(image)
    print(f"  FINAL (production predict_disease):", end=" ")
    if result.get("status") == "success":
        print(f"{result['predicted_disease']} ({result['confidence']}%) source={result['prediction_source']}")
        print(f"    majority_vote={result['majority_vote'].get('selected_class')} "
              f"soft_vote={result['soft_vote'].get('selected_class')}")
        ai = result.get("ai_verification", {})
        print(f"    ai_verification final_class={ai.get('final_class')} source={ai.get('source')}")
    else:
        print(f"ERROR: {result.get('message')}")
