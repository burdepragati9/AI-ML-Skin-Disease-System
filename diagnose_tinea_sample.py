"""Diagnose Tinea->Psoriasis misclassification on a small sample.

For each image (default 5) from analytics/misclassified/true_Tinea_pred_Psoriasis/,
prints:
  * Ground truth (always Tinea)
  * Per-model raw probabilities + predicted class (MobileNetV2, EfficientNetB0, DenseNet121)
  * Majority vote result
  * Soft vote result
  * Final prediction returned by predict_disease()

Also classifies the origin of the misclassification:
  1. individual model
  2. majority voting
  3. soft voting
  4. AI verification
  5. other backend issue

AI verification is disabled for this run (GEMINI_API_KEY forced empty) so the
diagnosis isolates the ML ensemble behavior.
"""
import os
import sys
from pathlib import Path

# Force-disable AI verification so no network calls are made during diagnosis.
os.environ["GEMINI_API_KEY"] = ""
os.environ["GOOGLE_API_KEY"] = ""
os.environ["API_KEY"] = ""

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from backend.services.prediction_service import predict_disease, _preprocess_image
from utils.model_manager import get_model_manager

MISCLASS_DIR = PROJECT_ROOT / "analytics" / "misclassified" / "true_Tinea_pred_Psoriasis"
GROUND_TRUTH = "Tinea"
SAMPLE_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def main() -> None:
    images = sorted(MISCLASS_DIR.glob("*.jpeg")) + sorted(MISCLASS_DIR.glob("*.jpg")) + \
             sorted(MISCLASS_DIR.glob("*.png"))
    if not images:
        print("No images found in", MISCLASS_DIR)
        sys.exit(1)

    sample = images[:SAMPLE_SIZE]

    manager = get_model_manager()
    active_model = manager.get_active_model()
    target_size = int(active_model.input_shape[1])
    class_names = manager.get_active_class_names()
    print(f"ACTIVE_MODEL={manager.get_active_model_name()} target_size={target_size}")
    print(f"class_names={class_names}")
    print(f"Sample size={len(sample)} (ground truth = {GROUND_TRUTH})")
    print("=" * 100)

    # Aggregate counters
    from collections import Counter
    per_model_votes = {m: Counter() for m in ["mobilenetv2", "efficientnetb0", "densenet121"]}
    majority_votes = Counter()
    soft_votes = Counter()
    final_votes = Counter()

    for img_path in sample:
        print(f"\nIMAGE: {img_path.name}  (true={GROUND_TRUTH})")
        image = Image.open(img_path).convert("RGB")
        arr = _preprocess_image(image, target_size)

        per_model = manager.predict_with_all_models(arr)
        model_preds = {}
        for p in per_model:
            probs = p["probabilities"]
            top = max(range(len(probs)), key=lambda i: probs[i])
            model_preds[p["model_name"]] = (class_names[top], p["confidence"])
            row = "  ".join(f"{class_names[i]}={v*100:.1f}%" for i, v in enumerate(probs))
            print(f"  [{p['model_type']}] pred={class_names[top]} ({p['confidence']:.2f}%) | {row}")
            per_model_votes[p["model_name"]][class_names[top]] += 1

        # Deterministic majority / soft vote (duplicate the production logic)
        vote_counts = Counter(pred for pred, _ in model_preds.values())
        majority_class = vote_counts.most_common(1)[0][0]
        majority_votes[majority_class] += 1
        print(f"  MAJORITY -> {majority_class} (votes={dict(vote_counts)})")

        # Soft vote: average probabilities across models
        prob_rows = []
        for p in per_model:
            prob_rows.append(p["probabilities"])
        import numpy as np
        avg = np.mean(np.stack(prob_rows, axis=0), axis=0)
        soft_top = class_names[int(np.argmax(avg))]
        soft_votes[soft_top] += 1
        print(f"  SOFT VOTE -> {soft_top} (avg={[f'{v*100:.1f}%' for v in avg]})")

        # Final production decision
        result = predict_disease(image)
        if result.get("status") == "success":
            final = result["predicted_disease"]
            final_votes[final] += 1
            mv = result["majority_vote"].get("selected_class")
            sv = result["soft_vote"].get("selected_class")
            ai = result.get("ai_verification", {})
            print(f"  FINAL predict_disease -> {final} ({result['confidence']}%) "
                  f"source={result['prediction_source']}")
            print(f"    majority_vote={mv} soft_vote={sv} "
                  f"ai_verification.final_class={ai.get('final_class')} "
                  f"ai_verification.source={ai.get('source')}")

            # Attribute the misclassification
            if final != GROUND_TRUTH:
                if final == majority_class:
                    origin = "MAJORITY VOTING"
                elif final == soft_top:
                    origin = "SOFT VOTING"
                elif final == ai.get("final_class"):
                    origin = "AI VERIFICATION"
                else:
                    origin = "OTHER BACKEND ISSUE"
                print(f"  >>> MISCLASSIFIED as {final}  [origin: {origin}]")
        else:
            print(f"  FINAL predict_disease -> ERROR: {result.get('message')}")

    print("\n" + "=" * 100)
    print("AGGREGATE CONFUSION (sample):")
    print("  Per-model predictions:")
    for m, c in per_model_votes.items():
        print(f"    {m}: {dict(c)}")
    print(f"  Majority vote: {dict(majority_votes)}")
    print(f"  Soft vote: {dict(soft_votes)}")
    print(f"  Final predict_disease: {dict(final_votes)}")


if __name__ == "__main__":
    main()
