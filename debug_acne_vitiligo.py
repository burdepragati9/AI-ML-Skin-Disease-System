"""
Debug script to trace where Acne becomes Vitiligo in the multi-model prediction pipeline.

Prints:
- Model loading details (.keras file, class_names, input_shape)
- Preprocessing details (resize, normalization)
- Individual model predictions (predicted_class, confidence, probability vector)
- Soft vote result (selected_class, selected_confidence, averaged probabilities)
- Majority vote result (vote_counts, selected_class)
"""
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.config import (
    AVAILABLE_MODELS,
    ACTIVE_MODEL,
    MOBILENET_MODEL_PATH,
    EFFICIENTNET_MODEL_PATH,
    DENSENET_MODEL_PATH,
    MODEL_PATH,
    CLASS_NAMES_PATH,
)
from utils.model_manager import get_model_manager
from utils.prediction_comparison import majority_vote, compare_predictions

# Pick an Acne test image
ACNE_IMAGE_PATH = ROOT_DIR / "CroppedData" / "Acne" / "Acne_3a2924e047.jpg"

print("=" * 80)
print("DEBUG: Tracing Acne -> Vitiligo in Multi-Model Prediction Pipeline")
print("=" * 80)

# =========================================================================
# STEP 1: Verify model files exist
# =========================================================================
print("\n" + "=" * 80)
print("STEP 1: Verify Model Files")
print("=" * 80)

model_paths = {
    "mobilenetv2": MOBILENET_MODEL_PATH,
    "efficientnetb0": EFFICIENTNET_MODEL_PATH,
    "densenet121": DENSENET_MODEL_PATH,
    "fallback (skin_model)": MODEL_PATH,
}

for name, path in model_paths.items():
    exists = path.exists()
    size_mb = path.stat().st_size / (1024 * 1024) if exists else 0
    print(f"  {name}: path={path} exists={exists} size={size_mb:.1f}MB")

# =========================================================================
# STEP 2: Verify class_names.json
# =========================================================================
print("\n" + "=" * 80)
print("STEP 2: Verify Class Names")
print("=" * 80)

class_names_data = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
print(f"  class_names.json path: {CLASS_NAMES_PATH}")
print(f"  class_names: {class_names_data}")
print(f"  num_classes: {len(class_names_data)}")

# Also check class_indices.json
class_indices_path = ROOT_DIR / "model" / "class_indices.json"
if class_indices_path.exists():
    class_indices_data = json.loads(class_indices_path.read_text(encoding="utf-8"))
    print(f"  class_indices.json: {class_indices_data}")
else:
    print(f"  class_indices.json: NOT FOUND at {class_indices_path}")

# Check class_mapping.json
class_mapping_path = ROOT_DIR / "model" / "class_mapping.json"
if class_mapping_path.exists():
    class_mapping_data = json.loads(class_mapping_path.read_text(encoding="utf-8"))
    print(f"  class_mapping.json: {class_mapping_data}")
else:
    print(f"  class_mapping.json: NOT FOUND at {class_mapping_path}")

# =========================================================================
# STEP 3: Load models and verify
# =========================================================================
print("\n" + "=" * 80)
print("STEP 3: Load Models and Verify")
print("=" * 80)

manager = get_model_manager()

for model_name in manager.list_available_models():
    info = manager.get_model_info(model_name)
    print(f"\n  Model: {model_name}")
    print(f"    .keras file: {AVAILABLE_MODELS[model_name]['path']}")
    print(f"    type: {info['type']}")
    print(f"    input_shape: {info['input_shape']}")
    print(f"    output_shape: {info['output_shape']}")
    print(f"    num_classes: {info['num_classes']}")
    print(f"    class_names: {info['class_names']}")
    print(f"    is_active: {info['is_active']}")

# Verify all models use the same class ordering
print("\n  Class ordering verification:")
all_class_names = {}
for model_name in manager.list_available_models():
    cn = manager.get_class_names(model_name)
    all_class_names[model_name] = cn
    print(f"    {model_name}: {cn}")

# Check if all are the same
unique_orderings = set(tuple(cn) for cn in all_class_names.values())
if len(unique_orderings) == 1:
    print(f"    ALL MODELS USE SAME CLASS ORDERING: {unique_orderings.pop()}")
else:
    print(f"    WARNING: DIFFERENT CLASS ORDERINGS DETECTED!")
    for name, cn in all_class_names.items():
        print(f"      {name}: {cn}")

# =========================================================================
# STEP 4: Load and preprocess image
# =========================================================================
print("\n" + "=" * 80)
print("STEP 4: Load and Preprocess Image")
print("=" * 80)

print(f"  Image path: {ACNE_IMAGE_PATH}")
print(f"  Image exists: {ACNE_IMAGE_PATH.exists()}")

image = Image.open(ACNE_IMAGE_PATH)
print(f"  Original image mode: {image.mode}")
print(f"  Original image size: {image.size}")

# Preprocess (same as prediction_service.py)
image_rgb = image.convert("RGB")
print(f"  After convert RGB - mode: {image_rgb.mode}, size: {image_rgb.size}")

# Get target size from active model
active_model = manager.get_active_model()
target_size = int(active_model.input_shape[1])
print(f"  Target size (from active model input_shape): {target_size}")

image_resized = image_rgb.resize((target_size, target_size))
print(f"  After resize: {image_resized.size}")

image_array = np.asarray(image_resized, dtype=np.float32)
print(f"  Array shape before expand_dims: {image_array.shape}")
print(f"  Array dtype: {image_array.dtype}")
print(f"  Array min: {image_array.min():.2f}, max: {image_array.max():.2f}")

image_array = np.expand_dims(image_array, axis=0)
print(f"  Array shape after expand_dims: {image_array.shape}")

# Check if normalization is applied
print(f"  NOTE: No normalization applied (values are 0-255, not 0-1)")
print(f"  This matches prediction_service._preprocess_image which uses np.asarray(dtype=float32)")

# =========================================================================
# STEP 5: Run prediction with ALL models
# =========================================================================
print("\n" + "=" * 80)
print("STEP 5: Individual Model Predictions")
print("=" * 80)

model_predictions = manager.predict_with_all_models(image_array)

for i, pred in enumerate(model_predictions):
    print(f"\n  --- Model {i + 1}: {pred['model_name']} ---")
    print(f"    model_name: {pred['model_name']}")
    print(f"    model_type: {pred['model_type']}")
    print(f"    predicted_class: {pred['predicted_class']}")
    print(f"    confidence: {pred['confidence']:.4f}%")

    # Get class index
    class_names = manager.get_class_names(pred['model_name'])
    predicted_class_index = class_names.index(pred['predicted_class']) if pred['predicted_class'] in class_names else -1
    print(f"    predicted_class_index: {predicted_class_index}")

    # Print probability vector
    probs = pred['probabilities']
    print(f"    probability vector (all {len(probs)} values):")
    for j, (cn, p) in enumerate(zip(class_names, probs)):
        pct = float(p) * 100.0
        marker = " <== BEST" if j == predicted_class_index else ""
        print(f"      [{j}] {cn}: {pct:.4f}%{marker}")

    # Top 5
    print(f"    Top 5 predictions:")
    for j, (cn, conf) in enumerate(pred['all_predictions'][:5]):
        print(f"      {j + 1}. {cn}: {conf:.4f}%")

# =========================================================================
# STEP 6: Soft Vote
# =========================================================================
print("\n" + "=" * 80)
print("STEP 6: Soft Vote Result")
print("=" * 80)

soft = manager.soft_vote(model_predictions)
print(f"  selected_class: {soft['selected_class']}")
print(f"  selected_confidence: {soft['selected_confidence']:.4f}%")
print(f"  method: {soft['method']}")
print(f"  supporting_models: {soft['supporting_models']}")

# Print averaged probability vector
avg_probs = soft['probabilities']
class_names = manager.get_active_class_names()
print(f"  Averaged probability vector ({len(avg_probs)} values):")
for j, (cn, p) in enumerate(zip(class_names, avg_probs)):
    pct = float(p) * 100.0
    best_idx = int(np.argmax(avg_probs))
    marker = " <== BEST" if j == best_idx else ""
    print(f"    [{j}] {cn}: {pct:.4f}%{marker}")

print(f"  Top 3:")
for j, (cn, conf) in enumerate(soft['top3']):
    print(f"    {j + 1}. {cn}: {conf:.4f}%")

# =========================================================================
# STEP 7: Majority Vote
# =========================================================================
print("\n" + "=" * 80)
print("STEP 7: Majority Vote Result")
print("=" * 80)

majority = majority_vote(model_predictions)
print(f"  selected_class: {majority['selected_class']}")
print(f"  selected_confidence: {majority['selected_confidence']:.4f}%")
print(f"  vote_counts: {majority['vote_counts']}")
print(f"  supporting_models: {majority['supporting_models']}")
print(f"  method: {majority['method']}")

# =========================================================================
# STEP 8: Comparison Summary
# =========================================================================
print("\n" + "=" * 80)
print("STEP 8: Comparison Summary")
print("=" * 80)

comparison = compare_predictions(model_predictions)
print(f"  consensus_class: {comparison['consensus_class']}")
print(f"  consensus_count: {comparison['consensus_count']}")
print(f"  total_models: {comparison['total_models']}")
print(f"  prediction_agreement: {comparison['prediction_agreement']}")
print(f"  model_vote_counts: {comparison['model_vote_counts']}")
print(f"  confidence_variance: {comparison['confidence_variance']:.4f}")

# =========================================================================
# STEP 9: Identify where Acne becomes Vitiligo
# =========================================================================
print("\n" + "=" * 80)
print("STEP 9: Root Cause Analysis - Where does Acne become Vitiligo?")
print("=" * 80)

print("\n  Individual model predictions:")
for pred in model_predictions:
    print(f"    {pred['model_name']}: predicted={pred['predicted_class']} confidence={pred['confidence']:.2f}%")

print(f"\n  Soft vote: {soft['selected_class']} ({soft['selected_confidence']:.2f}%)")
print(f"  Majority vote: {majority['selected_class']} ({majority['selected_confidence']:.2f}%)")

# Check if any model predicted Acne
acne_models = [p['model_name'] for p in model_predictions if p['predicted_class'] == 'Acne']
vitiligo_models = [p['model_name'] for p in model_predictions if p['predicted_class'] == 'Vitiligo']
print(f"\n  Models that predicted Acne: {acne_models if acne_models else 'NONE'}")
print(f"  Models that predicted Vitiligo: {vitiligo_models if vitiligo_models else 'NONE'}")

if soft['selected_class'] == 'Vitiligo':
    print("\n  >>> ISSUE IDENTIFIED: Soft vote selected Vitiligo for an Acne image <<<")
    print("  >>> This means the averaged probability for Vitiligo is higher than Acne <<<")
    print("  >>> Check individual model probability vectors above to see which model(s) <<<")
    print("  >>> are assigning high probability to Vitiligo <<<")

print("\n" + "=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)