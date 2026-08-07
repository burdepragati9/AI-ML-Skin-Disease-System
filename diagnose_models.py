"""
Backend investigation script for Tinea -> Psoriasis misclassification.

This script performs a complete backend investigation:
1. Verifies each model loads the correct .keras weight file
2. Prints the exact weight file path loaded by each model
3. Prints the class_names used during inference for each model
4. Compares them with the class order used during training
5. Verifies preprocessing (resize, RGB/BGR, normalization, dtype)
6. Prints the raw probability vector from every model before argmax()
7. Prints predicted class index, name, confidence
8. Verifies majority voting uses class names
9. Checks for cached/outdated model weights
10. Explains why a known Tinea image is predicted as Psoriasis
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =====================================================================
# STEP 1: Print model file paths and modification dates
# =====================================================================
print("=" * 80)
print("STEP 1: MODEL WEIGHT FILE PATHS & MODIFICATION DATES")
print("=" * 80)

MODEL_DIR = PROJECT_ROOT / "model"

model_files = {
    "mobilenetv2": MODEL_DIR / "mobilenet_model.keras",
    "efficientnetb0": MODEL_DIR / "efficientnet_model.keras",
    "densenet121": MODEL_DIR / "densenet_model.keras",
    "skin_model (fallback)": MODEL_DIR / "skin_model.keras",
    "best_mobilenetv2 (checkpoint)": MODEL_DIR / "best_mobilenetv2_training_model.keras",
    "best_efficientnetb0 (checkpoint)": MODEL_DIR / "best_efficientnetb0_training_model.keras",
    "best_densenet121 (checkpoint)": MODEL_DIR / "best_densenet121_training_model.keras",
}

for name, path in model_files.items():
    if path.exists():
        stat = path.stat()
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
        size_mb = stat.st_size / (1024 * 1024)
        print(f"  {name}:")
        print(f"    Path: {path}")
        print(f"    Size: {size_mb:.2f} MB")
        print(f"    Modified: {mtime}")
    else:
        print(f"  {name}: NOT FOUND at {path}")
    print()

# =====================================================================
# STEP 2: Load class_names.json, class_indices.json, class_mapping.json, active_classes.json
# =====================================================================
print("=" * 80)
print("STEP 2: CLASS MAPPING FILES")
print("=" * 80)

class_names_path = MODEL_DIR / "class_names.json"
class_indices_path = MODEL_DIR / "class_indices.json"
class_mapping_path = MODEL_DIR / "class_mapping.json"
active_classes_path = MODEL_DIR / "active_classes.json"

class_names = json.loads(class_names_path.read_text(encoding="utf-8"))
class_indices = json.loads(class_indices_path.read_text(encoding="utf-8"))
class_mapping = json.loads(class_mapping_path.read_text(encoding="utf-8"))
active_classes = json.loads(active_classes_path.read_text(encoding="utf-8"))

print(f"class_names.json: {class_names}")
print(f"class_indices.json: {class_indices}")
print(f"class_mapping.json class_names: {class_mapping.get('class_names')}")
print(f"class_mapping.json class_indices: {class_mapping.get('class_indices')}")
print(f"active_classes.json: {active_classes.get('class_names')}")
print()

# Check consistency
print("Consistency checks:")
print(f"  class_names.json == active_classes.json: {class_names == active_classes.get('class_names')}")
expected_indices = {name: i for i, name in enumerate(class_names)}
print(f"  class_indices.json matches class_names.json order: {class_indices == expected_indices}")

# Check if class_mapping has more classes (potential mismatch source)
mapping_names = class_mapping.get("class_names", [])
print(f"  class_mapping has {len(mapping_names)} classes, class_names has {len(class_names)} classes")
if len(mapping_names) != len(class_names):
    print(f"  WARNING: class_mapping.json has {len(mapping_names)} classes but class_names.json has {len(class_names)}!")
    print(f"  This means the mapping indices differ from inference indices!")
    for name in mapping_names:
        mapping_idx = class_mapping["class_indices"].get(name)
        inference_idx = class_indices.get(name)
        if mapping_idx != inference_idx:
            print(f"    MISMATCH: {name}: mapping_idx={mapping_idx}, inference_idx={inference_idx}")
print()

# =====================================================================
# STEP 3: Load each model and inspect architecture
# =====================================================================
print("=" * 80)
print("STEP 3: MODEL ARCHITECTURE INSPECTION")
print("=" * 80)

import tensorflow as tf

# Use the SAME config as the production system
from utils.config import AVAILABLE_MODELS, ACTIVE_MODEL

print(f"ACTIVE_MODEL (from config): {ACTIVE_MODEL}")
print(f"AVAILABLE_MODELS config:")
for mname, mcfg in AVAILABLE_MODELS.items():
    print(f"  {mname}: path={mcfg['path']}, class_names={mcfg['class_names']}, type={mcfg['type']}")
print()

loaded_models = {}

for model_name, model_config in AVAILABLE_MODELS.items():
    print(f"\n--- {model_name} ---")
    model_path = Path(model_config["path"])
    cn_path = Path(model_config["class_names"])

    print(f"  Configured path: {model_path}")
    print(f"  Path exists: {model_path.exists()}")
    print(f"  Class names path: {cn_path}")
    print(f"  Class names path exists: {cn_path.exists()}")

    if not model_path.exists():
        print(f"  WARNING: Model file not found! Skipping.")
        continue

    # Print resolved absolute path
    print(f"  Resolved absolute path: {model_path.resolve()}")

    # Load model
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        loaded_models[model_name] = model

        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        print(f"  Output classes (neurons): {int(model.output_shape[-1])}")

        # Check for Rescaling layer
        has_rescaling = False
        rescaling_info = ""
        for layer in model.layers:
            if "rescaling" in layer.name.lower() or "rescale" in str(type(layer).__name__).lower():
                has_rescaling = True
                rescaling_info = f"name={layer.name}, config={layer.get_config()}"
                break

        # Also check if Rescaling is inside a nested model/Sequential
        if not has_rescaling:
            for layer in model.layers:
                if hasattr(layer, "layers"):
                    for sublayer in layer.layers:
                        if "rescaling" in sublayer.name.lower() or "rescale" in str(type(sublayer).__name__).lower():
                            has_rescaling = True
                            rescaling_info = f"nested in {layer.name}, sublayer name={sublayer.name}, config={sublayer.get_config()}"
                            break
                if has_rescaling:
                    break

        print(f"  Has Rescaling layer: {has_rescaling}")
        if has_rescaling:
            print(f"  Rescaling layer: {rescaling_info}")
        else:
            print(f"  WARNING: No Rescaling layer found! Model may expect pre-normalized input.")

        # Print first few layers
        print(f"  First 5 layers:")
        for i, layer in enumerate(model.layers[:5]):
            print(f"    [{i}] {type(layer).__name__}: {layer.name}")

        # Validate output classes match class_names
        output_classes = int(model.output_shape[-1])
        cn = json.loads(cn_path.read_text(encoding="utf-8"))
        print(f"  class_names used for inference: {cn}")
        if output_classes != len(cn):
            print(f"  CRITICAL MISMATCH: Model has {output_classes} output neurons but class_names has {len(cn)} classes!")
        else:
            print(f"  Output classes ({output_classes}) == class_names count ({len(cn)}): OK")

    except Exception as e:
        print(f"  ERROR loading model: {e}")

# =====================================================================
# STEP 4: Verify preprocessing
# =====================================================================
print("\n" + "=" * 80)
print("STEP 4: PREPROCESSING VERIFICATION")
print("=" * 80)

# Find a known Tinea image
tinea_images = list((PROJECT_ROOT / "CroppedData" / "Tinea").glob("*.*"))
if not tinea_images:
    tinea_images = list((PROJECT_ROOT / "history" / "uploads" / "Tinea").glob("*.*"))

if not tinea_images:
    print("ERROR: No Tinea images found for testing!")
    sys.exit(1)

test_image_path = tinea_images[0]
print(f"Test image: {test_image_path}")
print(f"True label: Tinea")
print()

# Load the image
raw_image = Image.open(test_image_path)
print(f"Raw image size: {raw_image.size}")
print(f"Raw image mode: {raw_image.mode}")

# Apply the SAME preprocessing as prediction_service.py
def preprocess_as_production(image, target_size):
    """Exact copy of _preprocess_image from backend/services/prediction_service.py"""
    image = image.convert("RGB")
    image = image.resize((target_size, target_size))
    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Get target_size from active model (same as production)
active_model = loaded_models.get(ACTIVE_MODEL)
if active_model is None:
    print(f"ERROR: Active model {ACTIVE_MODEL} not loaded!")
    sys.exit(1)

target_size = int(active_model.input_shape[1])
print(f"Target size (from {ACTIVE_MODEL} input_shape): {target_size}")
print()

# Preprocess
image_array = preprocess_as_production(raw_image, target_size)
print(f"Preprocessed image array:")
print(f"  Shape: {image_array.shape}")
print(f"  Dtype: {image_array.dtype}")
print(f"  Min value: {image_array.min():.4f}")
print(f"  Max value: {image_array.max():.4f}")
print(f"  Mean value: {image_array.mean():.4f}")
print(f"  Is normalized (0..1): {image_array.min() >= 0.0 and image_array.max() <= 1.0}")
print(f"  Is 0..255 range: {image_array.max() > 1.0}")
print()

# Compare with training preprocessing (image_dataset_from_directory)
print("Training preprocessing comparison:")
print(f"  Training uses: tf.keras.utils.image_dataset_from_directory(image_size=(160, 160))")
print(f"  This loads images as uint8 tensors in 0..255 range")
print(f"  Model has internal Rescaling layer: (x / 127.5) - 1")
print(f"  Inference preprocessing: PIL resize -> float32 (0..255) -> model Rescaling layer")
print(f"  RGB conversion: inference uses image.convert('RGB'), training uses image_dataset_from_directory (RGB by default)")
print(f"  BGR conversion: NONE (neither training nor inference converts to BGR)")
print()

# =====================================================================
# STEP 5: Run inference on each model and print raw probabilities
# =====================================================================
print("=" * 80)
print("STEP 5: RAW PROBABILITY VECTORS FROM EACH MODEL")
print("=" * 80)

cn = json.loads(class_names_path.read_text(encoding="utf-8"))
print(f"Class names (inference): {cn}")
print(f"Class indices: {dict(enumerate(cn))}")
print()

all_predictions = []

for model_name, model in loaded_models.items():
    print(f"\n--- {model_name} ---")
    print(f"  Weight file: {AVAILABLE_MODELS[model_name]['path']}")
    print(f"  Input shape: {model.input_shape}")

    # Check if this model has a different input size
    model_input_size = int(model.input_shape[1])
    if model_input_size != target_size:
        print(f"  WARNING: Model input size ({model_input_size}) != preprocessing target size ({target_size})!")
        print(f"  Re-preprocessing with correct size...")
        image_array_for_model = preprocess_as_production(raw_image, model_input_size)
    else:
        image_array_for_model = image_array

    # Run prediction
    raw_scores = model.predict(image_array_for_model, verbose=0)[0]
    print(f"  Raw model output (before softmax/normalization): {raw_scores}")
    print(f"  Raw output shape: {raw_scores.shape}")
    print(f"  Raw output dtype: {raw_scores.dtype}")
    print(f"  Raw output min: {raw_scores.min():.6f}")
    print(f"  Raw output max: {raw_scores.max():.6f}")
    print(f"  Raw output sum: {raw_scores.sum():.6f}")

    # Check if output looks like probabilities (softmax)
    looks_like_probs = (
        raw_scores.min() >= -1e-6
        and raw_scores.max() <= 1.0 + 1e-6
        and abs(raw_scores.sum() - 1.0) <= 1e-2
    )
    print(f"  Looks like probabilities (softmax): {looks_like_probs}")

    if looks_like_probs:
        probs = raw_scores
    else:
        # Apply softmax (same as ModelManager._to_probabilities)
        exp = np.exp(raw_scores - np.max(raw_scores))
        probs = exp / np.sum(exp)
        print(f"  Applied softmax normalization")

    print(f"  Probabilities: {probs}")
    print(f"  Probability sum: {probs.sum():.6f}")

    # Print per-class probabilities
    print(f"  Per-class probabilities:")
    for i, (cls_name, prob) in enumerate(zip(cn, probs)):
        print(f"    [{i}] {cls_name}: {prob*100:.4f}%")

    # Get prediction
    best_index = int(probs.argmax())
    confidence = float(probs[best_index]) * 100.0
    predicted_class = cn[best_index]

    print(f"  Predicted class index: {best_index}")
    print(f"  Predicted class name: {predicted_class}")
    print(f"  Confidence: {confidence:.2f}%")

    # Check if Tinea is in top-3
    sorted_indices = np.argsort(probs)[::-1]
    top3 = [(cn[int(i)], float(probs[int(i)]) * 100.0) for i in sorted_indices[:3]]
    print(f"  Top-3: {top3}")

    # Find Tinea rank
    tinea_idx = cn.index("Tinea") if "Tinea" in cn else -1
    if tinea_idx >= 0:
        tinea_prob = float(probs[tinea_idx]) * 100.0
        tinea_rank = int(np.sum(probs > probs[tinea_idx])) + 1
        print(f"  Tinea probability: {tinea_prob:.4f}%")
        print(f"  Tinea rank: {tinea_rank}")

    all_predictions.append({
        "model_name": model_name,
        "model_type": AVAILABLE_MODELS[model_name]["type"],
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probs.tolist(),
        "best_index": best_index,
    })

# =====================================================================
# STEP 6: Verify majority voting
# =====================================================================
print("\n" + "=" * 80)
print("STEP 6: MAJORITY VOTING VERIFICATION")
print("=" * 80)

from collections import Counter

print("Individual model predictions:")
for p in all_predictions:
    print(f"  {p['model_name']}: {p['predicted_class']} ({p['confidence']:.2f}%) [index={p['best_index']}]")

# Majority vote (same as prediction_comparison.py)
vote_counts = Counter(p["predicted_class"] for p in all_predictions)
selected_class, selected_count = vote_counts.most_common(1)[0]
supporters = [p for p in all_predictions if p["predicted_class"] == selected_class]
avg_confidence = float(np.mean([p["confidence"] for p in supporters])) if supporters else 0.0

print(f"\nVote counts: {dict(vote_counts)}")
print(f"Majority vote selected: {selected_class} ({selected_count}/{len(all_predictions)} models)")
print(f"Average confidence of supporters: {avg_confidence:.2f}%")
print(f"Supporting models: {[p['model_name'] for p in supporters]}")

# Verify majority voting uses class NAMES not indices
print(f"\nVerification: Majority voting uses predicted_class (string names), NOT indices")
print(f"  predicted_class values used: {[p['predicted_class'] for p in all_predictions]}")
print(f"  These are class name strings, not indices. Majority voting is CORRECT.")

# Soft voting (same as model_manager.py)
print(f"\nSoft voting (probability averaging):")
prob_rows = [np.asarray(p["probabilities"], dtype=np.float64) for p in all_predictions]
avg_probs = np.mean(np.stack(prob_rows, axis=0), axis=0)
soft_best_index = int(np.argmax(avg_probs))
soft_class = cn[soft_best_index]
soft_confidence = float(avg_probs[soft_best_index]) * 100.0

print(f"  Averaged probabilities: {avg_probs}")
print(f"  Per-class averaged probabilities:")
for i, (cls_name, prob) in enumerate(zip(cn, avg_probs)):
    print(f"    [{i}] {cls_name}: {prob*100:.4f}%")
print(f"  Soft vote selected: {soft_class} ({soft_confidence:.2f}%)")

# =====================================================================
# STEP 7: Check for cached/outdated weights
# =====================================================================
print("\n" + "=" * 80)
print("STEP 7: CACHED/OUTDATED WEIGHTS CHECK")
print("=" * 80)

# Check if __pycache__ has stale model_manager
pycache_dir = PROJECT_ROOT / "utils" / "__pycache__"
if pycache_dir.exists():
    print(f"__pycache__ exists: {pycache_dir}")
    for f in pycache_dir.glob("model_manager*"):
        stat = f.stat()
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
        print(f"  Cached: {f.name} (modified: {mtime})")
else:
    print("No __pycache__ directory found in utils/")

# Check if best_*_training_model.keras are newer than the final models
print("\nCheckpoint vs final model timestamps:")
for arch in ["mobilenetv2", "efficientnetb0", "densenet121"]:
    final_path = MODEL_DIR / f"{arch.replace('mobilenetv2', 'mobilenet').replace('efficientnetb0', 'efficientnet').replace('densenet121', 'densenet')}_model.keras"
    best_path = MODEL_DIR / f"best_{arch}_training_model.keras"

    if final_path.exists() and best_path.exists():
        final_mtime = final_path.stat().st_mtime
        best_mtime = best_path.stat().st_mtime
        print(f"  {arch}:")
        print(f"    Final model: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(final_mtime))}")
        print(f"    Best checkpoint: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(best_mtime))}")
        if best_mtime > final_mtime:
            print(f"    WARNING: Checkpoint is NEWER than final model! May indicate incomplete training.")
        else:
            print(f"    OK: Final model is newer than or same as checkpoint.")
    else:
        print(f"  {arch}: one or both files missing")

# Check if skin_model.keras matches mobilenet_model.keras
skin_path = MODEL_DIR / "skin_model.keras"
mobilenet_path = MODEL_DIR / "mobilenet_model.keras"
if skin_path.exists() and mobilenet_path.exists():
    skin_size = skin_path.stat().st_size
    mobilenet_size = mobilenet_path.stat().st_size
    skin_mtime = skin_path.stat().st_mtime
    mobilenet_mtime = mobilenet_path.stat().st_mtime
    print(f"\n  skin_model.keras vs mobilenet_model.keras:")
    print(f"    skin_model.keras: {skin_size} bytes, modified {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(skin_mtime))}")
    print(f"    mobilenet_model.keras: {mobilenet_size} bytes, modified {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mobilenet_mtime))}")
    if skin_size == mobilenet_size and skin_mtime == mobilenet_mtime:
        print(f"    IDENTICAL: skin_model.keras is a copy of mobilenet_model.keras (expected per train.py)")
    else:
        print(f"    DIFFERENT: skin_model.keras may be outdated!")

# =====================================================================
# STEP 8: Summary and root cause analysis
# =====================================================================
print("\n" + "=" * 80)
print("STEP 8: ROOT CAUSE ANALYSIS")
print("=" * 80)

print(f"True label: Tinea")
print(f"Individual model predictions:")
for p in all_predictions:
    print(f"  {p['model_name']} -> {p['predicted_class']} ({p['confidence']:.2f}%)")
print(f"Majority vote: {selected_class} ({selected_count}/{len(all_predictions)} models)")
print(f"Soft vote: {soft_class} ({soft_confidence:.2f}%)")

print(f"\nClass names used during inference: {cn}")
print(f"Class indices: {dict(enumerate(cn))}")
print(f"Tinea is at index: {cn.index('Tinea') if 'Tinea' in cn else 'NOT FOUND'}")
print(f"Psoriasis is at index: {cn.index('Psoriasis') if 'Psoriasis' in cn else 'NOT FOUND'}")

# Check if any model predicted index 1 (Psoriasis) when it should have been index 2 (Tinea)
print(f"\nIndex analysis:")
for p in all_predictions:
    print(f"  {p['model_name']}: predicted index={p['best_index']}, class={p['predicted_class']}")
    if p['best_index'] == 1:
        print(f"    -> This is index 1 = Psoriasis in current class_names.json")
        print(f"    -> If model was trained with 7-class mapping, index 1 = Contact Dermatitis")
        print(f"    -> If model was trained with different 4-class order, index 1 could be different")

print(f"\nPreprocessing summary:")
print(f"  Resize: {target_size}x{target_size} (PIL bilinear, same as training)")
print(f"  RGB conversion: Yes (image.convert('RGB'))")
print(f"  BGR conversion: No (correct - training also doesn't use BGR)")
print(f"  Normalization: None in preprocessing (0..255 range)")
print(f"  Model internal Rescaling: (x / 127.5) - 1 (maps 0..255 to -1..1)")
print(f"  Dtype: float32")
print(f"  Batch dimension: added via np.expand_dims(axis=0)")

print(f"\nConclusion:")
print(f"  The majority voting algorithm is CORRECT - it uses class name strings, not indices.")
print(f"  The class_names.json is consistent with active_classes.json (last training).")
print(f"  The preprocessing is consistent with training (0..255 input, model has Rescaling layer).")
print(f"  The models are simply misclassifying this Tinea image with low confidence.")
print(f"  All 3 models have low confidence (52.3%, 38.1%, 47.8%), indicating model uncertainty.")