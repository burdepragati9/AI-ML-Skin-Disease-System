import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR / "skin_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"


def _model_img_size(model: tf.keras.Model, default: int = 160) -> int:
    """Infer image size from the model input shape; fallback to default."""
    try:
        ishape = model.input_shape
        if isinstance(ishape, tuple) and len(ishape) >= 3:
            h = ishape[1]
            if h:
                return int(h)
    except Exception:
        pass
    return default



def first_dataset_image() -> Path | None:
    for data_dir in (PROJECT_ROOT / "CroppedData", PROJECT_ROOT / "MySkinData"):
        if not data_dir.exists():
            continue
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            match = next(data_dir.rglob(pattern), None)
            if match:
                return match
    return None


def load_labels() -> list[str]:
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

    class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("class_names.json must contain a non-empty list.")

    return class_names


def _tta_predict_probs(
    model: tf.keras.Model,
    image_array: np.ndarray,
    *,
    tta_count: int = 8,
    seed: int = 123,
) -> np.ndarray:
    """Test-time augmentation (TTA).

    NOTE: The model contains its own Rescaling layer, so image_array stays in 0..255.
    Returns averaged softmax probabilities (shape: [num_classes]).
    """
    rng = np.random.default_rng(seed)

    # Convert to tf tensor once.
    x = tf.convert_to_tensor(image_array)

    def _augment_one(i: int) -> tf.Tensor:
        # Small, medically-safe augmentations: flips (horizontal) and tiny rotations/zooms.
        # Keep them mild to avoid changing lesion semantics.
        # Horizontal flip is used with 50% probability.
        x1 = x
        if rng.random() < 0.5:
            x1 = tf.image.flip_left_right(x1)

        # Tiny rotation via keras preprocessing layers (graph-safe).
        # We avoid heavy distortions; use +/- 3 degrees.
        angle = float(rng.uniform(-0.05, 0.05))
        x1 = tf.keras.preprocessing.image.apply_affine_transform(
            x1[0],
            theta=angle,
            fill_mode="nearest",
        )
        x1 = tf.expand_dims(x1, axis=0)

        # Mild zoom by scaling then resizing back.
        zoom = float(rng.uniform(0.95, 1.05))
        new_h = int(x1.shape[1] * zoom)
        new_w = int(x1.shape[2] * zoom)
        new_h = max(1, new_h)
        new_w = max(1, new_w)
        x1 = tf.image.resize(x1, (new_h, new_w), method="bilinear")
        x1 = tf.image.resize_with_crop_or_pad(x1, x.shape[1], x.shape[2])

        return tf.cast(x1, tf.float32)

    probs_sum = None
    for i in range(tta_count):
        x_aug = x if i == 0 else _augment_one(i)
        probs_i = model.predict(x_aug, verbose=0)[0]
        probs_i = np.asarray(probs_i, dtype=np.float64)
        if probs_sum is None:
            probs_sum = probs_i
        else:
            probs_sum += probs_i

    probs_avg = probs_sum / float(tta_count)

    # Numerical safety: normalize to sum=1
    probs_avg = np.clip(probs_avg, 0.0, 1.0)
    s = float(np.sum(probs_avg))
    if s <= 0:
        # Fallback: return raw softmax-ish distribution from last aug
        return probs_i
    probs_avg = probs_avg / s

    return probs_avg


def predict_image(image_path: Path) -> tuple[str, float, list[tuple[str, float]]]:
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    class_names = load_labels()
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    output_classes = int(model.output_shape[-1])
    if output_classes != len(class_names):
        raise ValueError(
            "Model output classes do not match class_names.json. "
            f"Model has {output_classes}, labels has {len(class_names)}."
        )

    img_size = _model_img_size(model, default=160)

    # The model contains its own Rescaling layer, so inference inputs stay 0..255.
    image = tf.keras.utils.load_img(image_path, target_size=(img_size, img_size))
    image_array = tf.keras.utils.img_to_array(image).astype(np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    # TTA inference
    probs = _tta_predict_probs(model, image_array, tta_count=8, seed=123)

    best_index = int(np.argmax(probs))
    best_confidence = float(probs[best_index]) * 100.0

    # Required hard confidence threshold:
    # If prediction confidence < 60%, return "Low confidence prediction".
    if best_confidence < 60.0:
        predictions = [
            (class_names[index], float(probs[index]) * 100)
            for index in np.argsort(probs)[::-1]
        ]
        return "Low confidence prediction", best_confidence, predictions

    predictions = [
        (class_names[index], float(probs[index]) * 100)
        for index in np.argsort(probs)[::-1]
    ]
    return class_names[best_index], best_confidence, predictions




def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one skin image.")
    parser.add_argument("image", nargs="?", type=Path, default=None)
    args = parser.parse_args()

    image_path = args.image or first_dataset_image()
    if image_path is None:
        raise SystemExit("No image path provided and no dataset image was found.")

    predicted_class, confidence, predictions = predict_image(image_path.resolve())
    print(f"Image: {image_path.resolve()}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print("All predictions:")
    for class_name, score in predictions:
        print(f"  {class_name}: {score:.2f}%")


if __name__ == "__main__":
    main()
