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
IMG_SIZE = 224


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

    image = tf.keras.utils.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    scores = model.predict(image_array, verbose=0)[0]
    best_index = int(np.argmax(scores))
    predictions = [
        (class_names[index], float(scores[index]) * 100)
        for index in np.argsort(scores)[::-1]
    ]
    return class_names[best_index], float(scores[best_index]) * 100, predictions


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
