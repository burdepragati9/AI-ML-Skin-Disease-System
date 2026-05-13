import json
import logging
import threading
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from database.db import execute, fetch_all, fetch_one, get_conn, utc_now
from utils.config import (
    CLASS_NAMES_PATH,
    DATASET_PATH,
    IMAGE_DUPLICATE_HASH_DISTANCE,
    INCREMENTAL_TRAINING_EPOCHS,
    MODEL_PATH,
    TRAINING_MAX_ATTEMPTS,
)
from utils.security import image_hash, image_hash_distance, safe_disease_slug, save_optimized_image, sanitize_text


LOGGER = logging.getLogger("self_learning")
LOGGER.setLevel(logging.INFO)
_QUEUE_LOCK = threading.Lock()
_WORKER_STARTED = False
IMG_SIZE = 160


def _log(event_type: str, status: str, message: str = "", image_path: str = "", disease: str = "", before=None, after=None) -> None:
    LOGGER.info("%s %s %s", event_type, status, message)
    execute(
        """
        INSERT INTO training_logs (
            event_type, status, message, image_path, disease, accuracy_before, accuracy_after, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (event_type, status, message, image_path, disease, before, after, utc_now()),
    )


def _find_similar_duplicate(img_hash: str) -> dict | None:
    candidates = fetch_all("SELECT id, image_hash, image_path FROM image_hashes")
    for candidate in candidates:
        try:
            if image_hash_distance(img_hash, candidate["image_hash"]) <= IMAGE_DUPLICATE_HASH_DISTANCE:
                return candidate
        except Exception:
            continue
    return None


def save_ai_prediction_for_learning(
    image,
    disease: str,
    confidence: float,
    original_name: str = "",
    doctor_id: int | None = None,
) -> dict:
    """Persist only AI-fallback images and queue one-image incremental learning.

    This function is intentionally called only from the low-confidence ML
    fallback path in app.py. It stores the image in dataset/<AI disease>/ and
    queues that exact file for incremental training.
    """
    disease_name = safe_disease_slug(disease)
    upload_name = sanitize_text(original_name or "ai_fallback_upload.jpg", 180)
    img_hash = image_hash(image)
    duplicate = fetch_one("SELECT id, image_path FROM image_hashes WHERE image_hash = ?", (img_hash,))
    if not duplicate:
        duplicate = _find_similar_duplicate(img_hash)

    if duplicate:
        execute(
            """
            INSERT OR IGNORE INTO ai_predictions (
                image_name, image_path, image_hash, predicted_disease, confidence,
                source, doctor_id, fallback_status, retraining_status, duplicate_of, created_at
            )
            VALUES (?, ?, ?, ?, ?, 'AI_FALLBACK', ?, 'AI_FALLBACK', 'duplicate', ?, ?)
            """,
            (
                upload_name or Path(duplicate["image_path"]).name,
                duplicate["image_path"],
                img_hash,
                disease_name,
                confidence,
                doctor_id,
                duplicate["id"],
                utc_now(),
            ),
        )
        return {"saved": False, "duplicate": True, "path": duplicate["image_path"]}

    target_dir = DATASET_PATH / disease_name
    saved_path = save_optimized_image(image, target_dir, disease_name)
    prediction_id = execute(
        """
        INSERT INTO ai_predictions (
            image_name, image_path, image_hash, predicted_disease, confidence,
            source, doctor_id, fallback_status, retraining_status, created_at
        )
        VALUES (?, ?, ?, ?, ?, 'AI_FALLBACK', ?, 'AI_FALLBACK', 'queued', ?)
        """,
        (saved_path.name, str(saved_path), img_hash, disease_name, confidence, doctor_id, utc_now()),
    )
    execute(
        """
        INSERT INTO image_hashes (image_hash, image_path, disease, source, created_at)
        VALUES (?, ?, ?, 'AI_FALLBACK', ?)
        """,
        (img_hash, str(saved_path), disease_name, utc_now()),
    )
    execute(
        """
        INSERT INTO training_queue (image_path, disease, doctor_id, source, status, created_at, updated_at)
        VALUES (?, ?, ?, 'AI_FALLBACK', 'queued', ?, ?)
        """,
        (str(saved_path), disease_name, doctor_id, utc_now(), utc_now()),
    )
    _log(
        "image_added",
        "completed",
        f"New AI fallback image added from upload '{upload_name}'.",
        str(saved_path),
        disease_name,
    )
    start_training_worker()
    return {"saved": True, "duplicate": False, "path": str(saved_path), "prediction_id": prediction_id}


def start_training_worker() -> None:
    global _WORKER_STARTED
    if _WORKER_STARTED:
        return
    _WORKER_STARTED = True
    thread = threading.Thread(target=process_training_queue, daemon=True)
    thread.start()


def process_training_queue() -> None:
    global _WORKER_STARTED
    with _QUEUE_LOCK:
        try:
            while True:
                job = fetch_one(
                    "SELECT * FROM training_queue WHERE status = 'queued' ORDER BY created_at LIMIT 1"
                )
                if not job:
                    break
                _run_job(dict(job))
        finally:
            _WORKER_STARTED = False


def _load_class_names() -> list[str]:
    if CLASS_NAMES_PATH.exists():
        return json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    return []


def _save_class_names(class_names: list[str]) -> None:
    CLASS_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLASS_NAMES_PATH.write_text(json.dumps(class_names, indent=2), encoding="utf-8")


def _prepare_model(class_names: list[str]):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    output_classes = int(model.output_shape[-1])
    if output_classes == len(class_names):
        return model

    # Expands only the classifier head for newly discovered labels while keeping
    # learned feature layers intact; this avoids a full retrain from scratch.
    penultimate = model.layers[-2].output
    new_output = keras.layers.Dense(len(class_names), activation="softmax", name="incremental_classifier")(penultimate)
    expanded = keras.Model(model.input, new_output)
    for layer in expanded.layers[:-1]:
        layer.trainable = False
    if model.layers[-1].get_weights():
        old_weights, old_bias = model.layers[-1].get_weights()
        new_weights, new_bias = expanded.layers[-1].get_weights()
        keep = min(output_classes, len(class_names), old_weights.shape[1])
        new_weights[:, :keep] = old_weights[:, :keep]
        new_bias[:keep] = old_bias[:keep]
        expanded.layers[-1].set_weights([new_weights, new_bias])
    return expanded


def _dataset_for_job(image_path: str, label_index: int):
    def load(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        image = tf.cast(image, tf.float32)
        return image, label

    paths = tf.constant([image_path])
    labels = tf.constant([label_index], dtype=tf.int64)
    return tf.data.Dataset.from_tensor_slices((paths, labels)).map(load).batch(1)


def _run_job(job: dict) -> None:
    execute(
        "UPDATE training_queue SET status = 'processing', attempts = attempts + 1, updated_at = ? WHERE id = ?",
        (utc_now(), job["id"]),
    )
    _log("training_started", "started", "Incremental training started.", job["image_path"], job["disease"])
    try:
        class_names = _load_class_names()
        if job["disease"] not in class_names:
            class_names.append(job["disease"])
            _save_class_names(class_names)

        label_index = class_names.index(job["disease"])
        model = _prepare_model(class_names)
        for layer in model.layers[:-1]:
            layer.trainable = False
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00005),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        ds = _dataset_for_job(job["image_path"], label_index)
        before = float(model.evaluate(ds, verbose=0)[1])
        history = model.fit(ds, epochs=INCREMENTAL_TRAINING_EPOCHS, verbose=0)
        after = float(history.history.get("accuracy", [before])[-1])
        model.save(MODEL_PATH)
        execute(
            "UPDATE training_queue SET status = 'completed', trained_at = ?, updated_at = ? WHERE id = ?",
            (utc_now(), utc_now(), job["id"]),
        )
        execute(
            """
            UPDATE ai_predictions
            SET retraining_status = 'trained'
            WHERE image_path = ? AND duplicate_of IS NULL
            """,
            (job["image_path"],),
        )
        _log("training_completed", "completed", "Incremental training completed.", job["image_path"], job["disease"], before, after)
    except Exception as exc:
        next_status = "failed"
        if int(job.get("attempts", 0)) < TRAINING_MAX_ATTEMPTS:
            next_status = "queued"
        execute(
            "UPDATE training_queue SET status = ?, error = ?, updated_at = ? WHERE id = ?",
            (next_status, str(exc), utc_now(), job["id"]),
        )
        execute(
            """
            UPDATE ai_predictions
            SET retraining_status = ?
            WHERE image_path = ? AND duplicate_of IS NULL
            """,
            (next_status, job["image_path"]),
        )
        _log("training_failed", "failed", str(exc), job["image_path"], job["disease"])
