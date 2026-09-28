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
    REPLAY_SAMPLES_PER_DISEASE,
    REPLAY_MAX_TOTAL_SAMPLES,
    REPLAY_SHUFFLE_SEED,
    CROPPED_DATASET_PATH,
)


from utils.security import image_hash, image_hash_distance, safe_disease_slug, save_optimized_image, sanitize_text

from model.registry_utils import ensure_label_exists, resolve_to_canonical

# The only disease classes supported by the ML system.
# Self-learning must NOT expand this scope via AI verification results.
SUPPORTED_CLASSES = ["Acne", "Psoriasis", "Tinea", "Vitiligo"]



def _normalize_for_matching(s: str) -> str:
    import re

    return re.sub(r"\s+", " ", (s or "").strip()).lower()



def _build_existing_disease_index(dataset_path: Path) -> dict[str, str]:
    """Map normalized folder names -> canonical folder names on disk."""
    index: dict[str, str] = {}
    if not dataset_path.exists():
        return index
    for p in dataset_path.iterdir():
        if not p.is_dir():
            continue
        canonical = p.name
        index[_normalize_for_matching(canonical)] = canonical
    return index


def _resolve_disease_folder(dataset_path: Path, ai_disease: str, alias_map: dict[str, str] | None = None) -> str | None:
    """Resolve AI disease label to an existing disease folder (or fall back to sanitized label).

    - First apply alias mapping.
    - Then try exact normalized folder match.
    - Then try fuzzy match via SequenceMatcher.
    - Finally, return a sanitized label (may create new folder).
    """
    import difflib

    alias_map = alias_map or {}
    target = (ai_disease or "").strip()
    if not target or target.lower() == "unknown":
        return None

    # Alias mapping (AI->canonical)
    normalized_target = _normalize_for_matching(target)
    if normalized_target in alias_map:
        target = alias_map[normalized_target]
        normalized_target = _normalize_for_matching(target)

    existing_index = _build_existing_disease_index(dataset_path)
    if not existing_index:
        return safe_disease_slug(target)

    if normalized_target in existing_index:
        return existing_index[normalized_target]

    # Fuzzy: best normalized folder name
    choices = list(existing_index.keys())
    best = difflib.get_close_matches(normalized_target, choices, n=1, cutoff=0.72)
    if best:
        return existing_index[best[0]]

    return safe_disease_slug(target)



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

    """Persist only AI-fallback images and queue incremental learning.

    This function is intentionally called only from the low-confidence ML
    fallback path in app.py.

    Per requirements: AI-added images are stored inside CroppedData/<Disease>/
    and queued for training.

    IMPORTANT: Only diseases in the supported 4-class scope (Acne, Psoriasis,
    Tinea, Vitiligo) are accepted. AI predictions outside this scope are
    rejected to prevent automatic expansion of the prediction classes.
    """

    # Safety check: only allow supported 4-class diseases for self-learning.
    if disease not in SUPPORTED_CLASSES:
        LOGGER.warning(
            "[Self-Learning] AI disease '%s' is outside the supported 4-class scope. "
            "Rejecting image for self-learning to prevent class expansion.",
            disease,
        )
        return {"saved": False, "duplicate": False, "path": None, "reason": "unsupported_disease"}

    # Resolve AI label -> canonical label (registry is source of truth).
    canonical = resolve_to_canonical(disease)
    if not canonical or str(canonical).strip().lower() == "unknown":
        return {"saved": False, "duplicate": False, "path": None}

    disease_name = ensure_label_exists(canonical)

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

    # Save into the canonical disease folder directly.
    # Requirement: store AI-added images inside CroppedData/<Disease>/
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

    # Rebuild classifier head deterministically and copy existing head weights/bias.
    # IMPORTANT: we must copy by (input_dim -> units) exact shape, and keep existing
    # class slots in the same order as `class_names`.
    penultimate = model.layers[-2].output

    old_head_layer = model.layers[-1]
    old_weights = old_head_layer.get_weights()
    if not old_weights or len(old_weights) != 2:
        raise ValueError("Existing model head weights not found; cannot do incremental expansion safely.")

    old_W, old_b = old_weights

    new_output = keras.layers.Dense(
        len(class_names),
        activation="softmax",
        name=old_head_layer.name,
    )(penultimate)
    expanded = keras.Model(model.input, new_output)

    # Freeze backbone
    for layer in expanded.layers[:-1]:
        layer.trainable = False

    # Copy weights into expanded head (only when output grew)
    new_head_layer = expanded.layers[-1]
    new_W, new_b = new_head_layer.get_weights()

    # Expected shapes:
    # old_W: (feature_dim, output_classes)
    # new_W: (feature_dim, len(class_names))
    if old_W.shape[0] != new_W.shape[0]:
        raise ValueError(
            f"Head feature dim mismatch: old {old_W.shape} new {new_W.shape}"
        )

    # Critical: do NOT assume the first N columns correspond to the same labels
    # unless class index mapping is stable. Incremental logic relies on stable
    # class_names ordering (we enforce stable append-only ordering), so this copy
    # remains safe.
    keep_classes = min(old_W.shape[1], new_W.shape[1])
    new_W[:, :keep_classes] = old_W[:, :keep_classes]
    new_b[:keep_classes] = old_b[:keep_classes]
    new_head_layer.set_weights([new_W, new_b])

    return expanded





def _dataset_from_paths_and_labels(image_paths: list[str], label_indices: list[int], batch_size: int = 8):
    def load(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        image = tf.cast(image, tf.float32)

        # The model contains its own Rescaling layer, so training inputs stay 0..255.

        return image, label

    paths = tf.constant(image_paths)
    labels = tf.constant(label_indices, dtype=tf.int64)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).map(load)
    return ds.shuffle(max(50, len(image_paths)), seed=REPLAY_SHUFFLE_SEED, reshuffle_each_iteration=True).batch(batch_size)


def _dataset_for_job(image_path: str, label_index: int):
    # Kept for backward compatibility; new replay-based pipeline uses
    # _dataset_from_paths_and_labels instead.
    return _dataset_from_paths_and_labels([image_path], [label_index], batch_size=1)



def _list_replay_samples_balanced(class_names: list[str], replay_samples_per_disease: int) -> list[tuple[str, int]]:

    """Balanced replay sampling from ALL existing disease folders.

    Returns list of (image_path, label_index).
    """
    samples: list[tuple[str, int]] = []
    cropped_root = CROPPED_DATASET_PATH
    for disease_name in class_names:
        disease_dir = cropped_root / disease_name
        if not disease_dir.exists():
            continue
        image_paths = [
            p
            for p in disease_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if not image_paths:
            continue

        # Deterministic-ish shuffle to avoid always picking same few images.
        rng = np.random.default_rng(REPLAY_SHUFFLE_SEED)
        rng.shuffle(image_paths)

        chosen = image_paths[: max(0, replay_samples_per_disease)]
        label_index = class_names.index(disease_name)
        samples.extend([(str(p), label_index) for p in chosen])

    return samples


def _run_job(job: dict) -> None:
    execute(
        "UPDATE training_queue SET status = 'processing', attempts = attempts + 1, updated_at = ? WHERE id = ?",
        (utc_now(), job["id"]),
    )
    _log("training_started", "started", "Incremental training started (replay).", job["image_path"], job["disease"])
    try:
        class_names = _load_class_names()
        # Ensure class label exists without reordering existing indices.
        # Critical: do NOT sort here; incremental training head weight copying
        # assumes existing class indices remain stable.
        if job["disease"] not in class_names:
            class_names.append(job["disease"])
            _save_class_names(class_names)

        # Runtime sanity checks to prevent silent label-index drift.
        # If mapping drifts, the model can collapse to a dominant class (e.g., Tinea).
        if not job.get("disease"):
            raise ValueError("job.disease is empty")

        new_disease = job["disease"]
        if new_disease not in class_names:
            raise ValueError("New disease not found after class_names update")

        # Log current label ordering for debugging.
        _log(
            "label_mapping",
            "checked",
            f"class_names_order={class_names}",
            job.get("image_path", ""),
            new_disease,
        )


        # New AI-corrected image


        new_disease = job["disease"]
        new_label_index = class_names.index(new_disease)

        # Replay samples from old disease folders (balanced)
        replay_pairs = _list_replay_samples_balanced(class_names, REPLAY_SAMPLES_PER_DISEASE)

        # Cap replay total size to control compute.
        if REPLAY_MAX_TOTAL_SAMPLES > 0 and len(replay_pairs) > REPLAY_MAX_TOTAL_SAMPLES:
            # Deterministic trim after shuffle already occurred.
            replay_pairs = replay_pairs[:REPLAY_MAX_TOTAL_SAMPLES]

        # Final training set: new image + replay
        image_paths = [job["image_path"]] + [p for (p, _) in replay_pairs]
        label_indices = [new_label_index] + [li for (_, li) in replay_pairs]

        model = _prepare_model(class_names)
        # Freeze backbone; keep current behavior (avoids large drift)
        for layer in model.layers[:-1]:
            layer.trainable = False
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00005),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        ds = _dataset_from_paths_and_labels(image_paths, label_indices, batch_size=8)
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
        _log("training_completed", "completed", "Replay incremental training completed.", job["image_path"], job["disease"], before, after)

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