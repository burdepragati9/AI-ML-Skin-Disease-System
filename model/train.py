# =========================================================
# train.py (two-stage fine-tuning + stable class mapping + class weights)
# =========================================================

import json
import os
import shutil
import sys
from pathlib import Path

# =========================================================
# TENSORFLOW SETTINGS
# =========================================================
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.utils.class_weight import compute_class_weight

try:
    from model.augmentation_layers import RandomErasing2D
except ImportError:  # when running as `python model/train.py`
    from augmentation_layers import RandomErasing2D


import numpy as np




# =========================================================
# LIMIT CPU THREADS
# =========================================================
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

# =========================================================
# PATHS
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "skin_model.keras"
BEST_MODEL_PATH = MODEL_DIR / "best_training_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"  # legacy (used by app)
CLASS_MAPPING_PATH = MODEL_DIR / "class_mapping.json"  # stable mapping

# Training dataset root (dynamic): all disease folders under CroppedData
MAIN_DATASET = PROJECT_ROOT / "CroppedData"

# (Legacy) AI-uncertain dataset merge support kept for backward compatibility.
UNCERTAIN_DATASET = PROJECT_ROOT / "dataset" / "uncertain"

# =========================================================
# SETTINGS
# =========================================================
IMG_SIZE = 160
BATCH_SIZE = 16
# Recommended epochs after changes:
# - Head training + fine-tuning with callbacks should converge in ~12-25 epochs total.
# - Keep EPOCHS modest to avoid overfitting on small dataset.
EPOCHS = 20

SEED = 123

# Fine-tuning (Stage 2)
# Enabled to reduce long plateau caused by domain shift.
# Toggle this to compare:
#   - Stage1 only (ENABLE_STAGE2_FINE_TUNING=False)
#   - Stage1 + Stage2 (ENABLE_STAGE2_FINE_TUNING=True)
ENABLE_STAGE2_FINE_TUNING = True

# Unfreeze last N layers (MobileNetV2 many small blocks; fewer = less overfitting)
FINE_TUNE_UNFREEZE_LAST_N_LAYERS = 20

STAGE1_LEARNING_RATE = 1e-4
# Much smaller LR for stable fine-tuning.
STAGE2_LEARNING_RATE = 1e-5

# Confidence threshold diagnostics (logging only; does NOT change inference logic)
CONFIDENCE_THRESHOLD = 0.65
LOW_CONFIDENCE_TOPK = 1



# Rare-class handling
# Requirement: Exclude ALL classes with <20 images (and explicitly remove Eczema)
RARE_CLASSES_MIN_IMAGES = 20
HARD_EXCLUDE_CLASSES = {"Eczema"}

# Balanced sampling
# Oversample rare classes so batches are not dominated by one class.
# Set to 0 to disable.
BALANCED_SAMPLER_OVERSAMPLE_POWER = 0.5

# Debug logging frequency (batches)
DEBUG_LOG_EVERY_N_BATCHES = 25

# Two-stage fine-tuning durations (small dataset => short head training)
STAGE1_EPOCHS = max(2, EPOCHS // 3)  # head-only
STAGE2_EPOCHS = max(1, EPOCHS - STAGE1_EPOCHS)


# =========================================================
# IMAGE COUNT
# =========================================================
def image_count(data_dir: Path) -> int:
    extensions = {".jpg", ".jpeg", ".png"}
    return sum(1 for path in data_dir.rglob("*") if path.suffix.lower() in extensions)


# =========================================================
# MERGE AI IMAGES (legacy)
# =========================================================
def merge_uncertain_images():
    if not UNCERTAIN_DATASET.exists():
        return
    print("\nMerging AI images...")

    for class_dir in UNCERTAIN_DATASET.iterdir():
        if not class_dir.is_dir():
            continue

        target_dir = MAIN_DATASET / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)

        for image_path in class_dir.glob("*"):
            destination = target_dir / image_path.name
            if not destination.exists():
                shutil.copy2(image_path, destination)


# =========================================================
# STABLE CLASS MAPPING
# =========================================================
def load_or_init_class_mapping() -> list[str]:
    """Return stable ordered class_names from model/class_mapping.json."""
    if not CLASS_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Stable class mapping not found: {CLASS_MAPPING_PATH}. "
            "Create/update model/class_mapping.json as the single source of truth."
        )

    mapping = json.loads(CLASS_MAPPING_PATH.read_text(encoding="utf-8"))
    class_names = mapping.get("class_names", [])
    class_indices = mapping.get("class_indices", {})

    if not isinstance(class_names, list) or not class_names:
        raise ValueError("class_mapping.json has invalid/empty class_names")
    if not isinstance(class_indices, dict) or not class_indices:
        raise ValueError("class_mapping.json has invalid/empty class_indices")

    expected = {name: i for i, name in enumerate(class_names)}
    if class_indices != expected:
        raise ValueError(
            "class_mapping.json class_indices must exactly match class_names ordering. "
            f"expected={expected} got={class_indices}"
        )

    return class_names


# =========================================================
# DATASET HELPERS
# =========================================================
def list_dataset_class_counts(
    dataset_root: Path,
    allowed_classes: set[str] | None = None,
) -> dict[str, int]:
    exts = {".jpg", ".jpeg", ".png"}
    counts: dict[str, int] = {}
    if not dataset_root.exists():
        return counts

    for disease_dir in dataset_root.iterdir():
        if not disease_dir.is_dir():
            continue
        label = disease_dir.name
        if allowed_classes is not None and label not in allowed_classes:
            continue

        c = sum(
            1
            for p in disease_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in exts
        )
        counts[label] = c

    return counts


def get_active_classes_from_mapping(
    stable_class_names: list[str],
    class_counts: dict[str, int],
) -> list[str]:
    """Requirement:
    - Completely remove Eczema from training.
    - Exclude ALL classes with <20 images.
    - Keep ordering ONLY from class_mapping.json.
    """
    allowed: list[str] = []
    for name in stable_class_names:
        if name in HARD_EXCLUDE_CLASSES:
            print(f"Hard-excluding class: {name}")
            continue
        c = class_counts.get(name, 0)
        if c < RARE_CLASSES_MIN_IMAGES:
            print(
                f"Excluding class (under {RARE_CLASSES_MIN_IMAGES} images): {name} (count={c})"
            )
            continue
        allowed.append(name)
    return allowed


def freeze_all_backbone(base_model: tf.keras.Model) -> None:
    for layer in base_model.layers:
        layer.trainable = False


def unfreeze_last_n_backbone_layers(base_model: tf.keras.Model, n: int) -> None:
    layers_list = list(base_model.layers)
    to_unfreeze = layers_list[-n:] if n < len(layers_list) else layers_list
    for layer in base_model.layers:
        layer.trainable = False
    for layer in to_unfreeze:
        layer.trainable = True


def build_model(num_classes: int) -> tuple[keras.Model, tf.keras.Model]:
    """Returns (full model, base_model)."""
    print("\nLoading MobileNetV2...")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        # Use full MobileNetV2 capacity. Low alpha (e.g. 0.35) can underfit
        # fine-grained lesion texture differences (Acne vs Psoriasis/Tinea).
        weights="imagenet",
        alpha=1.0,
    )


    # ------------------------------------------------------------------
    # Data augmentation
    # ------------------------------------------------------------------
    # Goal: improve robustness to lighting/background artifacts WITHOUT
    # destroying clinically relevant color/texture cues.
    #
    # Notes:
    # - Keep flips/rotations mild.
    # - Replace extreme brightness/contrast jitter with smaller ranges.
    # - Add random crop/zoom simulation to account for scale differences.
    # - Add optional random erasing (cutout-like) to reduce sensitivity to
    #   hair/occlusion and crop boundary artifacts.
    #
    # Medical caution: do NOT over-distort color space.
    # Serialization/deserialization fix notes:
    # - Some saved models may contain custom layer configs. Keras needs either
    #   a registered serializable custom layer (via @register_keras_serializable)
    #   OR a custom_objects mapping at load time.
    #
    # Medical caution during training:
    # - Reduce augmentation slightly to avoid confusing visually similar classes.
    # - Temporarily disable RandomErasing2D to reduce occlusion artifacts.
    data_augmentation = keras.Sequential(
        [
            # Geometry
            layers.RandomFlip("horizontal"),
            # Vertical flip can be medically questionable (body orientation),
            # so keep it out to reduce label noise.
            layers.RandomRotation(0.04),
            # Zoom in/out (medical images => keep conservative)
            layers.RandomZoom(0.08),

            # Color (small)
            layers.RandomContrast(0.01),

            layers.RandomBrightness(0.01),

            # Random erasing / cutout approximation (serialization-safe)
            # Temporarily disabled during training.
            # RandomErasing2D(p=0.03, erase_prob=0.02, fill_value=0.0),
        ],
        name="augmentation",
    )



    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)

    # Match inference: (x / 127.5) - 1
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    # Head regularization tuned for macro accuracy + stable training.
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)


    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model, base_model


def compute_class_weights_from_counts(
    class_names: list[str],
    class_counts: dict[str, int],
    *,
    cap_min: float = 0.25,
    cap_max: float = 4.0,
) -> dict[int, float]:
    """Compute normalized/capped class weights (safer against collapse)."""
    y: list[int] = []
    for idx, name in enumerate(class_names):
        c = class_counts.get(name, 0)
        if c > 0:
            y.extend([idx] * c)

    if not y:
        return {i: 1.0 for i in range(len(class_names))}

    classes = np.arange(len(class_names), dtype=np.int64)
    y_arr = np.asarray(y, dtype=np.int64)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_arr,
    ).astype(np.float32)

    weights = np.clip(weights, cap_min, cap_max)

    mean_w = float(np.mean(weights)) if len(weights) else 1.0
    if mean_w <= 0:
        mean_w = 1.0
    weights = weights / mean_w

    return {i: float(w) for i, w in enumerate(weights)}


def log_debug_batches_for_balance(model_for_debug: keras.Model, train_ds: tf.data.Dataset, class_names: list[str], *, n_batches: int = 3):

    """Print batch class distribution + y_true + y_pred + top softmax probs for a few batches."""
    from collections import Counter

    print("\n[Batch Debug] Verifying balanced batch composition (y_true + predicted labels)")
    debug_class_names = {i: name for i, name in enumerate(class_names)}

    # Take already-batched samples from train_ds to avoid shape mismatches.
    for bi, (x_b, y_b) in enumerate(train_ds.take(3)):
        y_np = y_b.numpy().astype(int).tolist()

        probs = model_for_debug.predict(x_b, verbose=0)
        pred_labels = np.argmax(probs, axis=1).astype(int).tolist()
        top1_probs = np.max(probs, axis=1).astype(float).tolist()


        dist = Counter(y_np)
        dist_str = {debug_class_names[k]: int(v) for k, v in sorted(dist.items())}

        print(f"\n[Batch Debug {bi+1}] y_true indices: {y_np}")
        print(f"[Batch Debug {bi+1}] y_true distribution: {dist_str}")
        print(f"[Batch Debug {bi+1}] pred indices: {pred_labels}")
        print(f"[Batch Debug {bi+1}] top-1 softmax probs: {top1_probs}")


def _sample_validation_predictions(model_for_debug: keras.Model, val_ds: tf.data.Dataset, class_names: list[str], *, n_batches: int = 2, top_k: int = 3) -> None:
    """Print sample val predictions to detect label mapping / collapse."""
    # Take n_batches already-batched from val_ds
    for bi, (x_b, y_b) in enumerate(val_ds.take(n_batches)):
        probs = model_for_debug.predict(x_b, verbose=0)
        y_true = y_b.numpy().astype(int)
        pred_idx = np.argmax(probs, axis=1).astype(int)

        for si in range(len(y_true)):
            t_idx = int(y_true[si])
            p_idx = int(pred_idx[si])
            # top_k indices
            top_indices = np.argsort(probs[si])[::-1][:top_k]
            top_items = [(class_names[int(j)], float(probs[si][int(j)])) for j in top_indices]

            print(
                f"\n[Val Sample] batch={bi} sample={si} true={class_names[t_idx]} pred={class_names[p_idx]} "
                f"top_probs={top_items}"
            )


def _compute_precision_recall_f1_from_confusion(cm: np.ndarray) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Per-class precision/recall/F1 from confusion matrix.

    cm rows=true, cols=pred.
    """
    eps = 1e-12
    n = cm.shape[0]
    precision: dict[int, float] = {}
    recall: dict[int, float] = {}
    f1: dict[int, float] = {}

    for i in range(n):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - cm[i, i])
        fn = float(cm[i, :].sum() - cm[i, i])

        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f = 2 * p * r / (p + r + eps)

        precision[i] = float(p)
        recall[i] = float(r)
        f1[i] = float(f)

    return precision, recall, f1


def _top_confusion_pairs(cm: np.ndarray, class_names: list[str], *, k: int = 5) -> list[tuple[str, str, int]]:
    """Return top K confusion pairs (true -> predicted) excluding diagonal."""
    n = cm.shape[0]
    pairs: list[tuple[str, str, int]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            count = int(cm[i, j])
            if count <= 0:
                continue
            pairs.append((class_names[i], class_names[j], count))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:k]


def main():
    keras.utils.set_random_seed(SEED)
    if BEST_MODEL_PATH.exists():
        BEST_MODEL_PATH.unlink()


    total_images = image_count(MAIN_DATASET)
    if total_images == 0:
        raise Exception("Dataset empty.")

    print(f"\nDataset: {MAIN_DATASET}")
    print(f"Total Images: {total_images}")

    stable_class_names = load_or_init_class_mapping()
    class_counts = list_dataset_class_counts(MAIN_DATASET)

    print("\nClass counts in CroppedData (total):")
    for k in stable_class_names:
        print(f"  {k}: {class_counts.get(k, 0)}")

    allowed_classes = get_active_classes_from_mapping(stable_class_names, class_counts)
    if len(allowed_classes) < 2:
        raise ValueError("Not enough allowed classes after rare-class exclusion.")

    print(
        "\nAllowed classes for this retraining run (excluded rare < %d if configured):" % RARE_CLASSES_MIN_IMAGES
    )
    print(sorted(list(allowed_classes)))

    # Stable label ordering only from mapping.json
    class_names = list(allowed_classes)

    missing = [c for c in class_names if class_counts.get(c, 0) == 0]
    if missing:
        raise ValueError(f"Allowed classes with 0 images found: {missing}")

    # Build datasets (training subset for random split)
    base_train_ds = tf.keras.utils.image_dataset_from_directory(
        MAIN_DATASET,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        class_names=class_names,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        MAIN_DATASET,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_names=class_names,
        shuffle=True,
    )

    # Balanced oversampling via tf.data
    # -------------------------------------------------------------
    # Problem in the old approach:
    # - It iterated base_train_ds and called `.numpy()` for each element,
    #   rebuilding big in-memory arrays. That is fragile, slow, and can
    #   reduce effective augmentation diversity.
    #
    # New approach:
    # - Build per-class datasets using `take_while` + filtering in tf.data
    #   without materializing all images into numpy arrays.
    # - Still use `sample_from_datasets` to balance classes.
    #
    # NOTE: tf.data.image_dataset_from_directory already loads/decodes.
    # We accept that dataset elements are tensors, but we avoid calling
    # `.numpy()` inside Python loops.
    # -------------------------------------------------------------

    if BALANCED_SAMPLER_OVERSAMPLE_POWER <= 0:
        print("Balanced sampler disabled (BALANCED_SAMPLER_OVERSAMPLE_POWER <= 0).")
        train_ds = (
            base_train_ds.unbatch()
            .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )
        steps_per_epoch = None
    else:
        # Create per-class datasets by filtering on label.
        # base_train_ds elements are (image, label_scalar).
        per_class_datasets: list[tf.data.Dataset] = []

        class_probs_arr = np.zeros((len(class_names),), dtype=np.float32)
        for idx, name in enumerate(class_names):
            c = class_counts.get(name, 0)
            class_probs_arr[idx] = (1.0 / max(float(c), 1.0)) ** BALANCED_SAMPLER_OVERSAMPLE_POWER

        s = float(np.sum(class_probs_arr))
        if s <= 0:
            class_probs_arr[:] = 1.0
            s = float(np.sum(class_probs_arr))
        class_probs_arr = class_probs_arr / s

        # base_train_ds is batched(1). Make it unbatched so filter can work on scalar labels.
        base_unbatched = base_train_ds.unbatch()

        for li in range(len(class_names)):
            ds_li = (
                base_unbatched
                .filter(lambda x, y, li=li: tf.equal(tf.cast(y, tf.int64), tf.cast(li, tf.int64)))
                .shuffle(512, seed=SEED + li, reshuffle_each_iteration=True)
                .repeat()
            )
            per_class_datasets.append(ds_li)

        if not per_class_datasets:
            raise RuntimeError("No training samples found for balanced sampler")

        mixed = tf.data.Dataset.sample_from_datasets(
            per_class_datasets,
            weights=class_probs_arr.tolist(),
            seed=SEED,
        )

        train_ds = mixed.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        steps_per_epoch = int(
            np.ceil(sum(class_counts.get(name, 0) for name in class_names) / BATCH_SIZE)
        )



    print("\nBalanced sampler class sampling probabilities:")
    if BALANCED_SAMPLER_OVERSAMPLE_POWER <= 0:
        for i, name in enumerate(class_names):
            print(f"  {name}: natural dataset frequency (count={class_counts.get(name,0)})")
    else:
        balanced_prob = 1.0 / max(len(per_class_datasets), 1)
        for name in class_names:
            print(f"  {name}: {balanced_prob:.4f} (count={class_counts.get(name,0)})")

    # Persist active classes for diagnostics
    active_classes_path = MODEL_DIR / "active_classes.json"
    active_classes_path.write_text(json.dumps({"class_names": class_names}, indent=2), encoding="utf-8")

    # Persist legacy class_names.json for inference compatibility
    CLASS_NAMES_PATH.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    print("\nStable class_names used for this training run:")
    print(class_names)

    # Compute class weights from counts (proxy)
    # If oversampling is enabled, disable class_weight to avoid double re-weighting bias.
    class_weight = None
    if BALANCED_SAMPLER_OVERSAMPLE_POWER <= 0:
        class_weight = compute_class_weights_from_counts(class_names, class_counts)

    print("\nComputed sklearn class_weight:")
    if class_weight is not None:
        print(class_weight)


    # Build model
    model, base_model = build_model(len(class_names))

    # Debug logs for balanced batch verification
    log_debug_batches_for_balance(model, train_ds, class_names)

    # Callback: per-class validation accuracy + debug samples each epoch
    class PerClassValAccuracyCallback(keras.callbacks.Callback):
        def __init__(self, val_ds, class_names, *, print_every_n_epochs: int = 1):
            super().__init__()
            self.val_ds = val_ds
            self.class_names = class_names
            self.print_every_n_epochs = print_every_n_epochs

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            if (epoch % self.print_every_n_epochs) != 0:
                return

            # Gather predictions on val_ds
            y_true_all = []
            y_pred_all = []
            for x_b, y_b in self.val_ds:
                probs = self.model.predict(x_b, verbose=0)
                preds = np.argmax(probs, axis=1).astype(int)
                y_true_all.extend(y_b.numpy().astype(int).tolist())
                y_pred_all.extend(preds.tolist())

            y_true_all = np.asarray(y_true_all, dtype=int)
            y_pred_all = np.asarray(y_pred_all, dtype=int)
            n_classes = len(self.class_names)

            per_class_acc = {}
            for i in range(n_classes):
                mask = y_true_all == i
                if np.any(mask):
                    per_class_acc[self.class_names[i]] = float(np.mean(y_pred_all[mask] == i))
                else:
                    per_class_acc[self.class_names[i]] = 0.0

            macro_acc = float(np.mean(list(per_class_acc.values()))) if per_class_acc else 0.0
            logs["val_macro_acc"] = macro_acc

            print("\n[PerClassValAccuracy] epoch", epoch, "overall_val_acc=", logs.get("val_accuracy"))
            print(f"  macro_avg: {macro_acc:.4f}")
            for k, v in per_class_acc.items():
                print(f"  {k}: {v:.4f}")

            # Print a few sample predictions for label mapping sanity
            _sample_validation_predictions(self.model, self.val_ds, self.class_names, n_batches=1, top_k=3)

    # -------------------------------------------------------------
    # Callbacks tuned for:
    # - stable convergence (avoid collapse)
    # - best checkpoint by macro validation accuracy
    # -------------------------------------------------------------
    # Misclassified-validation snapshot callback
    try:
        from model.misclassified_callback import SaveMisclassifiedValImages
    except ImportError:
        from misclassified_callback import SaveMisclassifiedValImages

    MISCLASSIFIED_OUTROOT = PROJECT_ROOT / "analytics" / "misclassified"

    callbacks = [
        PerClassValAccuracyCallback(val_ds, class_names, print_every_n_epochs=1),
        SaveMisclassifiedValImages(
            val_ds,
            class_names,
            MISCLASSIFIED_OUTROOT,
            max_images=200,
        ),
        ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor="val_macro_acc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        # Medical-image labels can be noisy/ambiguous => slightly higher patience.
        EarlyStopping(
            monitor="val_macro_acc",
            mode="max",
            patience=4,
            restore_best_weights=True,
        ),
        # Reduce LR when val_loss plateaus.
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]



    # One-time initial debug print (before any training)
    print("\n[Initial Validation Debug] sample predictions before training:")
    _sample_validation_predictions(model, val_ds, class_names, n_batches=1, top_k=3)


    # Stage 1
    print("\n=== Stage 1: head-only training (frozen backbone) ===")
    freeze_all_backbone(base_model)
    # -------------------------------------------------------------
    # Loss: label smoothing helps with ambiguous medical labels
    # (Acne vs Psoriasis/Tinea can overlap visually).
    # -------------------------------------------------------------
    label_smoothing = 0.1
    # TF/Keras SparseCategoricalCrossentropy label_smoothing keyword availability


    # varies across TF versions. Use the generic LossFunction with label_smoothing
    # if supported; otherwise fall back to plain sparse CE.
    try:
        loss_fn = keras.losses.SparseCategoricalCrossentropy(
            label_smoothing=label_smoothing
        )
    except TypeError:
        loss_fn = keras.losses.SparseCategoricalCrossentropy()


    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=STAGE1_LEARNING_RATE),
        loss=loss_fn,
        metrics=["accuracy"],
    )


    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        epochs=STAGE1_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    if ENABLE_STAGE2_FINE_TUNING:

        # Stage 2

        print("\n=== Stage 2: unfreeze last %d MobileNetV2 layers ===" % FINE_TUNE_UNFREEZE_LAST_N_LAYERS)
        unfreeze_last_n_backbone_layers(base_model, FINE_TUNE_UNFREEZE_LAST_N_LAYERS)
        # Reuse the same label-smoothed loss fn for fine-tuning.
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=STAGE2_LEARNING_RATE),
            loss=loss_fn,
            metrics=["accuracy"],
        )



        history = model.fit(
            train_ds,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            epochs=STAGE2_EPOCHS,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )
    else:
        print("\n=== Stage 2 skipped: fine-tuning disabled for stability ===")

    if BEST_MODEL_PATH.exists():
        # Fix custom layer deserialization for models saved with RandomErasing2D.
        # Why it happened:
        # - Keras couldn't locate the class by name during deserialization.
        # Why this works:
        # - Either @register_keras_serializable registers the class name globally,
        #   and/or providing custom_objects supplies the class at load time.
        model = keras.models.load_model(
            BEST_MODEL_PATH,
            compile=False,
            custom_objects={"RandomErasing2D": RandomErasing2D},
        )

    model.save(MODEL_PATH)

    old_model_path = MODEL_DIR / "skin_model.h5"
    if old_model_path.exists():
        old_model_path.unlink()

    best_accuracy = max(history.history.get("val_accuracy", [0.0]))

    print("\n=================================")
    print("Training completed.")
    print(f"Best Validation Accuracy (stage2): {best_accuracy:.4f}")
    print(f"\nModel saved at:\n{MODEL_PATH}")
    print(f"\nClass names saved at:\n{CLASS_NAMES_PATH}")
    print("=================================\n")

    try:
        import subprocess

        subprocess.run(
            [sys.executable, "analytics/model_diagnostics.py"],
            cwd=str(PROJECT_ROOT),
            check=False,
        )
    except Exception as exc:
        print(f"Diagnostics run failed: {exc}")


if __name__ == "__main__":
    main()
