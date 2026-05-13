# =========================================================
# FINAL UPDATED train.py
# SAME EPOCHS VERSION
# =========================================================

import json
import os
import shutil
from pathlib import Path

# =========================================================
# TENSORFLOW SETTINGS
# =========================================================
os.environ.setdefault(
    "TF_ENABLE_ONEDNN_OPTS",
    "0"
)

os.environ.setdefault(
    "TF_CPP_MIN_LOG_LEVEL",
    "2"
)

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
)

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

MODEL_DIR.mkdir(
    exist_ok=True
)

MODEL_PATH = (
    MODEL_DIR
    / "skin_model.keras"
)

CLASS_NAMES_PATH = (
    MODEL_DIR
    / "class_names.json"
)

MAIN_DATASET = (
    PROJECT_ROOT
    / "MySkinData"
)

UNCERTAIN_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "uncertain"
)

# =========================================================
# SETTINGS
# =========================================================
IMG_SIZE = 160

BATCH_SIZE = 2

# =========================================================
# KEEP SAME EPOCHS
# =========================================================
EPOCHS = 5

SEED = 123

# =========================================================
# IMAGE COUNT
# =========================================================
def image_count(data_dir):

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
    }

    return sum(
        1
        for path in data_dir.rglob("*")
        if path.suffix.lower() in extensions
    )

# =========================================================
# MERGE AI IMAGES
# =========================================================
def merge_uncertain_images():

    if not UNCERTAIN_DATASET.exists():
        return

    print("\nMerging AI images...")

    for class_dir in UNCERTAIN_DATASET.iterdir():

        if not class_dir.is_dir():
            continue

        target_dir = (
            MAIN_DATASET
            / class_dir.name
        )

        target_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for image_path in class_dir.glob("*"):

            destination = (
                target_dir
                / image_path.name
            )

            # avoid duplicates
            if not destination.exists():

                shutil.copy2(
                    image_path,
                    destination,
                )

# =========================================================
# BUILD MODEL
# =========================================================
def build_model(num_classes):

    print("\nLoading MobileNetV2...")

    base_model = tf.keras.applications.MobileNetV2(

        input_shape=(
            IMG_SIZE,
            IMG_SIZE,
            3,
        ),

        include_top=False,

        weights="imagenet",

        alpha=0.35,
    )

    base_model.trainable = False

    # =====================================================
    # DATA AUGMENTATION
    # =====================================================
    data_augmentation = keras.Sequential(
        [

            layers.RandomFlip(
                "horizontal"
            ),

            layers.RandomRotation(
                0.1
            ),

            layers.RandomZoom(
                0.1
            ),
        ]
    )

    # =====================================================
    # INPUT
    # =====================================================
    inputs = keras.Input(
        shape=(
            IMG_SIZE,
            IMG_SIZE,
            3,
        )
    )

    x = data_augmentation(inputs)

    # =====================================================
    # NORMALIZATION
    # =====================================================
    x = layers.Rescaling(
        1.0 / 127.5,
        offset=-1,
    )(x)

    # =====================================================
    # FEATURE EXTRACTION
    # =====================================================
    x = base_model(
        x,
        training=False,
    )

    # =====================================================
    # CLASSIFIER
    # =====================================================
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        256,
        activation="relu",
    )(x)

    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(

        num_classes,

        activation="softmax",
    )(x)

    # =====================================================
    # FINAL MODEL
    # =====================================================
    model = keras.Model(
        inputs,
        outputs,
    )

    # =====================================================
    # COMPILE
    # =====================================================
    model.compile(

        optimizer=keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"],
    )

    return model

# =========================================================
# MAIN
# =========================================================
def main():

    # =====================================================
    # MERGE AI IMAGES
    # =====================================================
    merge_uncertain_images()

    # =====================================================
    # CHECK DATASET
    # =====================================================
    total_images = image_count(
        MAIN_DATASET
    )

    if total_images == 0:

        raise Exception(
            "Dataset empty."
        )

    print(f"\nDataset: {MAIN_DATASET}")

    print(f"Total Images: {total_images}")

    # =====================================================
    # TRAIN DATASET
    # =====================================================
    train_ds = tf.keras.utils.image_dataset_from_directory(

        MAIN_DATASET,

        validation_split=0.2,

        subset="training",

        seed=SEED,

        image_size=(
            IMG_SIZE,
            IMG_SIZE,
        ),

        batch_size=BATCH_SIZE,
    )

    # =====================================================
    # VALIDATION DATASET
    # =====================================================
    val_ds = tf.keras.utils.image_dataset_from_directory(

        MAIN_DATASET,

        validation_split=0.2,

        subset="validation",

        seed=SEED,

        image_size=(
            IMG_SIZE,
            IMG_SIZE,
        ),

        batch_size=BATCH_SIZE,
    )

    # =====================================================
    # CLASS NAMES
    # =====================================================
    class_names = train_ds.class_names

    print("\nClasses:")

    print(class_names)

    CLASS_NAMES_PATH.write_text(

        json.dumps(
            class_names,
            indent=2,
        ),

        encoding="utf-8",
    )

    # =====================================================
    # LOW RAM SETTINGS
    # =====================================================
    train_ds = train_ds.shuffle(50)

    val_ds = val_ds

    # =====================================================
    # BUILD MODEL
    # =====================================================
    model = build_model(
        len(class_names)
    )

    # =====================================================
    # CALLBACKS
    # =====================================================
    callbacks = [

        EarlyStopping(

            monitor="val_loss",

            patience=2,

            restore_best_weights=True,
        ),

        ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.3,

            patience=1,
        ),
    ]

    # =====================================================
    # TRAINING
    # =====================================================
    print("\nStarting training...\n")

    history = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=EPOCHS,

        callbacks=callbacks,
    )

    # =====================================================
    # SAVE MODEL
    # =====================================================
    model.save(MODEL_PATH)

    best_accuracy = max(
        history.history["val_accuracy"]
    )

    print("\n=================================")

    print("Training completed.")

    print(
        f"Best Validation Accuracy: "
        f"{best_accuracy:.4f}"
    )

    print(
        f"\nModel saved at:\n{MODEL_PATH}"
    )

    print(
        f"\nClass names saved at:\n"
        f"{CLASS_NAMES_PATH}"
    )

    print("=================================")

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    main()
