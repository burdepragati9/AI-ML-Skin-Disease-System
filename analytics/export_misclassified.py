import argparse
import csv
import json
import os
import re
import shutil
from math import ceil
from pathlib import Path
from typing import Any

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - keeps script usable without tqdm installed.
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "skin_model.keras"
ACTIVE_CLASSES_PATH = PROJECT_ROOT / "model" / "active_classes.json"
CROPPED_DATASET_PATH = PROJECT_ROOT / "CroppedData"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analytics" / "misclassified"


def safe_name(name: str) -> str:
    value = str(name).strip()
    value = re.sub(r"[^\w.\- ]+", "_", value)
    value = re.sub(r"\s+", "_", value)
    return value or "unknown"


def load_active_class_names(path: Path = ACTIVE_CLASSES_PATH) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Active class file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    class_names = payload.get("class_names", [])
    if not isinstance(class_names, list) or not class_names:
        raise ValueError(f"{path} must contain a non-empty class_names list")

    return [str(name) for name in class_names]


def model_img_size(model: tf.keras.Model, default: int = 160) -> int:
    input_shape = model.input_shape
    if isinstance(input_shape, tuple) and len(input_shape) >= 3 and input_shape[1]:
        return int(input_shape[1])
    return default


def dataset_sample_count(dataset: tf.data.Dataset) -> int:
    return int(dataset.unbatch().reduce(0, lambda count, _: count + 1).numpy())


def progress_iter(iterable, *, total: int, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="batch")


def export_misclassified_images(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: list[str],
    file_paths: list[str | Path],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    batch_size: int = 32,
    clear_output: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if clear_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "misclassified.csv"

    file_paths = [Path(path) for path in file_paths]
    sample_count = dataset_sample_count(dataset)
    if sample_count != len(file_paths):
        raise ValueError(
            f"Dataset sample count ({sample_count}) does not match "
            f"file_paths count ({len(file_paths)}). Create the dataset with shuffle=False."
        )

    path_ds = tf.data.Dataset.from_tensor_slices([str(path) for path in file_paths])
    paired_ds = tf.data.Dataset.zip((path_ds, dataset.unbatch())).batch(batch_size)
    total_batches = ceil(sample_count / max(1, batch_size))

    summary: dict[str, Any] = {
        "total": 0,
        "correct": 0,
        "misclassified": 0,
        "copied": 0,
        "skipped_existing": 0,
        "confusion_counts": {},
        "output_dir": str(output_dir),
        "csv_path": str(csv_path),
    }

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image_path",
                "true_label",
                "predicted_label",
                "confidence_score",
            ],
        )
        writer.writeheader()

        for paths_batch, (images_batch, labels_batch) in progress_iter(
            paired_ds,
            total=total_batches,
            desc="Exporting misclassified images",
        ):
            probs_batch = model.predict(images_batch, verbose=0)
            pred_indices = np.argmax(probs_batch, axis=1).astype(int)
            true_indices = labels_batch.numpy().astype(int)
            decoded_paths = [Path(path.decode("utf-8")) for path in paths_batch.numpy()]

            for row_idx, (image_path, true_idx, pred_idx) in enumerate(
                zip(decoded_paths, true_indices, pred_indices)
            ):
                true_class = class_names[int(true_idx)]
                pred_class = class_names[int(pred_idx)]
                confidence_score = float(probs_batch[row_idx][int(pred_idx)])

                summary["total"] += 1
                if int(true_idx) == int(pred_idx):
                    summary["correct"] += 1
                    continue

                summary["misclassified"] += 1
                key = f"{true_class}_as_{pred_class}"
                summary["confusion_counts"][key] = int(summary["confusion_counts"].get(key, 0)) + 1

                writer.writerow(
                    {
                        "image_path": str(image_path),
                        "true_label": true_class,
                        "predicted_label": pred_class,
                        "confidence_score": f"{confidence_score:.8f}",
                    }
                )

                folder_name = f"{safe_name(true_class)}_as_{safe_name(pred_class)}"
                target_dir = output_dir / folder_name
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / image_path.name

                # Keep exports deterministic and duplicate-safe.
                if target_path.exists():
                    summary["skipped_existing"] += 1
                    continue

                shutil.copy2(image_path, target_path)
                summary["copied"] += 1

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export misclassified images for visual review.")
    parser.add_argument("--dataset", type=Path, default=CROPPED_DATASET_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.0)
    parser.add_argument("--subset", choices=["training", "validation"], default="validation")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--keep-old", action="store_true")
    args = parser.parse_args()

    class_names = load_active_class_names()
    model = tf.keras.models.load_model(args.model, compile=False)
    img_size = model_img_size(model)

    dataset_kwargs: dict[str, Any] = {
        "directory": args.dataset,
        "image_size": (img_size, img_size),
        "batch_size": args.batch_size,
        "class_names": class_names,
        # Critical: file_paths order only matches dataset samples when shuffle=False.
        "shuffle": False,
    }

    if args.validation_split and args.validation_split > 0:
        dataset_kwargs["validation_split"] = args.validation_split
        dataset_kwargs["subset"] = args.subset
        dataset_kwargs["seed"] = args.seed

    dataset = tf.keras.utils.image_dataset_from_directory(**dataset_kwargs)
    file_paths = list(getattr(dataset, "file_paths", []))
    if not file_paths:
        raise ValueError("Dataset does not expose file_paths. Use image_dataset_from_directory directly.")

    summary = export_misclassified_images(
        model=model,
        dataset=dataset,
        class_names=class_names,
        file_paths=file_paths,
        output_dir=args.output,
        batch_size=args.batch_size,
        clear_output=not args.keep_old,
    )

    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
