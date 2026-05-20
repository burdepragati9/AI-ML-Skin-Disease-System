import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "skin_model.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "model" / "class_names.json"
CROPPED_DATASET_PATH = PROJECT_ROOT / "CroppedData"


def load_class_names() -> List[str]:
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"class_names.json not found: {CLASS_NAMES_PATH}")
    names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    if not isinstance(names, list) or not names:
        raise ValueError("class_names.json must be a non-empty list")
    return names


def model_img_size(model: tf.keras.Model, default: int = 160) -> int:
    ishape = model.input_shape
    if isinstance(ishape, tuple) and len(ishape) >= 3:
        h = ishape[1]
        if h:
            return int(h)
    return default


def preprocess_for_model(image_path: Path, img_size: int) -> np.ndarray:
    # The model contains its own Rescaling layer, so diagnostics inputs stay 0..255.
    img = tf.keras.utils.load_img(image_path, target_size=(img_size, img_size))
    x = tf.keras.utils.img_to_array(img).astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return x


def list_dataset_samples(dataset_root: Path, allowed_classes: List[str] | None = None) -> List[Tuple[Path, str]]:
    samples: List[Tuple[Path, str]] = []
    exts = {".jpg", ".jpeg", ".png"}

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    for disease_dir in dataset_root.iterdir():
        if not disease_dir.is_dir():
            continue
        label = disease_dir.name
        if allowed_classes is not None and label not in allowed_classes:
            continue
        for p in disease_dir.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                samples.append((p, label))
    return samples


def confusion_matrix(y_true: List[int], y_pred: List[int], n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def entropy(probs: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostics: confusion matrix + prob stats using runtime preprocessing")
    parser.add_argument("--classes", nargs="*", default=None, help="Optional subset of classes (folder names) to evaluate")
    parser.add_argument("--max-per-class", type=int, default=0, help="Limit samples per class (0 = no limit)")
    parser.add_argument("--batch-size", type=int, default=32, help="Prediction batch size")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "analytics" / "diagnostics_results.json")
    args = parser.parse_args()


    # Diagnostics must evaluate ONLY classes used during the current training run.
    # train.py writes model/active_classes.json with the active class_names list.
    active_classes_path = PROJECT_ROOT / "model" / "active_classes.json"
    if not active_classes_path.exists():
        raise FileNotFoundError(
            "Missing model/active_classes.json. "
            "Re-train and re-run diagnostics so label ordering matches the trained Dense layer." 
        )

    active_payload = json.loads(active_classes_path.read_text(encoding="utf-8"))
    class_names = active_payload.get("class_names", [])

    if not class_names:
        raise ValueError("active_classes.json has no class_names")


    # Hard cap diagnostics evaluation set to active classes only.
    active_set = set(class_names)
    eval_classes = None
    if args.classes is not None and len(args.classes) > 0:
        eval_classes = [c for c in args.classes if c in active_set]
        if not eval_classes:
            raise ValueError(f"None of requested --classes are in active set: {sorted(active_set)}")
    else:
        eval_classes = list(class_names)

    n_classes = len(class_names)
    name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(class_names)}

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    img_size = model_img_size(model)

    samples = list_dataset_samples(CROPPED_DATASET_PATH, allowed_classes=eval_classes)

    # Debug: ensure dataset labels match active class set
    sample_labels = sorted({lbl for _, lbl in samples})
    unexpected = [lbl for lbl in sample_labels if lbl not in name_to_idx]
    if unexpected:
        print("\n[Diagnostics Debug] Found dataset folders not in active class_names: ", unexpected)

    print("\n[Diagnostics Debug] Evaluating on class folders:")
    for lbl in sample_labels:
        if lbl in name_to_idx:
            print(f"  {lbl} -> idx {name_to_idx[lbl]}")
        else:
            print(f"  {lbl} -> (unexpected)")


    # Optional limit per class

    if args.max_per_class and args.max_per_class > 0:
        counts: Dict[str, int] = {}
        limited: List[Tuple[Path, str]] = []
        for p, lbl in samples:
            c = counts.get(lbl, 0)
            if c >= args.max_per_class:
                continue
            counts[lbl] = c + 1
            limited.append((p, lbl))
        samples = limited

    if not samples:
        raise SystemExit("No samples found for evaluation")

    y_true: List[int] = []
    y_pred: List[int] = []
    top1_probs: List[float] = []
    entropies: List[float] = []

    # Per-class prob stats
    per_class_probs: List[List[float]] = [[] for _ in range(n_classes)]

    batch_size = max(1, int(args.batch_size))
    active_samples = [(path, lbl) for path, lbl in samples if lbl in name_to_idx]

    for start in range(0, len(active_samples), batch_size):
        batch = active_samples[start : start + batch_size]
        x_batch = np.concatenate(
            [preprocess_for_model(path, img_size=img_size) for path, _ in batch],
            axis=0,
        )
        probs_batch = model.predict(x_batch, verbose=0)
        pred_batch = np.argmax(probs_batch, axis=1).astype(int)
        entropy_batch = entropy(probs_batch)

        for sample_index, (_, lbl) in enumerate(batch):
            ti = name_to_idx[lbl]
            pi = int(pred_batch[sample_index])
            probs = probs_batch[sample_index]

            y_true.append(ti)
            y_pred.append(pi)

            top1_prob = float(probs[pi])
            top1_probs.append(top1_prob)
            entropies.append(float(entropy_batch[sample_index]))

            per_class_probs[ti].append(top1_prob)

    cm = confusion_matrix(y_true, y_pred, n_classes=n_classes)

    # Per-class accuracy + prediction percentage (requested)
    true_counts = cm.sum(axis=1)  # per row
    pred_counts = cm.sum(axis=0)  # per column

    per_class_accuracy: Dict[str, float] = {}
    per_class_pred_percentage: Dict[str, float] = {}

    total_preds = float(sum(pred_counts)) if n_classes else 0.0
    for i in range(n_classes):
        cls = class_names[i]
        if true_counts[i] > 0:
            per_class_accuracy[cls] = float(cm[i, i] / true_counts[i])
        else:
            per_class_accuracy[cls] = 0.0

        if total_preds > 0:
            per_class_pred_percentage[cls] = float(pred_counts[i] / total_preds) * 100.0
        else:
            per_class_pred_percentage[cls] = 0.0


    # Per-class precision/recall/F1 + macro/micro averages + classification report
    eps = 1e-12

    per_class_precision: Dict[str, float] = {}
    per_class_recall: Dict[str, float] = {}
    per_class_f1: Dict[str, float] = {}

    # macro avg
    for i in range(n_classes):
        cls = class_names[i]

        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - cm[i, i])
        fn = float(cm[i, :].sum() - cm[i, i])

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        per_class_precision[cls] = float(precision)
        per_class_recall[cls] = float(recall)
        per_class_f1[cls] = float(f1)

    macro_precision = float(np.mean(list(per_class_precision.values()))) if per_class_precision else 0.0
    macro_recall = float(np.mean(list(per_class_recall.values()))) if per_class_recall else 0.0
    macro_f1 = float(np.mean(list(per_class_f1.values()))) if per_class_f1 else 0.0

    # micro average (global)
    total_tp = float(np.trace(cm))
    total = float(np.sum(cm))
    micro_accuracy = total_tp / (total + eps)

    # In single-label classification:
    # micro precision == micro recall == micro f1 == accuracy
    micro_precision = micro_accuracy
    micro_recall = micro_accuracy
    micro_f1 = micro_accuracy

    # "classification_report" JSON compatible output
    classification_report = {
        "accuracy": float(micro_accuracy),
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1_score": macro_f1,
        },
        "micro_avg": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1_score": micro_f1,
        },
        "per_class": {
            class_names[i]: {
                "precision": per_class_precision[class_names[i]],
                "recall": per_class_recall[class_names[i]],
                "f1_score": per_class_f1[class_names[i]],
                "support": int(cm[i, :].sum()),
            }
            for i in range(n_classes)
        },
    }

    # Summary stats
    diag = {
        "model_path": str(MODEL_PATH),
        "img_size": img_size,
        "classes": class_names,
        "n_samples": len(y_true),
        "confusion_matrix": cm.tolist(),
        "true_distribution": {class_names[i]: int(cm[i, :].sum()) for i in range(n_classes)},
        "pred_distribution": {class_names[i]: int(cm[:, i].sum()) for i in range(n_classes)},
        "top1_prob_mean": float(np.mean(top1_probs)) if top1_probs else 0.0,
        "top1_prob_min": float(np.min(top1_probs)) if top1_probs else 0.0,
        "top1_prob_max": float(np.max(top1_probs)) if top1_probs else 0.0,
        "entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
        "per_true_class_top1_prob_mean": {
            class_names[i]: float(np.mean(per_class_probs[i])) if per_class_probs[i] else 0.0 for i in range(n_classes)
        },
        "per_class_accuracy": per_class_accuracy,
        "per_class_prediction_percentage": per_class_pred_percentage,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1_score": macro_f1,
        },
        "micro_avg": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1_score": micro_f1,
        },
        "classification_report": classification_report,
    }



    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(diag, indent=2), encoding="utf-8")

    # Console output (quick glance)
    print("\n=== Diagnostics ===")
    print(f"img_size: {img_size}")
    print(f"n_samples: {len(y_true)}")
    print("true_distribution:")
    for k, v in diag["true_distribution"].items():
        print(f"  {k}: {v}")
    print("pred_distribution:")
    for k, v in diag["pred_distribution"].items():
        print(f"  {k}: {v}")
    print("top1_prob_mean:", diag["top1_prob_mean"])
    print("entropy_mean:", diag["entropy_mean"])
    print(f"Saved: {args.out}")
    print("Confusion matrix (rows=true, cols=pred):")
    for i in range(n_classes):
        print(class_names[i], cm[i].tolist())


if __name__ == "__main__":
    main()

