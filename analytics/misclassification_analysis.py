import csv
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "skin_model.keras"
ACTIVE_CLASSES_PATH = PROJECT_ROOT / "model" / "active_classes.json"
CROPPED_DATASET_PATH = PROJECT_ROOT / "CroppedData"

MISCLASSIFIED_ROOT = PROJECT_ROOT / "analytics" / "misclassified"
MISCLASSIFIED_REPORT_CSV = PROJECT_ROOT / "analytics" / "misclassified_report.csv"


@dataclass(frozen=True)
class MisclassificationRow:
    image_path: str
    true_class: str
    predicted_class: str
    confidence_score: float
    true_idx: int
    pred_idx: int


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _model_img_size(model: tf.keras.Model, default: int = 160) -> int:
    try:
        ishape = model.input_shape
        if isinstance(ishape, tuple) and len(ishape) >= 3:
            h = ishape[1]
            if h:
                return int(h)
    except Exception:
        pass
    return default


def _load_active_class_names() -> List[str]:
    if not ACTIVE_CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {ACTIVE_CLASSES_PATH}. Re-train and re-run diagnostics so label ordering matches the trained Dense layer."
        )
    payload = json.loads(ACTIVE_CLASSES_PATH.read_text(encoding="utf-8"))
    class_names = payload.get("class_names", [])
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("active_classes.json has invalid/empty class_names")
    return [str(x) for x in class_names]


def _list_dataset_samples(dataset_root: Path, class_names: List[str]) -> List[Tuple[Path, int]]:
    """Return list of (path, class_index) for all images under CroppedData/<class>.

    IMPORTANT:
    - Assumes folder names match class_names ordering (source of truth).
    - Does not alter dataset loading logic; this is only for diagnostics export.
    """
    exts = {".jpg", ".jpeg", ".png"}
    samples: List[Tuple[Path, int]] = []

    name_to_idx = {name: i for i, name in enumerate(class_names)}
    for class_dir in dataset_root.iterdir():
        if not class_dir.is_dir():
            continue
        name = class_dir.name
        if name not in name_to_idx:
            continue
        idx = name_to_idx[name]
        for p in class_dir.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                samples.append((p, idx))
    return samples


def _preprocess_image_for_model(image_path: Path, img_size: int) -> np.ndarray:
    # The model contains its own Rescaling layer (see model/train.py).
    img = tf.keras.utils.load_img(image_path, target_size=(img_size, img_size))
    x = tf.keras.utils.img_to_array(img).astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return x


def _confidence_score_from_probs(probs: np.ndarray, pred_idx: int) -> float:
    return float(probs[int(pred_idx)])


def _confusion_folder_name(true_class: str, pred_class: str) -> str:
    # Requirement: true_<ActualClass>*pred*<PredictedClass>
    # Use underscores exactly as specified.
    safe_true = str(true_class).strip().replace(" ", "_")
    safe_pred = str(pred_class).strip().replace(" ", "_")
    return f"true_{safe_true}_pred_{safe_pred}"


def analyze_and_export_misclassifications(
    *,
    max_images: int = 0,
    clear_output: bool = True,
    batch_size: int = 32,
) -> Dict[str, object]:
    """After diagnostics evaluation, export per-image misclassification CSV and copy misclassified images.

    - Writes misclassified images to analytics/misclassified/<true>_pred_<pred>/...
    - Writes analytics/misclassified_report.csv with required columns.
    - Prints summary.

    Returns a summary dict.
    """

    class_names = _load_active_class_names()
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    n_classes = len(class_names)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if clear_output and MISCLASSIFIED_ROOT.exists():
        shutil.rmtree(MISCLASSIFIED_ROOT)
    _safe_mkdir(MISCLASSIFIED_ROOT)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    img_size = _model_img_size(model)

    samples = _list_dataset_samples(CROPPED_DATASET_PATH, class_names)
    if max_images and max_images > 0:
        samples = samples[: int(max_images)]

    if not samples:
        raise RuntimeError(f"No samples found under {CROPPED_DATASET_PATH} for {class_names}")

    y_true: List[int] = []
    y_pred: List[int] = []
    rows: List[MisclassificationRow] = []

    # Confusion matrix (rows=true, cols=pred)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    total = len(samples)
    batches = math.ceil(total / max(1, int(batch_size)))

    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        batch = samples[start:end]

        paths = [p for (p, _) in batch]
        true_idx = np.asarray([i for (_, i) in batch], dtype=np.int64)

        x_batch = np.concatenate(
            [_preprocess_image_for_model(p, img_size=img_size) for p in paths], axis=0
        )
        probs = model.predict(x_batch, verbose=0)
        pred_idx = np.argmax(probs, axis=1).astype(np.int64)

        for i in range(len(batch)):
            t = int(true_idx[i])
            p = int(pred_idx[i])
            cm[t, p] += 1

            if t != p:
                rows.append(
                    MisclassificationRow(
                        image_path=str(paths[i]),
                        true_class=class_names[t],
                        predicted_class=class_names[p],
                        confidence_score=_confidence_score_from_probs(probs[i], p),
                        true_idx=t,
                        pred_idx=p,
                    )
                )

        y_true.extend(true_idx.tolist())
        y_pred.extend(pred_idx.tolist())

    # Write CSV report with required columns.
    with MISCLASSIFIED_REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "true_class", "predicted_class", "confidence_score"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "image_path": r.image_path,
                    "true_class": r.true_class,
                    "predicted_class": r.predicted_class,
                    "confidence_score": f"{r.confidence_score:.8f}",
                }
            )

    # Copy misclassified images into required confusion folders.
    copied = 0
    for r in rows:
        folder_name = _confusion_folder_name(r.true_class, r.predicted_class)
        dest_dir = MISCLASSIFIED_ROOT / folder_name
        _safe_mkdir(dest_dir)
        dest_path = dest_dir / Path(r.image_path).name
        if dest_path.exists():
            continue
        shutil.copy2(r.image_path, dest_path)
        copied += 1

    # Summary prints
    total_misclassified = len(rows)

    # Top confusion pairs: true != pred, sorted by count
    pairs: List[Tuple[str, str, int]] = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:
                continue
            count = int(cm[i, j])
            if count <= 0:
                continue
            pairs.append((class_names[i], class_names[j], count))
    pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = pairs[:10]

    # confusion percentage per class:
    # percent of samples of each true class that are misclassified (row-wise)
    per_class_misclf_pct: Dict[str, float] = {}
    for i in range(n_classes):
        row_total = int(cm[i, :].sum())
        if row_total <= 0:
            per_class_misclf_pct[class_names[i]] = 0.0
        else:
            correct = int(cm[i, i])
            per_class_misclf_pct[class_names[i]] = float((row_total - correct) / row_total * 100.0)

    print("\n=== Misclassification analysis ===")
    print(f"Total misclassified images: {total_misclassified}")
    print(f"Copied images: {copied}")
    print(f"Report CSV: {MISCLASSIFIED_REPORT_CSV}")

    print("\nTop confusion pairs (true -> predicted):")
    for t, p, c in top_pairs:
        print(f"  {t} -> {p}: {c}")

    print("\nConfusion percentage per class (row-wise misclassification %):")
    for cls in class_names:
        print(f"  {cls}: {per_class_misclf_pct[cls]:.2f}%")

    # Extra targeted review (Psoriasis <-> Tinea)
    if "Psoriasis" in name_to_idx and "Tinea" in name_to_idx:
        i_ps = name_to_idx["Psoriasis"]
        i_ti = name_to_idx["Tinea"]
        ps_to_ti = int(cm[i_ps, i_ti])
        ti_to_ps = int(cm[i_ti, i_ps])
        total_ps = int(cm[i_ps, :].sum())
        total_ti = int(cm[i_ti, :].sum())
        ps_to_ti_pct = (ps_to_ti / total_ps * 100.0) if total_ps else 0.0
        ti_to_ps_pct = (ti_to_ps / total_ti * 100.0) if total_ti else 0.0

        print("\n=== Targeted review: Psoriasis vs Tinea confusion ===")
        print(
            f"  Psoriasis -> Tinea: {ps_to_ti} ({ps_to_ti_pct:.2f}% of Psoriasis row)"
        )
        print(
            f"  Tinea -> Psoriasis: {ti_to_ps} ({ti_to_ps_pct:.2f}% of Tinea row)"
        )
        print(
            "\nLikely causes (from typical derm-ML failure modes):\n"
            "  1) Label noise / inter-observer variability: clinically, psoriasis plaques and tinea (especially annular / chronic plaque-like tinea) can look similar.\n"
            "  2) Class overlap in dataset: if some images are borderline or mislabeled, the model will learn conflicting cues.\n"
            "  3) Preprocessing/inference mismatch: check that the same resizing/color handling is used for training/diagnostics/inference (this pipeline uses model_img_size + model Rescaling).\n"
            "  4) Background/lighting artifacts: MobileNet can pick up acquisition artifacts correlated with class folder rather than lesion morphology.\n"
            "  5) Under-represented subtypes: e.g., nail/hand/plantar tinea variants may resemble psoriasis patterns; psoriasis variants (guttate/inverse) may resemble tinea edges/rings.\n"
            "\nSuggested improvements based on the confusion matrix:\n"
            "  - Audit misclassified images in true_Psoriasis_pred_Tinea and true_Tinea_pred_Psoriasis folders; relabel borderline cases.\n"
            "  - Add more diverse samples for the confusing subtypes (plantar/digits/hand for tinea; inverse/guttate/chronic plaque for psoriasis).\n"
            "  - Consider adding a dedicated preprocessing step to normalize color/contrast consistently (without changing existing training logic here).\n"
            "  - Train with stronger regularization / reduce augmentation severity if overfitting to artifacts is observed (verify by per-class confidence: low-confidence widespread mistakes suggest overlap, high-confidence mistakes suggest systematic bias).\n"
        )

    return {
        "total_misclassified": total_misclassified,
        "copied": copied,
        "report_csv": str(MISCLASSIFIED_REPORT_CSV),
        "misclassified_root": str(MISCLASSIFIED_ROOT),
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    # Optional env vars / overrides can be added later; keep stable defaults for production.
    analyze_and_export_misclassifications()


if __name__ == "__main__":
    # Allow running standalone
    main()

