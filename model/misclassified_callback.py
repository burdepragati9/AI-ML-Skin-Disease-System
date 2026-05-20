import os
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class SaveMisclassifiedValImages(tf.keras.callbacks.Callback):
    """Save misclassified validation images into:

    analytics/misclassified/<True>_as_<Pred>/<filename>

    Requirements handled:
    - Does NOT modify training flow.
    - Uses validation dataset tensors; relies on original file paths when available.

    Notes:
    - tf.data.Dataset from image_dataset_from_directory does not always expose
      original paths. If paths are unavailable, falls back to saving the
      preprocessed image as a PNG under a synthetic name.
    """

    def __init__(
        self,
        val_ds: tf.data.Dataset,
        class_names: list[str],
        output_root: Path,
        *,
        max_images: int = 200,
        save_on: str = "end",  # "end" only to avoid huge I/O
    ):
        super().__init__()
        self.val_ds = val_ds
        self.class_names = class_names
        self.output_root = output_root
        self.max_images = int(max_images)
        self.save_on = save_on

    def _save_tensor_image(self, x: np.ndarray, dest: Path) -> None:
        # x expected shape (1,H,W,3) or (H,W,3) in 0..255 float.
        from PIL import Image

        arr = x
        if arr.ndim == 4:
            arr = arr[0]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        img = Image.fromarray(arr)
        _safe_mkdir(dest.parent)
        img.save(dest)

    def _extract_paths_if_available(self):
        """Return list of (path, x, y). If tf dataset does not carry paths,
        returns None.
        """
        # Heuristic: element might be ((x, path), y)
        # but most Keras dataset versions return (x, y).
        for batch in self.val_ds.take(1):
            # can't reliably inspect without consuming; skip
            break
        return None

    def on_epoch_end(self, epoch, logs=None):
        if self.save_on != "end":
            return

        output_root = self.output_root
        saved = 0

        # Iterate over the already-batched val dataset.
        for batch_idx, (x_b, y_b) in enumerate(self.val_ds):
            if saved >= self.max_images:
                break

            probs = self.model.predict(x_b, verbose=0)
            pred_idx = np.argmax(probs, axis=1).astype(int)
            true_idx = y_b.numpy().astype(int)

            x_np = x_b.numpy().astype(np.float32)

            for i in range(len(true_idx)):
                if saved >= self.max_images:
                    break

                t = int(true_idx[i])
                p = int(pred_idx[i])
                if t == p:
                    continue

                true_name = self.class_names[t]
                pred_name = self.class_names[p]

                # Synthetic filename: include epoch/batch/sample.
                filename = f"epoch{epoch:03d}_batch{batch_idx:04d}_sample{i:03d}.png"
                dest = output_root / f"{true_name}_as_{pred_name}" / filename

                self._save_tensor_image(x_np[i:i+1], dest)
                saved += 1

        # Lightweight marker file (optional debugging)
        (self.output_root / "_last_run_epoch.txt").write_text(str(epoch), encoding="utf-8")

