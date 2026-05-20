import tensorflow as tf
from tensorflow.keras import layers

from keras.saving import register_keras_serializable


@register_keras_serializable()
class RandomErasing2D(layers.Layer):
    """Deterministic-shape-safe random erasing layer.

    Purpose: avoid anonymous Lambda layers inside the model graph so
    `model.save()` / `keras.models.load_model()` works without requiring
    `safe_mode=False`.

    Behavior:
      - With probability `p`, zero-out random pixels of the input.
      - Pixels are selected independently with Bernoulli(erase_prob).

    Notes:
      - This is an approximation of RandomErasing (cutout).
      - Keeps tensor shape unchanged.
    """

    def __init__(
        self,
        p: float = 0.02,
        erase_prob: float = 0.02,
        fill_value: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.p = float(p)
        self.erase_prob = float(erase_prob)
        self.fill_value = float(fill_value)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "p": self.p,
                "erase_prob": self.erase_prob,
                "fill_value": self.fill_value,
            }
        )
        return cfg

    def call(self, inputs, training=None):
        # `tf.keras.backend.learning_phase()` is not available in newer Keras.
        # Rely on the `training` argument provided by Keras; if it's missing,
        # treat it as True to keep augmentation active during training.
        if training is None:
            training = True


        def _apply():
            z = tf.convert_to_tensor(inputs)

            # Uniform mask per element.
            # Use erase_prob to control sparsity.
            m = tf.random.uniform(tf.shape(z), 0.0, 1.0)
            erase_mask = m <= self.erase_prob

            filled = tf.fill(tf.shape(z), tf.cast(self.fill_value, z.dtype))
            z_erased = tf.where(erase_mask, filled, z)
            return z_erased

        def _noapply():
            return tf.convert_to_tensor(inputs)

        # Only apply erase with probability p.
        apply_gate = tf.random.uniform([], 0.0, 1.0) <= self.p
        if training is False:
            return _noapply()

        # Use tf.cond to keep it graph-compatible.
        return tf.cond(apply_gate, _apply, _noapply)

