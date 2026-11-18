"""
Loss functions for Diabetic Retinopathy multi-class classification.
Implements Categorical Crossentropy, Class-Weighted Loss, and Focal Loss.
Registered as Keras serializable for model checkpoints and saving.
"""

from typing import Dict, Optional, Union
import numpy as np

try:
    import keras
    _KERAS_SERIALIZABLE = True
except ImportError:
    _KERAS_SERIALIZABLE = False


class _BaseLoss:
    """Base class fallback when keras is not imported directly."""
    pass


BaseParent = keras.losses.Loss if _KERAS_SERIALIZABLE else _BaseLoss


def _register(cls):
    if _KERAS_SERIALIZABLE:
        return keras.saving.register_keras_serializable(package="dr_cdss")(cls)
    return cls


@_register
class FocalLoss(BaseParent):
    """Categorical Focal Loss for multi-class imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights easy examples and focuses gradient updates on hard-to-classify
    retinal lesion samples (e.g. Grade 1 vs Grade 2).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[float, list]] = None,
        name: str = "focal_loss",
        **kwargs,
    ):
        if _KERAS_SERIALIZABLE:
            super().__init__(name=name, **kwargs)
        self.gamma = float(gamma)
        self.alpha = alpha

    def call(self, y_true, y_pred):
        import tensorflow as tf

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Cross entropy
        ce = -y_true * tf.math.log(y_pred)

        # Focal weighting factor: (1 - p_t)^gamma
        weight = tf.pow(1.0 - y_pred, self.gamma)

        focal_loss = weight * ce

        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple, np.ndarray)):
                alpha_tensor = tf.constant(self.alpha, dtype=tf.float32)
                focal_loss = focal_loss * alpha_tensor
            else:
                focal_loss = self.alpha * focal_loss

        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

    def get_config(self):
        config = {}
        if _KERAS_SERIALIZABLE:
            config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@_register
class ClassWeightedCrossEntropy(BaseParent):
    """Categorical Crossentropy weighted by inverse class frequency."""

    def __init__(
        self,
        class_weights: Optional[Dict[int, float]] = None,
        name: str = "class_weighted_crossentropy",
        **kwargs,
    ):
        if _KERAS_SERIALIZABLE:
            super().__init__(name=name, **kwargs)
        self.class_weights = class_weights or {i: 1.0 for i in range(5)}

    def call(self, y_true, y_pred):
        import tensorflow as tf

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Weights tensor shape [num_classes]
        num_classes = len(self.class_weights)
        weights_arr = [float(self.class_weights.get(i, 1.0)) for i in range(num_classes)]
        weights_tensor = tf.constant(weights_arr, dtype=tf.float32)

        # Weight per sample: sum(y_true * weights)
        sample_weights = tf.reduce_sum(y_true * weights_tensor, axis=-1)

        # Standard cross entropy per sample
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

        weighted_loss = ce * sample_weights
        return tf.reduce_mean(weighted_loss)

    def get_config(self):
        config = {}
        if _KERAS_SERIALIZABLE:
            config = super().get_config()
        config.update({
            "class_weights": self.class_weights,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def get_loss_function(
    loss_name: str = "focal",
    class_weights: Optional[Dict[int, float]] = None,
    gamma: float = 2.0,
):
    """Factory function returning the specified loss callable."""
    loss_lower = loss_name.lower().strip()

    if loss_lower == "focal":
        alpha = list(class_weights.values()) if class_weights else None
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif loss_lower in ("class_weighted", "weighted"):
        if not class_weights:
            class_weights = {i: 1.0 for i in range(5)}
        return ClassWeightedCrossEntropy(class_weights=class_weights)
    elif loss_lower in ("categorical_crossentropy", "cce", "ce"):
        import tensorflow as tf
        return tf.keras.losses.CategoricalCrossentropy()
    else:
        raise ValueError(
            f"Unsupported loss '{loss_name}'. Choose 'focal', 'class_weighted', or 'categorical_crossentropy'."
        )
