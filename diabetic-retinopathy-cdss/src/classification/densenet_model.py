"""
DenseNet121 Model Architecture for Diabetic Retinopathy Classification.
Comparison architecture evaluated against EfficientNetB3 and ResNet50.
Dense connections are particularly effective at feature reuse for fine retinal microaneurysms.
"""

from typing import Tuple


def build_densenet121(
    input_shape: Tuple[int, int, int] = (300, 300, 3),
    num_classes: int = 5,
    freeze_backbone: bool = True,
    dropout_rate: float = 0.4,
    dense_units: int = 256,
):
    """Build the DenseNet121 DR classification model.

    Args:
        input_shape: (height, width, channels)
        num_classes: Number of disease categories (5)
        freeze_backbone: If True, freezes backbone layers initially
        dropout_rate: Regularization dropout rate
        dense_units: Dense layer projection dimension

    Returns:
        tf.keras.Model
    """
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    inputs = layers.Input(shape=input_shape, name="fundus_image_input")

    base_model = tf.keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs,
    )
    base_model._name = "densenet121_backbone"
    base_model.trainable = not freeze_backbone

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="projection_dense",
    )(x)
    x = layers.Dropout(dropout_rate, name="classifier_dropout")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="dr_predictions",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="DR_DenseNet121")
    return model
