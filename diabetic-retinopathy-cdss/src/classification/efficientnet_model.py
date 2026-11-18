"""
EfficientNetB3 Model Architecture for Diabetic Retinopathy Classification.
Primary deep learning backbone combining ImageNet transfer learning,
Global Average Pooling, Dropout regularization, and 5-class softmax output.
"""

from typing import Tuple


def build_efficientnet_b3(
    input_shape: Tuple[int, int, int] = (300, 300, 3),
    num_classes: int = 5,
    freeze_backbone: bool = True,
    dropout_rate: float = 0.4,
    dense_units: int = 256,
):
    """Build the primary EfficientNetB3 DR classification model.

    Architecture:
        Input (300, 300, 3)
            |
        EfficientNetB3 Backbone (Pre-trained ImageNet)
            |
        GlobalAveragePooling2D
            |
        BatchNormalization
            |
        Dense(256, ReLU, L2 Regularization)
            |
        Dropout(0.4)
            |
        Dense(5, Softmax) -> DR Grade 0-4

    Args:
        input_shape: Tuple representing (height, width, channels)
        num_classes: Number of disease categories (5 for DR)
        freeze_backbone: If True, freezes all backbone layers for Phase 1 head training
        dropout_rate: Dropout rate for regularization
        dense_units: Units in classification projection layer

    Returns:
        Compiled tf.keras.Model
    """
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    # 1. Input Layer
    inputs = layers.Input(shape=input_shape, name="fundus_image_input")

    # 2. EfficientNetB3 Backbone
    base_model = tf.keras.applications.EfficientNetB3(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs,
    )
    base_model._name = "efficientnet_b3_backbone"

    # Freeze backbone if requested for initial warmup/head training
    base_model.trainable = not freeze_backbone

    # 3. Classification Head
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

    # 4. Output Layer
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="dr_predictions",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="DR_EfficientNetB3")
    return model


def unfreeze_efficientnet_b3(
    model,
    fine_tune_layers: int = 50,
    learning_rate: float = 1e-5,
):
    """Unfreeze top layers of the EfficientNetB3 backbone for Phase 2 fine-tuning.

    Args:
        model: Trained or partially trained Keras model
        fine_tune_layers: Number of uppermost backbone layers to unfreeze
        learning_rate: Reduced learning rate for gentle fine-tuning
    """
    import tensorflow as tf

    # Find the backbone layer
    backbone = None
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            backbone = layer
            break

    if backbone is None:
        # Backbone is flattened into the model
        total_layers = len(model.layers)
        start_at = max(0, total_layers - fine_tune_layers)
        for i, layer in enumerate(model.layers):
            layer.trainable = (i >= start_at) and not isinstance(layer, tf.keras.layers.BatchNormalization)
    else:
        backbone.trainable = True
        total_sublayers = len(backbone.layers)
        start_at = max(0, total_sublayers - fine_tune_layers)
        for i, layer in enumerate(backbone.layers):
            # Keep BatchNormalization frozen to prevent mean/var collapse
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = (i >= start_at)

    return model
