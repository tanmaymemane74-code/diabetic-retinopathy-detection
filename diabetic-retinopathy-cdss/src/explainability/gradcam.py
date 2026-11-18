"""
Explainable AI (XAI) Grad-CAM Module for Diabetic Retinopathy Detection.
Computes Class Activation Maps via TensorFlow GradientTape to reveal
salient retinal regions (microaneurysms, hemorrhages, exudates, neovascularization)
driving the model's multi-class prediction.
"""

import base64
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.config import DR_CLASSES, DR_DESCRIPTIONS, GRADCAM_DIR

logger = logging.getLogger("dr_cdss.gradcam")


class DRGradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) for deep learning vision models."""

    def __init__(
        self,
        model,
        target_conv_layer_name: Optional[str] = None,
    ):
        """Initialize Grad-CAM for a given Keras model.

        Args:
            model: Trained tf.keras.Model
            target_conv_layer_name: Specific name of the final convolutional layer.
                                    If None, automatically identified.
        """
        self.model = model
        self.target_conv_layer_name = target_conv_layer_name or self._find_target_conv_layer()
        logger.info("DRGradCAM initialized with target layer: '%s'", self.target_conv_layer_name)

    def _find_target_conv_layer(self) -> str:
        """Automatically identify the final convolutional layer in the backbone."""
        import tensorflow as tf

        # 1. Search for nested backbone
        for layer in self.model.layers:
            if hasattr(layer, "layers"):
                # Nested functional model (e.g. EfficientNetB3 backbone)
                for sublayer in reversed(layer.layers):
                    if isinstance(sublayer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                        return sublayer.name
                    if "conv" in sublayer.name.lower():
                        return sublayer.name

        # 2. Search top-level layers in reverse
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
            if "conv" in layer.name.lower() and not "dropout" in layer.name.lower():
                return layer.name

        # Default fallback names for common architectures
        for fallback in ["top_conv", "conv5_block3_out", "conv5_block16_2_conv"]:
            for layer in self.model.layers:
                if layer.name == fallback or (hasattr(layer, "layers") and any(l.name == fallback for l in layer.layers)):
                    return fallback

        raise ValueError("Could not automatically locate the target convolutional layer for Grad-CAM.")

    def compute_heatmap(
        self,
        preprocessed_image: np.ndarray,
        pred_index: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float, np.ndarray]:
        """Compute the raw 2D Grad-CAM heatmap for a single preprocessed fundus image.

        Args:
            preprocessed_image: numpy array of shape [H, W, 3] or [1, H, W, 3], normalized [0, 1]
            pred_index: Target class index to explain (0-4). If None, uses top predicted class.

        Returns:
            Tuple of:
                - heatmap: 2D numpy array [H, W] normalized to [0, 1]
                - predicted_class: int (0 to 4)
                - confidence: float (0.0 to 1.0)
                - probabilities: 1D array of all 5 class probabilities
        """
        import tensorflow as tf

        # Ensure batch dimension [1, H, W, C]
        if preprocessed_image.ndim == 3:
            img_tensor = np.expand_dims(preprocessed_image, axis=0)
        else:
            img_tensor = preprocessed_image

        img_tensor = tf.cast(img_tensor, tf.float32)

        # 1. Inspect model structure to create gradient model
        # Check if model has a nested backbone
        backbone_model = None
        for layer in self.model.layers:
            if hasattr(layer, "layers") and any(l.name == self.target_conv_layer_name for l in layer.layers):
                backbone_model = layer
                break

        if backbone_model is not None:
            # Multi-stage model with nested backbone
            target_layer = backbone_model.get_layer(self.target_conv_layer_name)
            # Create sub-model mapping backbone inputs -> target conv outputs
            backbone_conv_model = tf.keras.Model(
                inputs=backbone_model.inputs,
                outputs=[target_layer.output, backbone_model.output],
            )

            # Reconstruct forward pass using GradientTape
            with tf.GradientTape() as tape:
                conv_outputs, backbone_out = backbone_conv_model(img_tensor)
                tape.watch(conv_outputs)

                # Feed backbone_out through remaining top classifier layers
                x = backbone_out
                passed_backbone = False
                for layer in self.model.layers:
                    if layer == backbone_model:
                        passed_backbone = True
                        continue
                    if passed_backbone:
                        x = layer(x)

                preds = x
                if pred_index is None:
                    pred_index = int(tf.argmax(preds[0]))
                class_channel = preds[:, pred_index]

            grads = tape.gradient(class_channel, conv_outputs)
        else:
            # Flat model
            target_layer = self.model.get_layer(self.target_conv_layer_name)
            grad_model = tf.keras.Model(
                inputs=self.model.inputs,
                outputs=[target_layer.output, self.model.output],
            )

            with tf.GradientTape() as tape:
                conv_outputs, preds = grad_model(img_tensor)
                if pred_index is None:
                    pred_index = int(tf.argmax(preds[0]))
                class_channel = preds[:, pred_index]

            grads = tape.gradient(class_channel, conv_outputs)

        # 2. Compute channel-wise importance weights alpha_k = mean(gradients)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # 3. Weighted combination of activation maps
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # 4. Apply ReLU to discard features that decrease target class score
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        heatmap_np = heatmap.numpy()

        probs = preds[0].numpy()
        confidence = float(probs[pred_index])

        return heatmap_np, int(pred_index), confidence, probs

    def generate_overlay(
        self,
        original_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Overlay the 2D Grad-CAM heatmap onto the original retinal image.

        Args:
            original_rgb: RGB uint8 retinal image of shape [H, W, 3]
            heatmap: 2D float array in range [0, 1]
            alpha: Transparency factor for heatmap overlay (0.0 to 1.0)
            colormap: OpenCV colormap constant (COLORMAP_JET by default)

        Returns:
            RGB uint8 overlaid image
        """
        h, w = original_rgb.shape[:2]

        # Resize heatmap to match image dimensions
        resized_heatmap = cv2.resize(heatmap, (w, h))

        # Convert to 8-bit [0, 255]
        heatmap_uint8 = np.uint8(255 * resized_heatmap)

        # Apply colormap (yields BGR)
        colored_heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_heatmap_rgb = cv2.cvtColor(colored_heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Blend
        overlay = cv2.addWeighted(
            original_rgb, 1.0 - alpha, colored_heatmap_rgb, alpha, 0
        )
        return overlay

    def explain_prediction(
        self,
        image_path: Union[str, Path],
        preprocessed_img: np.ndarray,
        save_figure: bool = True,
        output_filename: Optional[str] = None,
    ) -> dict:
        """Full XAI pipeline: generates heatmap, overlay, and clinical interpretation text.

        Args:
            image_path: Path to original retinal image file
            preprocessed_img: Preprocessed array ready for model input [H, W, 3]
            save_figure: Whether to save visualization figure to static/gradcam/
            output_filename: Optional filename for saved figure

        Returns:
            Dictionary with XAI results, class prediction, and clinical commentary.
        """
        # Load original image for visualization
        orig_bgr = cv2.imread(str(image_path))
        if orig_bgr is not None:
            orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        else:
            orig_rgb = (preprocessed_img * 255).astype(np.uint8)

        # Compute heatmap
        heatmap, pred_class, confidence, probs = self.compute_heatmap(preprocessed_img)

        # Resize original for consistent visualization
        h, w = preprocessed_img.shape[:2]
        orig_resized = cv2.resize(orig_rgb, (w, h))

        # Create overlay
        overlay = self.generate_overlay(orig_resized, heatmap, alpha=0.45)

        # Clinical interpretation
        grade_name = DR_CLASSES.get(pred_class, f"Grade {pred_class}")
        grade_desc = DR_DESCRIPTIONS.get(pred_class, "")

        explanation_text = (
            f"The deep learning model classified this retinal fundus photograph as "
            f"'{grade_name}' with {confidence * 100:.1f}% confidence. "
            f"The Grad-CAM heatmap highlights specific vascular and macular regions "
            f"that most strongly influenced the neural network's activation. "
            f"Clinical Note: Grad-CAM represents predictive model attention and does "
            f"not independently replace qualified ophthalmologic examination."
        )

        saved_path = None
        if save_figure:
            out_dir = Path(GRADCAM_DIR)
            out_dir.mkdir(parents=True, exist_ok=True)
            if output_filename is None:
                stem = Path(image_path).stem
                output_filename = f"gradcam_{stem}_grade{pred_class}.png"
            saved_path = str(out_dir / output_filename)

            # Generate 3-panel figure
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            fig.suptitle(
                f"Explainable AI (Grad-CAM): {grade_name} ({confidence * 100:.1f}% Confidence)",
                fontsize=14,
                fontweight="bold",
            )

            # Panel 1: Original Fundus
            axes[0].imshow(orig_resized)
            axes[0].set_title("1. Original Retinal Fundus", fontsize=11)
            axes[0].axis("off")

            # Panel 2: Grad-CAM Activation Heatmap
            im2 = axes[1].imshow(heatmap, cmap="jet")
            axes[1].set_title(f"2. Grad-CAM Activation Map\n(Layer: {self.target_conv_layer_name})", fontsize=11)
            axes[1].axis("off")
            plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

            # Panel 3: Heatmap Overlay
            axes[2].imshow(overlay)
            axes[2].set_title("3. Retinal Overlay\n(Model Attention Focus)", fontsize=11)
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(saved_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("Grad-CAM visualization saved to: %s", saved_path)

        b64_str = ""
        if saved_path and Path(saved_path).exists():
            with open(saved_path, "rb") as f:
                b64_str = base64.b64encode(f.read()).decode("utf-8")

        return {
            "predicted_grade": pred_class,
            "predicted_grade_label": grade_name,
            "confidence": round(confidence, 4),
            "class_probabilities": [round(float(p), 4) for p in probs],
            "target_layer": self.target_conv_layer_name,
            "gradcam_image_path": saved_path,
            "heatmap_base64": b64_str,
            "clinical_description": grade_desc,
            "xai_explanation": explanation_text,
        }
