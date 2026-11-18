"""
Unit tests for Deep Learning Models, Loss Functions, Metrics, and Grad-CAM.
"""

from pathlib import Path
import numpy as np
import pytest

from src.classification.densenet_model import build_densenet121
from src.classification.efficientnet_model import build_efficientnet_b3
from src.classification.loss_functions import ClassWeightedCrossEntropy, FocalLoss, get_loss_function
from src.classification.metrics import calculate_clinical_metrics, compute_quadratic_weighted_kappa
from src.classification.resnet_model import build_resnet50
from src.explainability.gradcam import DRGradCAM


class TestModelArchitectures:
    """Tests for model building and output shapes."""

    def test_build_efficientnet_b3(self):
        model = build_efficientnet_b3(input_shape=(300, 300, 3), num_classes=5, freeze_backbone=True)
        assert model is not None
        assert model.input_shape == (None, 300, 300, 3)
        assert model.output_shape == (None, 5)

    def test_build_resnet50(self):
        model = build_resnet50(input_shape=(300, 300, 3), num_classes=5, freeze_backbone=True)
        assert model is not None
        assert model.input_shape == (None, 300, 300, 3)
        assert model.output_shape == (None, 5)

    def test_build_densenet121(self):
        model = build_densenet121(input_shape=(300, 300, 3), num_classes=5, freeze_backbone=True)
        assert model is not None
        assert model.input_shape == (None, 300, 300, 3)
        assert model.output_shape == (None, 5)


class TestLossFunctions:
    """Tests for Focal Loss and Class-Weighted Crossentropy."""

    def test_focal_loss_callable(self):
        import tensorflow as tf

        loss_fn = FocalLoss(gamma=2.0)
        y_true = tf.constant([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])
        y_pred = tf.constant([[0.9, 0.05, 0.02, 0.02, 0.01], [0.1, 0.7, 0.1, 0.05, 0.05]])
        loss_val = loss_fn(y_true, y_pred)
        assert float(loss_val) > 0.0

    def test_class_weighted_loss(self):
        import tensorflow as tf

        weights = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0, 4: 2.5}
        loss_fn = ClassWeightedCrossEntropy(class_weights=weights)
        y_true = tf.constant([[0.0, 0.0, 1.0, 0.0, 0.0]])
        y_pred = tf.constant([[0.1, 0.1, 0.6, 0.1, 0.1]])
        val = loss_fn(y_true, y_pred)
        assert float(val) > 0.0


class TestClinicalMetrics:
    """Tests for multi-class and referable DR clinical metrics."""

    def test_quadratic_weighted_kappa(self):
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 3, 4, 0, 1, 2])
        qwk = compute_quadratic_weighted_kappa(y_true, y_pred)
        assert np.isclose(qwk, 1.0)

    def test_calculate_clinical_metrics(self):
        y_true = np.array([0, 1, 2, 3, 4, 0, 2, 3, 4, 1])
        # 10 samples x 5 classes
        y_probs = np.zeros((10, 5))
        for i, t in enumerate(y_true):
            y_probs[i, t] = 0.8
            y_probs[i, (t + 1) % 5] = 0.2

        metrics = calculate_clinical_metrics(y_true, y_probs)
        assert "accuracy" in metrics
        assert "quadratic_weighted_kappa" in metrics
        assert "referable_dr_screening" in metrics
        assert metrics["accuracy"] >= 0.80
        assert metrics["referable_dr_screening"]["sensitivity"] > 0.5


class TestGradCAM:
    """Tests for Explainable AI Grad-CAM generation."""

    def test_gradcam_execution(self, sample_image_path):
        from src.preprocessing.preprocessor import DRImagePreprocessor

        prep = DRImagePreprocessor(target_size=(300, 300))
        processed = prep.preprocess_single(sample_image_path)
        assert processed is not None

        # Build model
        model = build_efficientnet_b3(input_shape=(300, 300, 3), num_classes=5, freeze_backbone=False)
        gradcam = DRGradCAM(model=model)

        heatmap, pred_class, conf, probs = gradcam.compute_heatmap(processed)
        assert heatmap.ndim == 2
        assert 0 <= pred_class <= 4
        assert 0.0 <= conf <= 1.0
        assert len(probs) == 5

        # Test overlay
        raw_rgb = (processed * 255).astype(np.uint8)
        overlay = gradcam.generate_overlay(raw_rgb, heatmap)
        assert overlay.shape == raw_rgb.shape
