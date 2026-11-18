"""
End-to-End Deep Learning Training Pipeline for Diabetic Retinopathy.
Supports EfficientNetB3, ResNet50, DenseNet121, two-phase transfer learning,
Focal Loss / Class-Weighted Loss, and automatic metrics generation.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.classification.densenet_model import build_densenet121
from src.classification.efficientnet_model import build_efficientnet_b3, unfreeze_efficientnet_b3
from src.classification.loss_functions import get_loss_function
from src.classification.metrics import calculate_clinical_metrics
from src.classification.resnet_model import build_resnet50
from src.preprocessing.augmentation import DRAugmentor
from src.utils.config import (
    BATCH_SIZE,
    DR_CLASSES,
    EPOCHS,
    IMAGE_SIZE,
    LEARNING_RATE,
    MODELS_DIR,
    REPORTS_DIR,
)

logger = logging.getLogger("dr_cdss.trainer")


class DRModelTrainer:
    """Manages model building, two-stage fine-tuning, and metric evaluation."""

    def __init__(
        self,
        model_name: str = "efficientnet_b3",
        input_size: Tuple[int, int] = IMAGE_SIZE,
        num_classes: int = len(DR_CLASSES),
        loss_type: str = "focal",
        batch_size: int = BATCH_SIZE,
    ):
        self.model_name = model_name.lower().strip()
        self.input_size = input_size
        self.input_shape = (input_size[0], input_size[1], 3)
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.batch_size = batch_size
        self.model = None
        self.history = None

    def build_architecture(self, freeze_backbone: bool = True):
        """Instantiate selected neural network architecture."""
        logger.info("Building %s (freeze_backbone=%s)...", self.model_name, freeze_backbone)
        if "efficientnet" in self.model_name:
            self.model = build_efficientnet_b3(
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                freeze_backbone=freeze_backbone,
            )
        elif "resnet" in self.model_name:
            self.model = build_resnet50(
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                freeze_backbone=freeze_backbone,
            )
        elif "densenet" in self.model_name:
            self.model = build_densenet121(
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                freeze_backbone=freeze_backbone,
            )
        else:
            raise ValueError(f"Unknown architecture: {self.model_name}")
        return self.model

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        epochs_phase1: int = 10,
        epochs_phase2: int = 15,
        lr_phase1: float = 1e-3,
        lr_phase2: float = 1e-5,
        class_weights: Optional[Dict[int, float]] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train model using two-stage transfer learning."""
        import tensorflow as tf

        if save_path is None:
            save_path = str(Path(MODELS_DIR) / f"{self.model_name}_best.keras")

        # 1. Build initial model with frozen backbone
        self.build_architecture(freeze_backbone=True)

        # 2. Setup Loss & Optimizer
        loss_fn = get_loss_function(self.loss_type, class_weights=class_weights)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_phase1)

        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=["accuracy"],
        )

        # 3. Setup Augmentation & Data Generators
        train_df = train_df.copy()
        val_df = val_df.copy()
        train_df["diagnosis_str"] = train_df["diagnosis"].astype(str)
        val_df["diagnosis_str"] = val_df["diagnosis"].astype(str)
        classes = [str(i) for i in range(self.num_classes)]

        augmentor = DRAugmentor()
        train_gen = augmentor.create_train_generator(
            train_df,
            x_col="image_path",
            y_col="diagnosis_str",
            classes=classes,
            target_size=self.input_size,
            batch_size=self.batch_size,
            class_mode="categorical",
        )
        val_gen = augmentor.create_val_generator(
            val_df,
            x_col="image_path",
            y_col="diagnosis_str",
            classes=classes,
            target_size=self.input_size,
            batch_size=self.batch_size,
            class_mode="categorical",
        )

        # Callbacks
        callbacks_phase1 = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=save_path,
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
        ]

        logger.info("=== Starting Phase 1: Training Classification Head ===")
        hist1 = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs_phase1,
            callbacks=callbacks_phase1,
            verbose=1,
        )

        # Phase 2: Unfreeze and Fine-tune (if applicable)
        if epochs_phase2 > 0 and "efficientnet" in self.model_name:
            logger.info("=== Starting Phase 2: Fine-Tuning Backbone ===")
            self.model = unfreeze_efficientnet_b3(self.model, fine_tune_layers=50)

            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr_phase2),
                loss=loss_fn,
                metrics=["accuracy"],
            )

            callbacks_phase2 = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=6, restore_best_weights=True, verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=save_path,
                    monitor="val_accuracy",
                    save_best_only=True,
                    verbose=1,
                ),
            ]

            hist2 = self.model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=epochs_phase2,
                callbacks=callbacks_phase2,
                verbose=1,
            )
            # Combine histories
            combined_history = {
                k: hist1.history.get(k, []) + hist2.history.get(k, [])
                for k in hist1.history.keys()
            }
            self.history = combined_history
        else:
            self.history = hist1.history

        logger.info("Training complete. Best model checkpoint saved to: %s", save_path)
        self.plot_training_curves()
        return self.history

    def evaluate(
        self,
        test_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Generate predictions and compute clinical evaluation metrics."""
        test_df = test_df.copy()
        test_df["diagnosis_str"] = test_df["diagnosis"].astype(str)
        classes = [str(i) for i in range(self.num_classes)]
        augmentor = DRAugmentor()
        test_gen = augmentor.create_val_generator(
            test_df,
            x_col="image_path",
            y_col="diagnosis_str",
            classes=classes,
            target_size=self.input_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=False,
        )

        logger.info("Running evaluation predictions on %d test samples...", len(test_df))
        predictions = self.model.predict(test_gen, verbose=1)

        y_true = test_df["diagnosis"].values
        class_names = [DR_CLASSES.get(i, f"Grade {i}") for i in range(self.num_classes)]

        metrics = calculate_clinical_metrics(y_true, predictions, class_names=class_names)
        logger.info(
            "Evaluation complete. Accuracy: %.4f | QWK: %.4f | Referable DR Sens: %.4f Spec: %.4f",
            metrics["accuracy"],
            metrics["quadratic_weighted_kappa"],
            metrics["referable_dr_screening"]["sensitivity"],
            metrics["referable_dr_screening"]["specificity"],
        )
        return metrics

    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot loss and accuracy training curves and save as image."""
        if not self.history:
            return

        if save_path is None:
            figures_dir = Path(REPORTS_DIR) / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(figures_dir / f"{self.model_name}_training_curves.png")

        epochs = range(1, len(self.history["loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy
        axes[0].plot(epochs, self.history["accuracy"], "b-o", label="Training Accuracy")
        if "val_accuracy" in self.history:
            axes[0].plot(epochs, self.history["val_accuracy"], "r--s", label="Validation Accuracy")
        axes[0].set_title(f"{self.model_name.upper()} - Accuracy vs. Epochs", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # Loss
        axes[1].plot(epochs, self.history["loss"], "b-o", label="Training Loss")
        if "val_loss" in self.history:
            axes[1].plot(epochs, self.history["val_loss"], "r--s", label="Validation Loss")
        axes[1].set_title(f"{self.model_name.upper()} - Loss vs. Epochs", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Training curves saved to %s", save_path)
