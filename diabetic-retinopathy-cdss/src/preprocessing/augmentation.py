"""
Data Augmentation Module for Diabetic Retinopathy Classification.

Provides training and validation data generators with appropriate
augmentation strategies for fundus images, using TensorFlow / Keras.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import (  # type: ignore[attr-defined]
        ImageDataGenerator,
        load_img,
        img_to_array,
    )
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False
    ImageDataGenerator = None
    load_img = None
    img_to_array = None

logger = logging.getLogger(__name__)


class DRAugmentor:
    """Data augmentation helper for diabetic retinopathy fundus images.

    Wraps ``tf.keras.preprocessing.image.ImageDataGenerator`` to provide
    sensible defaults for training augmentation (rotation, flips, brightness
    jitter, zoom, shifts) and a no-augmentation path for validation.

    Attributes:
        train_datagen: ImageDataGenerator for training data.
        val_datagen: ImageDataGenerator for validation data.
    """

    def __init__(self) -> None:
        """Initialize training and validation data generators."""
        if not _TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for DRAugmentor. "
                "Install it with: pip install tensorflow>=2.15.0"
            )
        self.train_datagen: ImageDataGenerator = self._build_train_datagen()
        self.val_datagen: ImageDataGenerator = self._build_val_datagen()
        logger.info("DRAugmentor initialized.")

    # ------------------------------------------------------------------
    # Augmentation configurations
    # ------------------------------------------------------------------

    @staticmethod
    def get_train_augmentation() -> Dict[str, Any]:
        """Return the augmentation configuration dict for training.

        Returns:
            Dict of keyword arguments suitable for
            ``ImageDataGenerator(**config)``.
        """
        return {
            "rotation_range": 20,
            "horizontal_flip": True,
            "brightness_range": [0.9, 1.1],
            "zoom_range": 0.1,
            "width_shift_range": 0.05,
            "height_shift_range": 0.05,
            "fill_mode": "constant",
            "cval": 0,
            "rescale": 1.0 / 255.0,
        }

    @staticmethod
    def get_val_augmentation() -> Dict[str, Any]:
        """Return the augmentation configuration dict for validation.

        Only rescaling is applied — no geometric or photometric augmentation.

        Returns:
            Dict of keyword arguments suitable for
            ``ImageDataGenerator(**config)``.
        """
        return {
            "rescale": 1.0 / 255.0,
        }

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_train_datagen(self) -> ImageDataGenerator:
        """Build the training ``ImageDataGenerator``."""
        config = self.get_train_augmentation()
        logger.debug("Training augmentation config: %s", config)
        return ImageDataGenerator(**config)

    def _build_val_datagen(self) -> ImageDataGenerator:
        """Build the validation ``ImageDataGenerator``."""
        config = self.get_val_augmentation()
        logger.debug("Validation augmentation config: %s", config)
        return ImageDataGenerator(**config)

    # ------------------------------------------------------------------
    # Generator factories
    # ------------------------------------------------------------------

    def create_train_generator(
        self,
        dataframe: pd.DataFrame,
        image_dir: Optional[Union[str, Path]] = None,
        target_size: Tuple[int, int] = (300, 300),
        batch_size: int = 32,
        x_col: str = "id_code",
        y_col: str = "diagnosis",
        classes: Optional[List[str]] = None,
        class_mode: str = "raw",
        shuffle: bool = True,
        seed: int = 42,
    ) -> Any:
        """Create a training data generator from a DataFrame."""
        logger.info(
            "Creating training generator: %d samples, batch_size=%d, "
            "target_size=%s",
            len(dataframe), batch_size, target_size,
        )
        return self.train_datagen.flow_from_dataframe(
            dataframe=dataframe,
            directory=str(image_dir) if image_dir is not None else None,
            x_col=x_col,
            y_col=y_col,
            classes=classes,
            target_size=target_size,
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=shuffle,
            seed=seed,
        )

    def create_val_generator(
        self,
        dataframe: pd.DataFrame,
        image_dir: Optional[Union[str, Path]] = None,
        target_size: Tuple[int, int] = (300, 300),
        batch_size: int = 32,
        x_col: str = "id_code",
        y_col: str = "diagnosis",
        classes: Optional[List[str]] = None,
        class_mode: str = "raw",
        shuffle: bool = False,
        seed: int = 42,
    ) -> Any:
        """Create a validation data generator from a DataFrame."""
        return self.val_datagen.flow_from_dataframe(
            dataframe=dataframe,
            directory=str(image_dir) if image_dir is not None else None,
            x_col=x_col,
            y_col=y_col,
            classes=classes,
            target_size=target_size,
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=shuffle,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize_augmentations(
        self,
        image_path: Union[str, Path],
        n: int = 8,
        target_size: Tuple[int, int] = (300, 300),
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Display a grid of augmented versions of a single image.

        Args:
            image_path: Path to the source image.
            n: Number of augmented variants to generate.
            target_size: Resize dimensions for display.
            save_path: If provided, save the figure instead of showing it.
        """
        try:
            img = load_img(str(image_path), target_size=target_size)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, C)

            # Determine grid layout
            cols = 4
            rows = max(1, (n + cols - 1) // cols)
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            fig.suptitle(
                f"Augmented Variants — {Path(image_path).name}",
                fontsize=14,
                fontweight="bold",
            )

            axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

            # Use the training datagen (with rescale) to generate augmented images
            aug_iter = self.train_datagen.flow(
                img_array, batch_size=1, seed=None
            )

            for i, ax in enumerate(axes_flat):
                if i < n:
                    augmented = next(aug_iter)[0]  # (H, W, C) in [0, 1]
                    ax.imshow(np.clip(augmented, 0, 1))
                    ax.set_title(f"Variant {i + 1}", fontsize=10)
                ax.axis("off")

            plt.tight_layout()

            if save_path:
                fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
                logger.info("Augmentation visualization saved to %s", save_path)
                plt.close(fig)
            else:
                plt.show()

        except Exception as e:
            logger.error("Error visualizing augmentations for %s: %s", image_path, e)
