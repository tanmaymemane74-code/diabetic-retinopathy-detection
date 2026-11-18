"""
Tests for src.preprocessing.preprocessor – DRImagePreprocessor.
"""

import numpy as np
import pytest

from src.preprocessing.preprocessor import DRImagePreprocessor


class TestLoadImage:
    """Tests for image loading."""

    def test_load_valid_image(self, preprocessor: DRImagePreprocessor, sample_image_path):
        """Loading a valid image should return an ndarray with 3 dimensions."""
        img = preprocessor.load_image(str(sample_image_path))
        assert img is not None, "load_image returned None for a valid image"
        assert img.ndim == 3, f"Expected 3-dim array, got {img.ndim}"
        assert img.shape[2] == 3, f"Expected 3 channels, got {img.shape[2]}"

    def test_load_invalid_path(self, preprocessor: DRImagePreprocessor):
        """Loading from a non-existent path should return None (graceful)."""
        result = preprocessor.load_image("non_existent_path_12345.png")
        assert result is None, "Expected None for a missing file"

    def test_corrupted_image(self, preprocessor: DRImagePreprocessor, corrupted_image_path):
        """Loading a corrupted file should return None or raise without crashing."""
        result = preprocessor.preprocess_single(str(corrupted_image_path))
        # preprocess_single should return None on failure
        assert result is None, "Expected None when processing a corrupted image"


class TestColourConversion:
    """Tests for RGB conversion."""

    def test_rgb_conversion(self, preprocessor: DRImagePreprocessor, sample_image_path):
        """Converted image should have exactly 3 channels."""
        img = preprocessor.load_image(str(sample_image_path))
        rgb = preprocessor.convert_to_rgb(img)
        assert rgb.shape[2] == 3, f"Expected 3 channels, got {rgb.shape[2]}"


class TestResize:
    """Tests for image resizing."""

    def test_resize_image(self, preprocessor: DRImagePreprocessor, sample_image_path):
        """Resized image should match the target dimensions."""
        img = preprocessor.load_image(str(sample_image_path))
        target = (224, 224)
        resized = preprocessor.resize_image(img, target_size=target)
        assert resized.shape[0] == target[0], f"Height mismatch: {resized.shape[0]} != {target[0]}"
        assert resized.shape[1] == target[1], f"Width mismatch: {resized.shape[1]} != {target[1]}"


class TestCircularCrop:
    """Tests for circular ROI cropping."""

    def test_circular_crop(self, preprocessor: DRImagePreprocessor, sample_image_path):
        """Cropped image should be a valid 3D image with 3 channels."""
        img = preprocessor.load_image(str(sample_image_path))
        cropped = preprocessor.crop_circular_roi(img)
        assert cropped is not None, "crop_circular_roi returned None"
        assert cropped.ndim == 3, f"Expected 3-dim array, got {cropped.ndim}"
        assert cropped.shape[2] == 3, f"Expected 3 channels, got {cropped.shape[2]}"
        assert cropped.shape[0] > 0 and cropped.shape[1] > 0


class TestCLAHE:
    """Tests for CLAHE enhancement."""

    def test_clahe_enhancement(self, preprocessor: DRImagePreprocessor, sample_image_path):
        """CLAHE output should preserve shape and dtype."""
        img = preprocessor.load_image(str(sample_image_path))
        enhanced = preprocessor.apply_clahe(img)
        assert enhanced.shape == img.shape, "CLAHE changed shape"
        assert enhanced.dtype == img.dtype, "CLAHE changed dtype"


class TestNormalization:
    """Tests for pixel normalisation."""

    def test_normalize_pixels(self, preprocessor: DRImagePreprocessor, sample_image_path):
        """Normalised values should be in the [0, 1] range."""
        img = preprocessor.load_image(str(sample_image_path))
        normed = preprocessor.normalize_pixels(img)
        assert normed.min() >= 0.0, f"Min value {normed.min()} < 0"
        assert normed.max() <= 1.0, f"Max value {normed.max()} > 1"
        assert normed.dtype == np.float32 or normed.dtype == np.float64, (
            f"Expected float dtype, got {normed.dtype}"
        )


class TestFullPipeline:
    """End-to-end preprocessing test."""

    def test_full_pipeline(self, preprocessor: DRImagePreprocessor, sample_image_path):
        """Full pipeline should produce a normalised, correctly-shaped array."""
        result = preprocessor.preprocess_single(str(sample_image_path))
        assert result is not None, "Pipeline returned None for a valid image"
        # Should be target_size height × width × 3
        assert result.shape[0] == preprocessor.target_size[0]
        assert result.shape[1] == preprocessor.target_size[1]
        assert result.shape[2] == 3
        assert result.min() >= 0.0
        assert result.max() <= 1.0
