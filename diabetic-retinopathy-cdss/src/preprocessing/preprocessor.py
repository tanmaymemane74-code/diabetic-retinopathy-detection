"""
Diabetic Retinopathy Image Preprocessing Pipeline.

Implements a comprehensive preprocessing pipeline for retinal fundus images,
including circular ROI cropping, CLAHE enhancement, Ben Graham preprocessing,
noise reduction, and normalization.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DRImagePreprocessor:
    """Complete preprocessing pipeline for diabetic retinopathy fundus images.

    This pipeline implements best-practice preprocessing steps for retinal
    images, including techniques from winning Kaggle DR competition solutions.

    Attributes:
        target_size: Target dimensions (height, width) for output images.
    """

    def __init__(self, target_size: Tuple[int, int] = (300, 300)) -> None:
        """Initialize the preprocessor.

        Args:
            target_size: Target output image dimensions as (height, width).
        """
        self.target_size = target_size
        logger.info(
            "DRImagePreprocessor initialized with target_size=%s", target_size
        )

    def load_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """Load an image from disk using OpenCV.

        Args:
            image_path: Path to the image file.

        Returns:
            The loaded image as a BGR numpy array, or None if loading fails.
        """
        image_path = str(image_path)
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error("Failed to load image (returned None): %s", image_path)
                return None
            logger.debug("Loaded image %s with shape %s", image_path, image.shape)
            return image
        except Exception as e:
            logger.error("Error loading image %s: %s", image_path, e)
            return None

    def convert_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """Convert a BGR image to RGB color space.

        Args:
            image: Input image in BGR format.

        Returns:
            Image converted to RGB format.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def resize_image(
        self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Resize an image to the target dimensions.

        Uses INTER_AREA for downscaling and INTER_LINEAR for upscaling to
        maintain image quality.

        Args:
            image: Input image.
            target_size: Target (height, width). Uses instance default if None.

        Returns:
            Resized image.
        """
        target_size = target_size or self.target_size
        h, w = image.shape[:2]
        target_h, target_w = target_size

        # Choose interpolation method based on scaling direction
        if h * w > target_h * target_w:
            interpolation = cv2.INTER_AREA  # Shrinking
        else:
            interpolation = cv2.INTER_LINEAR  # Enlarging

        resized = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
        logger.debug("Resized image from (%d, %d) to (%d, %d)", h, w, target_h, target_w)
        return resized

    def crop_circular_roi(self, image: np.ndarray) -> np.ndarray:
        """Detect and crop the circular retinal region from a fundus image.

        Fundus images typically show a circular retinal field on a dark
        background. This method finds the retina boundary using contour
        detection, creates a tight circular mask, and crops to the bounding
        rectangle of that circle.

        Args:
            image: Input fundus image (BGR or RGB).

        Returns:
            Cropped image containing only the retinal region. Returns the
            original image if detection fails.
        """
        try:
            # Convert to grayscale for contour detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[2] == 3 else image[:, :, 0]
            else:
                gray = image.copy()

            # Threshold to separate retina from dark background
            _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

            # Clean up noise with morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

            # Find contours and pick the largest one (the retina)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                logger.warning("No contours found; returning original image.")
                return image

            largest_contour = max(contours, key=cv2.contourArea)

            # Fit a minimum enclosing circle
            (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
            cx, cy, radius = int(cx), int(cy), int(radius)

            if radius < 10:
                logger.warning("Detected radius too small (%d); returning original.", radius)
                return image

            # Create a circular mask
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (cx, cy), radius, 255, -1)

            # Apply mask
            result = cv2.bitwise_and(image, image, mask=mask)

            # Crop to bounding box of the circle
            x1 = max(0, cx - radius)
            y1 = max(0, cy - radius)
            x2 = min(image.shape[1], cx + radius)
            y2 = min(image.shape[0], cy + radius)

            cropped = result[y1:y2, x1:x2]

            if cropped.size == 0:
                logger.warning("Cropped region is empty; returning original.")
                return image

            logger.debug(
                "Cropped circular ROI: center=(%d,%d), radius=%d", cx, cy, radius
            )
            return cropped

        except Exception as e:
            logger.error("Error in circular ROI cropping: %s", e)
            return image

    def apply_clahe(
        self,
        image: np.ndarray,
        clip_limit: float = 2.0,
        grid_size: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

        Converts the image to LAB color space, applies CLAHE to the L
        (lightness) channel, then converts back to the original color space.

        Args:
            image: Input image in BGR or RGB format.
            clip_limit: CLAHE clip limit for contrast limiting.
            grid_size: Size of the grid for CLAHE tiles.

        Returns:
            Contrast-enhanced image.
        """
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            l_enhanced = clahe.apply(l_channel)

            lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
            result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

            logger.debug("Applied CLAHE with clip_limit=%.1f", clip_limit)
            return result
        except Exception as e:
            logger.error("Error applying CLAHE: %s", e)
            return image

    def reduce_noise(self, image: np.ndarray, strength: int = 10) -> np.ndarray:
        """Reduce noise using a bilateral filter.

        Bilateral filtering smooths the image while preserving edges, which
        is important for retaining microaneurysm and hemorrhage boundaries.

        Args:
            image: Input image.
            strength: Filter strength (higher = more smoothing).

        Returns:
            Denoised image.
        """
        try:
            denoised = cv2.bilateralFilter(image, d=9, sigmaColor=strength * 7.5, sigmaSpace=strength * 7.5)
            logger.debug("Applied bilateral noise reduction with strength=%d", strength)
            return denoised
        except Exception as e:
            logger.error("Error in noise reduction: %s", e)
            return image

    def normalize_pixels(self, image: np.ndarray) -> np.ndarray:
        """Normalize pixel values to the [0, 1] range.

        Args:
            image: Input image with uint8 pixel values (0-255).

        Returns:
            Normalized image as float32 with values in [0, 1].
        """
        return image.astype(np.float32) / 255.0

    def ben_graham_preprocessing(
        self, image: np.ndarray, sigmaX: int = 10
    ) -> np.ndarray:
        """Apply Ben Graham's preprocessing technique for fundus images.

        This technique was used in winning Kaggle Diabetic Retinopathy
        competition solutions. It subtracts the local average color from
        the image and scales to the detected retinal radius, enhancing
        local contrast and making features more visible.

        Steps:
            1. Compute a Gaussian-blurred version of the image.
            2. Subtract the blurred image from the original.
            3. Add 128 to shift values to a visible range.

        Args:
            image: Input fundus image (BGR).
            sigmaX: Gaussian blur sigma parameter controlling the scale
                    of the local average subtraction.

        Returns:
            Preprocessed image with enhanced local contrast.
        """
        try:
            # Ensure the image is uint8
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

            # Compute local average with Gaussian blur
            gaussian = cv2.GaussianBlur(image, (0, 0), sigmaX)

            # Subtract local average and add 128 to center values
            result = cv2.addWeighted(image, 4, gaussian, -4, 128)

            logger.debug("Applied Ben Graham preprocessing with sigmaX=%d", sigmaX)
            return result
        except Exception as e:
            logger.error("Error in Ben Graham preprocessing: %s", e)
            return image

    def preprocess_single(
        self,
        image_path: Union[str, Path],
        apply_ben_graham: bool = True,
    ) -> Optional[np.ndarray]:
        """Run the full preprocessing pipeline on a single image.

        Pipeline order: load → BGR→RGB → crop circular ROI → CLAHE →
        denoise → (optional Ben Graham) → resize → normalize.

        Args:
            image_path: Path to the input image.
            apply_ben_graham: Whether to apply Ben Graham preprocessing.

        Returns:
            Fully preprocessed image as a float32 array in [0, 1],
            or None if any critical step fails.
        """
        try:
            # Load
            image = self.load_image(image_path)
            if image is None:
                return None

            # Crop circular ROI (operates on BGR)
            image = self.crop_circular_roi(image)

            # CLAHE (operates on BGR internally)
            image = self.apply_clahe(image)

            # Denoise
            image = self.reduce_noise(image)

            # Ben Graham preprocessing (optional)
            if apply_ben_graham:
                image = self.ben_graham_preprocessing(image)

            # Convert to RGB
            image = self.convert_to_rgb(image)

            # Resize
            image = self.resize_image(image)

            # Normalize
            image = self.normalize_pixels(image)

            logger.info("Successfully preprocessed: %s", image_path)
            return image

        except Exception as e:
            logger.error("Pipeline failed for %s: %s", image_path, e)
            return None

    def preprocess_batch(
        self,
        image_paths: List[Union[str, Path]],
        num_workers: int = 4,
    ) -> List[Optional[np.ndarray]]:
        """Preprocess a batch of images using multithreaded execution.

        Args:
            image_paths: List of paths to images.
            num_workers: Number of parallel worker threads.

        Returns:
            List of preprocessed images (or None for failures),
            in the same order as the input paths.
        """
        results: List[Optional[np.ndarray]] = [None] * len(image_paths)
        logger.info(
            "Starting batch preprocessing of %d images with %d workers",
            len(image_paths),
            num_workers,
        )

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(self.preprocess_single, path): idx
                for idx, path in enumerate(image_paths)
            }

            with tqdm(total=len(image_paths), desc="Preprocessing") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        logger.error(
                            "Worker error for image %s: %s", image_paths[idx], e
                        )
                        results[idx] = None
                    pbar.update(1)

        successful = sum(1 for r in results if r is not None)
        logger.info(
            "Batch complete: %d/%d images processed successfully",
            successful,
            len(image_paths),
        )
        return results

    def visualize_pipeline(
        self,
        image_path: Union[str, Path],
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Visualize each stage of the preprocessing pipeline.

        Creates a figure with five panels showing the image at each step:
        Original | Cropped | CLAHE | Ben Graham | Final.

        Args:
            image_path: Path to the input image.
            save_path: If provided, save the figure to this path instead
                       of displaying it.
        """
        image = self.load_image(image_path)
        if image is None:
            logger.error("Cannot visualize pipeline: failed to load %s", image_path)
            return

        stages = {}

        # Stage 1: Original (convert to RGB for display)
        stages["Original"] = self.convert_to_rgb(image.copy())

        # Stage 2: Crop circular ROI
        cropped = self.crop_circular_roi(image)
        stages["Cropped ROI"] = self.convert_to_rgb(cropped.copy())

        # Stage 3: CLAHE
        clahe_img = self.apply_clahe(cropped)
        stages["CLAHE Enhanced"] = self.convert_to_rgb(clahe_img.copy())

        # Stage 4: Ben Graham
        denoised = self.reduce_noise(clahe_img)
        ben_graham_img = self.ben_graham_preprocessing(denoised)
        stages["Ben Graham"] = self.convert_to_rgb(ben_graham_img.copy())

        # Stage 5: Final (resized + normalized)
        final_bgr = ben_graham_img.copy()
        final_rgb = self.convert_to_rgb(final_bgr)
        final_resized = self.resize_image(final_rgb)
        final_normalized = self.normalize_pixels(final_resized)
        stages["Final"] = final_normalized

        # Plot
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        fig.suptitle(
            f"DR Preprocessing Pipeline: {Path(image_path).name}",
            fontsize=14,
            fontweight="bold",
        )

        for ax, (title, img) in zip(axes, stages.items()):
            ax.imshow(img)
            ax.set_title(title, fontsize=11)
            ax.axis("off")

        plt.tight_layout()

        if save_path:
            fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
            logger.info("Pipeline visualization saved to %s", save_path)
            plt.close(fig)
        else:
            plt.show()
