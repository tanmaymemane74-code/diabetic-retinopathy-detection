"""
Fundus Image Quality Assessment Module.

Provides automated quality assessment for retinal fundus images, evaluating
blur, illumination, contrast, field of view, and overall image quality to
determine suitability for DR grading.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageQualityAssessor:
    """Assess the quality of retinal fundus images.

    Evaluates multiple quality dimensions and produces an overall quality
    grade (good / acceptable / poor / ungradable) to determine whether an
    image is suitable for automated diabetic retinopathy screening.

    All thresholds are configurable via the constructor.

    Attributes:
        blur_threshold: Laplacian variance below this value indicates blur.
        brightness_low: Mean brightness below this is too dark.
        brightness_high: Mean brightness above this is too bright.
        contrast_threshold: Pixel-value std dev below this is low contrast.
        fov_threshold: Minimum fraction of image area that should be retina.
        fundus_min_circularity: Minimum circularity to accept as a fundus.
    """

    def __init__(
        self,
        blur_threshold: float = 100.0,
        brightness_low: float = 40.0,
        brightness_high: float = 220.0,
        contrast_threshold: float = 35.0,
        fov_threshold: float = 0.30,
        fundus_min_circularity: float = 0.60,
    ) -> None:
        """Initialize with configurable quality thresholds.

        Args:
            blur_threshold: Laplacian variance threshold for blur detection.
            brightness_low: Lower bound for acceptable mean brightness.
            brightness_high: Upper bound for acceptable mean brightness.
            contrast_threshold: Minimum acceptable standard deviation of pixels.
            fov_threshold: Minimum fraction (0-1) of image area that the
                           retinal field should cover.
            fundus_min_circularity: Minimum circularity (0-1) of the detected
                                   bright region to be considered a fundus.
        """
        self.blur_threshold = blur_threshold
        self.brightness_low = brightness_low
        self.brightness_high = brightness_high
        self.contrast_threshold = contrast_threshold
        self.fov_threshold = fov_threshold
        self.fundus_min_circularity = fundus_min_circularity
        logger.info("ImageQualityAssessor initialized with thresholds: "
                     "blur=%.1f, brightness=[%.1f, %.1f], contrast=%.1f, "
                     "fov=%.2f, circularity=%.2f",
                     blur_threshold, brightness_low, brightness_high,
                     contrast_threshold, fov_threshold, fundus_min_circularity)

    # ------------------------------------------------------------------
    # Individual assessments
    # ------------------------------------------------------------------

    def assess_blur(self, image: np.ndarray) -> Dict[str, Any]:
        """Assess image sharpness using the Laplacian variance method.

        A low Laplacian variance indicates a blurry image.

        Args:
            image: Input image (BGR or RGB, uint8).

        Returns:
            Dict with keys ``score`` (float, Laplacian variance) and
            ``is_blurry`` (bool).
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            is_blurry = laplacian_var < self.blur_threshold
            logger.debug("Blur score: %.2f, is_blurry: %s", laplacian_var, is_blurry)
            return {"score": float(laplacian_var), "is_blurry": bool(is_blurry)}
        except Exception as e:
            logger.error("Error assessing blur: %s", e)
            return {"score": 0.0, "is_blurry": True}

    def assess_illumination(self, image: np.ndarray) -> Dict[str, Any]:
        """Assess image illumination by checking mean brightness.

        Args:
            image: Input image (BGR or RGB, uint8).

        Returns:
            Dict with keys ``score`` (float, mean brightness 0-255) and
            ``is_poor`` (bool, True if too dark or too bright).
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            mean_brightness = float(np.mean(gray))
            is_poor = mean_brightness < self.brightness_low or mean_brightness > self.brightness_high
            logger.debug("Illumination score: %.2f, is_poor: %s", mean_brightness, is_poor)
            return {"score": mean_brightness, "is_poor": bool(is_poor)}
        except Exception as e:
            logger.error("Error assessing illumination: %s", e)
            return {"score": 0.0, "is_poor": True}

    def assess_contrast(self, image: np.ndarray) -> Dict[str, Any]:
        """Assess image contrast by computing pixel-value standard deviation.

        Args:
            image: Input image (BGR or RGB, uint8).

        Returns:
            Dict with keys ``score`` (float, std dev of grayscale pixels) and
            ``is_low`` (bool, True if below threshold).
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            contrast = float(np.std(gray))
            is_low = contrast < self.contrast_threshold
            logger.debug("Contrast score: %.2f, is_low: %s", contrast, is_low)
            return {"score": contrast, "is_low": bool(is_low)}
        except Exception as e:
            logger.error("Error assessing contrast: %s", e)
            return {"score": 0.0, "is_low": True}

    def assess_field_of_view(self, image: np.ndarray) -> Dict[str, Any]:
        """Assess whether the retinal field of view covers enough of the image.

        A valid fundus image should have the circular retinal area occupying
        at least ``fov_threshold`` (default 30%) of the total image area.

        Args:
            image: Input image (BGR or RGB, uint8).

        Returns:
            Dict with keys ``score`` (float, fraction of image covered) and
            ``is_insufficient`` (bool).
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

            # Threshold to find the bright retinal region
            _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

            # Clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

            retina_pixels = int(np.sum(thresh > 0))
            total_pixels = image.shape[0] * image.shape[1]
            coverage = retina_pixels / total_pixels if total_pixels > 0 else 0.0

            is_insufficient = coverage < self.fov_threshold
            logger.debug("FoV coverage: %.2f%%, is_insufficient: %s",
                         coverage * 100, is_insufficient)
            return {"score": float(coverage), "is_insufficient": bool(is_insufficient)}
        except Exception as e:
            logger.error("Error assessing field of view: %s", e)
            return {"score": 0.0, "is_insufficient": True}

    def is_valid_fundus(self, image: np.ndarray) -> Dict[str, Any]:
        """Check whether the image appears to be a retinal fundus photograph.

        Looks for a roughly circular bright region on a dark background,
        which is the hallmark of a fundus image.

        Args:
            image: Input image (BGR or RGB, uint8).

        Returns:
            Dict with keys ``is_fundus`` (bool) and ``circularity`` (float).
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

            # Threshold
            _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                return {"is_fundus": False, "circularity": 0.0}

            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)

            if perimeter == 0:
                return {"is_fundus": False, "circularity": 0.0}

            # Circularity: 1.0 for a perfect circle
            circularity = (4 * np.pi * area) / (perimeter * perimeter)

            is_fundus = circularity >= self.fundus_min_circularity
            logger.debug("Fundus check — circularity: %.3f, is_fundus: %s",
                         circularity, is_fundus)
            return {"is_fundus": bool(is_fundus), "circularity": float(circularity)}
        except Exception as e:
            logger.error("Error checking fundus validity: %s", e)
            return {"is_fundus": False, "circularity": 0.0}

    # ------------------------------------------------------------------
    # Composite report
    # ------------------------------------------------------------------

    def get_quality_report(
        self, image_path: Union[str, Path, np.ndarray]
    ) -> Dict[str, Any]:
        """Generate a comprehensive quality report for a fundus image.

        Runs all individual assessments and derives an overall quality grade:
        ``good``, ``acceptable``, ``poor``, or ``ungradable``.

        Args:
            image_path: Path to the image file, or an in-memory numpy array.

        Returns:
            Dict containing:
                - ``image_path``: str
                - ``blur``: blur assessment dict
                - ``illumination``: illumination assessment dict
                - ``contrast``: contrast assessment dict
                - ``field_of_view``: FoV assessment dict
                - ``fundus_validity``: fundus check dict
                - ``overall_quality``: one of good / acceptable / poor / ungradable
                - ``issues``: list of human-readable issue descriptions
        """
        if isinstance(image_path, np.ndarray):
            return self.assess_quality(image_path)
        image_path = str(image_path)
        report: Dict[str, Any] = {"image_path": image_path}

        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error("Cannot read image for quality report: %s", image_path)
                report.update({
                    "blur": {"score": 0.0, "is_blurry": True},
                    "illumination": {"score": 0.0, "is_poor": True},
                    "contrast": {"score": 0.0, "is_low": True},
                    "field_of_view": {"score": 0.0, "is_insufficient": True},
                    "fundus_validity": {"is_fundus": False, "circularity": 0.0},
                    "overall_quality": "ungradable",
                    "issues": ["Image could not be loaded."],
                })
                return report

            # Run individual assessments
            report["blur"] = self.assess_blur(image)
            report["illumination"] = self.assess_illumination(image)
            report["contrast"] = self.assess_contrast(image)
            report["field_of_view"] = self.assess_field_of_view(image)
            report["fundus_validity"] = self.is_valid_fundus(image)

            # Collect issues
            issues: list[str] = []
            if report["blur"]["is_blurry"]:
                issues.append(f"Image is blurry (Laplacian variance: {report['blur']['score']:.1f})")
            if report["illumination"]["is_poor"]:
                issues.append(f"Poor illumination (mean brightness: {report['illumination']['score']:.1f})")
            if report["contrast"]["is_low"]:
                issues.append(f"Low contrast (std dev: {report['contrast']['score']:.1f})")
            if report["field_of_view"]["is_insufficient"]:
                issues.append(f"Insufficient field of view ({report['field_of_view']['score'] * 100:.1f}%)")
            if not report["fundus_validity"]["is_fundus"]:
                issues.append("Image may not be a valid fundus photograph")

            report["issues"] = issues

            # Determine overall quality
            num_issues = len(issues)
            if not report["fundus_validity"]["is_fundus"]:
                overall = "ungradable"
            elif num_issues == 0:
                overall = "good"
            elif num_issues <= 2:
                overall = "acceptable"
            elif num_issues <= 3:
                overall = "poor"
            else:
                overall = "ungradable"

            report["overall_quality"] = overall
            report["is_gradable"] = (overall != "ungradable")
            logger.info("Quality report for %s: %s (%d issues)",
                        image_path, overall, num_issues)
            return report

        except Exception as e:
            logger.error("Error generating quality report for %s: %s", image_path, e)
            report.update({
                "overall_quality": "ungradable",
                "is_gradable": False,
                "issues": [f"Unexpected error: {e}"],
            })
            return report

    def assess_quality(
        self, image_or_path: Union[str, Path, np.ndarray]
    ) -> Dict[str, Any]:
        """Assess overall image quality from either an image array or file path.

        Convenience method that accepts either a file path or an in-memory numpy array.
        """
        if isinstance(image_or_path, (str, Path)):
            return self.get_quality_report(image_or_path)

        # In-memory ndarray
        report: Dict[str, Any] = {}
        report["blur"] = self.assess_blur(image_or_path)
        report["illumination"] = self.assess_illumination(image_or_path)
        report["contrast"] = self.assess_contrast(image_or_path)
        report["field_of_view"] = self.assess_field_of_view(image_or_path)
        report["fundus_validity"] = self.is_valid_fundus(image_or_path)

        issues: list[str] = []
        if report["blur"]["is_blurry"]:
            issues.append(f"Image is blurry (Laplacian variance: {report['blur']['score']:.1f})")
        if report["illumination"]["is_poor"]:
            issues.append(f"Poor illumination (mean brightness: {report['illumination']['score']:.1f})")
        if report["contrast"]["is_low"]:
            issues.append(f"Low contrast (std dev: {report['contrast']['score']:.1f})")
        if report["field_of_view"]["is_insufficient"]:
            issues.append(f"Insufficient field of view ({report['field_of_view']['score'] * 100:.1f}%)")
        if not report["fundus_validity"]["is_fundus"]:
            issues.append("Image may not be a valid fundus photograph")

        report["issues"] = issues

        num_issues = len(issues)
        if not report["fundus_validity"]["is_fundus"]:
            overall = "ungradable"
        elif num_issues == 0:
            overall = "good"
        elif num_issues <= 2:
            overall = "acceptable"
        elif num_issues <= 3:
            overall = "poor"
        else:
            overall = "ungradable"

        report["overall_quality"] = overall
        report["is_gradable"] = (overall != "ungradable")
        return report
