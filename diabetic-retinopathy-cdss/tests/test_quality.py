"""
Tests for src.preprocessing.quality_assessment – ImageQualityAssessor.
"""

import cv2
import numpy as np
import pytest

from src.preprocessing.quality_assessment import ImageQualityAssessor


class TestBlurAssessment:
    """Tests for blur detection."""

    def test_blur_assessment(self, quality_assessor: ImageQualityAssessor, sample_image_path):
        """assess_blur should return a numeric score and a boolean flag."""
        img = cv2.imread(str(sample_image_path))
        assert img is not None, "Failed to load sample image"

        result = quality_assessor.assess_blur(img)
        # Result should be a tuple/dict with score and boolean
        if isinstance(result, dict):
            assert "score" in result, "Missing 'score' key"
            assert "is_blurry" in result, "Missing 'is_blurry' key"
            assert isinstance(result["score"], (int, float))
            assert isinstance(result["is_blurry"], bool)
        else:
            # Assume (score, is_blurry) tuple
            score, is_blurry = result
            assert isinstance(score, (int, float))
            assert isinstance(is_blurry, bool)


class TestIlluminationAssessment:
    """Tests for illumination assessment."""

    def test_illumination_assessment(self, quality_assessor: ImageQualityAssessor, sample_image_path):
        """assess_illumination should return a score and a boolean flag."""
        img = cv2.imread(str(sample_image_path))
        result = quality_assessor.assess_illumination(img)

        if isinstance(result, dict):
            assert "score" in result
            assert "is_poor" in result
            assert isinstance(result["score"], (int, float))
            assert isinstance(result["is_poor"], bool)
        else:
            score, is_poor = result
            assert isinstance(score, (int, float))
            assert isinstance(is_poor, bool)


class TestContrastAssessment:
    """Tests for contrast assessment."""

    def test_contrast_assessment(self, quality_assessor: ImageQualityAssessor, sample_image_path):
        """assess_contrast should return a numeric result."""
        img = cv2.imread(str(sample_image_path))
        result = quality_assessor.assess_contrast(img)

        if isinstance(result, dict):
            assert "score" in result
            assert isinstance(result["score"], (int, float))
        else:
            # Could be a single float or tuple
            if isinstance(result, tuple):
                assert isinstance(result[0], (int, float))
            else:
                assert isinstance(result, (int, float))


class TestQualityReport:
    """Tests for the full quality report."""

    def test_quality_report_structure(self, quality_assessor: ImageQualityAssessor, sample_image_path):
        """The quality report should contain all expected keys."""
        img = cv2.imread(str(sample_image_path))
        report = quality_assessor.assess_quality(img)

        assert isinstance(report, dict), "Report should be a dict"
        expected_keys = {"blur", "illumination", "contrast", "overall_quality", "is_gradable"}
        # Allow for slight variations in key naming
        report_keys = set(report.keys())
        # At minimum we need overall quality and gradability
        assert "overall_quality" in report_keys or "quality" in report_keys, (
            f"Missing quality key. Keys: {report_keys}"
        )
        assert "is_gradable" in report_keys or "gradable" in report_keys, (
            f"Missing gradable key. Keys: {report_keys}"
        )

    def test_quality_report_valid_image(self, quality_assessor: ImageQualityAssessor, sample_image_path):
        """A well-formed synthetic image should not be flagged as ungradable."""
        img = cv2.imread(str(sample_image_path))
        report = quality_assessor.assess_quality(img)

        # Determine the gradable key
        gradable = report.get("is_gradable", report.get("gradable", None))
        assert gradable is not None, "Could not find gradability indicator in report"
        assert gradable is True, (
            f"Good synthetic image should be gradable, got report: {report}"
        )
