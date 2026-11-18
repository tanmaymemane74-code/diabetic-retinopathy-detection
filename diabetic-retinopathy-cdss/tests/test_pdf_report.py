"""
Unit tests for ReportLab PDF clinical screening report generation.
"""

from pathlib import Path
import pytest

from src.reporting.pdf_generator import DRPDFReportGenerator
from src.clinical_decision.decision_engine import DRClinicalDecisionEngine


class TestPDFReportGenerator:
    """Tests for medical PDF report compilation."""

    def test_pdf_creation(self, tmp_path: Path, sample_image_path: Path):
        generator = DRPDFReportGenerator(output_dir=tmp_path)
        engine = DRClinicalDecisionEngine()

        clinical_decision = engine.evaluate(
            predicted_grade=2,
            confidence=0.885,
            image_quality_report={"overall_quality": "good", "is_gradable": True},
        )

        pdf_path = generator.generate_report(
            patient_id="TEST-PT-9988",
            predicted_grade=2,
            confidence=0.885,
            probabilities=[0.02, 0.05, 0.885, 0.03, 0.015],
            image_path=str(sample_image_path),
            clinical_decision=clinical_decision,
            output_filename="test_screening_report.pdf",
        )

        assert Path(pdf_path).exists()
        assert Path(pdf_path).stat().st_size > 1000  # Non-empty PDF with content and tables
