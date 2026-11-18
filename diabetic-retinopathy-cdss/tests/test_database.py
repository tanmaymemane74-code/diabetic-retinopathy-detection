"""
Tests for src.utils.database – DRDatabase.
"""

import pytest

from src.utils.database import DRDatabase


class TestAddPatient:
    """Tests for patient creation."""

    def test_add_patient(self, temp_db: DRDatabase):
        """Adding a new patient should return a positive integer id."""
        pid = temp_db.add_patient("PAT-001")
        assert isinstance(pid, int)
        assert pid > 0

        # Verify retrieval
        patient = temp_db.get_patient("PAT-001")
        assert patient is not None
        assert patient["patient_id"] == "PAT-001"

    def test_duplicate_patient(self, temp_db: DRDatabase):
        """Adding the same patient_id twice should not raise and should return the same id."""
        pid1 = temp_db.add_patient("PAT-DUP")
        pid2 = temp_db.add_patient("PAT-DUP")
        assert pid1 == pid2, "Duplicate insert should return the existing id"


class TestScreening:
    """Tests for screening CRUD operations."""

    def _add_sample_screening(self, db: DRDatabase, patient: str = "PAT-100", grade: int = 2):
        """Helper to insert a screening record."""
        return db.add_screening(
            patient_id=patient,
            image_path="/images/fundus_001.png",
            eye="left",
            predicted_grade=grade,
            confidence=0.87,
            probabilities={0: 0.05, 1: 0.03, 2: 0.87, 3: 0.03, 4: 0.02},
            image_quality="good",
            gradcam_path="/gradcam/fundus_001.png",
            report_path="/reports/fundus_001.pdf",
        )

    def test_add_screening(self, temp_db: DRDatabase):
        """Adding a screening should return a positive integer id."""
        sid = self._add_sample_screening(temp_db)
        assert isinstance(sid, int)
        assert sid > 0

        screening = temp_db.get_screening(sid)
        assert screening is not None
        assert screening["predicted_grade"] == 2
        assert screening["confidence"] == pytest.approx(0.87)
        assert isinstance(screening["probabilities"], dict)

    def test_get_patient_history(self, temp_db: DRDatabase):
        """Patient history should return all screenings for that patient."""
        self._add_sample_screening(temp_db, patient="PAT-200", grade=0)
        self._add_sample_screening(temp_db, patient="PAT-200", grade=1)
        self._add_sample_screening(temp_db, patient="PAT-200", grade=3)

        history = temp_db.get_patient_history("PAT-200")
        assert len(history) == 3
        # Should be newest-first
        grades = [h["predicted_grade"] for h in history]
        assert 0 in grades
        assert 1 in grades
        assert 3 in grades

    def test_screening_stats(self, temp_db: DRDatabase):
        """Screening stats should accurately reflect inserted data."""
        self._add_sample_screening(temp_db, patient="PAT-S1", grade=0)
        self._add_sample_screening(temp_db, patient="PAT-S2", grade=0)
        self._add_sample_screening(temp_db, patient="PAT-S3", grade=2)
        self._add_sample_screening(temp_db, patient="PAT-S4", grade=4)

        stats = temp_db.get_screening_stats()
        assert stats["total"] == 4
        assert stats["grade_counts"][0] == 2
        assert stats["grade_counts"][2] == 1
        assert stats["grade_counts"][4] == 1
        assert stats["unreviewed"] == 4
        assert stats["reviewed"] == 0


class TestClinicianReview:
    """Tests for clinician review updates."""

    def test_clinician_review_update(self, temp_db: DRDatabase):
        """Updating a review should persist the flag and notes."""
        sid = temp_db.add_screening(
            patient_id="PAT-REV",
            image_path="/images/fundus_rev.png",
            eye="right",
            predicted_grade=1,
            confidence=0.72,
            probabilities={0: 0.15, 1: 0.72, 2: 0.08, 3: 0.03, 4: 0.02},
            image_quality="acceptable",
        )

        # Initially not reviewed
        screening = temp_db.get_screening(sid)
        assert screening["clinician_reviewed"] == 0

        # Mark as reviewed
        temp_db.update_clinician_review(sid, reviewed=True, notes="Confirmed mild NPDR")
        screening = temp_db.get_screening(sid)
        assert screening["clinician_reviewed"] == 1
        assert screening["clinician_notes"] == "Confirmed mild NPDR"


class TestModelMetadata:
    """Tests for model metadata storage."""

    def test_model_metadata(self, temp_db: DRDatabase):
        """Should store and retrieve model metadata correctly."""
        mid = temp_db.add_model_metadata(
            model_name="efficientnet_b3_dr_v1",
            architecture="EfficientNetB3",
            accuracy=0.83,
            f1_score=0.79,
            epochs=50,
            notes="Baseline model",
        )
        assert isinstance(mid, int)
        assert mid > 0

        models = temp_db.get_model_metadata()
        assert len(models) >= 1

        latest = models[0]
        assert latest["model_name"] == "efficientnet_b3_dr_v1"
        assert latest["architecture"] == "EfficientNetB3"
        assert latest["accuracy"] == pytest.approx(0.83)
        assert latest["f1_score"] == pytest.approx(0.79)
        assert latest["epochs"] == 50
        assert latest["notes"] == "Baseline model"
