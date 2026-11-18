"""
Database manager for the Diabetic Retinopathy CDSS project.

Provides a SQLite-backed storage layer for patients, screening results,
and model metadata.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DRDatabase:
    """SQLite database manager for diabetic retinopathy screening data."""

    def __init__(self, db_path: str = "dr_cdss.db") -> None:
        self.db_path = db_path
        self.init_database()

    # ------------------------------------------------------------------
    # Connection helper
    # ------------------------------------------------------------------
    def get_connection(self) -> sqlite3.Connection:
        """Return a new connection with Row factory enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------
    def init_database(self) -> None:
        """Create all required tables if they do not already exist."""
        conn = self.get_connection()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id    TEXT    UNIQUE NOT NULL,
                    created_at    TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS screenings (
                    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id         INTEGER NOT NULL
                                       REFERENCES patients(id) ON DELETE CASCADE,
                    image_path         TEXT    NOT NULL,
                    eye                TEXT    NOT NULL CHECK (eye IN ('left', 'right')),
                    predicted_grade    INTEGER NOT NULL,
                    confidence         REAL    NOT NULL,
                    probabilities      TEXT    NOT NULL,
                    image_quality      TEXT,
                    gradcam_path       TEXT,
                    report_path        TEXT,
                    clinician_reviewed INTEGER NOT NULL DEFAULT 0,
                    clinician_notes    TEXT,
                    created_at         TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_metadata (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name      TEXT    NOT NULL,
                    architecture    TEXT    NOT NULL,
                    accuracy        REAL,
                    f1_score        REAL,
                    training_date   TEXT,
                    epochs          INTEGER,
                    notes           TEXT,
                    created_at      TEXT    NOT NULL
                );
                """
            )
            conn.commit()
            logger.info("Database initialised at %s", self.db_path)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Patient operations
    # ------------------------------------------------------------------
    def add_patient(self, patient_id: str) -> int:
        """Insert a new patient and return the row id.

        If the patient already exists, return the existing row id.
        """
        conn = self.get_connection()
        try:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO patients (patient_id, created_at) VALUES (?, ?)",
                (patient_id, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            if cursor.lastrowid and cursor.rowcount > 0:
                logger.info("Added patient %s (id=%d)", patient_id, cursor.lastrowid)
                return cursor.lastrowid
            # Already exists – fetch existing id
            row = conn.execute(
                "SELECT id FROM patients WHERE patient_id = ?", (patient_id,)
            ).fetchone()
            return row["id"]
        finally:
            conn.close()

    def get_patient(self, patient_id: str) -> Optional[dict]:
        """Return patient record as a dict, or ``None`` if not found."""
        conn = self.get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Screening operations
    # ------------------------------------------------------------------
    def add_screening(
        self,
        patient_id: str,
        image_path: str,
        eye: str,
        predicted_grade: int,
        confidence: float,
        probabilities: dict,
        image_quality: str,
        gradcam_path: Optional[str] = None,
        report_path: Optional[str] = None,
    ) -> int:
        """Insert a screening result and return the new row id.

        ``patient_id`` is the *textual* patient identifier – the
        corresponding internal row id is resolved automatically.
        ``probabilities`` is serialised as JSON.
        """
        # Ensure the patient exists
        internal_id = self.add_patient(patient_id)

        # Normalize eye parameter to match DB CHECK constraint ('left' / 'right')
        eye_lower = str(eye).lower()
        normalized_eye = "left" if ("left" in eye_lower or "os" in eye_lower) else "right"

        conn = self.get_connection()
        try:
            cursor = conn.execute(
                """
                INSERT INTO screenings
                    (patient_id, image_path, eye, predicted_grade, confidence,
                     probabilities, image_quality, gradcam_path, report_path,
                     created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    internal_id,
                    image_path,
                    normalized_eye,
                    predicted_grade,
                    confidence,
                    json.dumps(probabilities),
                    image_quality,
                    gradcam_path,
                    report_path,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
            logger.info(
                "Added screening %d for patient %s", cursor.lastrowid, patient_id
            )
            return cursor.lastrowid
        finally:
            conn.close()

    def get_screening(self, screening_id: int) -> Optional[dict]:
        """Return a single screening record by its id."""
        conn = self.get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM screenings WHERE id = ?", (screening_id,)
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["probabilities"] = json.loads(result["probabilities"])
            return result
        finally:
            conn.close()

    def get_patient_history(self, patient_id: str) -> list[dict]:
        """Return all screenings for a patient, newest first."""
        patient = self.get_patient(patient_id)
        if patient is None:
            return []

        conn = self.get_connection()
        try:
            rows = conn.execute(
                """
                SELECT * FROM screenings
                WHERE patient_id = ?
                ORDER BY created_at DESC
                """,
                (patient["id"],),
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["probabilities"] = json.loads(d["probabilities"])
                results.append(d)
            return results
        finally:
            conn.close()

    def get_all_screenings(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """Return paginated screening records, newest first."""
        conn = self.get_connection()
        try:
            rows = conn.execute(
                """
                SELECT s.*, p.patient_id AS patient_id_text
                FROM screenings s
                JOIN patients p ON s.patient_id = p.id
                ORDER BY s.created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["probabilities"] = json.loads(d["probabilities"])
                results.append(d)
            return results
        finally:
            conn.close()

    def get_screening_stats(self) -> dict:
        """Return aggregate screening statistics.

        Returns a dict with keys ``total``, ``grade_counts`` (dict mapping
        grade int to count), ``reviewed``, and ``unreviewed``.
        """
        conn = self.get_connection()
        try:
            total = conn.execute("SELECT COUNT(*) FROM screenings").fetchone()[0]
            reviewed = conn.execute(
                "SELECT COUNT(*) FROM screenings WHERE clinician_reviewed = 1"
            ).fetchone()[0]

            grade_rows = conn.execute(
                "SELECT predicted_grade, COUNT(*) as cnt FROM screenings GROUP BY predicted_grade"
            ).fetchall()
            grade_counts = {row["predicted_grade"]: row["cnt"] for row in grade_rows}

            return {
                "total": total,
                "reviewed": reviewed,
                "unreviewed": total - reviewed,
                "grade_counts": grade_counts,
            }
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Clinician review
    # ------------------------------------------------------------------
    def update_clinician_review(
        self, screening_id: int, reviewed: bool, notes: Optional[str] = None
    ) -> None:
        """Mark a screening as reviewed (or not) by a clinician."""
        conn = self.get_connection()
        try:
            conn.execute(
                """
                UPDATE screenings
                SET clinician_reviewed = ?, clinician_notes = ?
                WHERE id = ?
                """,
                (int(reviewed), notes, screening_id),
            )
            conn.commit()
            logger.info("Updated clinician review for screening %d", screening_id)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Model metadata
    # ------------------------------------------------------------------
    def add_model_metadata(
        self,
        model_name: str,
        architecture: str,
        accuracy: float,
        f1_score: float,
        epochs: int,
        notes: Optional[str] = None,
    ) -> int:
        """Record metadata about a trained model."""
        conn = self.get_connection()
        try:
            cursor = conn.execute(
                """
                INSERT INTO model_metadata
                    (model_name, architecture, accuracy, f1_score,
                     training_date, epochs, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_name,
                    architecture,
                    accuracy,
                    f1_score,
                    datetime.now(timezone.utc).isoformat(),
                    epochs,
                    notes,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
            logger.info("Added model metadata: %s (id=%d)", model_name, cursor.lastrowid)
            return cursor.lastrowid
        finally:
            conn.close()

    def get_model_metadata(self) -> list[dict]:
        """Return all stored model metadata records."""
        conn = self.get_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM model_metadata ORDER BY created_at DESC"
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
