"""
Shared pytest fixtures for the Diabetic Retinopathy CDSS test suite.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Add project root to sys.path so ``src`` imports work regardless of cwd.
# ---------------------------------------------------------------------------
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.preprocessing.preprocessor import DRImagePreprocessor
from src.preprocessing.quality_assessment import ImageQualityAssessor
from src.utils.database import DRDatabase


# ---------------------------------------------------------------------------
# Image fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_image_path(tmp_path: Path) -> Path:
    """Create a synthetic 300×300 RGB fundus-like image.

    Dark background with a bright circle in the centre to mimic
    a retinal fundus photograph.
    """
    img = Image.new("RGB", (300, 300), color=(10, 10, 10))
    draw = ImageDraw.Draw(img)
    # Bright orange-ish circle to resemble the retinal disc
    draw.ellipse([50, 50, 250, 250], fill=(180, 100, 60))
    # Smaller bright spot for optic disc
    draw.ellipse([140, 120, 180, 160], fill=(240, 220, 180))

    path = tmp_path / "fundus_sample.png"
    img.save(str(path))
    return path


@pytest.fixture
def corrupted_image_path(tmp_path: Path) -> Path:
    """Create a file filled with random bytes (not a valid image)."""
    path = tmp_path / "corrupted.png"
    path.write_bytes(os.urandom(1024))
    return path


# ---------------------------------------------------------------------------
# Module fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def preprocessor() -> DRImagePreprocessor:
    """Return a default DRImagePreprocessor instance."""
    return DRImagePreprocessor()


@pytest.fixture
def quality_assessor() -> ImageQualityAssessor:
    """Return a default ImageQualityAssessor instance."""
    return ImageQualityAssessor()


# ---------------------------------------------------------------------------
# Database fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_db(tmp_path: Path) -> DRDatabase:
    """Create a temporary DRDatabase backed by a file in *tmp_path*."""
    db_path = str(tmp_path / "test_dr.db")
    db = DRDatabase(db_path=db_path)
    return db


# ---------------------------------------------------------------------------
# DataFrame fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Return a small DataFrame mimicking the APTOS 2019 dataset format.

    Columns: ``id_code``, ``diagnosis`` (grades 0-4).
    Contains enough samples per class for stratified splitting.
    """
    id_codes = [f"aptos_{i:04d}" for i in range(50)]
    # Imbalanced distribution reflecting APTOS: majority Grade 0, all grades >= 5 samples
    diagnoses = (
        [0] * 20 +  # No DR
        [1] * 8  +  # Mild
        [2] * 8  +  # Moderate
        [3] * 7  +  # Severe
        [4] * 7     # Proliferative
    )
    return pd.DataFrame({"id_code": id_codes, "diagnosis": diagnoses})
