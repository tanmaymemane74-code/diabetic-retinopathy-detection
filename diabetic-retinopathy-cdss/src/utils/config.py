"""
Configuration management for the Diabetic Retinopathy CDSS project.

Loads environment variables, defines project paths as constants,
and ensures required directories exist on import.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env (search upward from this file so it works from any cwd)
# ---------------------------------------------------------------------------
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

# ---------------------------------------------------------------------------
# Project root (two levels up from src/utils/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Paths derived from environment variables (with sensible defaults)
# ---------------------------------------------------------------------------
MODEL_PATH: Path = PROJECT_ROOT / os.getenv("MODEL_PATH", "models/efficientnet_b3_dr.h5")
MODELS_DIR: Path = PROJECT_ROOT / "models"
KNOWLEDGE_BASE_PATH: Path = PROJECT_ROOT / os.getenv("KNOWLEDGE_BASE_PATH", "knowledge_base/")
VECTOR_DB_PATH: Path = PROJECT_ROOT / os.getenv("VECTOR_DB_PATH", "vector_db/")
DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./dr_cdss.db")
DATASET_DIR: Path = PROJECT_ROOT / os.getenv("DATASET_DIR", "dataset/")
CSV_PATH: Path = DATASET_DIR / "raw" / "train.csv"
UPLOAD_DIR: Path = PROJECT_ROOT / os.getenv("UPLOAD_DIR", "static/uploads/")
GRADCAM_DIR: Path = PROJECT_ROOT / os.getenv("GRADCAM_DIR", "static/gradcam/")
REPORTS_DIR: Path = PROJECT_ROOT / os.getenv("REPORTS_DIR", "reports/")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Diabetic Retinopathy class definitions
# ---------------------------------------------------------------------------
DR_CLASSES: dict[int, str] = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}

DR_DESCRIPTIONS: dict[int, str] = {
    0: (
        "No abnormalities detected. The retina appears healthy with no signs "
        "of diabetic retinopathy. Routine screening recommended in 12 months."
    ),
    1: (
        "Mild Non-Proliferative Diabetic Retinopathy (NPDR). Presence of a few "
        "microaneurysms — small areas of balloon-like swelling in the retinal "
        "blood vessels. Follow-up recommended in 9–12 months."
    ),
    2: (
        "Moderate Non-Proliferative Diabetic Retinopathy (NPDR). More extensive "
        "microaneurysms, dot-and-blot hemorrhages, and possible hard exudates. "
        "Some blood vessels may show signs of blockage. Referral to an "
        "ophthalmologist recommended within 3–6 months."
    ),
    3: (
        "Severe Non-Proliferative Diabetic Retinopathy (NPDR). Widespread "
        "hemorrhages and microaneurysms in all four retinal quadrants, venous "
        "beading in two or more quadrants, and/or prominent intraretinal "
        "microvascular abnormalities (IRMA). High risk of progression to "
        "proliferative DR. Urgent referral to a retina specialist recommended."
    ),
    4: (
        "Proliferative Diabetic Retinopathy (PDR). Growth of new, fragile blood "
        "vessels (neovascularization) on the retina or optic disc. High risk of "
        "vitreous hemorrhage, tractional retinal detachment, and severe vision "
        "loss. Immediate referral for treatment (laser photocoagulation or "
        "anti-VEGF injections) is required."
    ),
}

# ---------------------------------------------------------------------------
# Training / model hyper-parameters
# ---------------------------------------------------------------------------
IMAGE_SIZE: tuple[int, int] = (300, 300)
BATCH_SIZE: int = 32
EPOCHS: int = 50
LEARNING_RATE: float = 1e-4

# ---------------------------------------------------------------------------
# Ensure required directories exist
# ---------------------------------------------------------------------------
_DIRS_TO_CREATE = [
    MODEL_PATH.parent,       # models/
    DATASET_DIR,
    KNOWLEDGE_BASE_PATH,
    VECTOR_DB_PATH,
    UPLOAD_DIR,
    GRADCAM_DIR,
    REPORTS_DIR,
]

for _dir in _DIRS_TO_CREATE:
    _dir.mkdir(parents=True, exist_ok=True)
