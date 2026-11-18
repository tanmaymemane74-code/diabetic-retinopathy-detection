"""
Dataset download helper for the APTOS 2019 Blindness Detection dataset.

Provides utilities to download the dataset from Kaggle, extract archives,
organise files into the expected directory layout, and verify integrity.
"""

import logging
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional

from src.utils.config import DATASET_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KAGGLE_DATASET = "mariaherrerot/aptos2019"
KAGGLE_COMPETITION = "aptos2019-blindness-detection"
EXPECTED_CSV = "train.csv"
EXPECTED_IMAGE_DIR = "train_images"


# ---------------------------------------------------------------------------
# Kaggle helpers
# ---------------------------------------------------------------------------

def _check_kaggle_configured() -> bool:
    """Return True if the Kaggle API credentials are available."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")

    if kaggle_json.is_file():
        return True
    if env_user and env_key:
        return True
    return False


def _print_kaggle_setup_instructions() -> None:
    """Print step-by-step instructions for Kaggle API setup."""
    instructions = """
╔══════════════════════════════════════════════════════════════════╗
║               Kaggle API Setup Instructions                     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. Go to https://www.kaggle.com/account                         ║
║  2. Scroll down to the "API" section                             ║
║  3. Click "Create New API Token"                                 ║
║  4. This downloads a kaggle.json file                            ║
║  5. Place it in:                                                 ║
║                                                                  ║
║     Windows:  C:\\Users\\<username>\\.kaggle\\kaggle.json           ║
║     Linux:    ~/.kaggle/kaggle.json                               ║
║     macOS:    ~/.kaggle/kaggle.json                               ║
║                                                                  ║
║  6. On Linux / macOS, set permissions:                           ║
║       chmod 600 ~/.kaggle/kaggle.json                             ║
║                                                                  ║
║  Alternatively, set environment variables:                       ║
║       KAGGLE_USERNAME=<your_username>                             ║
║       KAGGLE_KEY=<your_api_key>                                  ║
║                                                                  ║
║  Then install the Kaggle CLI:                                    ║
║       pip install kaggle                                         ║
║                                                                  ║
║  You may also need to accept the competition rules at:           ║
║   https://www.kaggle.com/c/aptos2019-blindness-detection/rules   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(instructions)


def _kaggle_cli_available() -> bool:
    """Check whether the kaggle CLI is installed."""
    try:
        subprocess.run(
            ["kaggle", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_aptos_dataset(destination: Optional[str | Path] = None) -> Path:
    """Download the APTOS 2019 dataset from Kaggle.

    Attempts to download via ``kaggle competitions download`` first;
    falls back to ``kaggle datasets download`` if the competition API
    is not accessible (e.g. rules not accepted).

    Parameters
    ----------
    destination : str | Path | None
        Directory to download into.  Defaults to ``DATASET_DIR/raw``.

    Returns
    -------
    Path
        Path to the downloaded zip file.

    Raises
    ------
    RuntimeError
        If the Kaggle CLI is not installed or credentials are missing.
    """
    dest = Path(destination) if destination else Path(DATASET_DIR) / "raw"
    dest.mkdir(parents=True, exist_ok=True)

    # Pre-flight checks
    if not _kaggle_cli_available():
        logger.error("Kaggle CLI not found.  Install with: pip install kaggle")
        _print_kaggle_setup_instructions()
        raise RuntimeError("Kaggle CLI is not installed.")

    if not _check_kaggle_configured():
        logger.error("Kaggle API credentials not configured.")
        _print_kaggle_setup_instructions()
        raise RuntimeError("Kaggle credentials not configured.")

    # Try competition download first
    logger.info("Downloading APTOS 2019 dataset to %s …", dest)
    try:
        subprocess.run(
            [
                "kaggle", "competitions", "download",
                "-c", KAGGLE_COMPETITION,
                "-p", str(dest),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        zip_path = dest / f"{KAGGLE_COMPETITION}.zip"
        if zip_path.is_file():
            logger.info("Downloaded competition archive → %s", zip_path)
            return zip_path
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "Competition download failed (%s). Trying dataset download …",
            exc.stderr.strip() if exc.stderr else exc,
        )

    # Fallback: dataset download
    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", KAGGLE_DATASET,
                "-p", str(dest),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("Dataset download failed: %s", exc.stderr)
        _print_kaggle_setup_instructions()
        raise RuntimeError(f"Download failed: {exc.stderr}") from exc

    # Locate the zip
    zips = list(dest.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No zip files found in {dest} after download.")

    zip_path = zips[0]
    logger.info("Downloaded dataset archive → %s", zip_path)
    return zip_path


# ---------------------------------------------------------------------------
# Extraction & organisation
# ---------------------------------------------------------------------------

def extract_and_organize(
    zip_path: str | Path,
    destination: Optional[str | Path] = None,
) -> Path:
    """Extract the downloaded archive and organise into the expected layout.

    Expected result::

        destination/
        ├── train.csv
        └── train_images/
            ├── 000c1434d8d7.png
            └── ...

    Parameters
    ----------
    zip_path : str | Path
        Path to the zip archive.
    destination : str | Path | None
        Target directory.  Defaults to ``DATASET_DIR/raw``.

    Returns
    -------
    Path
        Root directory containing the organised dataset.
    """
    zip_path = Path(zip_path)
    dest = Path(destination) if destination else Path(DATASET_DIR) / "raw"
    dest.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting %s → %s …", zip_path, dest)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    # The archive may contain a nested directory – flatten if necessary
    csv_candidates = list(dest.rglob(EXPECTED_CSV))
    if not csv_candidates:
        logger.warning("train.csv not found after extraction.  Archive layout may differ.")
        return dest

    csv_file = csv_candidates[0]
    dataset_root = csv_file.parent

    # If files are in a sub-directory, move them up
    if dataset_root != dest:
        for item in dataset_root.iterdir():
            target = dest / item.name
            if not target.exists():
                shutil.move(str(item), str(target))
        logger.info("Moved dataset files from %s to %s", dataset_root, dest)

    logger.info("Extraction and organisation complete.")
    return dest


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_dataset(data_dir: Optional[str | Path] = None) -> bool:
    """Verify that the extracted dataset contains the expected files.

    Parameters
    ----------
    data_dir : str | Path | None
        Root of the dataset directory.  Defaults to ``DATASET_DIR/raw``.

    Returns
    -------
    bool
        ``True`` if all expected files are present.
    """
    root = Path(data_dir) if data_dir else Path(DATASET_DIR) / "raw"
    logger.info("Verifying dataset at %s …", root)

    ok = True

    # Check CSV
    csv_path = root / EXPECTED_CSV
    if csv_path.is_file():
        import pandas as pd

        df = pd.read_csv(csv_path)
        logger.info("  ✓ %s found – %d rows", EXPECTED_CSV, len(df))
    else:
        logger.error("  ✗ %s not found!", EXPECTED_CSV)
        ok = False

    # Check images directory
    img_dir = root / EXPECTED_IMAGE_DIR
    if img_dir.is_dir():
        n_images = len(list(img_dir.glob("*")))
        logger.info("  ✓ %s/ found – %d files", EXPECTED_IMAGE_DIR, n_images)
        if n_images == 0:
            logger.warning("    ⚠ Image directory is empty.")
            ok = False
    else:
        # Images might be directly in root
        n_images = len(
            [f for f in root.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
        )
        if n_images > 0:
            logger.info("  ✓ Found %d images directly in %s", n_images, root)
        else:
            logger.error("  ✗ No image directory or images found!")
            ok = False

    if ok:
        logger.info("Dataset verification PASSED ✓")
    else:
        logger.error("Dataset verification FAILED ✗")

    return ok


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry-point: download → extract → verify."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    dest = Path(DATASET_DIR) / "raw"

    print("\n🔽 Step 1/3 – Download")
    print("-" * 40)
    try:
        zip_path = download_aptos_dataset(dest)
    except RuntimeError:
        sys.exit(1)

    print("\n📦 Step 2/3 – Extract & Organise")
    print("-" * 40)
    extract_and_organize(zip_path, dest)

    print("\n✅ Step 3/3 – Verify")
    print("-" * 40)
    success = verify_dataset(dest)

    if success:
        print("\n🎉 Dataset is ready for use!")
    else:
        print("\n⚠  Verification issues found – check logs above.")


if __name__ == "__main__":
    main()
