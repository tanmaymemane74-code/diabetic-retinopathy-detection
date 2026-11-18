# %% [markdown]
# # 01 – APTOS 2019 Diabetic Retinopathy Dataset Analysis
#
# This notebook provides a comprehensive exploration of the APTOS 2019
# Blindness Detection dataset used to train the DR-CDSS classification model.
#
# **Contents:**
# 1. Imports & setup
# 2. Load dataset
# 3. Dataset statistics & class distribution
# 4. Visualise class distribution
# 5. Display sample images per grade
# 6. Check for duplicates
# 7. Check for corrupted images
# 8. Create train / val / test splits
# 9. Visualise split distributions
# 10. Compute class weights
# 11. Image size analysis
# 12. Summary report

# %% Cell 1 – Imports and setup
import logging
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    sns = None
from PIL import Image

# Ensure the project root is on sys.path so src.* imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    DATASET_DIR,
    DR_CLASSES,
    IMAGE_SIZE,
    REPORTS_DIR,
)
from src.utils.dataset_loader import DRDatasetLoader

# Configure logging for notebook
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Matplotlib / seaborn defaults
if sns is not None:
    sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 150

print("✅ Imports complete")
print(f"   Project root : {PROJECT_ROOT}")
print(f"   Dataset dir  : {DATASET_DIR}")

# %% Cell 2 – Load APTOS dataset
# Point to the raw image directory and labels CSV.
# Adjust these paths if your dataset lives elsewhere.
DATA_DIR = Path(DATASET_DIR) / "raw" / "train_images"
CSV_PATH = Path(DATASET_DIR) / "raw" / "train.csv"

# Fall back to searching inside DATASET_DIR if the above doesn't exist
if not DATA_DIR.is_dir():
    # Try alternative layouts
    for candidate in [
        Path(DATASET_DIR) / "train_images",
        Path(DATASET_DIR) / "raw",
        Path(DATASET_DIR),
    ]:
        if candidate.is_dir() and any(candidate.glob("*.png")):
            DATA_DIR = candidate
            break
    logger.warning("Using fallback data directory: %s", DATA_DIR)

if not CSV_PATH.is_file():
    for candidate in [
        Path(DATASET_DIR) / "train.csv",
        Path(DATASET_DIR) / "raw" / "train.csv",
    ]:
        if candidate.is_file():
            CSV_PATH = candidate
            break

if not CSV_PATH.is_file():
    logger.warning("No real dataset found at %s. Creating synthetic demo dataset...", CSV_PATH)
    raw_dir = Path(DATASET_DIR) / "raw"
    images_dir = raw_dir / "train_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    CSV_PATH = raw_dir / "train.csv"
    DATA_DIR = images_dir

    demo_records = []
    # Generate 5 samples per grade for demo
    for grade in range(5):
        for i in range(5):
            id_code = f"demo_grade{grade}_{i:03d}"
            img_path = images_dir / f"{id_code}.png"
            if not img_path.exists():
                h, w = 300, 300
                demo_img = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.circle(demo_img, (w // 2, h // 2), int(w * 0.42), (20, 60, 180), -1)
                cv2.circle(demo_img, (int(w * 0.38), int(h * 0.50)), int(w * 0.08), (80, 200, 240), -1)
                cv2.imwrite(str(img_path), demo_img)
            demo_records.append({"id_code": id_code, "diagnosis": grade})

    pd.DataFrame(demo_records).to_csv(str(CSV_PATH), index=False)
    logger.info("Created synthetic demo dataset with %d samples at %s", len(demo_records), CSV_PATH)

loader = DRDatasetLoader(data_dir=DATA_DIR, csv_path=CSV_PATH)
df = loader.load_labels()

print(f"\nLoaded {len(df)} records.")
df.head(10)

# %% Cell 3 – Dataset statistics and class distribution
stats = loader.analyze_dataset()

print("\n📊 Dataset Statistics")
print("=" * 50)
print(f"  Total images   : {stats['total_images']}")
print(f"  Missing images : {stats['missing_images']}")
print()
print("  Class Distribution:")
print("  " + "-" * 40)
for grade in sorted(stats["class_distribution"].keys()):
    name = DR_CLASSES.get(grade, f"Grade {grade}")
    count = stats["class_distribution"][grade]
    pct = stats["class_percentages"][grade]
    print(f"    {grade} – {name:<25s} : {count:>5d}  ({pct:>5.1f}%)")

# %% Cell 4 – Visualise class distribution (bar chart)
save_dir = Path(REPORTS_DIR) / "figures"
save_dir.mkdir(parents=True, exist_ok=True)

loader.visualize_distribution(save_path=save_dir / "class_distribution.png")

# %% Cell 5 – Display sample images per grade (grid)
loader.visualize_samples(n_per_class=4, save_path=save_dir / "sample_images.png")

# %% Cell 6 – Check for duplicates
print("\n🔍 Checking for duplicate images …")
duplicates = loader.check_duplicates(method="hash")

if duplicates:
    print(f"\n⚠  Found {len(duplicates)} duplicate pair(s):")
    for a, b in duplicates[:10]:
        print(f"    • {Path(a).name}  ↔  {Path(b).name}")
    if len(duplicates) > 10:
        print(f"    … and {len(duplicates) - 10} more.")
else:
    print("✅ No duplicate images found.")

# %% Cell 7 – Check for corrupted images
print("\n🔍 Checking for corrupted images …")
corrupted = loader.check_corrupted()

if corrupted:
    print(f"\n⚠  Found {len(corrupted)} corrupted image(s):")
    for p in corrupted[:10]:
        print(f"    • {Path(p).name}")
    if len(corrupted) > 10:
        print(f"    … and {len(corrupted) - 10} more.")
else:
    print("✅ No corrupted images found.")

# %% Cell 8 – Create train / val / test splits
print("\n✂️  Creating stratified splits (70 / 15 / 15) …")
train_df, val_df, test_df = loader.create_splits(
    test_size=0.15, val_size=0.15, random_state=42
)

print(f"\n  Train : {len(train_df):>5d} images")
print(f"  Val   : {len(val_df):>5d} images")
print(f"  Test  : {len(test_df):>5d} images")
print(f"  Total : {len(train_df) + len(val_df) + len(test_df):>5d}")

# %% Cell 9 – Visualise split distributions
loader.visualize_splits(
    train_df, val_df, test_df,
    save_path=save_dir / "split_distribution.png",
)

# %% Cell 10 – Compute class weights
print("\n⚖️  Computing class weights for imbalanced training …")
class_weights = loader.get_class_weights(train_df)

print("\n  Class Weights:")
for grade, weight in sorted(class_weights.items()):
    name = DR_CLASSES.get(grade, f"Grade {grade}")
    print(f"    {grade} – {name:<25s} : {weight:.4f}")

# %% Cell 11 – Image size analysis
print("\n📐 Image size analysis …")
size_stats = stats.get("image_size_stats", {})

if size_stats:
    print(f"  Sampled          : {size_stats.get('sampled', 'N/A')}")
    print(
        f"  Width  (mean±std): {size_stats.get('width_mean', 0):.0f}"
        f" ± {size_stats.get('width_std', 0):.0f}"
        f"  [{size_stats.get('width_min', 0)} – {size_stats.get('width_max', 0)}]"
    )
    print(
        f"  Height (mean±std): {size_stats.get('height_mean', 0):.0f}"
        f" ± {size_stats.get('height_std', 0):.0f}"
        f"  [{size_stats.get('height_min', 0)} – {size_stats.get('height_max', 0)}]"
    )

    # Quick histogram of image widths
    existing = df.loc[df["image_exists"], "image_path"].tolist()[:200]
    widths, heights = [], []
    for p in existing:
        try:
            img = Image.open(p)
            w, h = img.size
            widths.append(w)
            heights.append(h)
        except Exception:
            continue

    if widths:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(widths, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
        axes[0].set_title("Image Width Distribution")
        axes[0].set_xlabel("Width (px)")
        axes[0].set_ylabel("Count")

        axes[1].hist(heights, bins=30, color="coral", edgecolor="black", alpha=0.8)
        axes[1].set_title("Image Height Distribution")
        axes[1].set_xlabel("Height (px)")
        axes[1].set_ylabel("Count")

        fig.tight_layout()
        fig.savefig(str(save_dir / "image_size_distribution.png"), dpi=150, bbox_inches="tight")
        plt.show()
else:
    print("  ⚠ No images available for size analysis.")

# %% Cell 12 – Summary report
print("\n" + loader.generate_dataset_report())

# Save report to file
report_path = Path(REPORTS_DIR) / "dataset_analysis_report.txt"
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(loader.generate_dataset_report(), encoding="utf-8")
print(f"\n📄 Report saved to {report_path}")
