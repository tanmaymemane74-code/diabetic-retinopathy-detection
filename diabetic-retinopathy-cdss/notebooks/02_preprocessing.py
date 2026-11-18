# %% [markdown]
# # 02 – Diabetic Retinopathy Fundus Preprocessing Pipeline
#
# This notebook demonstrates the end-to-end preprocessing, quality assessment,
# and data augmentation pipeline for fundus images.
#
# **Contents:**
# 1. Imports and setup
# 2. Load / generate sample fundus image
# 3. Full preprocessing pipeline visualization (ROI crop, CLAHE, Ben Graham, normalization)
# 4. Image quality assessment (blur, illumination, contrast, FOV, composite score)
# 5. Data augmentation visualization (rotations, flips, zoom, brightness)
# 6. Batch preprocessing test
# 7. Before vs. after comparison grid

# %% Cell 1: Imports
import logging
import os
import sys
import tempfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    DATASET_DIR,
    IMAGE_SIZE,
    REPORTS_DIR,
)
from src.preprocessing.preprocessor import DRImagePreprocessor
from src.preprocessing.quality_assessment import ImageQualityAssessor
from src.preprocessing.augmentation import DRAugmentor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Output directories for demonstration figures
FIG_DIR = Path(REPORTS_DIR) / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("✅ Preprocessing imports loaded successfully.")
print(f"   Project root : {PROJECT_ROOT}")
print(f"   Figures dir  : {FIG_DIR}")

# %% Cell 2: Load a sample image
def get_or_create_sample_image() -> Path:
    """Find a real sample image from the dataset, or create a synthetic fundus image."""
    raw_candidates = list(Path(DATASET_DIR).rglob("*.png")) + list(Path(DATASET_DIR).rglob("*.jpg"))
    if raw_candidates:
        logger.info("Found real sample image: %s", raw_candidates[0])
        return raw_candidates[0]

    # Create synthetic retinal fundus-like image for testing/demonstration
    logger.info("No real dataset image found. Creating synthetic fundus image for demonstration.")
    synthetic_dir = PROJECT_ROOT / "dataset" / "synthetic_samples"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    sample_path = synthetic_dir / "sample_fundus_demonstration.png"

    h, w = 600, 600
    img = np.zeros((h, w, 3), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = int(w * 0.42)

    # Fundus reddish-orange background circle
    cv2.circle(img, center, radius, (20, 60, 180), -1)

    # Add optic disc (bright yellowish circle)
    disc_center = (int(w * 0.38), int(h * 0.50))
    cv2.circle(img, disc_center, int(radius * 0.18), (80, 200, 240), -1)

    # Add macula (darker reddish spot)
    macula_center = (int(w * 0.60), int(h * 0.52))
    cv2.circle(img, macula_center, int(radius * 0.14), (10, 30, 120), -1)

    # Add simulated blood vessels (branching curves)
    for angle in [-0.5, -0.2, 0.2, 0.5, 1.0, -1.0]:
        pts = []
        for r in np.linspace(radius * 0.18, radius * 0.85, 30):
            theta = angle + np.sin(r / 25.0) * 0.3
            x = int(disc_center[0] + r * np.cos(theta))
            y = int(disc_center[1] + r * np.sin(theta))
            pts.append([x, y])
        pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=False, color=(10, 20, 90), thickness=2)

    # Add some simulated microaneurysms / hemorrhages (red dots)
    for _ in range(12):
        rx = np.random.randint(int(w * 0.3), int(w * 0.7))
        ry = np.random.randint(int(h * 0.3), int(h * 0.7))
        cv2.circle(img, (rx, ry), np.random.randint(2, 5), (5, 10, 110), -1)

    # Smooth the image slightly to emulate real photography
    img = cv2.GaussianBlur(img, (5, 5), 0)

    cv2.imwrite(str(sample_path), img)
    logger.info("Saved synthetic sample image to %s", sample_path)
    return sample_path

sample_image_path = get_or_create_sample_image()
print(f"Sample image ready: {sample_image_path}")

# Load and show raw sample
raw_bgr = cv2.imread(str(sample_image_path))
raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(5, 5))
plt.imshow(raw_rgb)
plt.title(f"Original Fundus ({raw_bgr.shape[1]}x{raw_bgr.shape[0]})", fontsize=11)
plt.axis("off")
plt.show()

# %% Cell 3: Run full preprocessing pipeline with visualization
preprocessor = DRImagePreprocessor(target_size=IMAGE_SIZE)

print("\n🚀 Running pipeline visualization...")
pipeline_fig_path = FIG_DIR / "preprocessing_pipeline_stages.png"
fig = preprocessor.visualize_pipeline(sample_image_path, save_path=pipeline_fig_path)
if fig is not None:
    plt.show()

# Demonstrate individual components
print("\nIndividual Preprocessing Operations:")
raw_loaded = preprocessor.load_image(sample_image_path)
rgb_img = preprocessor.convert_to_rgb(raw_loaded)
cropped_roi = preprocessor.crop_circular_roi(rgb_img)
clahe_img = preprocessor.apply_clahe(cropped_roi)
denoised_img = preprocessor.reduce_noise(clahe_img)
ben_graham_img = preprocessor.ben_graham_preprocessing(denoised_img)
normalized_img = preprocessor.normalize_pixels(ben_graham_img)

print(f"  • Original image shape    : {raw_loaded.shape}, dtype: {raw_loaded.dtype}")
print(f"  • Cropped ROI shape       : {cropped_roi.shape}")
print(f"  • CLAHE enhanced shape    : {clahe_img.shape}")
print(f"  • Ben Graham shape        : {ben_graham_img.shape}")
print(f"  • Final Normalized shape  : {normalized_img.shape}, min={normalized_img.min():.3f}, max={normalized_img.max():.3f}, dtype={normalized_img.dtype}")

# %% Cell 4: Quality assessment on multiple images
print("\n🔍 Evaluating Image Quality...")
assessor = ImageQualityAssessor(
    blur_threshold=100.0,
    brightness_low=40.0,
    brightness_high=220.0,
    contrast_threshold=35.0,
    fov_threshold=0.30,
)

# Test 1: Good image report
report_good = assessor.get_quality_report(sample_image_path)
print("\n--- Assessment Report: Good Fundus Sample ---")
for k, v in report_good.items():
    print(f"  {k:<20s}: {v}")

# Test 2: Simulated Blurry Image
blurry_img = cv2.GaussianBlur(raw_rgb, (35, 35), 10)
report_blur = assessor.get_quality_report(blurry_img)
print("\n--- Assessment Report: Blurry Image ---")
print(f"  Blur score           : {report_blur['blur']['score']:.2f} (is_blurry: {report_blur['blur']['is_blurry']})")
print(f"  Overall Grade        : {report_blur['overall_quality']}")

# Test 3: Simulated Dark / Underexposed Image
dark_img = (raw_rgb * 0.15).astype(np.uint8)
report_dark = assessor.get_quality_report(dark_img)
print("\n--- Assessment Report: Dark Image ---")
print(f"  Mean brightness      : {report_dark['illumination']['score']:.2f} (poor illumination: {report_dark['illumination']['is_poor']})")
print(f"  Overall Grade        : {report_dark['overall_quality']}")

# Quality comparison plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(raw_rgb)
axes[0].set_title(f"Standard Fundus\nGrade: {report_good['overall_quality'].upper()}", fontsize=11)
axes[0].axis("off")

axes[1].imshow(blurry_img)
axes[1].set_title(f"Degraded: Blurry\nGrade: {report_blur['overall_quality'].upper()}", fontsize=11)
axes[1].axis("off")

axes[2].imshow(dark_img)
axes[2].set_title(f"Degraded: Underexposed\nGrade: {report_dark['overall_quality'].upper()}", fontsize=11)
axes[2].axis("off")

plt.tight_layout()
quality_fig_path = FIG_DIR / "quality_assessment_comparison.png"
plt.savefig(str(quality_fig_path), dpi=150, bbox_inches="tight")
plt.show()

# %% Cell 5: Augmentation visualization
print("\n🔄 Generating data augmentations...")
try:
    augmentor = DRAugmentor()
    aug_fig_path = FIG_DIR / "sample_augmentations.png"
    augmentor.visualize_augmentations(
        image_path=sample_image_path,
        n=8,
        target_size=IMAGE_SIZE,
        save_path=aug_fig_path,
    )
    print(f"Augmentation preview saved to {aug_fig_path}")
except Exception as exc:
    print(f"Skipping augmentation visualization due to environment restriction: {exc}")

# %% Cell 6: Batch preprocessing test
print("\n⚡ Testing Batch Preprocessing...")
# Create a list of test image paths
test_batch_paths = [sample_image_path] * 6
print(f"Processing batch of {len(test_batch_paths)} images with DRImagePreprocessor.preprocess_batch...")

processed_batch = preprocessor.preprocess_batch(test_batch_paths, num_workers=2)
print(f"Successfully processed {len(processed_batch)} images in batch.")
assert len(processed_batch) == 6
assert processed_batch[0].shape == (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
print(f"Batch item shape: {processed_batch[0].shape}, range: [{processed_batch[0].min():.2f}, {processed_batch[0].max():.2f}]")

# %% Cell 7: Before/after comparison grid
print("\n📊 Generating Before / After Comparison Grid...")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Diabetic Retinopathy Preprocessing Pipeline: Before vs. After", fontsize=15, fontweight="bold")

# Show original vs processed for 4 sample views (original, flipped, slightly rotated)
for idx in range(4):
    ax_before = axes[0, idx]
    ax_after = axes[1, idx]

    # Generate slightly altered variant for demonstration
    if idx == 0:
        variant = raw_bgr.copy()
    elif idx == 1:
        variant = cv2.flip(raw_bgr, 1)
    elif idx == 2:
        variant = cv2.rotate(raw_bgr, cv2.ROTATE_90_CLOCKWISE)
    else:
        variant = cv2.convertScaleAbs(raw_bgr, alpha=1.2, beta=10)

    # Save temp variant to run preprocess_single
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_name = tmp.name
    cv2.imwrite(tmp_name, variant)

    try:
        processed = preprocessor.preprocess_single(tmp_name, apply_ben_graham=True)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

    # Display Before
    ax_before.imshow(cv2.cvtColor(variant, cv2.COLOR_BGR2RGB))
    ax_before.set_title(f"Sample {idx + 1} - Raw Input", fontsize=11)
    ax_before.axis("off")

    # Display After
    ax_after.imshow(processed)
    ax_after.set_title(f"Sample {idx + 1} - Standardized", fontsize=11)
    ax_after.axis("off")

plt.tight_layout()
comparison_fig_path = FIG_DIR / "preprocessing_before_after_grid.png"
plt.savefig(str(comparison_fig_path), dpi=150, bbox_inches="tight")
plt.show()

print(f"\n🎉 Demonstration complete! All visual artifacts saved to {FIG_DIR}")
