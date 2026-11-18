"""
Dataset loading, splitting, and analysis module for the DR-CDSS project.

Handles the APTOS 2019 Blindness Detection dataset: loading labels,
analyzing class distributions, checking data quality, creating stratified
splits, computing class weights, and generating visualizations.
"""

import hashlib
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import seaborn as sns
except ImportError:
    sns = None
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from src.utils.config import (
    DATASET_DIR,
    DR_CLASSES,
    IMAGE_SIZE,
    REPORTS_DIR,
)

logger = logging.getLogger(__name__)


class DRDatasetLoader:
    """Handles loading, splitting, and analyzing the APTOS 2019 DR dataset.

    Parameters
    ----------
    data_dir : str | Path
        Path to the directory containing retinal images.
    csv_path : str | Path | None
        Path to the labels CSV file (``train.csv`` from APTOS).  When *None*
        the loader searches for ``train.csv`` in *data_dir* and its parent.
    image_extensions : tuple[str, ...]
        File extensions recognised as images (case-insensitive).
    """

    IMAGE_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    def __init__(
        self,
        data_dir: str | Path,
        csv_path: Optional[str | Path] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.csv_path = self._resolve_csv(csv_path)
        self.labels_df: Optional[pd.DataFrame] = None
        self._class_names: Dict[int, str] = DR_CLASSES

        logger.info("DRDatasetLoader initialised – data_dir=%s, csv=%s", self.data_dir, self.csv_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_csv(self, csv_path: Optional[str | Path]) -> Path:
        """Locate the labels CSV, searching common locations when not given."""
        if csv_path is not None:
            p = Path(csv_path)
            if p.is_file():
                return p
            raise FileNotFoundError(f"CSV path does not exist: {p}")

        # Search in data_dir, its parent, and DATASET_DIR
        candidates = [
            self.data_dir / "train.csv",
            self.data_dir.parent / "train.csv",
            Path(DATASET_DIR) / "train.csv",
        ]
        for candidate in candidates:
            if candidate.is_file():
                logger.info("Auto-detected CSV at %s", candidate)
                return candidate

        raise FileNotFoundError(
            "Could not auto-detect train.csv.  Provide csv_path explicitly."
        )

    def _image_path(self, id_code: str) -> Path:
        """Return the full path for an image id, trying known extensions."""
        for ext in self.IMAGE_EXTENSIONS:
            p = self.data_dir / f"{id_code}{ext}"
            if p.is_file():
                return p
        return self.data_dir / f"{id_code}.png"  # fallback

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_labels(self) -> pd.DataFrame:
        """Load the APTOS CSV and add an ``image_path`` column.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``id_code``, ``diagnosis``, ``image_path``,
            and ``image_exists``.
        """
        logger.info("Loading labels from %s …", self.csv_path)
        df = pd.read_csv(self.csv_path)

        required_cols = {"id_code", "diagnosis"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"CSV must contain columns {required_cols}; found {set(df.columns)}"
            )

        df["image_path"] = df["id_code"].apply(lambda x: str(self._image_path(x)))
        df["image_exists"] = df["image_path"].apply(lambda x: Path(x).is_file())

        missing = (~df["image_exists"]).sum()
        if missing:
            logger.warning("%d images referenced in CSV are missing on disk.", missing)
        else:
            logger.info("All %d images verified on disk.", len(df))

        self.labels_df = df
        return df

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def analyze_dataset(self) -> Dict[str, Any]:
        """Return comprehensive dataset statistics.

        Returns
        -------
        dict
            Keys: ``total_images``, ``class_distribution``,
            ``class_percentages``, ``missing_images``, ``image_size_stats``.
        """
        if self.labels_df is None:
            self.load_labels()

        df = self.labels_df
        assert df is not None

        class_dist = df["diagnosis"].value_counts().sort_index().to_dict()
        total = len(df)
        class_pct = {k: round(v / total * 100, 2) for k, v in class_dist.items()}
        missing = int((~df["image_exists"]).sum())

        # Image-size statistics (sample up to 200 images for speed)
        size_stats = self._compute_image_size_stats(df, sample_n=200)

        stats: Dict[str, Any] = {
            "total_images": total,
            "class_distribution": class_dist,
            "class_percentages": class_pct,
            "missing_images": missing,
            "image_size_stats": size_stats,
        }
        logger.info("Dataset analysis complete – %d images, %d missing.", total, missing)
        return stats

    def _compute_image_size_stats(
        self, df: pd.DataFrame, sample_n: int = 200
    ) -> Dict[str, Any]:
        """Sample images and compute width/height statistics."""
        existing = df.loc[df["image_exists"], "image_path"].tolist()
        if not existing:
            return {}

        sample = existing[: min(sample_n, len(existing))]
        widths, heights = [], []
        for p in sample:
            try:
                img = Image.open(p)
                w, h = img.size
                widths.append(w)
                heights.append(h)
            except Exception:
                continue

        if not widths:
            return {}

        return {
            "sampled": len(widths),
            "width_mean": float(np.mean(widths)),
            "width_std": float(np.std(widths)),
            "width_min": int(np.min(widths)),
            "width_max": int(np.max(widths)),
            "height_mean": float(np.mean(heights)),
            "height_std": float(np.std(heights)),
            "height_min": int(np.min(heights)),
            "height_max": int(np.max(heights)),
        }

    # ------------------------------------------------------------------
    # Quality checks
    # ------------------------------------------------------------------
    def check_duplicates(self, method: str = "hash") -> List[Tuple[str, str]]:
        """Detect duplicate images by MD5 hash.

        Parameters
        ----------
        method : str
            Currently only ``'hash'`` is supported.

        Returns
        -------
        list[tuple[str, str]]
            Pairs of paths that share the same hash.
        """
        if self.labels_df is None:
            self.load_labels()
        assert self.labels_df is not None

        logger.info("Checking for duplicate images (method=%s) …", method)
        hash_map: Dict[str, List[str]] = defaultdict(list)
        paths = self.labels_df.loc[self.labels_df["image_exists"], "image_path"].tolist()

        for p in tqdm(paths, desc="Hashing images", leave=False):
            try:
                h = hashlib.md5(Path(p).read_bytes()).hexdigest()
                hash_map[h].append(p)
            except Exception as exc:
                logger.debug("Could not hash %s: %s", p, exc)

        duplicates: List[Tuple[str, str]] = []
        for h, group in hash_map.items():
            if len(group) > 1:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        duplicates.append((group[i], group[j]))

        logger.info("Found %d duplicate pairs.", len(duplicates))
        return duplicates

    def check_corrupted(self) -> List[str]:
        """Identify images that cannot be opened.

        Returns
        -------
        list[str]
            Paths of corrupted or unreadable images.
        """
        if self.labels_df is None:
            self.load_labels()
        assert self.labels_df is not None

        logger.info("Checking for corrupted images …")
        corrupted: List[str] = []
        paths = self.labels_df.loc[self.labels_df["image_exists"], "image_path"].tolist()

        for p in tqdm(paths, desc="Validating images", leave=False):
            try:
                img = Image.open(p)
                img.verify()
            except Exception as exc:
                logger.debug("Corrupted image %s: %s", p, exc)
                corrupted.append(p)

        logger.info("Found %d corrupted images.", len(corrupted))
        return corrupted

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------
    def create_splits(
        self,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create stratified train / validation / test splits.

        Parameters
        ----------
        test_size : float
            Fraction of data for the test set.
        val_size : float
            Fraction of *total* data for the validation set.
        random_state : int
            Seed for reproducibility.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            ``(train_df, val_df, test_df)`` DataFrames.
        """
        if self.labels_df is None:
            self.load_labels()
        assert self.labels_df is not None

        df = self.labels_df[self.labels_df["image_exists"]].copy()

        # Check if stratification is mathematically possible given sample count
        n_classes = df["diagnosis"].nunique()
        min_class_count = df["diagnosis"].value_counts().min()
        can_stratify = (min_class_count >= 2) and (int(len(df) * test_size) >= n_classes)
        strat = df["diagnosis"] if can_stratify else None

        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=strat,
        )

        # Second split: train vs val (adjust proportion relative to train+val)
        relative_val_size = val_size / (1 - test_size)
        can_stratify_val = (
            can_stratify
            and (int(len(train_val_df) * relative_val_size) >= train_val_df["diagnosis"].nunique())
        )
        val_strat = train_val_df["diagnosis"] if can_stratify_val else None

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            random_state=random_state,
            stratify=val_strat,
        )

        # Log statistics
        for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            dist = split_df["diagnosis"].value_counts().sort_index().to_dict()
            logger.info(
                "%s split: %d images – distribution: %s", name, len(split_df), dist
            )

        logger.info(
            "Split sizes → Train: %d (%.1f%%), Val: %d (%.1f%%), Test: %d (%.1f%%)",
            len(train_df),
            len(train_df) / len(df) * 100,
            len(val_df),
            len(val_df) / len(df) * 100,
            len(test_df),
            len(test_df) / len(df) * 100,
        )

        return train_df, val_df, test_df

    # ------------------------------------------------------------------
    # Class weights
    # ------------------------------------------------------------------
    def get_class_weights(self, train_df: pd.DataFrame) -> Dict[int, float]:
        """Compute balanced class weights using scikit-learn.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training split with a ``diagnosis`` column.

        Returns
        -------
        dict[int, float]
            Mapping from class label (0–4) to weight.
        """
        classes = np.array(sorted(train_df["diagnosis"].unique()))
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=train_df["diagnosis"].values,
        )
        weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
        logger.info("Class weights: %s", weight_dict)
        return weight_dict

    # ------------------------------------------------------------------
    # Sample images
    # ------------------------------------------------------------------
    def get_sample_images(self, n_per_class: int = 3) -> Dict[int, List[str]]:
        """Return a dictionary mapping each DR grade to sample image paths.

        Parameters
        ----------
        n_per_class : int
            Number of sample images per class.

        Returns
        -------
        dict[int, list[str]]
            ``{0: [path, ...], 1: [...], ...}``
        """
        if self.labels_df is None:
            self.load_labels()
        assert self.labels_df is not None

        df = self.labels_df[self.labels_df["image_exists"]]
        samples: Dict[int, List[str]] = {}
        for grade in sorted(df["diagnosis"].unique()):
            grade_df = df[df["diagnosis"] == grade]
            sampled = grade_df.sample(n=min(n_per_class, len(grade_df)), random_state=42)
            samples[int(grade)] = sampled["image_path"].tolist()

        return samples

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------
    def visualize_distribution(self, save_path: Optional[str | Path] = None) -> None:
        """Bar chart of class distribution with counts and percentages.

        Parameters
        ----------
        save_path : str | Path | None
            If given, save the figure to this path.
        """
        if self.labels_df is None:
            self.load_labels()
        assert self.labels_df is not None

        df = self.labels_df
        counts = df["diagnosis"].value_counts().sort_index()
        total = counts.sum()

        fig, ax = plt.subplots(figsize=(10, 6))
        if sns is not None:
            sns.set_style("whitegrid")
            palette = sns.color_palette("viridis", n_colors=len(counts))
        else:
            ax.grid(True, linestyle="--", alpha=0.6)
            palette = plt.cm.viridis(np.linspace(0.2, 0.9, len(counts)))

        bars = ax.bar(
            [self._class_names.get(i, str(i)) for i in counts.index],
            counts.values,
            color=palette,
            edgecolor="black",
            linewidth=0.8,
        )

        # Annotate bars
        for bar, count in zip(bars, counts.values):
            pct = count / total * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.005,
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_title("APTOS 2019 – DR Grade Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("DR Grade", fontsize=12)
        ax.set_ylabel("Number of Images", fontsize=12)
        ax.set_ylim(0, counts.max() * 1.2)
        fig.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
            logger.info("Distribution chart saved to %s", save_path)

        plt.show()

    def visualize_samples(
        self, n_per_class: int = 3, save_path: Optional[str | Path] = None
    ) -> None:
        """Display a grid of sample retinal images per DR grade.

        Parameters
        ----------
        n_per_class : int
            Number of images per row.
        save_path : str | Path | None
            If given, save the figure to this path.
        """
        samples = self.get_sample_images(n_per_class)
        n_classes = len(samples)

        fig, axes = plt.subplots(
            n_classes, n_per_class, figsize=(4 * n_per_class, 4 * n_classes)
        )
        if n_classes == 1:
            axes = [axes]

        for row_idx, grade in enumerate(sorted(samples.keys())):
            paths = samples[grade]
            for col_idx in range(n_per_class):
                ax = axes[row_idx][col_idx] if n_per_class > 1 else axes[row_idx]
                if col_idx < len(paths):
                    try:
                        img = Image.open(paths[col_idx]).convert("RGB")
                        ax.imshow(img)
                    except Exception:
                        ax.text(0.5, 0.5, "Error", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(
                    f"Grade {grade}: {self._class_names.get(grade, '?')}",
                    fontsize=10,
                )
                ax.axis("off")

        fig.suptitle(
            "Sample Retinal Images by DR Grade",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )
        fig.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
            logger.info("Sample images grid saved to %s", save_path)

        plt.show()

    def visualize_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        save_path: Optional[str | Path] = None,
    ) -> None:
        """Grouped bar chart comparing class distributions across splits.

        Parameters
        ----------
        train_df, val_df, test_df : pd.DataFrame
            DataFrames from :meth:`create_splits`.
        save_path : str | Path | None
            If given, save the figure to this path.
        """
        splits = {"Train": train_df, "Validation": val_df, "Test": test_df}
        records = []
        for split_name, split_df in splits.items():
            counts = split_df["diagnosis"].value_counts().sort_index()
            for grade, count in counts.items():
                records.append(
                    {
                        "DR Grade": self._class_names.get(int(grade), str(grade)),
                        "Count": count,
                        "Split": split_name,
                    }
                )

        plot_df = pd.DataFrame(records)

        fig, ax = plt.subplots(figsize=(12, 6))
        if sns is not None:
            sns.set_style("whitegrid")
            sns.barplot(
                data=plot_df,
                x="DR Grade",
                y="Count",
                hue="Split",
                palette="Set2",
                edgecolor="black",
                linewidth=0.5,
                ax=ax,
            )
        else:
            pivot_df = plot_df.pivot(index="DR Grade", columns="Split", values="Count").fillna(0)
            pivot_df.plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.5, colormap="tab10")
            ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_title(
            "Class Distribution Across Train / Val / Test Splits",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("DR Grade", fontsize=12)
        ax.set_ylabel("Number of Images", fontsize=12)
        ax.legend(title="Split", fontsize=10, title_fontsize=11)
        fig.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
            logger.info("Split distribution chart saved to %s", save_path)

        plt.show()

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def generate_dataset_report(self) -> str:
        """Return a formatted text report with all dataset statistics.

        Returns
        -------
        str
            Multi-line report string.
        """
        stats = self.analyze_dataset()
        lines = [
            "=" * 60,
            "  APTOS 2019 Diabetic Retinopathy Dataset Report",
            "=" * 60,
            "",
            f"  Total images        : {stats['total_images']}",
            f"  Missing images      : {stats['missing_images']}",
            "",
            "  Class Distribution:",
            "  " + "-" * 40,
        ]

        for grade in sorted(stats["class_distribution"].keys()):
            name = self._class_names.get(grade, f"Grade {grade}")
            count = stats["class_distribution"][grade]
            pct = stats["class_percentages"][grade]
            lines.append(f"    {grade} – {name:<25s} : {count:>5d}  ({pct:>5.1f}%)")

        lines.append("")
        lines.append("  Image Size Statistics:")
        lines.append("  " + "-" * 40)
        sz = stats.get("image_size_stats", {})
        if sz:
            lines.append(f"    Sampled          : {sz.get('sampled', 'N/A')}")
            lines.append(
                f"    Width  (mean±std): {sz.get('width_mean', 0):.0f} ± {sz.get('width_std', 0):.0f}  "
                f"[{sz.get('width_min', 0)} – {sz.get('width_max', 0)}]"
            )
            lines.append(
                f"    Height (mean±std): {sz.get('height_mean', 0):.0f} ± {sz.get('height_std', 0):.0f}  "
                f"[{sz.get('height_min', 0)} – {sz.get('height_max', 0)}]"
            )
        else:
            lines.append("    (no images available for size analysis)")

        lines.append("")
        lines.append("=" * 60)

        report = "\n".join(lines)
        logger.info("Dataset report generated.")
        return report
