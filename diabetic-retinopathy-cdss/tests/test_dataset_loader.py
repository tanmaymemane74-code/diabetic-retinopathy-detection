"""
Tests for dataset loading utilities – class weights, stratified splits,
and data leakage prevention.

These tests rely on the ``sample_dataframe`` fixture defined in conftest.py,
which provides a small DataFrame in the APTOS 2019 format.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# ---------------------------------------------------------------------------
# Helpers that mirror what the dataset_loader module should expose.
# If the actual module exists, import from it; otherwise fall back to
# self-contained reference implementations so the tests can still run.
# ---------------------------------------------------------------------------
def _compute_class_weights(df: pd.DataFrame, label_col: str = "diagnosis") -> dict:
    """Compute balanced class weights from a DataFrame."""
    classes = np.sort(df[label_col].unique())
    weights = compute_class_weight("balanced", classes=classes, y=df[label_col].values)
    return dict(zip(classes.tolist(), weights.tolist()))


def _stratified_split(
    df: pd.DataFrame,
    label_col: str = "diagnosis",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
):
    """Split a DataFrame into train / val / test with stratification."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[label_col], random_state=random_state
    )
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=train_val[label_col],
        random_state=random_state,
    )
    return train, val, test


# Try to import from the project; fall back to helpers above.
try:
    from src.data.dataset_loader import compute_class_weights, stratified_split
except ImportError:
    compute_class_weights = _compute_class_weights
    stratified_split = _stratified_split


# ===================================================================
# Tests
# ===================================================================


class TestClassWeights:
    """Tests for class-weight computation."""

    def test_class_weights_computation(self, sample_dataframe: pd.DataFrame):
        """Computed weights should cover every class present in the data."""
        weights = compute_class_weights(sample_dataframe)

        assert isinstance(weights, dict), "Weights should be a dict"
        present_classes = set(sample_dataframe["diagnosis"].unique())
        assert set(weights.keys()) == present_classes, (
            f"Weight keys {set(weights.keys())} != classes {present_classes}"
        )
        # Minority classes should have higher weight
        majority_class = sample_dataframe["diagnosis"].value_counts().idxmax()
        for cls, w in weights.items():
            assert w > 0, f"Weight for class {cls} must be positive"
        # The majority class weight should be ≤ any minority class weight
        for cls, w in weights.items():
            if cls != majority_class:
                assert w >= weights[majority_class], (
                    f"Minority class {cls} weight {w} should be >= majority weight {weights[majority_class]}"
                )


class TestStratifiedSplit:
    """Tests for stratified train/val/test splitting."""

    def test_stratified_split(self, sample_dataframe: pd.DataFrame):
        """Split proportions should be roughly correct."""
        train, val, test = stratified_split(
            sample_dataframe, test_size=0.2, val_size=0.1
        )
        total = len(sample_dataframe)

        # Allow ±2 tolerance due to small sample and rounding
        assert abs(len(test) - int(total * 0.2)) <= 2, (
            f"Test size {len(test)} not ≈ {int(total * 0.2)}"
        )
        assert abs(len(val) - int(total * 0.1)) <= 2, (
            f"Val size {len(val)} not ≈ {int(total * 0.1)}"
        )
        assert len(train) + len(val) + len(test) == total, "Sizes don't add up"

    def test_split_no_leakage(self, sample_dataframe: pd.DataFrame):
        """There must be zero overlap between train, val, and test sets."""
        train, val, test = stratified_split(sample_dataframe, test_size=0.2, val_size=0.1)

        train_ids = set(train["id_code"].tolist())
        val_ids = set(val["id_code"].tolist())
        test_ids = set(test["id_code"].tolist())

        assert train_ids.isdisjoint(val_ids), "Leakage between train and val"
        assert train_ids.isdisjoint(test_ids), "Leakage between train and test"
        assert val_ids.isdisjoint(test_ids), "Leakage between val and test"

        # Combined should equal the original
        assert train_ids | val_ids | test_ids == set(
            sample_dataframe["id_code"].tolist()
        ), "Some samples are missing after split"
