"""
VisionTrack - Dataset Partitioning Engine
=========================================
Leak-free, deterministic dataset splitting developed during Day 2.
"""

import random
from pathlib import Path


def split_dataset(
    sample_ids: list[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    """
    Splits sample IDs into train, val, and test sets reproducibly.

    Args:
        sample_ids: List of file stem identifiers.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for deterministic reproducibility.

    Returns:
        {"train": [...], "val": [...], "test": [...]}
    """
    total_ratio = train_ratio + test_ratio + val_ratio
    if not abs(total_ratio - 1.0) < 1e-4:
        raise ValueError(f"Sum of split ratios must be 1.0, got: {total_ratio}")

    random.seed(seed)

    shuffled_data = sample_ids.copy()
    random.shuffle(shuffled_data)
    n = len(shuffled_data)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_data = sorted(shuffled_data[:n_train])
    val_data = sorted(shuffled_data[n_train : n_train + n_val])
    test_data = sorted(shuffled_data[n_train + n_val :])

    return {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }
