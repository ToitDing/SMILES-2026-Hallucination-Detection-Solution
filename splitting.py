"""
Train, validation, and test splits for SMILES-2026.

The official evaluation accepts a list of folds.  This solution uses a
stratified outer K-fold split for stable reporting, with a stratified validation
split inside each training fold for threshold selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.20,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Return reproducible stratified folds.

    Args:
        y: Label array of shape ``(N,)`` with values in ``{0, 1}``.
        df: Unused; kept for API compatibility with the fixed runner.
        test_size: Approximate held-out fraction.  The implementation maps the
            default 20 percent to 5 outer folds.
        val_size: Fraction of all samples used for validation inside each fold.
        random_state: Seed for every randomized split.

    Returns:
        A list of ``(idx_train, idx_val, idx_test)`` tuples.
    """
    del df

    labels = np.asarray(y, dtype=np.int64)
    idx = np.arange(labels.shape[0])
    min_class = int(np.bincount(labels, minlength=2).min())
    requested_folds = int(round(1.0 / max(test_size, 1e-6)))
    n_splits = max(2, min(5, requested_folds, min_class))

    outer = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    for fold_id, (train_val_idx, test_idx) in enumerate(outer.split(idx, labels)):
        relative_val = val_size / (1.0 - (1.0 / n_splits))
        relative_val = float(np.clip(relative_val, 0.05, 0.30))

        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=relative_val,
            random_state=random_state + fold_id,
            stratify=labels[train_val_idx],
        )

        splits.append(
            (
                np.sort(train_idx.astype(np.int64)),
                np.sort(val_idx.astype(np.int64)),
                np.sort(test_idx.astype(np.int64)),
            )
        )

    return splits
