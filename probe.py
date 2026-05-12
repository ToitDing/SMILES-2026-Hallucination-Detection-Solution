"""
Deterministic hallucination probe for SMILES-2026.

The probe is implemented as a small ensemble of regularized linear classifiers.
This is a good fit for the challenge setting: hidden-state features are rich,
the labelled set is small, and linear probes are usually strong and stable for
representation-analysis tasks.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


@dataclass(frozen=True)
class _ModelSpec:
    """Configuration for one member of the probe ensemble."""

    name: str
    pipeline: Pipeline
    weight: float


def _clean_array(X: np.ndarray) -> np.ndarray:
    """Convert feature matrices to finite float32 arrays."""
    X_arr = np.asarray(X, dtype=np.float32)
    return np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)


def _logistic(C: float, solver: str = "liblinear") -> LogisticRegression:
    """Create a reproducible balanced logistic classifier."""
    return LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
        solver=solver,
    )


def _make_pipeline(steps: list[tuple[str, object]]) -> Pipeline:
    """Attach common imputation and scaling before model-specific steps."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            *steps,
        ]
    )


def _candidate_thresholds(probs: np.ndarray) -> np.ndarray:
    """Return a compact set of thresholds to evaluate."""
    finite_probs = np.asarray(probs, dtype=np.float64)
    finite_probs = finite_probs[np.isfinite(finite_probs)]
    if finite_probs.size == 0:
        return np.array([0.5], dtype=np.float64)
    grid = np.linspace(0.05, 0.95, 181)
    quantiles = np.quantile(finite_probs, np.linspace(0.02, 0.98, 97))
    return np.unique(np.concatenate([grid, quantiles, finite_probs]))


class HallucinationProbe(nn.Module):
    """Binary classifier with the API expected by ``evaluate.py``."""

    def __init__(self) -> None:
        super().__init__()
        self._models: list[tuple[Pipeline, float]] = []
        self._threshold: float = 0.5
        self._input_dim: int | None = None
        self._positive_prior: float = 0.5

    def _build_specs(self, input_dim: int, n_samples: int) -> list[_ModelSpec]:
        """Create ensemble members after the feature dimensionality is known."""
        select_k = max(32, min(input_dim, 4096, n_samples * 8))
        pca_components = max(2, min(192, n_samples - 2, input_dim))

        specs = [
            _ModelSpec(
                name="raw_l2_logistic",
                pipeline=_make_pipeline([("clf", _logistic(C=0.08))]),
                weight=0.40,
            ),
            _ModelSpec(
                name="selected_l2_logistic",
                pipeline=_make_pipeline(
                    [
                        ("select", SelectKBest(score_func=f_classif, k=select_k)),
                        ("clf", _logistic(C=0.20)),
                    ]
                ),
                weight=0.35,
            ),
        ]

        if pca_components >= 2:
            specs.append(
                _ModelSpec(
                    name="pca_l2_logistic",
                    pipeline=_make_pipeline(
                        [
                            (
                                "pca",
                                PCA(
                                    n_components=pca_components,
                                    random_state=RANDOM_STATE,
                                    svd_solver="randomized",
                                    whiten=True,
                                ),
                            ),
                            ("clf", _logistic(C=0.60, solver="lbfgs")),
                        ]
                    ),
                    weight=0.25,
                )
            )

        return specs

    def _fit_pipelines(self, X: np.ndarray, y: np.ndarray) -> list[tuple[Pipeline, float]]:
        """Fit all ensemble members and return fitted pipelines with weights."""
        specs = self._build_specs(X.shape[1], X.shape[0])
        fitted: list[tuple[Pipeline, float]] = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for spec in specs:
                model = clone(spec.pipeline)
                model.fit(X, y)
                fitted.append((model, spec.weight))

        weight_sum = sum(weight for _, weight in fitted)
        return [(model, weight / weight_sum) for model, weight in fitted]

    def _predict_with_models(
        self,
        X: np.ndarray,
        models: list[tuple[Pipeline, float]],
    ) -> np.ndarray:
        """Average positive-class probabilities from fitted models."""
        if not models:
            return np.full(X.shape[0], self._positive_prior, dtype=np.float64)

        probs = np.zeros(X.shape[0], dtype=np.float64)
        for model, weight in models:
            model_probs = model.predict_proba(X)[:, 1]
            probs += weight * model_probs

        return np.clip(np.nan_to_num(probs, nan=self._positive_prior), 1e-6, 1 - 1e-6)

    def _tune_threshold(self, probs: np.ndarray, y: np.ndarray) -> float:
        """Choose the threshold that maximizes accuracy, then F1 as tie-breaker."""
        best_threshold = 0.5
        best_accuracy = -1.0
        best_f1 = -1.0

        for threshold in _candidate_thresholds(probs):
            pred = (probs >= threshold).astype(np.int64)
            acc = accuracy_score(y, pred)
            f1 = f1_score(y, pred, zero_division=0)
            if (acc > best_accuracy) or (
                np.isclose(acc, best_accuracy) and f1 > best_f1
            ):
                best_accuracy = float(acc)
                best_f1 = float(f1)
                best_threshold = float(threshold)

        return best_threshold

    def _out_of_fold_probabilities(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Estimate training-set probabilities without using in-fold labels."""
        class_counts = np.bincount(y, minlength=2)
        min_class = int(class_counts.min())
        n_splits = max(2, min(5, min_class))
        if n_splits < 2:
            return np.full(len(y), self._positive_prior, dtype=np.float64)

        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        oof = np.zeros(len(y), dtype=np.float64)

        for train_idx, val_idx in splitter.split(X, y):
            fold_models = self._fit_pipelines(X[train_idx], y[train_idx])
            oof[val_idx] = self._predict_with_models(X[val_idx], fold_models)

        return oof

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for compatibility with ``torch.nn.Module``."""
        probs = self.predict_proba(x.detach().cpu().numpy())[:, 1]
        logits = np.log(probs / (1.0 - probs))
        return torch.from_numpy(logits.astype(np.float32))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Fit the probe on labelled hidden-state features."""
        X_clean = _clean_array(X)
        y_clean = np.asarray(y, dtype=np.int64)
        self._input_dim = X_clean.shape[1]
        self._positive_prior = float(np.clip(y_clean.mean(), 1e-6, 1 - 1e-6))

        np.random.seed(RANDOM_STATE)
        torch.manual_seed(RANDOM_STATE)

        oof_probs = self._out_of_fold_probabilities(X_clean, y_clean)
        self._threshold = self._tune_threshold(oof_probs, y_clean)
        self._models = self._fit_pipelines(X_clean, y_clean)
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set."""
        X_clean = _clean_array(X_val)
        y_clean = np.asarray(y_val, dtype=np.int64)
        probs = self.predict_proba(X_clean)[:, 1]
        self._threshold = self._tune_threshold(probs, y_clean)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels, where 1 means hallucinated."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self._threshold).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ``[P(label=0), P(label=1)]`` for each sample."""
        X_clean = _clean_array(X)
        prob_pos = self._predict_with_models(X_clean, self._models)
        return np.column_stack([1.0 - prob_pos, prob_pos])
