from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline


class HallucinationProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._threshold = 0.5
        self._model = None

    def _build_network(self, input_dim: int) -> None:
        # Kept only for API compatibility.
        self._net = nn.Sequential(
            nn.Linear(input_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("This solution uses sklearn models via fit/predict_proba.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        n_samples, n_features = X.shape
        n_components = min(128, n_samples - 1, n_features)

        self._model = make_pipeline(
            StandardScaler(),
            PCA(n_components=n_components, random_state=42),
            ExtraTreesClassifier(
                n_estimators=600,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        )

        self._model.fit(X, y.astype(int))
        return self

    def fit_hyperparameters(self, X_val: np.ndarray, y_val: np.ndarray) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(np.concatenate([probs, np.linspace(0.05, 0.95, 181)]))

        best_threshold = 0.5
        best_f1 = -1.0

        for t in candidates:
            pred = (probs >= t).astype(int)
            score = f1_score(y_val, pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict_proba().")
        return self._model.predict_proba(X)
