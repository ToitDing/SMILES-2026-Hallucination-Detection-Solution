"""
Hidden-state aggregation for the SMILES-2026 hallucination detector.

The official runner passes all hidden states for one sample with shape
``(n_layers, seq_len, hidden_dim)``.  This module turns that tensor into a
single deterministic feature vector.  The feature vector intentionally mixes
upper-layer token summaries with compact geometric statistics: the linear probe
in ``probe.py`` can then decide which signals are useful without requiring any
changes to the fixed extraction loop.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _selected_layer_indices(n_layers: int) -> list[int]:
    """Return stable mid-to-late layer indices for any decoder depth."""
    raw = [
        round((n_layers - 1) * 0.35),
        round((n_layers - 1) * 0.50),
        round((n_layers - 1) * 0.65),
        round((n_layers - 1) * 0.80),
        n_layers - 1,
    ]
    indices: list[int] = []
    for idx in raw:
        idx = int(max(0, min(n_layers - 1, idx)))
        if idx not in indices:
            indices.append(idx)
    return indices


def _tail_mean(layer_tokens: torch.Tensor, window: int) -> torch.Tensor:
    """Mean-pool the last ``window`` non-padding tokens of one layer."""
    n_tokens = layer_tokens.shape[0]
    width = min(window, n_tokens)
    return layer_tokens[-width:].mean(dim=0)


def _safe_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity as a one-element tensor with finite output."""
    value = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1)
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-layer token states into one flat feature vector.

    The representation contains five groups of features:

    1. last-token vectors from selected mid and upper layers;
    2. short-tail means, which focus on the end of the answer;
    3. medium-tail means, which smooth over more answer tokens;
    4. full-sequence means, which preserve prompt-plus-answer context;
    5. compact scalar statistics describing norms, dispersion, and layer drift.

    Args:
        hidden_states: Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: Tensor of shape ``(seq_len,)`` with 1 for real tokens.

    Returns:
        A 1-D float tensor with deterministic dimensionality for a fixed model.
    """
    device = hidden_states.device
    mask = attention_mask.to(device=device).bool()
    if not bool(mask.any()):
        mask = torch.ones(hidden_states.shape[1], dtype=torch.bool, device=device)

    states = hidden_states.float()
    valid_states = states[:, mask, :]
    n_layers, n_tokens, _ = valid_states.shape
    layer_indices = _selected_layer_indices(n_layers)

    dense_features: list[torch.Tensor] = []
    scalar_features: list[torch.Tensor] = []
    last_vectors: list[torch.Tensor] = []
    tail_vectors: list[torch.Tensor] = []

    for layer_idx in layer_indices:
        layer = valid_states[layer_idx]
        last = layer[-1]
        tail_8 = _tail_mean(layer, 8)
        tail_32 = _tail_mean(layer, 32)
        full_mean = layer.mean(dim=0)

        dense_features.extend([last, tail_8, tail_32, full_mean])
        last_vectors.append(last)
        tail_vectors.append(tail_32)

        token_norms = torch.linalg.vector_norm(layer, dim=1)
        dispersion = layer.std(dim=0, unbiased=False).mean()
        scalar_features.extend(
            [
                torch.log1p(torch.tensor(float(n_tokens), device=device)),
                token_norms.mean(),
                token_norms.std(unbiased=False),
                torch.linalg.vector_norm(last),
                torch.linalg.vector_norm(tail_8),
                torch.linalg.vector_norm(tail_32),
                torch.linalg.vector_norm(full_mean),
                dispersion,
                _safe_cosine(last, tail_8).squeeze(0),
                _safe_cosine(tail_32, full_mean).squeeze(0),
            ]
        )

    for left, right in zip(last_vectors, last_vectors[1:]):
        scalar_features.extend(
            [
                _safe_cosine(left, right).squeeze(0),
                torch.linalg.vector_norm(right - left)
                / torch.sqrt(torch.tensor(float(right.numel()), device=device)),
            ]
        )

    for left, right in zip(tail_vectors, tail_vectors[1:]):
        scalar_features.extend(
            [
                _safe_cosine(left, right).squeeze(0),
                torch.linalg.vector_norm(right - left)
                / torch.sqrt(torch.tensor(float(right.numel()), device=device)),
            ]
        )

    scalars = torch.stack([s.reshape(()) for s in scalar_features]).float()
    features = torch.cat([*dense_features, scalars], dim=0)
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Return optional extra geometric features.

    The main ``aggregate`` function already includes the geometric statistics
    used by the final solution because the fixed ``solution.py`` keeps
    ``USE_GEOMETRIC`` set to ``False``.  This function remains available for
    experiments that explicitly enable that flag.
    """
    device = hidden_states.device
    mask = attention_mask.to(device=device).bool()
    if not bool(mask.any()):
        mask = torch.ones(hidden_states.shape[1], dtype=torch.bool, device=device)

    states = hidden_states.float()[:, mask, :]
    final_layer = states[-1]
    token_norms = torch.linalg.vector_norm(final_layer, dim=1)
    features = torch.tensor(
        [
            float(final_layer.shape[0]),
            float(token_norms.mean()),
            float(token_norms.std(unbiased=False)),
            float(final_layer.std(dim=0, unbiased=False).mean()),
        ],
        dtype=torch.float32,
        device=device,
    )
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append extra scalar features."""
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
