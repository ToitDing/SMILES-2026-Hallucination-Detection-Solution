from __future__ import annotations
import torch
import torch.nn.functional as F


SELECTED_LAYERS = [-1, -2, -4, -8]


def _last_real_index(attention_mask: torch.Tensor) -> int:
    real_positions = attention_mask.nonzero(as_tuple=False).flatten()
    return int(real_positions[-1].item())


def aggregate(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    real = attention_mask.bool()
    last_pos = _last_real_index(attention_mask)

    feats = []

    for layer_idx in SELECTED_LAYERS:
        h = hidden_states[layer_idx]
        h_real = h[real]

        last_tok = h[last_pos]
        mean_pool = h_real.mean(dim=0)
        max_pool = h_real.max(dim=0).values

        feats.extend([last_tok, mean_pool, max_pool])

    return torch.cat(feats, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    real = attention_mask.bool()
    last_pos = _last_real_index(attention_mask)

    stats = []

    selected = [hidden_states[i] for i in SELECTED_LAYERS]

    for h in selected:
        h_real = h[real]
        last = h[last_pos]
        mean = h_real.mean(dim=0)

        stats.append(last.norm())
        stats.append(mean.norm())
        stats.append(h_real.std(dim=0).mean())
        stats.append(F.cosine_similarity(last, mean, dim=0))

    for a, b in zip(selected[:-1], selected[1:]):
        va = a[last_pos]
        vb = b[last_pos]
        stats.append(F.cosine_similarity(va, vb, dim=0))
        stats.append((va - vb).norm())

    stats.append(torch.tensor(float(real.sum()), dtype=hidden_states.dtype))

    return torch.stack(stats).float()


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    agg_features = aggregate(hidden_states, attention_mask)
    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)
    return agg_features
