"""Losses for endpoint refinement and sparse directed topology."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def _valid_mask(targets: Tensor, valid_mask: Tensor | None) -> Tensor:
    if valid_mask is None:
        mask = torch.ones_like(targets, dtype=torch.bool)
    else:
        mask = valid_mask.to(dtype=torch.bool)
        if mask.shape != targets.shape:
            raise ValueError("valid_mask must match topology target shape")
    if targets.shape[-1] == targets.shape[-2]:
        diagonal = torch.eye(targets.shape[-1], dtype=torch.bool, device=targets.device)
        mask = mask & ~diagonal.unsqueeze(0)
    return mask


def topology_classification_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    valid_mask: Tensor | None = None,
    max_pos_weight: float = 20.0,
) -> Tensor:
    targets = targets.to(dtype=logits.dtype)
    mask = _valid_mask(targets, valid_mask)
    selected_targets = targets[mask]
    selected_logits = logits[mask]
    if selected_targets.numel() == 0:
        return logits.sum() * 0.0
    positives = selected_targets.sum()
    negatives = selected_targets.numel() - positives
    pos_weight = torch.clamp(negatives / positives.clamp_min(1.0), min=1.0, max=max_pos_weight)
    return F.binary_cross_entropy_with_logits(selected_logits, selected_targets, pos_weight=pos_weight)


def endpoint_regression_loss(
    refined_start: Tensor,
    refined_end: Tensor,
    gt_start: Tensor,
    gt_end: Tensor,
    *,
    matched_mask: Tensor | None = None,
    beta: float = 1.0,
) -> Tensor:
    loss = F.smooth_l1_loss(refined_start, gt_start, reduction="none", beta=beta).sum(-1)
    loss = loss + F.smooth_l1_loss(refined_end, gt_end, reduction="none", beta=beta).sum(-1)
    if matched_mask is not None:
        weights = matched_mask.to(dtype=loss.dtype)
        return (loss * weights).sum() / weights.sum().clamp_min(1.0)
    return loss.mean()


def connection_consistency_loss(
    refined_lanes: Tensor, adjacency: Tensor, *, valid_mask: Tensor | None = None
) -> Tensor:
    if refined_lanes.ndim == 3:
        refined_lanes = refined_lanes.reshape(*refined_lanes.shape[:2], -1, 3)
    start = refined_lanes[..., 0, :]
    end = refined_lanes[..., -1, :]
    distances = torch.abs(end.unsqueeze(2) - start.unsqueeze(1)).sum(-1)
    weights = (adjacency.bool() & _valid_mask(adjacency, valid_mask)).to(dtype=distances.dtype)
    return (distances * weights).sum() / weights.sum().clamp_min(1.0)


def longitudinal_overshoot_loss(
    refined_end: Tensor,
    gt_end: Tensor,
    gt_end_tangent: Tensor,
    *,
    matched_mask: Tensor,
    margin_m: float = 0.0,
    transition_end_mask: Tensor | None = None,
) -> Tensor:
    """Penalize forward endpoint error, optionally only at transition boundaries."""
    if margin_m < 0:
        raise ValueError("margin_m must be nonnegative")
    mask = matched_mask.bool()
    if transition_end_mask is not None:
        mask = mask & transition_end_mask.bool()
    if not mask.any():
        return refined_end.sum() * 0.0
    longitudinal = ((refined_end - gt_end) * gt_end_tangent).sum(-1)
    return F.relu(longitudinal - margin_m)[mask].mean()


def hard_negative_ranking_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    endpoint_distances: Tensor | None = None,
    valid_mask: Tensor | None = None,
    margin: float = 0.2,
    negatives_per_source: int = 8,
) -> Tensor:
    """Rank every positive above nearby (or high-scoring) negatives."""
    mask = _valid_mask(targets, valid_mask)
    terms = []
    for batch_index in range(logits.shape[0]):
        for source_index in range(logits.shape[1]):
            positive_mask = (targets[batch_index, source_index] > 0.5) & mask[batch_index, source_index]
            negative_mask = (targets[batch_index, source_index] <= 0.5) & mask[batch_index, source_index]
            positive_logits = logits[batch_index, source_index][positive_mask]
            negative_logits = logits[batch_index, source_index][negative_mask]
            if positive_logits.numel() == 0 or negative_logits.numel() == 0:
                continue
            keep = min(int(negatives_per_source), negative_logits.numel())
            if endpoint_distances is None:
                hard_negative_logits = torch.topk(negative_logits, keep).values
            else:
                negative_distances = endpoint_distances[batch_index, source_index][negative_mask]
                indices = torch.topk(negative_distances, keep, largest=False).indices
                hard_negative_logits = negative_logits[indices]
            pair_losses = F.relu(margin - positive_logits.unsqueeze(-1) + hard_negative_logits.unsqueeze(0))
            terms.append(pair_losses.mean())
    if not terms:
        return logits.sum() * 0.0
    return torch.stack(terms).mean()
