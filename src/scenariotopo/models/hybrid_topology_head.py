"""Learned semantic/geometric topology fusion with bounded pair chunks."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _mlp(input_dim: int, hidden_dim: int, output_dim: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


def pairwise_geometry_features(
    lanes: Tensor,
    confidence: Tensor,
    *,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    """Return [distance, direction penalty, dz, lateral, ci, cj] and distance."""
    if lanes.ndim != 4 or lanes.shape[-1] < 2 or lanes.shape[-2] < 2:
        raise ValueError("lanes must be [B,N,P,D] with P>=2 and D>=2")
    if confidence.ndim == 3 and confidence.shape[-1] == 1:
        confidence = confidence.squeeze(-1)
    if confidence.shape != lanes.shape[:2]:
        raise ValueError("confidence must have shape [B,N] or [B,N,1]")

    start = lanes[..., 0, :]
    end = lanes[..., -1, :]
    start_tangent = F.normalize(lanes[..., 1, :] - start, dim=-1, eps=eps)
    end_tangent = F.normalize(end - lanes[..., -2, :], dim=-1, eps=eps)
    displacement = start.unsqueeze(1) - end.unsqueeze(2)
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    direction_penalty = 1.0 - torch.sum(end_tangent.unsqueeze(2) * start_tangent.unsqueeze(1), dim=-1)
    if lanes.shape[-1] >= 3:
        dz = torch.abs(displacement[..., 2])
    else:
        dz = torch.zeros_like(distance)
    lateral = torch.abs(
        end_tangent[..., 0].unsqueeze(2) * displacement[..., 1]
        - end_tangent[..., 1].unsqueeze(2) * displacement[..., 0]
    )
    source_conf = confidence.unsqueeze(2).expand_as(distance)
    target_conf = confidence.unsqueeze(1).expand_as(distance)
    geometry = torch.stack(
        [distance, direction_penalty, dz, lateral, source_conf, target_conf],
        dim=-1,
    )
    return geometry, distance


class HybridTopologyHead(nn.Module):
    geometry_dim = 6

    def __init__(
        self,
        query_dim: int = 256,
        hidden_dim: int = 128,
        pair_chunk_size: int = 16,
        mask_self_edges: bool = True,
    ) -> None:
        super().__init__()
        self.query_dim = int(query_dim)
        self.pair_chunk_size = int(pair_chunk_size)
        self.mask_self_edges = bool(mask_self_edges)
        self.semantic_head = _mlp(4 * query_dim, hidden_dim)
        self.geometry_head = _mlp(self.geometry_dim, max(32, hidden_dim // 2))
        self.gate_head = _mlp(2 * query_dim + self.geometry_dim, hidden_dim)

    def forward(
        self,
        query_features: Tensor,
        lanes: Tensor,
        confidence: Tensor,
        *,
        return_aux: bool = False,
    ):
        if query_features.ndim != 3:
            raise ValueError("query_features must be [B,N,C]")
        if query_features.shape[-1] != self.query_dim:
            raise ValueError(f"expected query_dim={self.query_dim}, got {query_features.shape[-1]}")
        if lanes.ndim == 3:
            if lanes.shape[-1] % 3:
                raise ValueError("flat lanes must contain 3D points")
            lanes = lanes.reshape(*lanes.shape[:2], -1, 3)
        if lanes.shape[:2] != query_features.shape[:2]:
            raise ValueError("lanes and query_features must share [B,N]")

        geometry, distance = pairwise_geometry_features(lanes, confidence)
        batch_size, query_count, _ = query_features.shape
        semantic_parts = []
        geometry_parts = []
        gate_parts = []
        fused_parts = []

        source = query_features.unsqueeze(2)
        for begin in range(0, query_count, self.pair_chunk_size):
            end = min(begin + self.pair_chunk_size, query_count)
            target = query_features[:, begin:end, :].unsqueeze(1)
            source_expanded = source.expand(-1, -1, end - begin, -1)
            target_expanded = target.expand(-1, query_count, -1, -1)
            semantic_input = torch.cat(
                [
                    source_expanded,
                    target_expanded,
                    source_expanded - target_expanded,
                    source_expanded * target_expanded,
                ],
                dim=-1,
            )
            geometry_chunk = geometry[:, :, begin:end, :]
            semantic_logit = self.semantic_head(semantic_input).squeeze(-1)
            geometry_logit = self.geometry_head(geometry_chunk).squeeze(-1)
            gate_input = torch.cat([source_expanded, target_expanded, geometry_chunk], dim=-1)
            gate = torch.sigmoid(self.gate_head(gate_input).squeeze(-1))
            fused = gate * semantic_logit + (1.0 - gate) * geometry_logit
            semantic_parts.append(semantic_logit)
            geometry_parts.append(geometry_logit)
            gate_parts.append(gate)
            fused_parts.append(fused)

        semantic_logits = torch.cat(semantic_parts, dim=2)
        geometry_logits = torch.cat(geometry_parts, dim=2)
        gates = torch.cat(gate_parts, dim=2)
        logits = torch.cat(fused_parts, dim=2)
        if self.mask_self_edges:
            diagonal = torch.eye(query_count, dtype=torch.bool, device=logits.device).unsqueeze(0)
            logits = logits.masked_fill(diagonal, -1e4)

        if not return_aux:
            return logits
        return {
            "logits": logits,
            "semantic_logits": semantic_logits,
            "geometry_logits": geometry_logits,
            "geometry_features": geometry,
            "endpoint_distances": distance,
            "semantic_gates": gates,
        }

