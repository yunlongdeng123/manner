"""Lightweight residual endpoint refinement head."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn


class EndpointRefiner(nn.Module):
    def __init__(
        self,
        query_dim: int = 256,
        hidden_dim: int = 128,
        point_dim: int = 3,
        max_offset_m: float = 3.0,
        endpoint_span: int = 4,
        warp_gamma: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.point_dim = int(point_dim)
        self.max_offset_m = float(max_offset_m)
        self.endpoint_span = int(endpoint_span)
        self.warp_gamma = float(warp_gamma)
        if self.endpoint_span < 1 or self.warp_gamma <= 0:
            raise ValueError("endpoint_span must be >=1 and warp_gamma must be positive")
        self.network = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * point_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def _reshape_lanes(self, lanes: Tensor) -> Tuple[Tensor, bool]:
        if lanes.ndim == 4 and lanes.shape[-1] == self.point_dim:
            return lanes, False
        if lanes.ndim == 3 and lanes.shape[-1] % self.point_dim == 0:
            return lanes.reshape(*lanes.shape[:2], -1, self.point_dim), True
        raise ValueError(
            "base_lanes must have shape [B,N,P,D] or [B,N,P*D]; "
            f"received {tuple(lanes.shape)}"
        )

    def forward(self, query_features: Tensor, base_lanes: Tensor) -> Dict[str, Tensor]:
        if query_features.ndim != 3:
            raise ValueError(f"query_features must be [B,N,C], got {tuple(query_features.shape)}")
        lane_points, was_flat = self._reshape_lanes(base_lanes)
        if lane_points.shape[-2] < 2:
            raise ValueError("base_lanes must have at least two points")
        if query_features.shape[:2] != lane_points.shape[:2]:
            raise ValueError("query_features and base_lanes must share [B,N]")

        raw_offsets = self.network(query_features)
        offsets = torch.tanh(raw_offsets).reshape(*query_features.shape[:2], 2, self.point_dim)
        offsets = offsets * self.max_offset_m
        point_count = lane_points.shape[-2]
        span = min(self.endpoint_span, point_count)
        positions = torch.arange(point_count, device=lane_points.device, dtype=lane_points.dtype)
        start_weights = ((span - 1 - positions).clamp(min=0) / max(span - 1, 1)) ** self.warp_gamma
        end_weights = ((positions - (point_count - span)).clamp(min=0) / max(span - 1, 1)) ** self.warp_gamma
        if span == 1:
            start_weights = (positions == 0).to(lane_points.dtype)
            end_weights = (positions == point_count - 1).to(lane_points.dtype)
        refined = (
            lane_points
            + start_weights.view(1, 1, -1, 1) * offsets[..., 0, :].unsqueeze(-2)
            + end_weights.view(1, 1, -1, 1) * offsets[..., 1, :].unsqueeze(-2)
        )
        refined_output = refined.flatten(-2) if was_flat else refined
        return {
            "refined_lanes": refined_output,
            "refined_start": refined[..., 0, :],
            "refined_end": refined[..., -1, :],
            "endpoint_offsets": offsets,
        }
