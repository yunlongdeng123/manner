"""Composable head-only model used on frozen TopoLogic feature caches."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from torch import Tensor, nn

from .endpoint_refiner import EndpointRefiner
from .hybrid_topology_head import HybridTopologyHead


class ScenarioTopoModel(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        modules = config.get("modules") or {}
        endpoint_config = dict(config.get("endpoint_refiner") or {})
        topology_config = dict(config.get("hybrid_topology") or {})
        self.endpoint_refiner = EndpointRefiner(**endpoint_config) if modules.get("endpoint_refiner", True) else None
        self.topology_head = HybridTopologyHead(**topology_config) if modules.get("hybrid_topology", True) else None

    def forward(self, query_features: Tensor, lanes: Tensor, confidence: Tensor) -> Dict[str, Tensor]:
        if self.endpoint_refiner is None:
            refined = {
                "refined_lanes": lanes,
                "refined_start": lanes[..., 0, :],
                "refined_end": lanes[..., -1, :],
            }
        else:
            refined = self.endpoint_refiner(query_features, lanes)
        if self.topology_head is None:
            return refined
        topology = self.topology_head(query_features, refined["refined_lanes"], confidence, return_aux=True)
        return {**refined, **topology}
