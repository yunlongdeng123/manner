"""Dataset for one-file-per-frame TopoLogic feature caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from .manifest import iter_jsonl


REQUIRED_ARRAYS = (
    "query_features",
    "lanes",
    "confidence",
    "query_adjacency",
    "matched_mask",
    "gt_start",
    "gt_end",
    "gt_end_tangent",
    "transition_end_mask",
    "connector_end_mask",
    "split_end_mask",
    "merge_end_mask",
)


class CachedOpenLaneDataset:
    def __init__(self, index_path: Path, split: str | None = None) -> None:
        self.index_path = Path(index_path).resolve()
        records = list(iter_jsonl(self.index_path))
        if split is not None:
            records = [record for record in records if (record.get("model_split") or record.get("source_split")) == split]
        self.records = records

    @property
    def sample_tags(self):
        return [record.get("tags") or [] for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is required to load feature-cache training tensors") from error
        record = self.records[index]
        cache_path = Path(record["cache_path"])
        if not cache_path.is_absolute():
            cache_path = self.index_path.parent / cache_path
        with np.load(cache_path, allow_pickle=False) as arrays:
            missing = [name for name in REQUIRED_ARRAYS if name not in arrays]
            if missing:
                raise KeyError(f"{cache_path} is missing arrays: {missing}")
            item = {
                name: torch.from_numpy(np.asarray(arrays[name])).float()
                for name in REQUIRED_ARRAYS
            }
            if "base_topology_scores" in arrays:
                item["base_topology_scores"] = torch.from_numpy(np.asarray(arrays["base_topology_scores"])).float()
        item["matched_mask"] = item["matched_mask"].bool()
        for name in ("transition_end_mask", "connector_end_mask", "split_end_mask", "merge_end_mask"):
            item[name] = item[name].bool()
        item["frame_key"] = str(record["frame_key"])
        item["tags"] = list(record.get("tags") or [])
        return item


def collate_cached_frames(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required to collate feature caches") from error
    tensor_keys = REQUIRED_ARRAYS
    output = {key: torch.stack([item[key] for item in batch], dim=0) for key in tensor_keys}
    if all("base_topology_scores" in item for item in batch):
        output["base_topology_scores"] = torch.stack([item["base_topology_scores"] for item in batch], dim=0)
    elif any("base_topology_scores" in item for item in batch):
        raise ValueError("Mixed cache batches with and without base_topology_scores")
    output["frame_key"] = [item["frame_key"] for item in batch]
    output["tags"] = [item["tags"] for item in batch]
    return output
