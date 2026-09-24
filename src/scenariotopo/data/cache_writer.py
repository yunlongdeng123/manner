"""Atomic writer and GT-to-query mapping for frozen TopoLogic caches."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


def map_ground_truth_to_queries(
    query_to_gt,
    gt_lanes: Sequence[np.ndarray],
    gt_adjacency,
    *,
    point_dim: int = 3,
    gt_is_intersection_or_connector: Sequence[bool] | None = None,
) -> dict[str, np.ndarray]:
    query_to_gt = np.asarray(query_to_gt, dtype=np.int64)
    gt_adjacency = np.asarray(gt_adjacency, dtype=np.float32)
    query_count = query_to_gt.shape[0]
    if gt_adjacency.shape != (len(gt_lanes), len(gt_lanes)):
        raise ValueError("gt_adjacency must be square with one row per GT lane")
    if gt_is_intersection_or_connector is None:
        connector = np.zeros(len(gt_lanes), dtype=np.bool_)
    else:
        connector = np.asarray(gt_is_intersection_or_connector, dtype=np.bool_)
        if connector.shape != (len(gt_lanes),):
            raise ValueError("gt_is_intersection_or_connector must have one value per GT lane")
    matched = query_to_gt >= 0
    gt_start = np.zeros((query_count, point_dim), dtype=np.float32)
    gt_end = np.zeros((query_count, point_dim), dtype=np.float32)
    gt_end_tangent = np.zeros((query_count, point_dim), dtype=np.float32)
    transition_end_mask = np.zeros(query_count, dtype=np.bool_)
    connector_end_mask = np.zeros(query_count, dtype=np.bool_)
    split_end_mask = np.zeros(query_count, dtype=np.bool_)
    merge_end_mask = np.zeros(query_count, dtype=np.bool_)
    query_adjacency = np.zeros((query_count, query_count), dtype=np.float32)

    for query_index in np.flatnonzero(matched):
        gt_index = int(query_to_gt[query_index])
        if gt_index >= len(gt_lanes):
            raise IndexError(f"query_to_gt references missing GT lane {gt_index}")
        lane = np.asarray(gt_lanes[gt_index], dtype=np.float32)
        if lane.ndim != 2 or lane.shape[0] < 2 or lane.shape[1] < point_dim:
            raise ValueError(f"GT lane {gt_index} has invalid shape {lane.shape}")
        gt_start[query_index] = lane[0, :point_dim]
        gt_end[query_index] = lane[-1, :point_dim]
        delta = lane[-1, :point_dim] - lane[-2, :point_dim]
        norm = np.linalg.norm(delta)
        if norm > 1e-6:
            gt_end_tangent[query_index] = delta / norm
        successors = gt_adjacency[gt_index] > 0
        split_end_mask[query_index] = np.count_nonzero(successors) > 1
        merge_end_mask[query_index] = np.any(successors & (np.count_nonzero(gt_adjacency > 0, axis=0) > 1))
        # Connector boundaries are a measurable proxy, not a polygon region mask.
        connector_end_mask[query_index] = bool(not connector[gt_index] and np.any(successors & connector))
        transition_end_mask[query_index] = bool(
            split_end_mask[query_index]
            or merge_end_mask[query_index]
            or connector_end_mask[query_index]
        )

    matched_queries = np.flatnonzero(matched)
    for source_query in matched_queries:
        source_gt = int(query_to_gt[source_query])
        for target_query in matched_queries:
            target_gt = int(query_to_gt[target_query])
            query_adjacency[source_query, target_query] = gt_adjacency[source_gt, target_gt]
    return {
        "matched_mask": matched.astype(np.bool_),
        "gt_start": gt_start,
        "gt_end": gt_end,
        "gt_end_tangent": gt_end_tangent,
        "transition_end_mask": transition_end_mask,
        "connector_end_mask": connector_end_mask,
        "split_end_mask": split_end_mask,
        "merge_end_mask": merge_end_mask,
        "query_adjacency": query_adjacency,
    }


class FeatureCacheWriter:
    def __init__(self, output_root: Path) -> None:
        self.output_root = Path(output_root).resolve()
        self.frames_root = self.output_root / "frames"
        self.frames_root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.output_root / "index.jsonl"

    def write(
        self,
        *,
        frame_key: str,
        query_features,
        lanes,
        confidence,
        query_to_gt,
        gt_lanes: Sequence[np.ndarray],
        gt_adjacency,
        gt_is_intersection_or_connector: Sequence[bool] | None = None,
        base_topology_scores=None,
        semantic_topology_scores=None,
        tags: Sequence[str],
        split: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        query_features = np.asarray(query_features, dtype=np.float16)
        lanes = np.asarray(lanes, dtype=np.float32)
        confidence = np.asarray(confidence, dtype=np.float32)
        if query_features.ndim != 2:
            raise ValueError("query_features must be [N,C]")
        if lanes.ndim != 3 or lanes.shape[-1] != 3:
            raise ValueError("lanes must be [N,P,3]")
        if confidence.shape != (query_features.shape[0],):
            raise ValueError("confidence must be [N]")
        if lanes.shape[0] != query_features.shape[0]:
            raise ValueError("query_features and lanes must have the same query count")
        mapped = map_ground_truth_to_queries(
            query_to_gt, gt_lanes, gt_adjacency,
            gt_is_intersection_or_connector=gt_is_intersection_or_connector,
        )
        topology_scores = {}
        for name, values in (
            ("base_topology_scores", base_topology_scores),
            ("semantic_topology_scores", semantic_topology_scores),
        ):
            if values is None:
                continue
            values = np.asarray(values, dtype=np.float32)
            if values.shape != (query_features.shape[0], query_features.shape[0]):
                raise ValueError(f"{name} must be [N,N]")
            topology_scores[name] = values

        safe_name = frame_key.replace("/", "__").replace("\\", "__")
        final_path = self.frames_root / f"{safe_name}.npz"
        with tempfile.NamedTemporaryFile(dir=self.frames_root, prefix=f".{safe_name}.", suffix=".npz", delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            payload = dict(query_features=query_features, lanes=lanes, confidence=confidence, **mapped)
            payload.update(topology_scores)
            np.savez_compressed(temp_path, **payload)
            os.replace(temp_path, final_path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise

        record = {
            "frame_key": frame_key,
            "cache_path": final_path.relative_to(self.output_root).as_posix(),
            "source_split": split,
            "tags": list(tags),
            **dict(metadata or {}),
        }
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record
