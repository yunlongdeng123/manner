"""Leakage-resistant, segment-level geographic clustering and splitting."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from .manifest import iter_jsonl


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if self.rank[first_root] < self.rank[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        if self.rank[first_root] == self.rank[second_root]:
            self.rank[first_root] += 1


def _stable_hash(text: str, seed: int) -> str:
    return hashlib.sha1(f"{seed}:{text}".encode("utf-8")).hexdigest()


def _segment_anchors(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    aggregates: Dict[str, Dict[str, Any]] = {}
    for record in records:
        segment_id = str(record["segment_id"])
        entry = aggregates.setdefault(
            segment_id,
            {"x": 0.0, "y": 0.0, "count": 0, "frames": 0, "source_id": str(record.get("source_id", "unknown"))},
        )
        entry["frames"] += 1
        pose = record.get("pose_translation")
        if pose and len(pose) >= 2:
            entry["x"] += float(pose[0])
            entry["y"] += float(pose[1])
            entry["count"] += 1
    for entry in aggregates.values():
        if entry["count"]:
            entry["x"] /= entry["count"]
            entry["y"] /= entry["count"]
        else:
            entry["x"] = None
            entry["y"] = None
    return aggregates


def cluster_segments(records: Sequence[Mapping[str, Any]], radius_m: float) -> Dict[str, str]:
    """Cluster segment anchors using a spatial grid, avoiding an O(N^2) matrix."""
    anchors = _segment_anchors(records)
    union_find = UnionFind(anchors)
    grid: Dict[Tuple[str, int, int], List[str]] = defaultdict(list)

    for segment_id in sorted(anchors):
        anchor = anchors[segment_id]
        if anchor["x"] is None:
            continue
        source_id = str(anchor["source_id"])
        cell_x = math.floor(float(anchor["x"]) / radius_m)
        cell_y = math.floor(float(anchor["y"]) / radius_m)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other_id in grid.get((source_id, cell_x + dx, cell_y + dy), []):
                    other = anchors[other_id]
                    if math.hypot(float(anchor["x"]) - float(other["x"]), float(anchor["y"]) - float(other["y"])) <= radius_m:
                        union_find.union(segment_id, other_id)
        grid[(source_id, cell_x, cell_y)].append(segment_id)

    groups: Dict[str, List[str]] = defaultdict(list)
    for segment_id in sorted(anchors):
        groups[union_find.find(segment_id)].append(segment_id)

    segment_clusters: Dict[str, str] = {}
    for members in groups.values():
        canonical = min(members)
        cluster_id = f"geo-{hashlib.sha1(canonical.encode('utf-8')).hexdigest()[:12]}"
        for segment_id in members:
            segment_clusters[segment_id] = cluster_id
    return segment_clusters


def assign_clusters(
    records: Sequence[Mapping[str, Any]],
    segment_clusters: Mapping[str, str],
    ratios: Mapping[str, float],
    seed: int,
) -> Dict[str, str]:
    cluster_sizes: Dict[str, int] = defaultdict(int)
    for record in records:
        cluster_sizes[segment_clusters[str(record["segment_id"])]] += 1

    ratio_sum = sum(float(value) for value in ratios.values())
    normalized = {name: float(value) / ratio_sum for name, value in ratios.items()}
    total = sum(cluster_sizes.values())
    targets = {name: total * ratio for name, ratio in normalized.items()}
    assigned_sizes = {name: 0 for name in normalized}
    assignments: Dict[str, str] = {}

    clusters = sorted(cluster_sizes, key=lambda cid: (-cluster_sizes[cid], _stable_hash(cid, seed)))
    for cluster_id in clusters:
        split = max(
            normalized,
            key=lambda name: (
                targets[name] - assigned_sizes[name],
                -assigned_sizes[name],
                _stable_hash(f"{cluster_id}:{name}", seed),
            ),
        )
        assignments[cluster_id] = split
        assigned_sizes[split] += cluster_sizes[cluster_id]
    return assignments


def create_geo_split(
    manifest_path: Path,
    output_manifest_path: Path,
    split_map_path: Path,
    *,
    radius_m: float = 80.0,
    ratios: Mapping[str, float] | None = None,
    seed: int = 20260905,
) -> Dict[str, Any]:
    ratios = ratios or {"train": 0.8, "val": 0.1, "test": 0.1}
    records = list(iter_jsonl(Path(manifest_path)))
    def is_labeled(record: Mapping[str, Any]) -> bool:
        return bool(record.get("has_annotation", record.get("source_split") != "test"))

    labeled_records = [record for record in records if is_labeled(record)]
    benchmark_records = [record for record in records if not is_labeled(record)]

    labeled_clusters = cluster_segments(labeled_records, radius_m)
    benchmark_clusters = cluster_segments(benchmark_records, radius_m)
    segment_clusters = {**labeled_clusters, **benchmark_clusters}
    assignments = assign_clusters(labeled_records, labeled_clusters, ratios, seed)
    for cluster_id in set(benchmark_clusters.values()):
        assignments[cluster_id] = "benchmark_test"

    split_map = {
        "seed": seed,
        "radius_m": radius_m,
        "ratios": dict(ratios),
        "benchmark_split": "benchmark_test",
        "segment_to_cluster": segment_clusters,
        "cluster_to_split": assignments,
    }
    split_map_path = Path(split_map_path).resolve()
    split_map_path.parent.mkdir(parents=True, exist_ok=True)
    split_map_path.write_text(json.dumps(split_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    output_manifest_path = Path(output_manifest_path).resolve()
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_manifest_path.parent, delete=False, prefix=".geo-", suffix=".tmp"
    ) as handle:
        temp_path = Path(handle.name)
        counts: Dict[str, int] = defaultdict(int)
        for record in records:
            cluster_id = segment_clusters[str(record["segment_id"])]
            split = assignments[cluster_id]
            enriched = dict(record)
            enriched["physical_cluster_id"] = cluster_id
            enriched["model_split"] = split
            handle.write(json.dumps(enriched, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts[split] += 1
    os.replace(temp_path, output_manifest_path)
    return {
        "frames": len(records),
        "labeled_frames": len(labeled_records),
        "benchmark_test_frames": len(benchmark_records),
        "segments": len(segment_clusters),
        "clusters": len(assignments),
        "split_counts": dict(sorted(counts.items())),
        "manifest": str(output_manifest_path),
        "split_map": str(split_map_path),
    }
