"""Validation helpers for ScenarioTopo JSONL manifests."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Set

from .manifest import iter_jsonl


def validate_manifest(manifest_path: Path, data_root: Path | None = None) -> Dict[str, Any]:
    seen: Set[str] = set()
    duplicates = []
    missing_info = []
    tag_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    cluster_splits: Dict[str, Set[str]] = defaultdict(set)
    frame_count = 0

    for record in iter_jsonl(Path(manifest_path)):
        frame_count += 1
        frame_key = str(record["frame_key"])
        if frame_key in seen:
            duplicates.append(frame_key)
        seen.add(frame_key)
        tag_counts.update(record.get("tags") or [])
        split = str(record.get("model_split") or record.get("source_split") or "unknown")
        split_counts[split] += 1
        cluster_id = record.get("physical_cluster_id")
        if cluster_id:
            cluster_splits[str(cluster_id)].add(split)
        if data_root is not None:
            info_path = Path(data_root) / str(record["relative_info_path"])
            if not info_path.is_file() and len(missing_info) < 100:
                missing_info.append(str(info_path))

    leaking_clusters = {
        cluster_id: sorted(splits)
        for cluster_id, splits in cluster_splits.items()
        if len(splits) > 1
    }
    return {
        "valid": not duplicates and not missing_info and not leaking_clusters,
        "frames": frame_count,
        "duplicate_frame_keys": duplicates[:100],
        "missing_info_files": missing_info,
        "leaking_clusters": leaking_clusters,
        "split_counts": dict(sorted(split_counts.items())),
        "tag_counts": dict(sorted(tag_counts.items())),
    }

