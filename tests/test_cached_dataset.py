from pathlib import Path
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scenariotopo.data.cached_dataset import CachedOpenLaneDataset, collate_cached_frames


def test_cached_dataset_round_trip(tmp_path: Path):
    arrays = {
        "query_features": np.zeros((3, 4), dtype=np.float32),
        "lanes": np.zeros((3, 2, 3), dtype=np.float32),
        "confidence": np.ones(3, dtype=np.float32),
        "query_adjacency": np.eye(3, dtype=np.float32),
        "matched_mask": np.ones(3, dtype=np.bool_),
        "gt_start": np.zeros((3, 3), dtype=np.float32),
        "gt_end": np.ones((3, 3), dtype=np.float32),
        "gt_end_tangent": np.ones((3, 3), dtype=np.float32),
        "transition_end_mask": np.zeros(3, dtype=np.bool_),
        "connector_end_mask": np.zeros(3, dtype=np.bool_),
        "split_end_mask": np.zeros(3, dtype=np.bool_),
        "merge_end_mask": np.zeros(3, dtype=np.bool_),
        "base_topology_scores": np.zeros((3, 3), dtype=np.float32),
    }
    np.savez(tmp_path / "frame.npz", **arrays)
    record = {"frame_key": "fixture/0", "cache_path": "frame.npz", "source_split": "train", "tags": ["Split"]}
    (tmp_path / "index.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    dataset = CachedOpenLaneDataset(tmp_path / "index.jsonl", split="train")
    item = dataset[0]
    batch = collate_cached_frames([item])
    assert batch["query_features"].shape == (1, 3, 4)
    assert batch["matched_mask"].dtype == torch.bool
    assert batch["transition_end_mask"].dtype == torch.bool
    assert batch["base_topology_scores"].shape == (1, 3, 3)
