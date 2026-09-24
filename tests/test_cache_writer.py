import json

import numpy as np

from scenariotopo.data.cache_writer import FeatureCacheWriter, map_ground_truth_to_queries


def test_gt_mapping_and_atomic_cache(tmp_path):
    mapping = np.asarray([1, -1, 0])
    gt_lanes = [
        np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        np.asarray([[1, 0, 0], [2, 0, 0]], dtype=np.float32),
    ]
    adjacency = np.asarray([[0, 1], [0, 0]], dtype=np.float32)
    mapped = map_ground_truth_to_queries(
        mapping, gt_lanes, adjacency, gt_is_intersection_or_connector=[False, True]
    )
    assert mapped["matched_mask"].tolist() == [True, False, True]
    assert mapped["query_adjacency"][2, 0] == 1
    assert mapped["transition_end_mask"].tolist() == [False, False, True]
    np.testing.assert_allclose(mapped["gt_end_tangent"][2], [1, 0, 0])

    writer = FeatureCacheWriter(tmp_path)
    record = writer.write(
        frame_key="val/segment/123",
        query_features=np.zeros((3, 4), dtype=np.float32),
        lanes=np.zeros((3, 2, 3), dtype=np.float32),
        confidence=np.ones(3, dtype=np.float32),
        query_to_gt=mapping,
        gt_lanes=gt_lanes,
        gt_adjacency=adjacency,
        gt_is_intersection_or_connector=[False, True],
        tags=["Split"],
        split="val",
    )
    assert (tmp_path / record["cache_path"]).is_file()
    index_record = json.loads((tmp_path / "index.jsonl").read_text(encoding="utf-8"))
    assert index_record["frame_key"] == "val/segment/123"
