from __future__ import annotations

import json
from pathlib import Path

from scenariotopo.data.geosplit import create_geo_split
from scenariotopo.data.manifest import build_manifest, iter_jsonl
from scenariotopo.data.scene_tags import derive_scene_tags
from scenariotopo.data.validate import validate_manifest


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "openlanev2"


def test_scene_tags_cover_structural_cases():
    frame = json.loads(
        (FIXTURE_ROOT / "train" / "seg_split_merge" / "info" / "100.json").read_text(encoding="utf-8")
    )
    result = derive_scene_tags(frame)
    assert {"Split", "Merge", "Curve", "FarField", "ShortLane", "ElevationOverlap"} <= set(result["tags"])
    assert result["stats"]["edge_count"] == 4


def test_manifest_is_streamed_and_validated(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    report = build_manifest(
        FIXTURE_ROOT,
        FIXTURE_ROOT / "data_dict_subset_A.json",
        manifest,
        progress_every=0,
    )
    assert report["frames"] == 1
    record = next(iter_jsonl(manifest))
    assert record["frame_key"] == "train/seg_split_merge/100"
    assert validate_manifest(manifest, FIXTURE_ROOT)["valid"] is True


def test_geo_split_keeps_clusters_disjoint(tmp_path):
    source = tmp_path / "source.jsonl"
    records = [
        {"frame_key": "a/1", "segment_id": "a", "source_id": "city", "source_split": "train", "pose_translation": [0, 0, 0], "tags": []},
        {"frame_key": "b/1", "segment_id": "b", "source_id": "city", "source_split": "train", "pose_translation": [20, 0, 0], "tags": ["Split"]},
        {"frame_key": "c/1", "segment_id": "c", "source_id": "city", "source_split": "train", "pose_translation": [500, 0, 0], "tags": []},
    ]
    source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    output = tmp_path / "split.jsonl"
    split_map = tmp_path / "map.json"
    create_geo_split(source, output, split_map, radius_m=80.0)
    enriched = list(iter_jsonl(output))
    assert enriched[0]["physical_cluster_id"] == enriched[1]["physical_cluster_id"]
    assert enriched[0]["model_split"] == enriched[1]["model_split"]
    assert validate_manifest(output)["valid"] is True


def test_unlabeled_official_test_stays_out_of_training_splits(tmp_path):
    fixture = tmp_path / "openlanev2"
    info = fixture / "test" / "benchmark_seg" / "info"
    info.mkdir(parents=True)
    (info / "1.json").write_text(
        json.dumps(
            {
                "meta_data": {"source_id": "benchmark-log"},
                "pose": {"translation": [0, 0, 0]},
                "sensor": {},
            }
        ),
        encoding="utf-8",
    )
    data_dict = fixture / "data_dict_subset_A.json"
    data_dict.write_text(json.dumps({"test": {"benchmark_seg": ["1"]}}), encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    build_manifest(fixture, data_dict, manifest, progress_every=0)
    record = next(iter_jsonl(manifest))
    assert record["has_annotation"] is False
    assert record["tags"] == []

    output = tmp_path / "split.jsonl"
    create_geo_split(manifest, output, tmp_path / "map.json", radius_m=80.0)
    enriched = next(iter_jsonl(output))
    assert enriched["model_split"] == "benchmark_test"
