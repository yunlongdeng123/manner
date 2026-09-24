# GPU hand-off boundary

The CPU-only phase ends after these checks pass:

- Official subset-A annotations pass MD5 verification and are extracted.
- All 33,642 subset-A val images are selectively materialized from the three
  AutoDL Hub AV2 val shards, with zero missing requests and valid JPEG markers.
- `openlanev2_manifest.jsonl` and `openlanev2_geo_split.jsonl` exist.
- `data_validation.json` reports `"valid": true`.
- `av2_camera_requests.jsonl` contains the train/val shard request index.
- Unit tests for tagging, splitting, sampling, model heads, and losses pass.

The next operation is official TopoLogic checkpoint inference to cache lane
query features and predictions.  That operation requires an NVIDIA GPU and the
legacy CUDA 11.1 / PyTorch 1.9.1 / MMCV environment.  Do not attempt to solve
or build that stack in the 2GB no-GPU container.

The exporter must pass to `FeatureCacheWriter.write` one-to-one query-to-GT
indices (`-1` for unmatched queries), predicted query centerlines, query
features, lane confidence, GT centerlines, GT LCLC, and the lane-level
`is_intersection_or_connector` flags. Pass official predicted LCLC
probabilities as `base_topology_scores`. Do not threshold the official
probabilities before saving. The writer derives matched-query adjacency, GT
endpoint tangents, and transition-boundary labels. Train/evaluation code now
requires this cache schema; caches from before this change need regeneration.

When a GPU instance is available, run the guard first:

```bash
source /root/autodl-tmp/ScenarioTopo/tools/env.sh
python /root/autodl-tmp/ScenarioTopo/tools/preflight_gpu.py \
  --checkpoint /root/autodl-tmp/checkpoints/topologic_r50_8x1_24e_olv2_subset_A.pth \
  --config /root/autodl-tmp/third_party/TopoLogic/projects/configs/topologic_r50_8x1_24e_olv2_subset_A.py \
  --data-root /root/autodl-tmp/datasets/openlanev2/OpenLane-V2
```

Create a separate `topologic-gpu` environment at that time.  The lightweight
`scenariotopo` venv is intentionally not mutated into the legacy framework
environment.

Training images do not fit on the 50GB disk as a full copy.  Once the GPU
environment is ready, use the compact request index and `--allow-partial` to
materialize one `train-NNN.tar` shard, export and validate its frozen feature
cache, then remove only that shard's materialized image copies before moving
to the next shard.
