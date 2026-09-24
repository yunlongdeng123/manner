# AutoDL Hub data preparation

Camera images come from AutoDL **Public Data** (the user called it AutoDL Hub).
The available Hub item is the raw Argoverse 2 Sensor dataset; OpenLane-V2's
centerline and topology label overlay is not present there, so that annotation
layer must retain the official OpenLane-V2 release and checksum.

## Installed sources

The AutoDL Hub image source is mounted read-only at:

```text
/root/autodl-pub/argoverse2.0-sensor
```

It contains uncompressed `train-*.tar` and `val-*.tar` AV2 shards of roughly
52–54GB each. Never extract a complete shard into this instance's 50GB data
disk.

OpenLane-V2 subset A is the Argoverse 2 based subset.  Its topology annotation
layer is installed from `OpenLane-V2_subset_A_info.tar`, after checking the
official MD5 `95bf28ccf22583d20434d75800be065d`:

```bash
bash tools/install_openlane_annotations.sh \
  /root/autodl-tmp/datasets/_archives/OpenLane-V2_subset_A_info.tar \
  /root/autodl-tmp/datasets/openlanev2/OpenLane-V2
```

The installer refuses a checksum mismatch and refuses to overlay a non-empty
target.  The official `info.tar` contains only the split directories, so the
installer copies `data_dict_subset_A.json` and `openlanev2.md5` from the pinned
OpenLane-V2 devkit checkout into the data root.  After a verified extraction,
the reproducible 9.5GB download archive can be removed to make room for
selected images; the extracted annotation layer is retained.

## Selective image materialization

Extract only OpenLane-referenced validation images from the Hub shards:

```bash
python tools/extract_av2_hub_images.py \
  --data-root /root/autodl-tmp/datasets/openlanev2/OpenLane-V2 \
  --data-dict /root/autodl-tmp/datasets/openlanev2/OpenLane-V2/data_dict_subset_A.json \
  --hub-root /root/autodl-pub/argoverse2.0-sensor \
  --archive-glob 'val-*.tar' \
  --splits val \
  --report results/av2_hub_val_extraction.json
```

The extractor is resumable, single-process, and writes each image atomically.
It seeks past unrelated tar members instead of expanding the full AV2 dataset.

Do not materialize all training images on the 50GB disk.  During the later GPU
feature-export phase, process training shards incrementally: extract the
referenced images for one shard, export frozen features, validate the cache,
then remove only those materialized image copies before advancing.  The Hub
tars themselves remain untouched.

Build the compact request index once during the CPU phase:

```bash
python tools/build_av2_request_index.py \
  --manifest results/openlanev2_manifest.jsonl \
  --output results/av2_camera_requests.jsonl \
  --splits train val
```

Later, materialize one training shard without treating the intentionally
unseen requests from other shards as an error:

```bash
python tools/extract_av2_hub_images.py \
  --data-root "$OPENLANEV2_ROOT" \
  --request-index results/av2_camera_requests.jsonl \
  --hub-root /root/autodl-pub/argoverse2.0-sensor \
  --archive-glob train-000.tar \
  --splits train \
  --allow-partial \
  --report results/av2_hub_train_000_extraction.json
```

## Build the CPU-side assets

Set the root containing `data_dict_subset_A.json`, then run:

```bash
export OPENLANEV2_ROOT=/root/autodl-tmp/datasets/openlanev2/OpenLane-V2
export OPENLANEV2_DATA_DICT="$OPENLANEV2_ROOT/data_dict_subset_A.json"
bash tools/reproduce_cpu.sh
```

Generated files are written to `results/` and remain excluded from Git because
they are dataset-derived artifacts.
