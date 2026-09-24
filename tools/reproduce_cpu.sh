#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${SCENARIOTOPO_ROOT:-/root/autodl-tmp/ScenarioTopo}"
source "$PROJECT_ROOT/tools/env.sh"

DATA_ROOT="${OPENLANEV2_ROOT}"
DATA_DICT="${OPENLANEV2_DATA_DICT:-$DATA_ROOT/data_dict_subset_A.json}"
OUTPUT_ROOT="${SCENARIOTOPO_OUTPUT:-$PROJECT_ROOT/results}"

mkdir -p "$OUTPUT_ROOT"
python "$PROJECT_ROOT/tools/check_resources.py"
python "$PROJECT_ROOT/tools/build_scene_manifest.py" \
  --data-root "$DATA_ROOT" \
  --data-dict "$DATA_DICT" \
  --config "$PROJECT_ROOT/configs/data.yaml" \
  --output "$OUTPUT_ROOT/openlanev2_manifest.jsonl"
python "$PROJECT_ROOT/tools/make_geo_splits.py" \
  --manifest "$OUTPUT_ROOT/openlanev2_manifest.jsonl" \
  --output-manifest "$OUTPUT_ROOT/openlanev2_geo_split.jsonl" \
  --split-map "$OUTPUT_ROOT/geo_split_map.json" \
  --radius-m 80
python "$PROJECT_ROOT/tools/validate_data.py" \
  --manifest "$OUTPUT_ROOT/openlanev2_geo_split.jsonl" \
  --data-root "$DATA_ROOT" \
  --report "$OUTPUT_ROOT/data_validation.json"

