#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/manner-repro
export PATH=/root/miniconda3/envs/umpe/bin:$PATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export TORCH_HOME="$ROOT/weights/torch"
cd "$ROOT/repos/UMPE/Maptrv2"
export PYTHONPATH="$PWD:$PWD/mmdetection3d:${PYTHONPATH:-}"
test -f "$ROOT/manifests/umpe_gpu_smoke_passed.json" || { echo 'GPU smoke must pass before full evaluation.'; exit 2; }
test -f "$ROOT/manifests/nuscenes_coverage.json"
test -f "$ROOT/manifests/umpe_infos_ready.json"
test -f "$ROOT/manifests/umpe_extensions_ready"
CHECKPOINT="${UMPE_CHECKPOINT:-$ROOT/weights/umpe/maptrv2_umpe_ft.pth}"
NAME="$(basename "$CHECKPOINT" .pth)"
OUT="${UMPE_OUT_DIR:-$ROOT/results/umpe/${NAME}_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
bash tools/dist_test_map.sh "$ROOT/configs/umpe_reproduce.py" \
  "$CHECKPOINT" 1 \
  --out "$OUT/predictions.pkl" "$@"
python "$ROOT/scripts/verify_prediction_coverage.py" umpe "$OUT/predictions.pkl"
