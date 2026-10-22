#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/manner-repro
export PATH=/root/miniconda3/envs/umpe/bin:$PATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TORCH_HOME="$ROOT/weights/torch"
cd "$ROOT/repos/UMPE/Maptrv2"
export PYTHONPATH="$PWD:$PWD/mmdetection3d:${PYTHONPATH:-}"
test -f "$ROOT/manifests/umpe_infos_ready.json"
test -f "$ROOT/manifests/umpe_extensions_ready"
test -f "$ROOT/manifests/umpe_prior_keys_verified.json"
test -f "$ROOT/manifests/umpe_smoke_selection.json"
OUT="$ROOT/results/smoke/umpe_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
python "$ROOT/scripts/umpe_smoke.py" --out "$OUT" --seconds 480 2>&1 | tee "$OUT/train.log"
python "$ROOT/scripts/umpe_smoke.py" --out "$OUT" --resume "$OUT/train.pth" 2>&1 | tee "$OUT/resume.log"
# Do not load the training-only hook when running the standalone official test CLI.
python "$ROOT/scripts/make_umpe_smoke_eval.py" "$OUT"
bash tools/dist_test_map.sh "$OUT/eval_config.py" "$ROOT/weights/umpe/maptrv2_umpe_ft.pth" 1 \
  --out "$OUT/predictions.pkl" 2>&1 | tee "$OUT/eval.log"
python "$ROOT/scripts/finish_umpe_smoke.py" "$OUT"
