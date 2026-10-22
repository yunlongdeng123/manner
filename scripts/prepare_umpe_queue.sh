#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/manner-repro
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
while ! test -f "$ROOT/manifests/nuscenes_coverage.json"; do sleep 10; done
/root/miniconda3/envs/umpe/bin/python "$ROOT/scripts/prepare_umpe_infos.py" --probe
exec /root/miniconda3/envs/umpe/bin/python "$ROOT/scripts/prepare_umpe_infos.py"
