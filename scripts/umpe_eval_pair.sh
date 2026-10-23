#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/manner-repro
test -f "$ROOT/manifests/umpe_gpu_smoke_passed.json"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export UMPE_OUT_DIR="$ROOT/results/umpe/finetuned_$STAMP"
export UMPE_CHECKPOINT="$ROOT/weights/umpe/maptrv2_umpe_ft.pth"
bash "$ROOT/scripts/umpe_eval.sh" 2>&1 | tee "$ROOT/logs/umpe_full_eval_20261023.log"
export UMPE_OUT_DIR="$ROOT/results/umpe/base_$STAMP"
export UMPE_CHECKPOINT="$ROOT/weights/umpe/maptrv2_umpe.pth"
bash "$ROOT/scripts/umpe_eval.sh" 2>&1 | tee "$ROOT/logs/umpe_base_eval_20261023.log"
printf '%s\n' "$STAMP" > "$ROOT/manifests/umpe_both_checkpoints_evaluated"
