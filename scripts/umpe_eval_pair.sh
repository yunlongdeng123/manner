#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/manner-repro
test -f "$ROOT/manifests/umpe_gpu_smoke_passed.json"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export UMPE_OUT_DIR="$ROOT/results/umpe/finetuned_$STAMP"
export UMPE_CHECKPOINT="$ROOT/weights/umpe/maptrv2_umpe_ft.pth"
mkdir -p "$UMPE_OUT_DIR"
bash "$ROOT/scripts/umpe_eval.sh" 2>&1 | tee "$UMPE_OUT_DIR/evaluation.log"
export UMPE_OUT_DIR="$ROOT/results/umpe/base_$STAMP"
export UMPE_CHECKPOINT="$ROOT/weights/umpe/maptrv2_umpe.pth"
mkdir -p "$UMPE_OUT_DIR"
bash "$ROOT/scripts/umpe_eval.sh" 2>&1 | tee "$UMPE_OUT_DIR/evaluation.log"
printf '%s\n' "$STAMP" > "$ROOT/manifests/umpe_both_checkpoints_evaluated"
