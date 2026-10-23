#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/manner-repro
export PORT=29505
/root/miniconda3/bin/python -c 'import socket; s=socket.socket(); s.bind(("0.0.0.0",29505)); s.close()'
export UMPE_OUT_DIR="$ROOT/results/umpe/base_parallel_$(date -u +%Y%m%dT%H%M%SZ)"
export UMPE_CHECKPOINT="$ROOT/weights/umpe/maptrv2_umpe.pth"
bash "$ROOT/scripts/umpe_eval.sh"
