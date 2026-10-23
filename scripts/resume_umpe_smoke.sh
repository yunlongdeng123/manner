#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/manner-repro
SOURCE="$1"
export PATH=/root/miniconda3/envs/umpe/bin:$PATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TORCH_HOME="$ROOT/weights/torch"
cd "$ROOT/repos/UMPE/Maptrv2"
export PYTHONPATH="$PWD:$PWD/mmdetection3d:${PYTHONPATH:-}"
OUT="$ROOT/results/smoke/umpe_recovery_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
cp "$SOURCE/train_audit.json" "$OUT/train_audit.json"
ln -s "$SOURCE/train.pth" "$OUT/train.pth"
printf '%s\n' "$SOURCE" > "$OUT/original_smoke_directory.txt"
sha256sum "$SOURCE/train.pth" > "$OUT/source_checkpoint.sha256"
python "$ROOT/scripts/umpe_smoke.py" --out "$OUT" --resume "$SOURCE/train.pth" 2>&1 | tee "$OUT/resume.log"
python "$ROOT/scripts/make_umpe_smoke_eval.py" "$OUT"
bash tools/dist_test_map.sh "$OUT/eval_config.py" "$ROOT/weights/umpe/maptrv2_umpe_ft.pth" 1 \
  --out "$OUT/predictions.pkl" 2>&1 | tee "$OUT/eval.log"
python "$ROOT/scripts/finish_umpe_smoke.py" "$OUT"
