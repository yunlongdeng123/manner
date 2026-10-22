#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/manner-repro
export PATH=/root/miniconda3/envs/topogpt/bin:$PATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TORCH_HOME="$ROOT/weights/torch"
cd "$ROOT/repos/TopoGPT"
export PYTHONPATH="$PWD:$PWD/MapPrior/tools/eval/graph:${PYTHONPATH:-}"
test -f "$ROOT/manifests/openlane_smoke_coverage.json"
OUT="/root/manner-repro-smoke/topogpt_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
python "$ROOT/scripts/topogpt_smoke.py" --out "$OUT" --seconds 480 2>&1 | tee "$OUT/train.log"
python "$ROOT/scripts/topogpt_smoke.py" --out "$OUT" --resume "$OUT/train.ckpt" 2>&1 | tee "$OUT/resume.log"
python MapPrior/scripts/train.py 'tags=[gpu-smoke]' exp_name=smoke_evaluation \
  model=finetune data=finetune_subA train=False +val=True \
  ckpt_path="$ROOT/weights/topogpt/finetune.ckpt" \
  model.sample_cfg.save_pred_path="$OUT/predictions.pkl" \
  paths.data_dir="$ROOT/data/topogpt_smoke" paths.log_dir="$OUT/eval_logs" \
  data.num_workers=0 data.batch_size=1 trainer.devices=1 \
  model.eval_cfg.eval_seq_decode=True 2>&1 | tee "$OUT/eval.log"
python MapPrior/tools/eval/metric.py --task pr topo graph \
  --gt-file-path "$ROOT/data/topogpt_smoke/openlane_subA/data_dict_subset_A_val.pkl" \
  --pred-file-path "$OUT/predictions.pkl" --save-path "$OUT/metrics.json" 2>&1 | tee "$OUT/metric.log"
python "$ROOT/scripts/finish_topogpt_smoke.py" "$OUT"
