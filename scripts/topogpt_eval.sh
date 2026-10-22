#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/manner-repro
export PATH=/root/miniconda3/envs/topogpt/bin:$PATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 HYDRA_FULL_ERROR=1
export TORCH_HOME="$ROOT/weights/torch"
cd "$ROOT/repos/TopoGPT"
export PYTHONPATH="$PWD:$PWD/MapPrior/tools/eval/graph:${PYTHONPATH:-}"
test -f "$ROOT/manifests/topogpt_gpu_smoke_passed.json" || { echo 'GPU smoke must pass before full evaluation.'; exit 2; }
test -f "$ROOT/manifests/openlane_coverage.json"
OUT="${TOPOGPT_OUT_DIR:-$ROOT/results/topogpt/run_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
python MapPrior/scripts/train.py 'tags=[ar,predict,reproduction]' exp_name=reproduce_subset_A \
  model=finetune data=finetune_subA train=False +val=True \
  ckpt_path="$ROOT/weights/topogpt/finetune.ckpt" \
  model.sample_cfg.save_pred_path="$OUT/predictions.pkl" \
  paths.data_dir="$ROOT/data/topogpt" paths.log_dir="$OUT/logs" \
  data.num_workers=4 data.batch_size=1 trainer.devices=1 model.eval_cfg.eval_seq_decode=True "$@"
python "$ROOT/scripts/verify_prediction_coverage.py" topogpt "$OUT/predictions.pkl"
python MapPrior/tools/eval/metric.py --task pr topo graph \
  --gt-file-path "$ROOT/data/topogpt/openlane_subA/data_dict_subset_A_val.pkl" \
  --pred-file-path "$OUT/predictions.pkl" \
  --save-path "$OUT/metrics.json"
