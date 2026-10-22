#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/manner-repro
export PATH=/root/miniconda3/envs/topogpt/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_HOME="$ROOT/weights/torch"
pip install -i https://mirrors.aliyun.com/pypi/simple setuptools==69.5.1
cd "$ROOT/repos/TopoGPT"
export PYTHONPATH="$PWD:$PWD/MapPrior/tools/eval/graph"
python -c 'import torch,lightning; import MapPrior.modules.autoregression.vector_ar; import MapPrior.tools.eval.metric; print(torch.__version__,lightning.__version__,"imports OK")'
pip freeze --all > "$ROOT/manifests/topogpt_pip_freeze.txt"
pip check > "$ROOT/manifests/topogpt_pip_check.txt"
touch "$ROOT/manifests/topogpt_deps_ready"
python "$ROOT/scripts/verify_collect.py"
python "$ROOT/scripts/checkpoint_inventory.py"
