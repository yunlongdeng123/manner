#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${SCENARIOTOPO_ROOT:-/root/autodl-tmp/ScenarioTopo}"
ENV_ROOT="${SCENARIOTOPO_ENV:-/root/autodl-tmp/envs/scenariotopo}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PIP_CACHE_DIR="${SCENARIOTOPO_CACHE:-/root/autodl-tmp/cache}/pip"
export TMPDIR=/root/autodl-tmp/tmp

"$ENV_ROOT/bin/python" -m pip install \
  --disable-pip-version-check \
  --no-input \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r "$PROJECT_ROOT/requirements-model-cpu.txt"

