#!/usr/bin/env bash

export SCENARIOTOPO_ROOT="${SCENARIOTOPO_ROOT:-/root/autodl-tmp/ScenarioTopo}"
export SCENARIOTOPO_ENV="${SCENARIOTOPO_ENV:-/root/autodl-tmp/envs/scenariotopo}"
export OPENLANEV2_ROOT="${OPENLANEV2_ROOT:-/root/autodl-tmp/datasets/openlanev2/OpenLane-V2}"
export TOPOLOGIC_ROOT="${TOPOLOGIC_ROOT:-/root/autodl-tmp/third_party/TopoLogic}"
export OPENLANEV2_DEVKIT_ROOT="${OPENLANEV2_DEVKIT_ROOT:-/root/autodl-tmp/third_party/OpenLane-V2}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/cache/pip}"
export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$SCENARIOTOPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$SCENARIOTOPO_ENV/bin:$PATH"
