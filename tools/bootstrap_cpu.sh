#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${SCENARIOTOPO_ROOT:-/root/autodl-tmp/ScenarioTopo}"
ENV_ROOT="${SCENARIOTOPO_ENV:-/root/autodl-tmp/envs/scenariotopo}"
CACHE_ROOT="${SCENARIOTOPO_CACHE:-/root/autodl-tmp/cache}"
BASE_PYTHON="${BASE_PYTHON:-/root/miniconda3/bin/python}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export TMPDIR=/root/autodl-tmp/tmp

mkdir -p "$CACHE_ROOT/pip" "$TMPDIR" "$(dirname "$ENV_ROOT")"

if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
  "$BASE_PYTHON" -m venv "$ENV_ROOT"
fi

"$ENV_ROOT/bin/python" -m pip install \
  --disable-pip-version-check \
  --no-input \
  -r "$PROJECT_ROOT/requirements-cpu.txt"
"$ENV_ROOT/bin/python" -m pip install \
  --disable-pip-version-check \
  --no-input \
  --no-build-isolation \
  -e "$PROJECT_ROOT"

printf 'CPU environment ready: %s\n' "$ENV_ROOT"

