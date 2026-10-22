#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/manner-repro
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$ROOT/wheels"
case "$1" in
topogpt) VER='2.1.0'; CU='118'; PY='310'; TV='0.16.0';;
umpe) VER='1.9.1'; CU='111'; PY='38'; TV='0.10.1';;
*) exit 2;;
esac
NAME="torch-${VER}+cu${CU}-cp${PY}-cp${PY}-linux_x86_64.whl"
curl --http1.1 -fL -x '' --connect-timeout 12 --retry 4 --speed-limit 1048576 --speed-time 30 -C - -o "$ROOT/wheels/$NAME" "https://download.pytorch.org/whl/cu${CU}/torch-${VER}%2Bcu${CU}-cp${PY}-cp${PY}-linux_x86_64.whl"
/root/miniconda3/envs/"$1"/bin/pip install "$ROOT/wheels/$NAME" "torchvision==${TV}+cu${CU}" -f https://download.pytorch.org/whl/torch_stable.html -i https://mirrors.aliyun.com/pypi/simple
touch "$ROOT/manifests/${1}_torch_ready"
