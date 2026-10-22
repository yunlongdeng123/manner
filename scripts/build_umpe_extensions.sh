#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/manner-repro
export PATH=/root/miniconda3/envs/umpe/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8 TORCH_CUDA_ARCH_LIST=8.6 FORCE_CUDA=1 MAX_JOBS=4
python -c 'import torch; assert torch.cuda.is_available(), "Switch the instance to GPU mode before compiling and validating CUDA extensions"; print(torch.cuda.get_device_name(0))'
cd "$ROOT/repos/UMPE/Maptrv2/mmdetection3d"
python setup.py build_ext --inplace
pip install --no-deps --no-build-isolation -e .
cd "$ROOT/repos/UMPE/Maptrv2/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn"
python setup.py build install
cd "$ROOT/repos/UMPE/Maptrv2"
PYTHONPATH="$PWD:$PWD/mmdetection3d" python -c 'import mmdet3d; import projects.mmdet3d_plugin; import GeometricKernelAttention; print("UMPE extension imports OK")'
pip freeze --all > "$ROOT/manifests/umpe_pip_freeze_after_build.txt"
touch "$ROOT/manifests/umpe_extensions_ready"
