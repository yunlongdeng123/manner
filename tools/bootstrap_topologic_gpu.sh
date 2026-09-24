#!/usr/bin/env bash
set -euo pipefail

# Run on the AutoDL image with Python 3.10, torch 2.1.2+cu118 and CUDA GPU.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base_python="${BASE_PYTHON:-/root/miniconda3/bin/python}"
gpu_venv="${GPU_VENV:-/root/autodl-tmp/envs/topologic-gpu}"

"$base_python" - <<'PY'
import sys
import torch
assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__ == '2.1.2+cu118', torch.__version__
assert torch.cuda.is_available(), 'CUDA GPU unavailable'
print('Base CUDA PyTorch:', torch.__version__, torch.cuda.get_device_name(0))
PY

"$base_python" -m venv --system-site-packages "$gpu_venv"
gpu_python="$gpu_venv/bin/python"
"$gpu_python" -m pip install --timeout 120 --retries 4 --no-deps \
  --only-binary=mmcv-full 'mmcv-full==1.7.2' \
  -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
"$gpu_python" -m pip install --timeout 120 --retries 4 --no-deps \
  -r "$project_root/requirements-topologic-gpu.txt"
"$gpu_python" "$project_root/tools/patch_topologic_gpu_compat.py" --venv "$gpu_venv"
"$gpu_python" - <<'PY'
import torch
import mmcv, mmdet, mmseg, mmdet3d
from mmcv.ops import nms
boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device='cuda')
scores = torch.tensor([0.9, 0.8], device='cuda')
assert nms(boxes, scores, 0.5)[1].tolist() == [0]
print('GPU framework and CUDA NMS verified')
PY
