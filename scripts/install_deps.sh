#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/manner-repro
ENV=$1
export PATH="/root/miniconda3/envs/$ENV/bin:$PATH"
export PIP_DISABLE_PIP_VERSION_CHECK=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
while ! test -f "$ROOT/manifests/${ENV}_torch_ready"; do sleep 10; done
if [[ "$ENV" == topogpt ]]; then
  pip install -i https://mirrors.aliyun.com/pypi/simple setuptools==69.5.1
  cat > "$ROOT/manifests/topogpt_constraints.txt" <<'EOF'
torch==2.1.0+cu118
torchvision==0.16.0+cu118
pytorch-lightning==2.0.3
torchmetrics==1.3.2
numpy==1.23.4
fsspec==2023.6.0
EOF
  pip install -i https://mirrors.aliyun.com/pypi/simple -c "$ROOT/manifests/topogpt_constraints.txt" -r "$ROOT/repos/TopoGPT/requirements.txt" tensorboard
  cd "$ROOT/repos/TopoGPT"
  PYTHONPATH="$PWD:$PWD/MapPrior/tools/eval/graph" python -c 'import torch,lightning; import MapPrior.modules.autoregression.vector_ar; import MapPrior.tools.eval.metric; print(torch.__version__,lightning.__version__,"imports OK")'
else
  cat > "$ROOT/manifests/umpe_constraints.txt" <<'EOF'
numpy==1.23.5
scipy==1.10.1
numba==0.56.4
matplotlib==3.5.3
networkx==2.8.8
jupyter==1.0.0
notebook==6.5.7
ipykernel==6.29.5
jupyter-client==7.4.9
EOF
  export PIP_CONSTRAINT="$ROOT/manifests/umpe_constraints.txt"
  pip install -i https://mirrors.aliyun.com/pypi/simple 'pip<25' 'setuptools<60' wheel 'numpy==1.23.5' 'scipy==1.10.1' 'numba==0.56.4' 'networkx==2.8.8' 'matplotlib==3.5.3' 'pillow<10' 'yapf==0.40.1' 'opencv-python==4.8.1.78'
  pip install mmcv-full==1.4.0 --only-binary mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html -i https://mirrors.aliyun.com/pypi/simple
  pip install -i https://mirrors.aliyun.com/pypi/simple mmdet==2.14.0 mmsegmentation==0.14.1 timm==0.6.13 nuscenes-devkit==1.1.9 shapely==1.8.5.post1 lyft_dataset_sdk==0.0.8 'plyfile<1' 'scikit-image==0.19.3' tensorboard trimesh==2.35.39 terminaltables einops tqdm future 'pyquaternion==0.9.9' 'imageio<2.35'
  python -c 'import torch,mmcv,mmdet,mmseg,nuscenes; from mmcv.ops import MultiScaleDeformableAttention; print(torch.__version__, mmcv.__version__, "imports OK")'
fi
pip freeze --all > "$ROOT/manifests/${ENV}_pip_freeze.txt"
pip check > "$ROOT/manifests/${ENV}_pip_check.txt"
touch "$ROOT/manifests/${ENV}_deps_ready"
