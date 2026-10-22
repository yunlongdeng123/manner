#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/manner-repro
export PATH=/root/miniconda3/envs/umpe/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cat > "$ROOT/manifests/umpe_sdk_constraints.txt" <<'EOF'
numpy==1.23.5
scipy==1.10.1
numba==0.56.4
matplotlib==3.5.3
networkx==2.8.8
opencv-python==4.8.1.78
opencv-contrib-python==4.8.1.78
pyarrow==12.0.1
EOF
pip install -i https://mirrors.aliyun.com/pypi/simple -c "$ROOT/manifests/umpe_sdk_constraints.txt" av2==0.2.1 ninja==1.11.1.4
python -c 'from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader; from av2.map.map_api import ArgoverseStaticMap; from av2.geometry.se3 import SE3; import av2.geometry.interpolate; print("AV2 transitive imports OK")'
pip check > "$ROOT/manifests/umpe_pip_check.txt"
pip freeze --all > "$ROOT/manifests/umpe_pip_freeze.txt"
touch "$ROOT/manifests/umpe_sdk_ready"
