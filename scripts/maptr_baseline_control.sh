#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/manner-repro
export PATH=/root/miniconda3/envs/umpe/bin:$PATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1
export TORCH_HOME="$ROOT/weights/torch" PORT=29506
cd "$ROOT/repos/UMPE/Maptrv2"
export PYTHONPATH="$PWD:$PWD/mmdetection3d:${PYTHONPATH:-}"
python -c 'import socket; s=socket.socket(); s.bind(("0.0.0.0",29506)); s.close()'
test -f "$ROOT/manifests/umpe_gpu_smoke_passed.json"
python "$ROOT/scripts/audit_maptr_baseline.py"
OUT="$ROOT/results/diagnostic/maptrv2_baseline_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT/preflight"
export MAPTR_CONTROL_OUT="$OUT"
python - <<'PY'
import os,pathlib
r=pathlib.Path('/root/autodl-tmp/manner-repro'); out=pathlib.Path(os.environ['MAPTR_CONTROL_OUT'])
source="_base_ = '/root/autodl-tmp/manner-repro/configs/maptr_baseline_control.py'\ndata = dict(test=dict(ann_file='data/nuscenes/nuscenes_map_infos_temporal_val_smoke_diverse.pkl', map_ann_file="+repr(str(out/'preflight/gt.json'))+",sdmap_pkl="+repr(str(r/'data/umpe_priors/selected/sd_val_smoke.pkl'))+",hdmap_pkl="+repr(str(r/'data/umpe_priors/selected/hd_val_smoke.pkl'))+"))\n"
(out/'preflight/config.py').write_text(source)
(r/'manifests/maptr_baseline_control_started.json').write_text(__import__('json').dumps(dict(directory=str(out),purpose='Official MapTRv2 diagnostic baseline on identical validation data'),indent=2))
PY
bash tools/dist_test_map.sh "$OUT/preflight/config.py" "$ROOT/weights/diagnostic/maptrv2_nusc_r50_24e.pth" 1 --out "$OUT/preflight/predictions.pkl" > "$OUT/preflight/evaluation.log" 2>&1
python - <<'PY'
import json,os,pathlib,pickle,numpy as np
out=pathlib.Path(os.environ['MAPTR_CONTROL_OUT'])
with (out/'preflight/predictions.pkl').open('rb') as f: pred=pickle.load(f)
assert len(pred)==8
metrics=json.loads((out/'preflight/predictions.metrics.json').read_text())
assert all(np.isfinite(v) for v in metrics.values())
print('8-frame actual GPU inference and official evaluation passed',flush=True)
PY
bash tools/dist_test_map.sh "$ROOT/configs/maptr_baseline_control.py" "$ROOT/weights/diagnostic/maptrv2_nusc_r50_24e.pth" 1 --out "$OUT/predictions.pkl" > "$OUT/evaluation.log" 2>&1
python "$ROOT/scripts/verify_prediction_coverage.py" umpe "$OUT/predictions.pkl"
