#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/manner-repro
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
while ! test -f "$ROOT/manifests/umpe_infos_ready.json"; do sleep 10; done
/root/miniconda3/envs/umpe/bin/python "$ROOT/scripts/refine_umpe_smoke.py"
for script in "$ROOT"/scripts/*.sh; do bash -n "$script"; done
/root/miniconda3/envs/topogpt/bin/python -m compileall -q "$ROOT/scripts"
/root/miniconda3/bin/python "$ROOT/scripts/preparation_audit.py"
/root/miniconda3/bin/python "$ROOT/status.py" > "$ROOT/manifests/final_status.json"
cd "$ROOT"
if ! test -d .git; then git init -b main; fi
git config --local user.name Codex
git config --local user.email codex@localhost
git config --local core.autocrlf false
git add .gitignore README.md status.py scripts configs patches manifests
if ! git diff --cached --quiet; then
  git commit -m 'Prepare TopoGPT and UMPE reproducible evaluation environment'
fi
git bundle create reproduction.bundle --all
git bundle verify reproduction.bundle
tar --exclude='manifests/code-inspection.tgz' --exclude='*.bundle' --exclude='__pycache__' \
  -czf repro-delivery.tgz README.md status.py scripts configs patches manifests logs .gitignore
sha256sum reproduction.bundle repro-delivery.tgz > SHA256SUMS-delivery
sync
printf 'No-GPU preparation and delivery snapshot complete.\n'
