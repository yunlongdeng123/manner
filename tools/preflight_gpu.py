#!/usr/bin/env python3
"""Fail closed before any step that needs TopoLogic CUDA inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    missing = [str(path) for path in (args.checkpoint, args.config, args.data_root) if not path.exists()]
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5, check=False)
        gpu_lines = [line for line in result.stdout.splitlines() if line.strip()]
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired):
        gpu_lines = []
    report = {"gpu_available": bool(gpu_lines), "gpus": gpu_lines, "missing_paths": missing}
    print(json.dumps(report, indent=2))
    if missing:
        print("INPUT_REQUIRED: prepare the listed files before feature caching.", file=sys.stderr)
        sys.exit(2)
    if not gpu_lines:
        print("GPU_REQUIRED: start the instance in GPU mode before TopoLogic feature export.", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()

