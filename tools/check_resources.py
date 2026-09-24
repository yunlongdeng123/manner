#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess


def _read_first(paths):
    for path in paths:
        candidate = Path(path)
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()
    return None


def _memory_limit_bytes():
    raw = _read_first([
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ])
    if raw and raw != "max":
        return int(raw)
    return None


def _gpu_available():
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired):
        return False, []
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return result.returncode == 0 and bool(lines), lines


def main():
    memory_limit = _memory_limit_bytes()
    gpu, gpu_lines = _gpu_available()
    system = shutil.disk_usage("/")
    data = shutil.disk_usage("/root/autodl-tmp")
    report = {
        "memory_limit_bytes": memory_limit,
        "memory_limit_gib": round(memory_limit / 2**30, 3) if memory_limit else None,
        "system_disk_free_gib": round(system.free / 2**30, 3),
        "data_disk_free_gib": round(data.free / 2**30, 3),
        "gpu_available": gpu,
        "gpus": gpu_lines,
        "low_memory_mode": bool(memory_limit and memory_limit <= 3 * 2**30),
        "recommended_workers": 0 if memory_limit and memory_limit <= 3 * 2**30 else 2,
        "recommended_threads": 1,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

