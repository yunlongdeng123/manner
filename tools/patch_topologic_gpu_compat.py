#!/usr/bin/env python3
"""Apply the two verified PyTorch 2.1/MMCV 1.7.2 compatibility edits to a venv.

Only files under the explicitly supplied virtual environment are changed.
An untouched .orig copy is retained beside each file.
"""

from __future__ import annotations

import argparse
from pathlib import Path


PATCHES = (
    (
        "mmdet3d/__init__.py",
        "mmcv_maximum_version = '1.7.0'",
        "mmcv_maximum_version = '1.7.2'",
    ),
    (
        "mmcv/parallel/_functions.py",
        "streams = [_get_stream(device) for device in target_gpus]",
        'streams = [_get_stream(torch.device("cuda", device)) for device in target_gpus]',
    ),
)


def patch(site_packages: Path) -> None:
    for relative_path, old, new in PATCHES:
        target = site_packages / relative_path
        source = target.read_text(encoding="utf-8")
        if new in source:
            print(f"already patched: {target}")
            continue
        if source.count(old) != 1:
            raise RuntimeError(f"Expected exactly one compatibility target in {target}")
        backup = target.with_name(target.name + ".orig")
        if backup.exists() and backup.read_text(encoding="utf-8") != source:
            raise RuntimeError(f"Existing backup differs from current package: {backup}")
        if not backup.exists():
            backup.write_text(source, encoding="utf-8")
        target.write_text(source.replace(old, new), encoding="utf-8")
        print(f"patched: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, required=True)
    args = parser.parse_args()
    candidates = list(args.venv.resolve().glob("lib/python*/site-packages"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one site-packages in {args.venv}: {candidates}")
    patch(candidates[0])


if __name__ == "__main__":
    main()
