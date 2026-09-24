#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from scenariotopo.data.manifest import build_manifest
from scenariotopo.data.scene_tags import SceneTagThresholds


def parse_args():
    parser = argparse.ArgumentParser(description="Stream OpenLane-V2 annotations into a scene-tag manifest")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-dict", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    thresholds = SceneTagThresholds()
    if args.config:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        thresholds = SceneTagThresholds(**(config.get("scene_tags") or {}))
    report = build_manifest(
        args.data_root,
        args.data_dict,
        args.output,
        thresholds=thresholds,
        max_frames=args.max_frames,
        progress_every=args.progress_every,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

