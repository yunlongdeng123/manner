#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scenariotopo.data.geosplit import create_geo_split


def main():
    parser = argparse.ArgumentParser(description="Create physical-cluster-disjoint train/val/test splits")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--split-map", type=Path, required=True)
    parser.add_argument("--radius-m", type=float, default=80.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    report = create_geo_split(
        args.manifest,
        args.output_manifest,
        args.split_map,
        radius_m=args.radius_m,
        ratios={"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

