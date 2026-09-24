#!/usr/bin/env python3
"""Create the official OpenLane-V2 pickle expected by TopoLogic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openlane-source", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--output-stem", required=True, help="File name without .pkl")
    parser.add_argument("--limit", type=int, default=0, help="0 selects every frame")
    args = parser.parse_args()
    root = args.data_root.resolve()
    destination = root / f"{args.output_stem}.pkl"
    if destination.exists():
        raise FileExistsError(destination)
    if args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    data_dict = json.loads((root / "data_dict_subset_A.json").read_text(encoding="utf-8"))
    segments = data_dict[args.split]
    if args.limit:
        remaining = args.limit
        selected = {}
        for segment, timestamps in segments.items():
            selected[segment] = timestamps[:remaining]
            remaining -= len(selected[segment])
            if remaining == 0:
                break
        segments = selected
    sys.path.insert(0, str(args.openlane_source.resolve()))
    from openlanev2.centerline.preprocessing import collect

    collect(str(root), {args.split: segments}, args.output_stem,
            point_interval=1 if args.split == "train" else 20,
            with_sd_map=False)
    print(json.dumps({"output": str(destination), "frames": sum(map(len, segments.values()))}))


if __name__ == "__main__":
    main()
