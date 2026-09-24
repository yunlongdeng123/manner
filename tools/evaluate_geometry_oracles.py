#!/usr/bin/env python3
"""CPU-only B0/E1/GT-endpoint topology probes on frozen feature caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scenariotopo.data.manifest import iter_jsonl
from scenariotopo.evaluation.geometry import distance_heading_scores, replace_matched_endpoints
from scenariotopo.evaluation.research_metrics import EndpointAccumulator, MatchedGraphAccumulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distance-scale-m", type=float, default=2.0)
    parser.add_argument("--geometry-weight", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overshoot-margin-m", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.geometry_weight <= 1:
        parser.error("--geometry-weight must be between 0 and 1")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    accumulators = {key: MatchedGraphAccumulator(args.threshold) for key in ("base", "geometry", "base_plus_geometry", "gt_endpoint_oracle")}
    endpoint = EndpointAccumulator(args.overshoot_margin_m)
    frames = 0
    base_frames = 0
    for record in iter_jsonl(args.index):
        if (record.get("model_split") or record.get("source_split")) != args.split:
            continue
        path = Path(record["cache_path"])
        if not path.is_absolute():
            path = args.index.parent / path
        with np.load(path, allow_pickle=False) as data:
            lanes = data["lanes"]
            matched = data["matched_mask"]
            targets = data["query_adjacency"]
            tags = record.get("tags") or []
            geometry = distance_heading_scores(lanes, distance_scale_m=args.distance_scale_m)
            accumulators["geometry"].update(geometry, targets, matched, tags)
            oracle_lanes = replace_matched_endpoints(lanes, data["gt_start"], data["gt_end"], matched)
            oracle = distance_heading_scores(oracle_lanes, distance_scale_m=args.distance_scale_m)
            accumulators["gt_endpoint_oracle"].update(oracle, targets, matched, tags)
            if "base_topology_scores" in data:
                base = data["base_topology_scores"]
                accumulators["base"].update(base, targets, matched, tags)
                blended = (1 - args.geometry_weight) * base + args.geometry_weight * geometry
                accumulators["base_plus_geometry"].update(blended, targets, matched, tags)
                base_frames += 1
            endpoint.update(lanes[:, -1, :], data["gt_end"], data["gt_end_tangent"], matched,
                            data["transition_end_mask"], tags)
        frames += 1
    report = {
        "split": args.split, "frames": frames, "base_frames": base_frames,
        "scope": "GT-matched queries; GT endpoint oracle uses ground truth and is diagnostic only",
        "distance_scale_m": args.distance_scale_m, "geometry_weight": args.geometry_weight,
        "topology": {key: acc.compute() for key, acc in accumulators.items() if key not in ("base", "base_plus_geometry") or base_frames},
        "raw_endpoint": endpoint.compute(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
