#!/usr/bin/env python3
"""Create tiny deterministic cache fixtures; never use them as benchmark data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--query-dim", type=int, default=16)
    parser.add_argument("--points", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    records = []
    for frame_index in range(args.frames):
        features = rng.normal(size=(args.queries, args.query_dim)).astype(np.float32)
        lanes = np.zeros((args.queries, args.points, 3), dtype=np.float32)
        for query in range(args.queries):
            x = np.linspace(query * 3.0, query * 3.0 + 12.0, args.points, dtype=np.float32)
            lanes[query, :, 0] = x
            lanes[query, :, 1] = query * 0.4 + 0.2 * np.sin(x / 4.0)
        confidence = rng.uniform(0.5, 1.0, size=args.queries).astype(np.float32)
        adjacency = np.zeros((args.queries, args.queries), dtype=np.float32)
        for query in range(args.queries - 1):
            adjacency[query, query + 1] = 1.0
        if args.queries >= 4:
            adjacency[0, 2] = 1.0
        gt_start = lanes[:, 0, :] + rng.normal(0, 0.1, size=(args.queries, 3)).astype(np.float32)
        gt_end = lanes[:, -1, :] + rng.normal(0, 0.1, size=(args.queries, 3)).astype(np.float32)
        matched_mask = np.ones(args.queries, dtype=np.bool_)
        gt_end_tangent = np.zeros((args.queries, 3), dtype=np.float32)
        gt_end_tangent[:, 0] = 1.0
        transition_end_mask = np.zeros(args.queries, dtype=np.bool_)
        connector_end_mask = np.zeros(args.queries, dtype=np.bool_)
        split_end_mask = np.zeros(args.queries, dtype=np.bool_)
        merge_end_mask = np.zeros(args.queries, dtype=np.bool_)
        if args.queries >= 4:
            transition_end_mask[0] = True
            split_end_mask[0] = True
        cache_name = f"frame-{frame_index:04d}.npz"
        np.savez_compressed(
            args.output / cache_name,
            query_features=features,
            lanes=lanes,
            confidence=confidence,
            query_adjacency=adjacency,
            base_topology_scores=0.1 + 0.8 * adjacency,
            matched_mask=matched_mask,
            gt_start=gt_start,
            gt_end=gt_end,
            gt_end_tangent=gt_end_tangent,
            transition_end_mask=transition_end_mask,
            connector_end_mask=connector_end_mask,
            split_end_mask=split_end_mask,
            merge_end_mask=merge_end_mask,
        )
        tags = ["Split"] if frame_index % 4 == 0 else (["Curve"] if frame_index % 3 == 0 else [])
        records.append({
            "frame_key": f"synthetic/{frame_index}",
            "cache_path": cache_name,
            "source_split": "train" if frame_index < max(1, args.frames - 2) else "val",
            "tags": tags,
            "synthetic": True,
        })
    index_path = args.output / "index.jsonl"
    index_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    print(json.dumps({"index": str(index_path), "frames": len(records), "synthetic": True}, indent=2))


if __name__ == "__main__":
    main()
