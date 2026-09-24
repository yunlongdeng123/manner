#!/usr/bin/env python3
"""Summarize a ScenarioTopo JSONL manifest without loading the dataset into RAM."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def describe(values):
    ordered = sorted(values)
    if not ordered:
        return None
    last = len(ordered) - 1
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": ordered[round(0.25 * last)],
        "median": ordered[round(0.5 * last)],
        "p75": ordered[round(0.75 * last)],
        "p90": ordered[round(0.9 * last)],
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 2),
    }


def summarize(path: Path):
    work_splits = Counter()
    source_splits = Counter()
    unannotated = Counter()
    tags = Counter()
    segments = defaultdict(set)
    values = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            frame = json.loads(line)
            split = frame.get("model_split") or frame["source_split"]
            work_splits[split] += 1
            source_splits[frame["source_split"]] += 1
            segments[split].add(frame["segment_id"])
            if not frame.get("has_annotation"):
                unannotated[split] += 1
                continue
            tags.update(frame.get("tags") or [])
            for name in ("lane_count", "edge_count", "max_in_degree", "max_out_degree"):
                values[name].append(frame["stats"][name])
    return {
        "total_frames": sum(work_splits.values()),
        "annotated_frames": sum(work_splits.values()) - sum(unannotated.values()),
        "source_splits": dict(sorted(source_splits.items())),
        "model_splits": dict(sorted(work_splits.items())),
        "unannotated_by_split": dict(sorted(unannotated.items())),
        "segments_by_split": {key: len(value) for key, value in sorted(segments.items())},
        "scene_tag_counts": dict(sorted(tags.items())),
        "annotated_frame_quantiles": {key: describe(value) for key, value in sorted(values.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(summarize(args.manifest), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
