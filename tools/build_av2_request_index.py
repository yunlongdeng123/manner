#!/usr/bin/env python3
"""Build a compact AV2 camera request index from the ScenarioTopo manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import tempfile

from scenariotopo.data.manifest import iter_jsonl


SUBSET_A_CAMERAS = (
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_rear_left",
    "ring_rear_right",
    "ring_side_left",
    "ring_side_right",
)


def build_request_index(manifest: Path, output: Path, splits: set[str]) -> dict[str, object]:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    split_frames: Counter[str] = Counter()
    image_count = 0
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        try:
            for record in iter_jsonl(manifest):
                split = str(record["source_split"])
                if split not in splits:
                    continue
                segment_id = str(record["segment_id"])
                timestamp = str(record["timestamp"])
                source_id = str(record["source_id"])
                split_frames[split] += 1
                for camera in SUBSET_A_CAMERAS:
                    request = {
                        "frame_key": str(record["frame_key"]),
                        "source_split": split,
                        "source_id": source_id,
                        "camera": camera,
                        "filename": f"{timestamp}.jpg",
                        "image_path": f"{split}/{segment_id}/image/{camera}/{timestamp}.jpg",
                    }
                    handle.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
                    image_count += 1
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temp_path, output)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    return {
        "output": str(output),
        "splits": sorted(splits),
        "frames": sum(split_frames.values()),
        "split_frames": dict(sorted(split_frames.items())),
        "images": image_count,
        "cameras_per_frame": len(SUBSET_A_CAMERAS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()
    print(json.dumps(build_request_index(args.manifest, args.output, set(args.splits)), indent=2))


if __name__ == "__main__":
    main()
