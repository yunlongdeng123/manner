"""Stream OpenLane-V2 raw annotations into a compact JSONL manifest."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional

from .scene_tags import SceneTagThresholds, derive_scene_tags


@dataclass(frozen=True)
class FrameRef:
    source_split: str
    segment_id: str
    timestamp: str
    path: Path

    @property
    def frame_key(self) -> str:
        return f"{self.source_split}/{self.segment_id}/{self.timestamp}"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error


def iter_frame_refs(data_root: Path, data_dict_path: Path) -> Iterator[FrameRef]:
    data_root = Path(data_root)
    data_dict = load_json(Path(data_dict_path))
    for split in sorted(data_dict):
        segments = data_dict[split]
        for segment_id in sorted(segments):
            for filename in sorted(segments[segment_id]):
                timestamp = str(filename)
                if timestamp.endswith(".json"):
                    timestamp = timestamp[:-5]
                yield FrameRef(
                    source_split=str(split),
                    segment_id=str(segment_id),
                    timestamp=timestamp,
                    path=data_root / str(split) / str(segment_id) / "info" / f"{timestamp}.json",
                )


def _pose_translation(frame: Mapping[str, Any]) -> Optional[list[float]]:
    translation = (frame.get("pose") or {}).get("translation")
    if not isinstance(translation, list) or len(translation) < 2:
        return None
    values = [float(value) for value in translation[:3]]
    while len(values) < 3:
        values.append(0.0)
    return values


def build_manifest(
    data_root: Path,
    data_dict_path: Path,
    output_path: Path,
    *,
    thresholds: SceneTagThresholds | None = None,
    max_frames: int | None = None,
    progress_every: int = 500,
) -> Dict[str, Any]:
    """Build a manifest atomically while holding at most one frame in memory."""
    data_root = Path(data_root).resolve()
    data_dict_path = Path(data_dict_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = thresholds or SceneTagThresholds()
    tag_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    missing_frames = 0
    written = 0

    temp_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(temp_handle.name)
    try:
        with temp_handle:
            for frame_ref in iter_frame_refs(data_root, data_dict_path):
                if max_frames is not None and written >= max_frames:
                    break
                if not frame_ref.path.is_file():
                    missing_frames += 1
                    continue
                frame = load_json(frame_ref.path)
                has_annotation = isinstance(frame.get("annotation"), Mapping)
                derived = derive_scene_tags(frame, thresholds)
                # The official benchmark test split intentionally has no GT.
                # Empty GT must not be mistaken for a sparse-positive scene.
                tags = derived["tags"] if has_annotation else []
                source_id = str((frame.get("meta_data") or {}).get("source_id", "unknown"))
                record = {
                    "frame_key": frame_ref.frame_key,
                    "source_split": frame_ref.source_split,
                    "segment_id": frame_ref.segment_id,
                    "timestamp": frame_ref.timestamp,
                    "source_id": source_id,
                    "relative_info_path": frame_ref.path.relative_to(data_root).as_posix(),
                    "pose_translation": _pose_translation(frame),
                    "has_annotation": has_annotation,
                    "tags": tags,
                    "stats": derived["stats"],
                }
                temp_handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                written += 1
                split_counts[frame_ref.source_split] += 1
                tag_counts.update(tags)
                if progress_every and written % progress_every == 0:
                    print(f"processed={written} missing={missing_frames}", flush=True)
        os.replace(temp_path, output_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise

    return {
        "output": str(output_path),
        "frames": written,
        "missing_frames": missing_frames,
        "source_splits": dict(sorted(split_counts.items())),
        "tag_counts": dict(sorted(tag_counts.items())),
        "thresholds": thresholds.to_dict(),
    }
