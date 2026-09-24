#!/usr/bin/env python3
"""Extract only OpenLane-referenced AV2 images from AutoDL Hub tar shards.

The Hub stores Argoverse 2 as uncompressed 50+ GiB tar shards.  This tool walks
tar headers sequentially, seeks over unrelated files, and writes only camera
images referenced by the OpenLane-V2 metadata.  It never expands a full AV2
shard and uses one process.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from typing import DefaultDict, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from scenariotopo.data.manifest import iter_frame_refs, iter_jsonl, load_json


CameraKey = Tuple[str, str, str]


def safe_target(root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe image_path in OpenLane metadata: {relative_path}")
    target = (root / Path(*relative.parts)).resolve()
    if root.resolve() not in target.parents:
        raise ValueError(f"image_path escapes dataset root: {relative_path}")
    return target


def build_requests(
    data_root: Path,
    data_dict: Path,
    splits: Sequence[str],
    *,
    max_frames: int | None = None,
) -> Tuple[DefaultDict[CameraKey, List[Path]], Dict[str, int]]:
    requests: DefaultDict[CameraKey, List[Path]] = defaultdict(list)
    counts = defaultdict(int)
    allowed_splits = set(splits)
    selected_frames = 0
    for ref in iter_frame_refs(data_root, data_dict):
        if ref.source_split not in allowed_splits:
            continue
        if max_frames is not None and selected_frames >= max_frames:
            break
        frame = load_json(ref.path)
        source_id = str((frame.get("meta_data") or {}).get("source_id", ""))
        if not source_id:
            raise ValueError(f"Missing meta_data.source_id: {ref.path}")
        for camera, sensor in (frame.get("sensor") or {}).items():
            image_path = sensor.get("image_path")
            if not image_path:
                continue
            filename = PurePosixPath(str(image_path).replace("\\", "/")).name
            key = (source_id, str(camera), filename)
            target = safe_target(data_root, str(image_path))
            if target not in requests[key]:
                requests[key].append(target)
                counts["images_requested"] += 1
        selected_frames += 1
        counts["frames"] += 1
    counts["unique_source_images"] = len(requests)
    return requests, dict(counts)


def build_requests_from_index(
    data_root: Path,
    request_index: Path,
    splits: Sequence[str],
    *,
    max_frames: int | None = None,
) -> Tuple[DefaultDict[CameraKey, List[Path]], Dict[str, int]]:
    requests: DefaultDict[CameraKey, List[Path]] = defaultdict(list)
    counts = defaultdict(int)
    allowed_splits = set(splits)
    selected_frames: set[str] = set()
    for record in iter_jsonl(request_index):
        if str(record["source_split"]) not in allowed_splits:
            continue
        frame_key = str(record["frame_key"])
        if frame_key not in selected_frames:
            if max_frames is not None and len(selected_frames) >= max_frames:
                continue
            selected_frames.add(frame_key)
            counts["frames"] += 1
        key = (str(record["source_id"]), str(record["camera"]), str(record["filename"]))
        target = safe_target(data_root, str(record["image_path"]))
        if target not in requests[key]:
            requests[key].append(target)
            counts["images_requested"] += 1
    counts["unique_source_images"] = len(requests)
    return requests, dict(counts)


def member_camera_key(name: str) -> CameraKey | None:
    parts = PurePosixPath(name).parts
    # sensor/{train,val}/{source_id}/sensors/cameras/{camera}/{timestamp}.jpg
    try:
        sensors_index = parts.index("sensors")
    except ValueError:
        return None
    if sensors_index < 1 or len(parts) <= sensors_index + 3:
        return None
    if parts[sensors_index + 1] != "cameras":
        return None
    source_id = parts[sensors_index - 1]
    camera = parts[sensors_index + 2]
    filename = parts[-1]
    return source_id, camera, filename


def write_member(archive: tarfile.TarFile, member: tarfile.TarInfo, targets: Sequence[Path]) -> int:
    existing = [target for target in targets if target.is_file() and target.stat().st_size == member.size]
    if len(existing) == len(targets):
        return 0
    first_target = existing[0] if existing else targets[0]
    if not existing:
        first_target.parent.mkdir(parents=True, exist_ok=True)
        extracted = archive.extractfile(member)
        if extracted is None:
            raise OSError(f"Could not read tar member: {member.name}")
        with tempfile.NamedTemporaryFile(dir=first_target.parent, prefix=f".{first_target.name}.", suffix=".part", delete=False) as handle:
            temp_path = Path(handle.name)
            with extracted:
                shutil.copyfileobj(extracted, handle, length=1024 * 1024)
        if temp_path.stat().st_size != member.size:
            temp_path.unlink(missing_ok=True)
            raise OSError(f"Truncated extraction for {member.name}")
        os.replace(temp_path, first_target)
    written = 0 if existing else member.size
    for target in targets:
        if target == first_target or (target.is_file() and target.stat().st_size == member.size):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(first_target, target)
        except OSError:
            shutil.copy2(first_target, target)
        written += member.size
    return written


def scan_archives(
    archives: Sequence[Path],
    requests: MutableMapping[CameraKey, List[Path]],
    *,
    progress_every: int = 1000,
) -> Dict[str, object]:
    extracted_images = 0
    written_bytes = 0
    scanned_archives = []
    for archive_path in archives:
        if not requests:
            break
        matched_in_archive = 0
        with tarfile.open(archive_path, mode="r:") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                key = member_camera_key(member.name)
                if key is None or key not in requests:
                    continue
                targets = requests.pop(key)
                written_bytes += write_member(archive, member, targets)
                extracted_images += len(targets)
                matched_in_archive += len(targets)
                if progress_every and extracted_images % progress_every == 0:
                    print(
                        f"extracted={extracted_images} pending_unique={len(requests)} archive={archive_path.name}",
                        flush=True,
                    )
        scanned_archives.append({"archive": str(archive_path), "matched": matched_in_archive})
        print(
            f"archive_done={archive_path.name} matched={matched_in_archive} pending_unique={len(requests)}",
            flush=True,
        )
    missing_examples = ["/".join(key) for key in list(requests)[:100]]
    return {
        "extracted_images": extracted_images,
        "written_bytes": written_bytes,
        "pending_unique_images": len(requests),
        "missing_examples": missing_examples,
        "archives": scanned_archives,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Extracted OpenLane-V2 annotation root")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-dict", type=Path)
    source.add_argument("--request-index", type=Path)
    parser.add_argument("--hub-root", type=Path, default=Path("/root/autodl-pub/argoverse2.0-sensor"))
    parser.add_argument("--archive-glob", default="val-*.tar")
    parser.add_argument("--splits", nargs="+", default=["val"])
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exit successfully when one selected shard contains only part of the request index",
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    if args.request_index:
        requests, request_report = build_requests_from_index(
            args.data_root.resolve(), args.request_index.resolve(), args.splits, max_frames=args.max_frames
        )
    else:
        requests, request_report = build_requests(
            args.data_root.resolve(), args.data_dict.resolve(), args.splits, max_frames=args.max_frames
        )
    archives = sorted(args.hub_root.resolve().glob(args.archive_glob))
    if not archives:
        raise FileNotFoundError(f"No Hub archives match {args.hub_root / args.archive_glob}")
    report = {
        "source": "AutoDL Hub argoverse2.0-sensor",
        "request": request_report,
        **scan_archives(archives, requests, progress_every=args.progress_every),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    if report["pending_unique_images"] and not args.allow_partial:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
