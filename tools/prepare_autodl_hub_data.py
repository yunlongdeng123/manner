#!/usr/bin/env python3
"""Prepare a user-selected AutoDL public-data path without high concurrency."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import zipfile


MARKER_FILES = ("data_dict_subset_A.json", "data_dict_sample.json")


def discover_dataset_root(source: Path, max_depth: int = 4) -> Path | None:
    source = source.resolve()
    candidates = [source, source / "OpenLane-V2", source / "data" / "OpenLane-V2"]
    for candidate in candidates:
        if any((candidate / marker).is_file() for marker in MARKER_FILES):
            return candidate
    source_depth = len(source.parts)
    for current, directories, files in os.walk(source):
        current_path = Path(current)
        depth = len(current_path.parts) - source_depth
        if depth >= max_depth:
            directories[:] = []
        if any(marker in files for marker in MARKER_FILES):
            return current_path
    return None


def is_annotation_member(name: str) -> bool:
    normalized = PurePosixPath(name.replace("\\", "/"))
    filename = normalized.name
    if filename.startswith("data_dict_") and filename.endswith(".json"):
        return True
    if filename in {"openlanev2.md5", "preprocess.py", "preprocess-ls.py", "sdmap.json"}:
        return True
    return filename.endswith(".json") and "info" in normalized.parts


def safe_destination(target: Path, member_name: str) -> Path:
    relative = PurePosixPath(member_name.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe archive member: {member_name}")
    destination = (target / Path(*relative.parts)).resolve()
    if target.resolve() not in destination.parents and destination != target.resolve():
        raise ValueError(f"Archive member escapes target: {member_name}")
    return destination


def extract_tar_annotations(source: Path, target: Path) -> dict:
    files = 0
    bytes_written = 0
    with tarfile.open(source, mode="r|*") as archive:
        for member in archive:
            if not member.isfile() or not is_annotation_member(member.name):
                continue
            destination = safe_destination(target, member.name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            with extracted, destination.open("wb") as output:
                shutil.copyfileobj(extracted, output, length=1024 * 1024)
            files += 1
            bytes_written += member.size
            if files % 1000 == 0:
                print(f"extracted_files={files}", flush=True)
    return {"files": files, "bytes": bytes_written}


def extract_zip_annotations(source: Path, target: Path) -> dict:
    files = 0
    bytes_written = 0
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            if member.is_dir() or not is_annotation_member(member.filename):
                continue
            destination = safe_destination(target, member.filename)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as extracted, destination.open("wb") as output:
                shutil.copyfileobj(extracted, output, length=1024 * 1024)
            files += 1
            bytes_written += member.file_size
            if files % 1000 == 0:
                print(f"extracted_files={files}", flush=True)
    return {"files": files, "bytes": bytes_written}


def copy_directory_annotations(source: Path, target: Path) -> dict:
    files = 0
    bytes_written = 0
    for current, directories, filenames in os.walk(source):
        current_path = Path(current)
        if "image" in current_path.parts:
            directories[:] = []
            continue
        directories[:] = [name for name in directories if name != "image"]
        for filename in filenames:
            path = current_path / filename
            relative = path.relative_to(source).as_posix()
            if not is_annotation_member(relative):
                continue
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            files += 1
            bytes_written += path.stat().st_size
            if files % 1000 == 0:
                print(f"copied_files={files}", flush=True)
    return {"files": files, "bytes": bytes_written}


def ensure_empty_target(target: Path) -> None:
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty target: {target}")
    target.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Use an AutoDL public-data path as OpenLane-V2 input (single process, streaming I/O)"
    )
    parser.add_argument("--hub-source", type=Path, required=True, help="Path copied from AutoDL public-data page")
    parser.add_argument("--target", type=Path, default=Path("/root/autodl-tmp/datasets/openlanev2"))
    parser.add_argument("--mode", choices=("inspect", "link", "annotations"), default="inspect")
    args = parser.parse_args()
    source = args.hub_source.resolve()
    if not source.exists():
        raise FileNotFoundError(f"AutoDL Hub path does not exist: {source}")

    if args.mode == "inspect":
        dataset_root = discover_dataset_root(source) if source.is_dir() else None
        print(json.dumps({
            "source": str(source),
            "kind": "directory" if source.is_dir() else "archive",
            "size_bytes": source.stat().st_size if source.is_file() else None,
            "detected_dataset_root": str(dataset_root) if dataset_root else None,
        }, indent=2))
        return

    target = args.target.resolve()
    if args.mode == "link":
        if not source.is_dir():
            raise ValueError("link mode requires a directory source")
        dataset_root = discover_dataset_root(source)
        if dataset_root is None:
            raise FileNotFoundError("Could not find data_dict_subset_A.json under the Hub path")
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to replace existing target: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(dataset_root, target_is_directory=True)
        print(json.dumps({"mode": "link", "source": str(dataset_root), "target": str(target)}, indent=2))
        return

    ensure_empty_target(target)
    if source.is_dir():
        dataset_root = discover_dataset_root(source)
        if dataset_root is None:
            raise FileNotFoundError("Could not find an OpenLane-V2 dataset root under the Hub path")
        report = copy_directory_annotations(dataset_root, target)
    elif zipfile.is_zipfile(source):
        report = extract_zip_annotations(source, target)
    elif tarfile.is_tarfile(source):
        report = extract_tar_annotations(source, target)
    else:
        raise ValueError(f"Unsupported Hub resource format: {source}")
    extracted_root = discover_dataset_root(target)
    report.update({
        "mode": "annotations",
        "source": str(source),
        "target": str(target),
        "detected_dataset_root": str(extracted_root) if extracted_root else None,
    })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
