#!/usr/bin/env python3
"""Restore pinned official sources and recorded local patches; --check writes nothing."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


SOURCES = {
    "TopoGPT": "https://github.com/buaa-colalab/TopoGPT.git",
    "UMPE": "https://github.com/Ethan-Zheng136/UMPE.git",
    "OpenLane-V2": "https://github.com/OpenDriveLab/OpenLane-V2.git",
}


def git(repo, *args, allowed=(0,)):
    command = ["git", "-c", "color.ui=false", "--no-pager"]
    if repo is not None:
        command += ["-C", str(repo)]
    command += list(args)
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    result = subprocess.run(command, capture_output=True, env=env)
    if result.returncode not in allowed:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{' '.join(command)} failed ({result.returncode}): {detail}")
    return result


def tracked_diff(repo):
    return git(repo, "diff", "--binary", "--no-ext-diff", "--no-textconv", "--").stdout


def verify_repository(repo, commit):
    top = Path(git(repo, "rev-parse", "--show-toplevel").stdout.decode().strip())
    if top.resolve() != repo.resolve():
        raise RuntimeError(f"{repo}: expected a repository rooted at this exact directory")
    head = git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    if head != commit:
        raise RuntimeError(f"{repo}: HEAD is {head}; expected {commit}; refusing checkout")
    staged = git(repo, "diff", "--cached", "--quiet", "--", allowed=(0, 1))
    if staged.returncode:
        raise RuntimeError(f"{repo}: staged changes present; refusing to change sources")


def restore_one(root, name, record, check):
    commit = record.get("commit", "")
    digest = record.get("patch_sha256", "")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"{name}: manifest must specify a full commit SHA")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{name}: manifest must specify a patch SHA-256")
    patch = root / "source_artifacts" / f"{name}_local.patch"
    patch_bytes = patch.read_bytes()
    if hashlib.sha256(patch_bytes).hexdigest() != digest:
        raise RuntimeError(f"{patch}: SHA-256 differs from the source manifest")

    repo = root / "repos" / name
    if not repo.exists() and not repo.is_symlink():
        if check:
            raise RuntimeError(f"{repo}: missing repository; run without --check to restore")
        repo.parent.mkdir(parents=True, exist_ok=True)
        print(f"{name}: cloning official source", flush=True)
        git(None, "clone", "--no-checkout", "--", SOURCES[name], str(repo))
        git(repo, "checkout", "--detach", commit)
    if not repo.is_dir():
        raise RuntimeError(f"{repo}: expected a repository directory; refusing to replace it")

    verify_repository(repo, commit)
    actual = tracked_diff(repo)
    if hashlib.sha256(actual).hexdigest() == digest:
        print(f"{name}: verified {commit} and recorded patch", flush=True)
        return
    if actual:
        raise RuntimeError(f"{repo}: tracked changes differ from the recorded patch")
    if check:
        raise RuntimeError(f"{repo}: recorded patch has not been applied")

    # git apply refuses collisions; untracked backup files are left untouched.
    git(repo, "apply", "--check", str(patch))
    git(repo, "apply", str(patch))
    verify_repository(repo, commit)
    if hashlib.sha256(tracked_diff(repo)).hexdigest() != digest:
        raise RuntimeError(f"{repo}: applied changes do not match the recorded patch")
    print(f"{name}: restored {commit} and applied recorded patch", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="project root (default: parent of scripts/)")
    parser.add_argument("--check", action="store_true",
                        help="verify existing repositories without writing or cloning")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = root / "manifests" / "source_archives.json"
    records = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(records, dict):
        raise ValueError(f"{manifest}: expected an object keyed by repository name")
    for name in SOURCES:
        if not isinstance(records.get(name), dict):
            raise ValueError(f"{manifest}: missing or invalid {name} record")
    errors = []
    for name in SOURCES:
        try:
            restore_one(root, name, records[name], args.check)
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(f"{name}: {exc}")
    if errors:
        raise RuntimeError("\n".join(errors))
    print("All pinned source repositories and patches verified.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
