#!/usr/bin/env python3
"""Select a topology threshold on training scenes only, then hold it fixed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scenariotopo.data.cached_dataset import CachedOpenLaneDataset
from scenariotopo.models import ScenarioTopoModel


def best_threshold(scores: np.ndarray, targets: np.ndarray, maximum: float) -> dict:
    scores = np.asarray(scores, dtype=np.float32)
    targets = np.asarray(targets, dtype=bool)
    if scores.ndim != 1 or targets.shape != scores.shape or not scores.size:
        raise ValueError("Expected nonempty one-dimensional scores and targets")
    grid = np.round(np.arange(0.05, maximum + 0.001, 0.05), 2)
    positive = int(targets.sum())
    candidates = []
    for threshold in grid:
        predicted = scores >= threshold
        tp = int(np.count_nonzero(predicted & targets))
        fp = int(np.count_nonzero(predicted & ~targets))
        fn = positive - tp
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        candidates.append((f1, float(threshold), tp, fp, fn))
    # A higher threshold wins a tie to avoid unnecessary false connections.
    f1, threshold, tp, fp, fn = max(candidates, key=lambda item: (item[0], item[1]))
    return {"threshold": threshold, "train_f1": f1, "tp": tp, "fp": fp,
            "fn": fn, "positive_edges": positive, "valid_pairs": int(scores.size)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--score-key", choices=("base_topology_scores", "semantic_topology_scores"),
                        default="base_topology_scores")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = CachedOpenLaneDataset(args.index, split="train")
    if not len(dataset):
        raise ValueError("Calibration requires model_split=train frames")
    torch.set_num_threads(1)
    model = None
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        model = ScenarioTopoModel(state["config"]).to(args.device)
        model.load_state_dict(state["model"])
        if model.topology_head is None:
            raise ValueError("Checkpoint has no topology head; calibrate cached score instead")
        model.eval()

    score_parts = []
    target_parts = []
    with torch.no_grad():
        for frame in dataset:
            matched = frame["matched_mask"].numpy()
            valid = matched[:, None] & matched[None, :] & ~np.eye(len(matched), dtype=bool)
            if model is None:
                if args.score_key not in frame:
                    raise ValueError(f"Cache lacks {args.score_key}")
                scores = frame[args.score_key].numpy()
            else:
                output = model(
                    frame["query_features"].unsqueeze(0).to(args.device),
                    frame["lanes"].unsqueeze(0).to(args.device),
                    frame["confidence"].unsqueeze(0).to(args.device),
                )
                scores = output["logits"].sigmoid()[0].cpu().numpy()
            score_parts.append(scores[valid])
            target_parts.append(frame["query_adjacency"].numpy()[valid] > 0)
    result = best_threshold(np.concatenate(score_parts), np.concatenate(target_parts),
                            2.0 if model is None and args.score_key == "base_topology_scores" else 1.0)
    report = {"frames": len(dataset), "source": str(args.checkpoint or args.score_key), **result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
