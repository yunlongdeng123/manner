#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from scenariotopo.data.cached_dataset import CachedOpenLaneDataset, collate_cached_frames
from scenariotopo.evaluation.research_metrics import EndpointAccumulator, MatchedGraphAccumulator
from scenariotopo.models import ScenarioTopoModel


def main():
    parser = argparse.ArgumentParser(description="Evaluate cached topology by scene slices")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path, help="Required without --checkpoint; use baseline.yaml for raw scores")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overshoot-margin-m", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        config = checkpoint["config"]
    elif args.config:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    else:
        parser.error("provide --checkpoint or --config")
    model = ScenarioTopoModel(config).to(args.device)
    if args.checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif any(parameter.requires_grad for parameter in model.parameters()):
        parser.error("an untrained head requires --checkpoint")
    model.eval()
    dataset = CachedOpenLaneDataset(args.index, split=args.split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_cached_frames)
    metrics = MatchedGraphAccumulator(args.threshold)
    endpoints = EndpointAccumulator(args.overshoot_margin_m)
    with torch.no_grad():
        for batch in loader:
            tensors = {key: value.to(args.device) for key, value in batch.items() if torch.is_tensor(value)}
            output = model(tensors["query_features"], tensors["lanes"], tensors["confidence"])
            if model.topology_head is None:
                if "base_topology_scores" not in tensors:
                    raise ValueError("This cache has no base_topology_scores for baseline evaluation")
                scores = tensors["base_topology_scores"].cpu().numpy()
            else:
                scores = torch.sigmoid(output["logits"]).cpu().numpy()
            targets = tensors["query_adjacency"].cpu().numpy()
            metrics.update(scores[0], targets[0], tensors["matched_mask"][0].cpu().numpy(), batch["tags"][0])
            endpoints.update(
                output["refined_end"][0].cpu().numpy(), tensors["gt_end"][0].cpu().numpy(),
                tensors["gt_end_tangent"][0].cpu().numpy(), tensors["matched_mask"][0].cpu().numpy(),
                tensors["transition_end_mask"][0].cpu().numpy(), batch["tags"][0],
            )
    report = {
        "split": args.split,
        "frames": len(dataset),
        "scope": "GT-matched queries only; not official OpenLane-V2 TOP_ll or full detector frame pass",
        "topology": metrics.compute(),
        "endpoint": endpoints.compute(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
