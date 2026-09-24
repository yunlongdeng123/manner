#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from scenariotopo.data.cached_dataset import CachedOpenLaneDataset, collate_cached_frames
from scenariotopo.losses import (
    connection_consistency_loss,
    endpoint_regression_loss,
    hard_negative_ranking_loss,
    longitudinal_overshoot_loss,
    topology_classification_loss,
)
from scenariotopo.models import ScenarioTopoModel
from scenariotopo.samplers import ScenarioBalancedSampler


def parse_args():
    parser = argparse.ArgumentParser(description="Train small ScenarioTopo heads on frozen feature caches")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    seed = int(config.get("seed", 20260905))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    dataset = CachedOpenLaneDataset(args.index, split="train")
    if len(dataset) == 0:
        raise ValueError("No training frames found in cache index")
    sampler_config = config.get("sampler") or {}
    sampler = None
    if sampler_config.get("scenario_balanced", False):
        sampler = ScenarioBalancedSampler(
            dataset.sample_tags,
            seed=seed,
            alpha=float(sampler_config.get("alpha", 0.5)),
            beta=float(sampler_config.get("beta", 0.5)),
            min_weight=float(sampler_config.get("min_weight", 0.25)),
            max_weight=float(sampler_config.get("max_weight", 4.0)),
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=0,
        collate_fn=collate_cached_frames,
        drop_last=False,
    )
    model = ScenarioTopoModel(config).to(args.device)
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("This baseline has no trainable head; evaluate cached base_topology_scores directly")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_weights = config.get("loss_weights") or {}
    hard_negative = config.get("hard_negative") or {}
    overshoot = config.get("overshoot") or {}
    modules = config.get("modules") or {}
    history = []
    global_step = 0

    model.train()
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            tensors = {key: value.to(args.device) for key, value in batch.items() if torch.is_tensor(value)}
            output = model(tensors["query_features"], tensors["lanes"], tensors["confidence"])
            matched = tensors["matched_mask"]
            valid_pairs = matched.unsqueeze(2) & matched.unsqueeze(1)
            losses = {}
            if model.topology_head is not None:
                losses["relation"] = topology_classification_loss(
                    output["logits"], tensors["query_adjacency"], valid_mask=valid_pairs
                )
            if model.endpoint_refiner is not None:
                losses["endpoint"] = endpoint_regression_loss(
                    output["refined_start"],
                    output["refined_end"],
                    tensors["gt_start"],
                    tensors["gt_end"],
                    matched_mask=tensors["matched_mask"],
                )
                losses["connection"] = connection_consistency_loss(
                    output["refined_lanes"], tensors["query_adjacency"], valid_mask=valid_pairs
                )
                losses["overshoot"] = longitudinal_overshoot_loss(
                    output["refined_end"], tensors["gt_end"], tensors["gt_end_tangent"],
                    matched_mask=matched, margin_m=float(overshoot.get("margin_m", 0.5)),
                )
                losses["transition"] = longitudinal_overshoot_loss(
                    output["refined_end"], tensors["gt_end"], tensors["gt_end_tangent"],
                    matched_mask=matched, transition_end_mask=tensors["transition_end_mask"],
                    margin_m=float(overshoot.get("transition_margin_m", 0.0)),
                )
            if modules.get("hard_negative_ranking", False) and model.topology_head is not None:
                losses["ranking"] = hard_negative_ranking_loss(
                    output["logits"],
                    tensors["query_adjacency"],
                    endpoint_distances=output["endpoint_distances"],
                    valid_mask=valid_pairs,
                    margin=float(hard_negative.get("margin", 0.2)),
                    negatives_per_source=int(hard_negative.get("negatives_per_source", 8)),
                )
            total = sum(float(loss_weights.get(name, 1.0)) * value for name, value in losses.items())
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            global_step += 1
            scalar = {name: float(value.detach().cpu()) for name, value in losses.items()}
            scalar.update({"total": float(total.detach().cpu()), "epoch": epoch, "step": global_step})
            history.append(scalar)
            print(json.dumps(scalar), flush=True)
            if args.max_steps is not None and global_step >= args.max_steps:
                break
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config, "steps": global_step}, args.output)
    args.output.with_suffix(".history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.output), "steps": global_step}, indent=2))


if __name__ == "__main__":
    main()
