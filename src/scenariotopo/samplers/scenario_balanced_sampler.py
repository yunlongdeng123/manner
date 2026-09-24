"""Scenario-aware sample weighting without framework-specific data formats."""

from __future__ import annotations

from collections import Counter
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


def compute_sample_weights(
    sample_tags: Sequence[Sequence[str]],
    *,
    recent_losses: Sequence[float] | None = None,
    alpha: float = 0.5,
    beta: float = 0.5,
    min_weight: float = 0.25,
    max_weight: float = 4.0,
) -> np.ndarray:
    """Compute normalized inverse-frequency and difficulty-aware weights.

    Multi-label samples take the largest tag weight so rare structural scenes
    are not diluted by a common co-occurring tag.
    """
    normalized_tags = [tuple(tags) if tags else ("Normal",) for tags in sample_tags]
    counts = Counter(tag for tags in normalized_tags for tag in set(tags))
    median_count = float(np.median(list(counts.values()))) if counts else 1.0
    tag_weights = {
        tag: (median_count / max(count, 1)) ** alpha
        for tag, count in counts.items()
    }
    weights = np.asarray(
        [max(tag_weights[tag] for tag in tags) for tags in normalized_tags],
        dtype=np.float64,
    )

    if recent_losses is not None:
        losses = np.asarray(recent_losses, dtype=np.float64)
        if losses.shape != weights.shape:
            raise ValueError(f"recent_losses shape {losses.shape} does not match {weights.shape}")
        finite = losses[np.isfinite(losses)]
        scale = float(np.median(finite)) if finite.size else 1.0
        scale = max(scale, 1e-8)
        difficulty = np.nan_to_num(losses / scale, nan=1.0, posinf=1.0, neginf=0.0)
        weights *= 1.0 + beta * difficulty

    weights = np.clip(weights, min_weight, max_weight)
    mean = float(weights.mean()) if weights.size else 1.0
    return (weights / max(mean, 1e-8)).astype(np.float32)


class ScenarioBalancedSampler:
    """A deterministic PyTorch sampler with lazy torch import."""

    def __init__(
        self,
        sample_tags: Sequence[Sequence[str]],
        *,
        num_samples: int | None = None,
        replacement: bool = True,
        seed: int = 20260905,
        recent_losses: Sequence[float] | None = None,
        alpha: float = 0.5,
        beta: float = 0.5,
        min_weight: float = 0.25,
        max_weight: float = 4.0,
    ) -> None:
        self.weights = compute_sample_weights(
            sample_tags,
            recent_losses=recent_losses,
            alpha=alpha,
            beta=beta,
            min_weight=min_weight,
            max_weight=max_weight,
        )
        self.num_samples = int(num_samples if num_samples is not None else len(self.weights))
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.epoch = 0
        if not self.replacement and self.num_samples > len(self.weights):
            raise ValueError("num_samples cannot exceed dataset size without replacement")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is required to iterate ScenarioBalancedSampler") from error
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            torch.as_tensor(self.weights, dtype=torch.double),
            self.num_samples,
            self.replacement,
            generator=generator,
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples

