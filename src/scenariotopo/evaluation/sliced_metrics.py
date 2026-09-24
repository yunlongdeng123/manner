"""Small cached-topology metrics for per-scene regression checks."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


def binary_topology_metrics(scores, targets, threshold: float = 0.5) -> Dict[str, float]:
    scores = np.asarray(scores)
    targets = np.asarray(targets).astype(bool)
    predictions = scores >= threshold
    if predictions.ndim >= 2 and predictions.shape[-1] == predictions.shape[-2]:
        diagonal = np.eye(predictions.shape[-1], dtype=bool)
        valid = np.broadcast_to(~diagonal, predictions.shape)
        predictions = predictions[valid]
        targets = targets[valid]
    tp = int(np.logical_and(predictions, targets).sum())
    fp = int(np.logical_and(predictions, ~targets).sum())
    fn = int(np.logical_and(~predictions, targets).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


class SlicedTopologyMetrics:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = float(threshold)
        self.counts = defaultdict(lambda: np.zeros(3, dtype=np.int64))

    def update(self, scores, targets, tags: Sequence[str]) -> None:
        metrics = binary_topology_metrics(scores, targets, self.threshold)
        groups = ["Overall", *(tags or ["Normal"])]
        for group in groups:
            self.counts[group] += np.asarray([metrics["tp"], metrics["fp"], metrics["fn"]])

    def compute(self) -> Dict[str, Dict[str, float]]:
        output = {}
        for group in sorted(self.counts):
            tp, fp, fn = (int(value) for value in self.counts[group])
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
            output[group] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        return output
