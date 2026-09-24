"""Diagnostic metrics on GT-matched queries, not official TOP_ll scores."""

from __future__ import annotations

from collections import defaultdict
import numpy as np


def matched_graph_metrics(scores, targets, matched_mask, threshold=0.5):
    scores, targets, matched = np.asarray(scores), np.asarray(targets, dtype=bool), np.asarray(matched_mask, dtype=bool)
    if scores.shape != targets.shape or scores.shape != (len(matched), len(matched)):
        raise ValueError("Expected scores/targets [N,N] and matched_mask [N]")
    valid = matched[:, None] & matched[None, :] & ~np.eye(len(matched), dtype=bool)
    pred = scores >= threshold
    tp = int(np.count_nonzero(pred & targets & valid))
    fp = int(np.count_nonzero(pred & ~targets & valid))
    fn = int(np.count_nonzero(~pred & targets & valid))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    nodes = int(matched.sum())
    exact = (pred == targets) | ~valid
    return {
        "tp": tp, "fp": fp, "fn": fn, "matched_nodes": nodes,
        "successor_exact": int(np.count_nonzero(exact.all(axis=1) & matched)),
        "predecessor_exact": int(np.count_nonzero(exact.all(axis=0) & matched)),
        "frame_pass": int(nodes > 0 and bool(exact.all())),
        "evaluable_frame": int(nodes > 0),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


class MatchedGraphAccumulator:
    def __init__(self, threshold=0.5):
        self.threshold = float(threshold)
        self.counts = defaultdict(lambda: np.zeros(8, dtype=np.int64))

    def update(self, scores, targets, matched_mask, tags):
        metrics = matched_graph_metrics(scores, targets, matched_mask, self.threshold)
        values = np.asarray([metrics[key] for key in (
            "tp", "fp", "fn", "matched_nodes", "successor_exact", "predecessor_exact",
            "frame_pass", "evaluable_frame",
        )])
        for group in ("Overall", *(tags or ["Normal"])):
            self.counts[group] += values

    def compute(self):
        result = {}
        for group, values in sorted(self.counts.items()):
            tp, fp, fn, nodes, successor, predecessor, passing, frames = map(int, values)
            precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            result[group] = {
                "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
                "f1": 2 * precision * recall / max(precision + recall, 1e-12),
                "matched_nodes": nodes, "evaluable_frames": frames,
                "successor_exact_accuracy": successor / max(nodes, 1),
                "predecessor_exact_accuracy": predecessor / max(nodes, 1),
                "frame_topology_pass_rate": passing / max(frames, 1),
            }
        return result


class EndpointAccumulator:
    def __init__(self, overshoot_margin_m=0.5):
        if overshoot_margin_m < 0:
            raise ValueError("overshoot_margin_m must be nonnegative")
        self.margin = float(overshoot_margin_m)
        self.stats = defaultdict(lambda: np.zeros(8, dtype=np.float64))

    def update(self, predicted_end, gt_end, gt_end_tangent, matched_mask, transition_end_mask, tags):
        predicted_end, gt_end, tangent = map(np.asarray, (predicted_end, gt_end, gt_end_tangent))
        matched = np.asarray(matched_mask, dtype=bool)
        transition = np.asarray(transition_end_mask, dtype=bool) & matched
        if predicted_end.shape != gt_end.shape or tangent.shape != gt_end.shape:
            raise ValueError("Endpoint arrays must share [N,D]")
        if matched.shape != predicted_end.shape[:1] or transition.shape != matched.shape:
            raise ValueError("Endpoint masks must be [N]")
        error = np.linalg.norm(predicted_end - gt_end, axis=-1)
        forward = np.sum((predicted_end - gt_end) * tangent, axis=-1)
        over = forward > self.margin
        values = np.asarray([
            matched.sum(), error[matched].sum(), np.maximum(forward[matched], 0).sum(),
            np.count_nonzero(over & matched), transition.sum(), np.count_nonzero(over & transition),
            int(bool(matched.any()) and not bool(np.any(over & matched))), int(bool(matched.any())),
        ], dtype=np.float64)
        for group in ("Overall", *(tags or ["Normal"])):
            self.stats[group] += values

    def compute(self):
        result = {}
        for group, values in sorted(self.stats.items()):
            nodes, error, forward, overs, boundaries, penetrations, passes, frames = values
            result[group] = {
                "matched_endpoints": int(nodes), "evaluable_frames": int(frames),
                "mean_end_error_m": float(error / max(nodes, 1)),
                "mean_positive_longitudinal_error_m": float(forward / max(nodes, 1)),
                "overshoot_rate": float(overs / max(nodes, 1)),
                "transition_boundaries": int(boundaries),
                "transition_penetration_proxy_rate": float(penetrations / max(boundaries, 1)),
                "frame_overextension_pass_rate": float(passes / max(frames, 1)),
            }
        return result
