"""CPU distance and heading topology probe; not TopoLogic's learned head."""

from __future__ import annotations

import numpy as np


def distance_heading_scores(lanes, *, distance_scale_m=2.0, eps=1e-8):
    lanes = np.asarray(lanes, dtype=np.float64)
    if lanes.ndim != 3 or lanes.shape[1] < 2 or lanes.shape[2] < 2:
        raise ValueError("lanes must be [N,P,D], P>=2, D>=2")
    if distance_scale_m <= 0:
        raise ValueError("distance_scale_m must be positive")
    start, end = lanes[:, 0, :2], lanes[:, -1, :2]
    start_dir = lanes[:, 1, :2] - start
    end_dir = end - lanes[:, -2, :2]
    start_dir /= np.maximum(np.linalg.norm(start_dir, axis=1, keepdims=True), eps)
    end_dir /= np.maximum(np.linalg.norm(end_dir, axis=1, keepdims=True), eps)
    distance = np.linalg.norm(end[:, None, :] - start[None, :, :], axis=-1)
    heading = np.maximum(end_dir @ start_dir.T, 0.0)
    scores = np.exp(-distance / distance_scale_m) * heading
    np.fill_diagonal(scores, 0.0)
    return scores.astype(np.float32)


def replace_matched_endpoints(lanes, gt_start, gt_end, matched_mask):
    lanes = np.array(lanes, copy=True)
    matched = np.asarray(matched_mask, dtype=bool)
    if gt_start.shape != gt_end.shape or gt_start.shape != (len(lanes), lanes.shape[-1]):
        raise ValueError("GT endpoints must have shape [N,D]")
    lanes[matched, 0, :] = gt_start[matched]
    lanes[matched, -1, :] = gt_end[matched]
    return lanes
