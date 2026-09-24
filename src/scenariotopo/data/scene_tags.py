"""Derive reproducible structure tags from one OpenLane-V2 frame.

The implementation intentionally uses only the Python standard library.  A
frame is processed and released before the next frame is opened, keeping peak
memory nearly independent of dataset size.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


Point = Sequence[float]


@dataclass(frozen=True)
class SceneTagThresholds:
    split_degree: int = 2
    merge_degree: int = 2
    complex_lane_count: int = 20
    complex_edge_count: int = 20
    dense_edge_count: int = 12
    curve_p95_per_meter: float = 0.025
    far_field_m: float = 50.0
    short_lane_m: float = 10.0
    sparse_positive_max_edges: int = 1
    elevation_delta_m: float = 2.5
    overlap_xy_margin_m: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _xyz(point: Point) -> Tuple[float, float, float]:
    x = float(point[0])
    y = float(point[1])
    z = float(point[2]) if len(point) > 2 else 0.0
    return x, y, z


def polyline_length(points: Sequence[Point]) -> float:
    length = 0.0
    for first, second in zip(points, points[1:]):
        ax, ay, az = _xyz(first)
        bx, by, bz = _xyz(second)
        length += math.sqrt((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2)
    return length


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def curvature_p95(points: Sequence[Point]) -> float:
    """Return a robust discrete XY curvature proxy in radians per metre."""
    curvatures: List[float] = []
    for previous, current, following in zip(points, points[1:], points[2:]):
        px, py, _ = _xyz(previous)
        cx, cy, _ = _xyz(current)
        fx, fy, _ = _xyz(following)
        first = (cx - px, cy - py)
        second = (fx - cx, fy - cy)
        first_len = math.hypot(*first)
        second_len = math.hypot(*second)
        local_length = 0.5 * (first_len + second_len)
        if local_length <= 1e-6 or first_len <= 1e-6 or second_len <= 1e-6:
            continue
        cosine = (first[0] * second[0] + first[1] * second[1]) / (first_len * second_len)
        angle = math.acos(max(-1.0, min(1.0, cosine)))
        curvatures.append(angle / local_length)
    return _percentile(curvatures, 0.95)


def _bbox(points: Sequence[Point]) -> Tuple[float, float, float, float]:
    xy = [_xyz(point)[:2] for point in points]
    if not xy:
        return (math.inf, math.inf, -math.inf, -math.inf)
    xs, ys = zip(*xy)
    return min(xs), min(ys), max(xs), max(ys)


def _median_z(points: Sequence[Point]) -> float:
    values = sorted(_xyz(point)[2] for point in points)
    if not values:
        return 0.0
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return 0.5 * (values[middle - 1] + values[middle])


def _boxes_overlap(
    first: Tuple[float, float, float, float],
    second: Tuple[float, float, float, float],
    margin: float,
) -> bool:
    return not (
        first[2] + margin < second[0]
        or second[2] + margin < first[0]
        or first[3] + margin < second[1]
        or second[3] + margin < first[1]
    )


def has_elevation_overlap(
    lanes: Sequence[Sequence[Point]],
    *,
    min_delta_m: float,
    xy_margin_m: float,
) -> bool:
    boxes = [_bbox(lane) for lane in lanes]
    heights = [_median_z(lane) for lane in lanes]
    for i in range(len(lanes)):
        for j in range(i + 1, len(lanes)):
            if abs(heights[i] - heights[j]) < min_delta_m:
                continue
            if _boxes_overlap(boxes[i], boxes[j], xy_margin_m):
                return True
    return False


def _degrees(adjacency: Sequence[Sequence[Any]], lane_count: int) -> Tuple[List[int], List[int], int]:
    out_degree = [0] * lane_count
    in_degree = [0] * lane_count
    edge_count = 0
    for i, row in enumerate(adjacency[:lane_count]):
        for j, value in enumerate(row[:lane_count]):
            if bool(value):
                out_degree[i] += 1
                in_degree[j] += 1
                edge_count += 1
    return out_degree, in_degree, edge_count


def derive_scene_tags(
    frame: Mapping[str, Any],
    thresholds: SceneTagThresholds | None = None,
) -> Dict[str, Any]:
    """Return deterministic scene tags and scalar diagnostics for one frame."""
    thresholds = thresholds or SceneTagThresholds()
    annotation = frame.get("annotation") or {}
    lane_items = annotation.get("lane_centerline") or []
    lanes = [lane.get("points") or [] for lane in lane_items]
    adjacency = annotation.get("topology_lclc") or []
    lane_count = len(lanes)
    out_degree, in_degree, edge_count = _degrees(adjacency, lane_count)

    lengths = [polyline_length(lane) for lane in lanes]
    curve_values = [curvature_p95(lane) for lane in lanes]
    farthest_xy = max(
        (math.hypot(_xyz(point)[0], _xyz(point)[1]) for lane in lanes for point in lane),
        default=0.0,
    )
    maximum_degree = max(out_degree + in_degree, default=0)

    tags: List[str] = []
    if max(out_degree, default=0) >= thresholds.split_degree:
        tags.append("Split")
    if max(in_degree, default=0) >= thresholds.merge_degree:
        tags.append("Merge")
    if (
        lane_count >= thresholds.complex_lane_count
        or edge_count >= thresholds.complex_edge_count
        or maximum_degree >= 3
    ):
        tags.append("ComplexJunction")
    if max(curve_values, default=0.0) >= thresholds.curve_p95_per_meter:
        tags.append("Curve")
    if farthest_xy >= thresholds.far_field_m:
        tags.append("FarField")
    if lengths and min(lengths) <= thresholds.short_lane_m:
        tags.append("ShortLane")
    if edge_count >= thresholds.dense_edge_count:
        tags.append("DenseTopology")
    if edge_count <= thresholds.sparse_positive_max_edges:
        tags.append("SparsePositive")
    if has_elevation_overlap(
        lanes,
        min_delta_m=thresholds.elevation_delta_m,
        xy_margin_m=thresholds.overlap_xy_margin_m,
    ):
        tags.append("ElevationOverlap")

    return {
        "tags": tags,
        "stats": {
            "lane_count": lane_count,
            "edge_count": edge_count,
            "traffic_element_count": len(annotation.get("traffic_element") or []),
            "max_in_degree": max(in_degree, default=0),
            "max_out_degree": max(out_degree, default=0),
            "min_lane_length_m": min(lengths, default=0.0),
            "max_lane_length_m": max(lengths, default=0.0),
            "max_curve_p95_per_meter": max(curve_values, default=0.0),
            "farthest_xy_m": farthest_xy,
        },
    }

