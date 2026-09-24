import numpy as np

from scenariotopo.evaluation import SlicedTopologyMetrics, binary_topology_metrics
from scenariotopo.samplers import compute_sample_weights


def test_rare_scene_gets_more_weight():
    tags = [["Normal"], ["Normal"], ["Normal"], ["Split"]]
    weights = compute_sample_weights(tags, alpha=1.0, beta=0.0)
    assert weights[-1] > weights[0]
    assert np.isclose(weights.mean(), 1.0)


def test_sliced_metrics_excludes_self_edges():
    scores = np.asarray([[0.9, 0.9], [0.1, 0.9]])
    targets = np.asarray([[0, 1], [0, 0]])
    metrics = binary_topology_metrics(scores, targets)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 0
    accumulator = SlicedTopologyMetrics()
    accumulator.update(scores, targets, ["Split"])
    assert accumulator.compute()["Split"]["f1"] == 1.0

