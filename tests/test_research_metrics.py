import numpy as np

from scenariotopo.evaluation.research_metrics import (
    EndpointAccumulator,
    MatchedGraphAccumulator,
    matched_graph_metrics,
)


def test_matched_graph_exact_sets_ignore_unmatched_queries():
    target = np.zeros((3, 3), dtype=int)
    target[0, 1] = 1
    scores = np.zeros((3, 3), dtype=float)
    scores[0, 1] = 0.9
    scores[0, 2] = 0.9
    mask = np.array([True, True, False])
    m = matched_graph_metrics(scores, target, mask)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 0, 0)
    assert (m["successor_exact"], m["predecessor_exact"], m["frame_pass"]) == (2, 2, 1)
    acc = MatchedGraphAccumulator()
    acc.update(scores, target, mask, ["Split"])
    assert acc.compute()["Split"]["frame_topology_pass_rate"] == 1.0


def test_overshoot_and_transition_proxy():
    end = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 0.0]])
    gt = np.zeros_like(end)
    tangent = np.tile([1.0, 0.0], (3, 1))
    acc = EndpointAccumulator(overshoot_margin_m=0.5)
    acc.update(end, gt, tangent, [True, True, True], [True, False, False], [],
               endpoint_buckets={"Connector": [True, False, False]})
    m = acc.compute()["Overall"]
    assert np.isclose(m["overshoot_rate"], 1 / 3)
    assert m["transition_penetration_proxy_rate"] == 1.0
    assert m["frame_overextension_pass_rate"] == 0.0
    assert acc.compute()["Endpoint/Connector"]["overshoot_rate"] == 1.0
    assert acc.compute()["Endpoint/Ordinary"]["overshoot_rate"] == 0.0
