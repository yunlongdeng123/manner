import numpy as np

from scenariotopo.evaluation.geometry import distance_heading_scores, replace_matched_endpoints


def test_directional_geometry_and_oracle_copy():
    lanes = np.array([
        [[0, 0, 0], [1, 0, 0]],
        [[1, 0, 0], [2, 0, 0]],
        [[2, 0, 0], [1, 0, 0]],
    ], dtype=float)
    scores = distance_heading_scores(lanes)
    assert scores[0, 1] == 1.0
    assert scores[1, 0] < scores[0, 1]
    assert scores[0, 2] == 0.0
    oracle = replace_matched_endpoints(lanes, lanes[:, 0], lanes[:, -1] + 0.1, [True, False, False])
    assert np.allclose(lanes[0, -1], [1, 0, 0])
    assert np.allclose(oracle[0, -1], [1.1, 0.1, 0.1])
