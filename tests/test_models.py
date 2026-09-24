import pytest

torch = pytest.importorskip("torch")

from scenariotopo.losses import (
    connection_consistency_loss,
    endpoint_regression_loss,
    hard_negative_ranking_loss,
    longitudinal_overshoot_loss,
    topology_classification_loss,
)
from scenariotopo.models import EndpointRefiner, HybridTopologyHead


def test_heads_and_losses_backward():
    torch.manual_seed(7)
    batch, queries, points, channels = 2, 5, 4, 16
    features = torch.randn(batch, queries, channels, requires_grad=True)
    lanes = torch.randn(batch, queries, points, 3)
    confidence = torch.sigmoid(torch.randn(batch, queries))
    target = torch.zeros(batch, queries, queries)
    target[:, 0, 1] = 1
    target[:, 1, 2] = 1

    refiner = EndpointRefiner(query_dim=channels, hidden_dim=12)
    refined = refiner(features, lanes)
    assert torch.allclose(refined["refined_lanes"], lanes)
    warp_refiner = EndpointRefiner(query_dim=channels, hidden_dim=12)
    with torch.no_grad():
        warp_refiner.network[-1].bias[3] = 0.5
    warped = warp_refiner(features, lanes)["refined_lanes"]
    assert torch.all(warped[..., -1, 0] > lanes[..., -1, 0])
    assert torch.all(warped[..., -2, 0] > lanes[..., -2, 0])
    assert torch.allclose(warped[..., 0, :], lanes[..., 0, :])

    head = HybridTopologyHead(query_dim=channels, hidden_dim=12, pair_chunk_size=2)
    output = head(features, refined["refined_lanes"], confidence, return_aux=True)
    assert output["logits"].shape == (batch, queries, queries)
    assert output["semantic_gates"].min() >= 0
    assert output["semantic_gates"].max() <= 1

    gt_start = lanes[..., 0, :] + 0.1
    gt_end = lanes[..., -1, :] - 0.1
    loss = topology_classification_loss(output["logits"], target)
    loss = loss + endpoint_regression_loss(
        refined["refined_start"], refined["refined_end"], gt_start, gt_end
    )
    loss = loss + 0.1 * connection_consistency_loss(refined["refined_lanes"], target)
    loss = loss + 0.1 * hard_negative_ranking_loss(
        output["logits"], target, endpoint_distances=output["endpoint_distances"]
    )
    loss = loss + longitudinal_overshoot_loss(
        refined["refined_end"], gt_end, torch.ones_like(gt_end) / 3**0.5,
        matched_mask=torch.ones(batch, queries, dtype=torch.bool),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert features.grad is not None


def test_overshoot_is_directional_and_masked():
    end = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    gt = torch.zeros_like(end)
    tangent = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    mask = torch.tensor([[True, False]])
    assert torch.isclose(longitudinal_overshoot_loss(end, gt, tangent, matched_mask=mask, margin_m=0.2), torch.tensor(0.8))
    assert longitudinal_overshoot_loss(end, gt, tangent, matched_mask=mask, transition_end_mask=torch.zeros_like(mask)) == 0
