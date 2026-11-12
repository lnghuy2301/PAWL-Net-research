import numpy as np
import pytest
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pawl_net.inference import reconstruct_scene
from pawl_net.losses import AutomaticWeightedLoss, CharbonnierLoss, EdgeLoss
from pawl_net.metrics import ReconstructionMetrics
from pawl_net.models.unet import StandardUNet


@pytest.mark.parametrize("channels", [2, 10])
def test_unet_output_shape_and_range(channels):
    model = StandardUNet(in_channels=channels).eval()
    with torch.inference_mode():
        output = model(torch.rand(1, channels, 32, 32))
    assert output.shape == (1, 1, 32, 32)
    assert torch.all((output >= 0) & (output <= 1))


def test_losses_backpropagate_into_model_and_log_variances():
    prediction = torch.rand(2, 1, 16, 16, requires_grad=True)
    target = torch.rand_like(prediction)
    weighting = AutomaticWeightedLoss()
    loss = weighting(CharbonnierLoss()(prediction, target), EdgeLoss()(prediction, target))
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.isfinite(weighting.params.grad).all()


def test_empty_cloud_loss_is_zero_and_number_of_losses_is_checked():
    prediction = torch.ones(1, 1, 8, 8)
    assert CharbonnierLoss()(prediction, prediction, torch.zeros_like(prediction)) == 0
    with pytest.raises(ValueError, match="Number of losses"):
        AutomaticWeightedLoss()(torch.tensor(1.0))


def test_cloud_only_streaming_metrics_match_sklearn():
    rng = np.random.default_rng(42)
    target = rng.uniform(20, 50, size=(4, 16, 16))
    prediction = target + rng.normal(size=target.shape)
    mask = rng.integers(0, 2, size=target.shape)
    accumulator = ReconstructionMetrics()
    accumulator.update(prediction[:2], target[:2], mask[:2])
    accumulator.update(prediction[2:], target[2:], mask[2:])
    metrics = accumulator.compute()
    truth, pred = target[mask == 0], prediction[mask == 0]
    assert metrics["cloud_pixels"] == len(truth)
    assert metrics["mae_celsius"] == pytest.approx(mean_absolute_error(truth, pred))
    assert metrics["rmse_celsius"] == pytest.approx(np.sqrt(mean_squared_error(truth, pred)))
    assert metrics["r2"] == pytest.approx(r2_score(truth, pred))
    assert metrics["ssim_patches"] == 4


def test_metrics_ignore_observed_errors_and_can_exclude_unknown_truth():
    truth = np.arange(64, dtype=np.float32).reshape(8, 8)
    mask = np.ones((8, 8), dtype=np.float32)
    mask[2:5, 2:5] = 0
    prediction = truth.copy()
    prediction[mask == 1] = 1000
    valid = np.ones((8, 8), dtype=bool)
    valid[2, 2] = False
    prediction[2, 2] = 1000
    metrics = ReconstructionMetrics()
    metrics.update(prediction, truth, mask, valid=valid)
    result = metrics.compute()
    assert result["mae_celsius"] == 0
    assert result["cloud_pixels"] == 8
    assert result["ssim"] is None


def test_empty_evaluation_fails_clearly():
    with pytest.raises(ValueError, match="No cloud pixels"):
        ReconstructionMetrics().compute()


class ConstantModel(torch.nn.Module):
    def forward(self, tensor):
        return torch.full_like(tensor[:, :1], 0.5)


@pytest.mark.parametrize("shape", [(7, 9), (45, 65)])
def test_sliding_window_covers_edges_and_preserves_raw_observations(shape):
    image = np.zeros((*shape, 9), dtype=np.float32)
    image[:, :, 0] = 330  # Observed temperatures outside clipping range must be preserved.
    mask = np.zeros(shape, dtype=np.float32)
    mask[0, 0] = 1
    model = ConstantModel().train()
    result = reconstruct_scene(model, image, mask, patch_size=32, stride=16)
    assert result.shape == shape
    assert model.training
    assert result[0, 0] == pytest.approx(330 - 273.15, abs=1e-5)
    np.testing.assert_allclose(result[mask == 0], 34.85, atol=1e-5)
