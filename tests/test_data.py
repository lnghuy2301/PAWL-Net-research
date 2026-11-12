import numpy as np
import pytest
import torch

from pawl_net.config import DEFAULT_CONFIG, load_config
from pawl_net.dataset import UHIDataset
from pawl_net.masks import STRATEGIES, create_cloud_mask
from pawl_net.normalization import make_input, normalize_image, to_celsius


def test_config_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    expected = load_config(DEFAULT_CONFIG)["paths"]
    monkeypatch.chdir(tmp_path)
    assert load_config()["paths"] == expected
    assert all(path.is_absolute() for path in expected.values())


def test_normalization_preserves_input_and_physical_units(patch_array):
    original = patch_array.copy()
    normalized = normalize_image(patch_array)
    np.testing.assert_array_equal(patch_array, original)
    np.testing.assert_allclose(normalized[0, [0, -1], 0], [0, 1])
    np.testing.assert_allclose(
        to_celsius(normalized[:, :, 0]), original[:, :, 0] - 273.15, atol=2e-5
    )
    np.testing.assert_allclose(normalized[:, :, 4], 0.3)


def test_mask_hides_six_channels_but_retains_spatial_features(patch_array):
    normalized = normalize_image(patch_array)
    mask = np.ones((32, 32), dtype=np.float32)
    mask[5:15, 10:20] = 0
    inputs = make_input(normalized, mask)
    assert inputs.shape == (32, 32, 10)
    np.testing.assert_array_equal(inputs[5:15, 10:20, :6], 0)
    np.testing.assert_array_equal(inputs[:, :, 6:9], normalized[:, :, 6:9])
    np.testing.assert_array_equal(inputs[:, :, 9], mask)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_masks_are_binary_and_repeatable_on_rectangular_images(strategy):
    first = create_cloud_mask(16, 32, rng=np.random.default_rng(8), strategy=strategy)
    second = create_cloud_mask(16, 32, rng=np.random.default_rng(8), strategy=strategy)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (16, 32)
    assert first.dtype == np.float32
    assert set(np.unique(first)) <= {0.0, 1.0}


def test_optional_coverage_is_enforced():
    mask = create_cloud_mask(
        128, 128, strategy="squares", rng=np.random.default_rng(42), cloud_fraction=(0.1, 0.7)
    )
    assert 0.1 <= 1 - mask.mean() <= 0.7


def test_split_and_deterministic_evaluation(patch_dir):
    train = UHIDataset(patch_dir, "train", mask_strategy="squares")
    val = UHIDataset(patch_dir, "val", mask_strategy="squares")
    test = UHIDataset(patch_dir, "test", mask_strategy="squares")
    assert (len(train), len(val), len(test)) == (8, 2, 10)
    assert set(train.files).isdisjoint(val.files)
    assert set(train.files) | set(val.files) == set(test.files)
    assert all(torch.equal(a, b) for a, b in zip(test[2], test[2]))
    assert test[0][0].shape == (10, 32, 32)


@pytest.mark.parametrize("contents", [b"not a numpy array", b""])
def test_bad_patches_are_reported_instead_of_replaced_with_zeros(patch_dir, contents):
    path = sorted(patch_dir.glob("*.npy"))[0]
    path.write_bytes(contents)
    with pytest.raises(ValueError, match="Cannot load patch"):
        UHIDataset(patch_dir, "test")[0]


def test_missing_data_and_invalid_modes_fail_early(tmp_path):
    with pytest.raises(FileNotFoundError):
        UHIDataset(tmp_path)
    with pytest.raises(ValueError, match="mode"):
        UHIDataset(tmp_path, "typo")


def test_nan_population_and_wrong_shape_are_rejected(patch_array):
    bad = patch_array.copy()
    bad[0, 0, 8] = -1
    with pytest.raises(ValueError, match="Population"):
        normalize_image(bad)
    bad[0, 0, 8] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        normalize_image(bad)
    with pytest.raises(ValueError, match="9"):
        normalize_image(patch_array[:, :, :8])
