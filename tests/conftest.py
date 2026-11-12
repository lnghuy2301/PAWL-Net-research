import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    torch.set_num_threads(1)


@pytest.fixture
def patch_array():
    data = np.zeros((32, 32, 9), dtype=np.float32)
    data[:, :, 0] = np.linspace(293, 323, 32)[None, :]
    data[:, :, 1:4] = 0.2
    data[:, :, 4] = 0.3
    data[:, :, 5] = 0.98
    data[:, :, 6] = 1000
    data[:, :, 7] = 10
    data[:, :, 8] = 100
    return data


@pytest.fixture
def patch_dir(tmp_path, patch_array):
    directory = tmp_path / "patches"
    directory.mkdir()
    for index in range(10):
        np.save(directory / f"CanTho_2020_m01_y0_x{index:02d}.npy", patch_array)
    return directory
