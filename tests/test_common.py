import numpy as np
import torch

from engine.common import get_device, seed_everything


def test_seed_everything_is_reproducible():
    seed_everything(0)
    torch_first = torch.rand(3)
    numpy_first = np.random.rand(3)

    seed_everything(0)
    torch_second = torch.rand(3)
    numpy_second = np.random.rand(3)

    assert torch.equal(torch_first, torch_second)
    assert np.array_equal(numpy_first, numpy_second)


def test_get_device_respects_cpu_preference():
    assert get_device("cpu") == torch.device("cpu")
