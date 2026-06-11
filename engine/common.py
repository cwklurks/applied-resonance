"""Shared utilities for reproducibility and device selection."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 1337) -> None:
    """Seed random, numpy, torch (+cuda all), set PYTHONHASHSEED env,
    torch.use_deterministic_algorithms(True, warn_only=True),
    and set cudnn.deterministic=True / cudnn.benchmark=False."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer: str | None = None) -> torch.device:
    """If prefer is given ('cpu'|'cuda'|'mps'), return that device.
    Otherwise auto-detect: cuda if torch.cuda.is_available(),
    else mps if torch.backends.mps.is_available(), else cpu."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
