"""Feature extraction dispatch."""

import numpy as np
import torch

from engine.features.logmel import logmel
from engine.features.panns import panns_embedding


def extract_features(
    wav: np.ndarray, kind: str, device: torch.device | None = None
) -> np.ndarray:
    """Extract features of the requested kind.

    kind='logmel' -> (64, T); kind='panns' -> (2048,).
    Raises ValueError on any other kind.
    """
    if kind == "logmel":
        return logmel(wav)
    if kind == "panns":
        return panns_embedding(wav, device=device)
    raise ValueError(f"Unknown feature kind: {kind!r} (expected 'logmel' or 'panns')")
