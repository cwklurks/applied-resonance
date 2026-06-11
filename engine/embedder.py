"""Embedder seam: turn waveforms into fixed-size embedding vectors.

The PyTorch PANNs backend is the default. An ONNX backend (faster, no torch
at inference) lands as a parallel work item; ``get_embedder("onnx")`` wires it
in lazily so this module does not depend on it existing yet.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from engine.common import get_device
from engine.features.panns import panns_embeddings


@runtime_checkable
class Embedder(Protocol):
    def embed(self, wavs: Sequence[np.ndarray]) -> np.ndarray:  # (N, 2048) float32
        ...


class TorchPannsEmbedder:
    """PyTorch CNN14_16k backend. device=None -> engine.common.get_device()."""

    def __init__(self, device=None, batch_size: int = 8):
        self.device = get_device(device)
        self.batch_size = batch_size

    def embed(self, wavs: Sequence[np.ndarray]) -> np.ndarray:  # (N, 2048) float32
        return panns_embeddings(
            wavs, device=self.device, batch_size=self.batch_size
        )


def get_embedder(backend: str = "torch", device=None) -> Embedder:
    """backend='torch' -> TorchPannsEmbedder. backend='onnx' -> lazily
    `from engine.onnx_embedder import OnnxPannsEmbedder` (module lands in a
    parallel work item; if the import fails, raise a clear RuntimeError
    naming the missing module). Anything else -> ValueError."""
    if backend == "torch":
        return TorchPannsEmbedder(device=device)
    if backend == "onnx":
        try:
            from engine.onnx_embedder import OnnxPannsEmbedder
        except ImportError as exc:
            raise RuntimeError(
                "ONNX backend unavailable: could not import "
                "engine.onnx_embedder (module not yet implemented)"
            ) from exc
        return OnnxPannsEmbedder(device=device)
    raise ValueError(f"Unknown embedder backend: {backend!r} (expected 'torch' or 'onnx')")
