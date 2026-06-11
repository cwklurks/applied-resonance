"""Audio I/O helpers."""

from pathlib import Path

import numpy as np
import soundfile as sf


def load_wav(path: str | Path, expected_sr: int = 16000) -> np.ndarray:
    """Read with soundfile as float32. If multichannel, take channel 0.

    Raise ValueError naming the path if samplerate != expected_sr.
    Returns 1-D float32 array.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != expected_sr:
        raise ValueError(
            f"Unexpected samplerate {sr} (expected {expected_sr}) for {path}"
        )
    if data.ndim > 1:
        data = data[:, 0]
    return np.ascontiguousarray(data, dtype=np.float32)
