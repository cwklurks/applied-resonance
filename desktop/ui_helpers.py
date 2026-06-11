"""Pure-numpy helpers for the desktop demo UI.

These are deliberately free of streamlit and torch so they can be unit-tested
without a running server, a microphone, or a model load. The only heavyweight
import (``engine.features.logmel`` -> librosa) is done lazily inside
:func:`spectrogram_image`, so merely importing this module stays cheap and
never pulls torch.
"""

import numpy as np

SR = 16000
N_MELS = 64

# A compact viridis-like LUT (256 entries, RGB uint8). Hand-sampled control
# points linearly interpolated -- "no matplotlib" but still perceptual-ish.
_VIRIDIS_STOPS = np.array(
    [
        [68, 1, 84],
        [72, 40, 120],
        [62, 74, 137],
        [49, 104, 142],
        [38, 130, 142],
        [31, 158, 137],
        [53, 183, 121],
        [109, 205, 89],
        [180, 222, 44],
        [253, 231, 37],
    ],
    dtype=np.float64,
)


def _viridis_lut() -> np.ndarray:
    """Build a (256, 3) uint8 viridis-ish lookup table via linear interp."""
    xs = np.linspace(0.0, 1.0, _VIRIDIS_STOPS.shape[0])
    grid = np.linspace(0.0, 1.0, 256)
    lut = np.stack(
        [np.interp(grid, xs, _VIRIDIS_STOPS[:, c]) for c in range(3)], axis=1
    )
    return lut.round().astype(np.uint8)


_LUT = _viridis_lut()


def pcm_concat(chunks: list[np.ndarray]) -> np.ndarray:
    """Concatenate 1-D float32 chunks into one waveform (empty -> length 0)."""
    if not chunks:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(
        [np.asarray(c, dtype=np.float32).ravel() for c in chunks]
    ).astype(np.float32, copy=False)


def slice_windows(
    chunks: list[np.ndarray], window_s: float = 3.0, hop_s: float = 1.0
) -> list[np.ndarray]:
    """Slice a list of 1 s chunks into ``window_s`` windows hopped ``hop_s``.

    30 one-second chunks -> 28 windows of 48000 samples (3 s window, 1 s hop).
    Returns ``[]`` when the audio is shorter than a single window.
    """
    wav = pcm_concat(chunks)
    window_n = int(round(window_s * SR))
    hop_n = int(round(hop_s * SR))
    if wav.shape[0] < window_n or hop_n <= 0:
        return []
    starts = range(0, wav.shape[0] - window_n + 1, hop_n)
    return [
        np.ascontiguousarray(wav[s : s + window_n], dtype=np.float32) for s in starts
    ]


def spectrogram_image(chunks: list[np.ndarray]) -> np.ndarray:
    """Render the concatenated chunks as a colorized log-mel image.

    Returns a (64, T, 3) uint8 RGB array with low frequencies at the BOTTOM
    (rows flipped) and a viridis-ish colormap applied. When there is no audio
    a flat (64, 1, 3) image is returned so callers never special-case empties.
    """
    wav = pcm_concat(chunks)
    if wav.shape[0] < N_MELS:  # nothing meaningful to transform
        return np.zeros((N_MELS, 1, 3), dtype=np.uint8)

    from engine.features.logmel import logmel  # lazy: librosa only, no torch

    mel = logmel(wav)  # (64, T) float32, dB
    lo, hi = float(mel.min()), float(mel.max())
    span = hi - lo
    if span <= 0:
        norm = np.zeros_like(mel)
    else:
        norm = (mel - lo) / span
    idx = np.clip((norm * 255.0).round().astype(np.int64), 0, 255)
    idx = np.flipud(idx)  # low mel bands -> bottom rows
    return _LUT[idx]  # (64, T, 3) uint8
