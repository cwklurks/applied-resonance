"""Tests for desktop.ui_helpers (pure numpy; no streamlit, no torch, no mic).

These exercise only the testable helper functions. They must NOT import
streamlit or torch -- the import-time-cleanliness test enforces this in a
subprocess.
"""

import subprocess
import sys

import numpy as np

from desktop.app import _Capture
from desktop.ui_helpers import pcm_concat, slice_windows, spectrogram_image

SR = 16000


def _chunk(seed: int, n: int = SR) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


# --------------------------------------------------------- slice_windows ----


def test_slice_windows_count_and_length():
    chunks = [_chunk(i) for i in range(30)]
    windows = slice_windows(chunks)
    assert len(windows) == 28
    for w in windows:
        assert w.shape == (48000,)
        assert w.dtype == np.float32


def test_slice_windows_content_matches_concat():
    chunks = [_chunk(i) for i in range(30)]
    full = np.concatenate(chunks)
    windows = slice_windows(chunks)
    for k, w in enumerate(windows):
        expected = full[k * SR : k * SR + 48000]
        assert np.array_equal(w, expected)


def test_slice_windows_too_short_returns_empty():
    # Fewer than 3 one-second chunks cannot form a single 3 s window.
    assert slice_windows([_chunk(0), _chunk(1)]) == []
    assert slice_windows([]) == []


def test_slice_windows_custom_geometry():
    # 4 chunks, 2 s window, 1 s hop -> windows at starts 0,1,2 -> 3 windows.
    chunks = [_chunk(i) for i in range(4)]
    windows = slice_windows(chunks, window_s=2.0, hop_s=1.0)
    assert len(windows) == 3
    for w in windows:
        assert w.shape == (32000,)


# ------------------------------------------------------- spectrogram_image ----


def test_spectrogram_image_shape_and_dtype():
    chunks = [_chunk(100 + i) for i in range(10)]
    img = spectrogram_image(chunks)
    assert img.dtype == np.uint8
    assert img.shape[0] == 64  # mel bands on the first axis
    # Grayscale (64, T) or RGB (64, T, 3) are both acceptable.
    assert img.ndim in (2, 3)
    if img.ndim == 3:
        assert img.shape[2] == 3


def test_spectrogram_image_has_contrast():
    chunks = [_chunk(200 + i) for i in range(10)]
    img = spectrogram_image(chunks)
    assert np.unique(img).size > 1  # more than one distinct level


def test_spectrogram_image_empty_is_safe():
    img = spectrogram_image([])
    assert img.dtype == np.uint8
    assert img.shape[0] == 64


# --------------------------------------------------------------- pcm_concat ----


def test_pcm_concat_joins_in_order():
    chunks = [_chunk(i, n=100) for i in range(5)]
    out = pcm_concat(chunks)
    assert out.shape == (500,)
    assert out.dtype == np.float32
    assert np.array_equal(out, np.concatenate(chunks))


def test_pcm_concat_empty():
    out = pcm_concat([])
    assert out.shape == (0,)
    assert out.dtype == np.float32


# ----------------------------------------------------- _Capture ring buffer ----


def _push(cap, n):
    """Simulate the capture thread appending ``n`` chunks (no mic, no thread)."""
    for _ in range(n):
        with cap.lock:
            cap.seq += 1
            cap.buffer.append((cap.seq, _chunk(cap.seq, n=10)))


def test_capture_chunks_since_returns_only_new_in_order():
    cap = _Capture(device=None, backend_label="t")
    _push(cap, 5)
    latest, chunks = cap.chunks_since(2)
    # seqs 3,4,5 are new.
    assert latest == 5
    assert len(chunks) == 3
    # Draining again from the new cursor yields nothing.
    latest2, chunks2 = cap.chunks_since(latest)
    assert chunks2 == []
    assert latest2 == 5  # cursor does not move past what exists


def test_capture_chunks_since_never_skips_buffered_when_backlog_exceeds_ring():
    # Ring holds only the last 30; push 40 so 10 are evicted.
    cap = _Capture(device=None, backend_label="t")
    _push(cap, 40)
    latest, chunks = cap.chunks_since(0)
    # Only the surviving 30 (seqs 11..40) are returned, and the cursor advances
    # exactly to the last returned seq -- NOT to self.seq -- so a subsequent
    # chunk (seq 41) is still picked up rather than skipped.
    assert len(chunks) == 30
    assert latest == 40
    _push(cap, 1)  # seq 41
    latest2, chunks2 = cap.chunks_since(latest)
    assert len(chunks2) == 1
    assert latest2 == 41


def test_capture_snapshot_seq_consistent_with_buffer():
    cap = _Capture(device=None, backend_label="t")
    _push(cap, 7)
    latest, chunks = cap.snapshot()
    assert latest == 7
    assert len(chunks) == 7


# -------------------------------------------------- import-time cleanliness ----


def test_helpers_import_does_not_pull_torch():
    """Importing the helpers must not drag torch into the process."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import desktop.ui_helpers, sys; "
            "assert 'torch' not in sys.modules, sorted(sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
