"""ONNX backend parity, determinism, and RTF tests.

Parity is the acceptance criterion: the onnxruntime embeddings must match the
torch embeddings on real audio to < 1e-4 max abs difference, both at full
clip length and at the 3 s streaming window shape.
"""

import numpy as np
import pytest

from engine.audio import load_wav
from engine.embedder import TorchPannsEmbedder, get_embedder
from engine.export_onnx import DEFAULT_ARTIFACT, export, rtf_benchmark
from engine.features.panns import CHECKPOINT_PATH
from engine.onnx_embedder import OnnxPannsEmbedder
from engine.paths import MIMII_DIR

_NO_CKPT = not CHECKPOINT_PATH.exists()
_CKPT_REASON = "checkpoint not downloaded"

_MIMII_WAVS = sorted(MIMII_DIR.rglob("*.wav")) if MIMII_DIR.exists() else []
_NO_MIMII = len(_MIMII_WAVS) < 20
_MIMII_REASON = "MIMII data not available"

_PARITY_TOL = 1e-4
_WINDOW_SAMPLES = 48000  # 3 s at 16 kHz


@pytest.fixture(scope="module")
def onnx_model_path():
    """Export the ONNX artifact once (reuse if present; export is slow)."""
    if not DEFAULT_ARTIFACT.exists():
        export(out_path=DEFAULT_ARTIFACT)
    assert DEFAULT_ARTIFACT.exists()
    return DEFAULT_ARTIFACT


def _select_clips(n: int = 20) -> list[np.ndarray]:
    """Deterministic spread of n clips across the MIMII set."""
    step = max(len(_MIMII_WAVS) // n, 1)
    chosen = _MIMII_WAVS[::step][:n]
    return [load_wav(p) for p in chosen]


# --------------------------------------------------------------- parity ------


@pytest.mark.skipif(_NO_CKPT, reason=_CKPT_REASON)
@pytest.mark.skipif(_NO_MIMII, reason=_MIMII_REASON)
def test_parity_full_clip(onnx_model_path):
    wavs = _select_clips(20)
    torch_emb = TorchPannsEmbedder(device="cpu").embed(wavs)
    onnx_emb = OnnxPannsEmbedder(model_path=onnx_model_path).embed(wavs)
    max_delta = np.max(np.abs(torch_emb - onnx_emb))
    print(f"PARITY[full]=max|Δ|={max_delta:.3e}")
    assert torch_emb.shape == onnx_emb.shape == (20, 2048)
    assert max_delta < _PARITY_TOL


@pytest.mark.skipif(_NO_CKPT, reason=_CKPT_REASON)
@pytest.mark.skipif(_NO_MIMII, reason=_MIMII_REASON)
def test_parity_streaming_window(onnx_model_path):
    wavs = [w[:_WINDOW_SAMPLES] for w in _select_clips(20)]
    torch_emb = TorchPannsEmbedder(device="cpu").embed(wavs)
    onnx_emb = OnnxPannsEmbedder(model_path=onnx_model_path).embed(wavs)
    max_delta = np.max(np.abs(torch_emb - onnx_emb))
    print(f"PARITY[window]=max|Δ|={max_delta:.3e}")
    assert torch_emb.shape == onnx_emb.shape == (20, 2048)
    assert max_delta < _PARITY_TOL


# ------------------------------------------------------ wiring + shape -------


@pytest.mark.skipif(_NO_CKPT, reason=_CKPT_REASON)
def test_get_embedder_onnx_returns_instance(onnx_model_path):
    embedder = get_embedder("onnx", device="cpu")
    assert isinstance(embedder, OnnxPannsEmbedder)
    wav = np.random.default_rng(0).standard_normal(_WINDOW_SAMPLES).astype("float32")
    emb = embedder.embed([wav])
    assert emb.shape == (1, 2048)
    assert emb.dtype == np.float32
    assert np.all(np.isfinite(emb))


@pytest.mark.skipif(_NO_CKPT, reason=_CKPT_REASON)
def test_onnx_deterministic(onnx_model_path):
    wav = np.random.default_rng(1).standard_normal(_WINDOW_SAMPLES).astype("float32")
    embedder = OnnxPannsEmbedder(model_path=onnx_model_path)
    first = embedder.embed([wav])
    second = embedder.embed([wav])
    assert np.array_equal(first, second)


# ------------------------------------------------------------------ RTF ------


@pytest.mark.skipif(_NO_CKPT, reason=_CKPT_REASON)
def test_rtf_under_target(onnx_model_path):
    embedder = OnnxPannsEmbedder(model_path=onnx_model_path)
    rtfs = rtf_benchmark({"onnx": embedder}, reps=10)
    rtf = rtfs["onnx"]
    print(f"RTF[onnx]={rtf:.3f}")
    # If this fails on this machine, report DONE_WITH_CONCERNS rather than
    # deleting the assert; documented fallback is hop_s=2.0.
    assert rtf < 0.5
