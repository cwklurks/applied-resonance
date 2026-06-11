import numpy as np
import pytest

from engine.embedder import TorchPannsEmbedder, get_embedder
from engine.features.panns import CHECKPOINT_PATH

_NO_CKPT = not CHECKPOINT_PATH.exists()
_CKPT_REASON = "checkpoint not downloaded"


# -------------------------------------------------------- get_embedder ------


def test_get_embedder_rejects_bogus_backend():
    with pytest.raises(ValueError):
        get_embedder("bogus")


def test_get_embedder_onnx_raises_runtime_error_naming_module():
    with pytest.raises(RuntimeError, match="engine.onnx_embedder"):
        get_embedder("onnx")


# ----------------------------------------------------- TorchPannsEmbedder ----


@pytest.mark.skipif(_NO_CKPT, reason=_CKPT_REASON)
def test_torch_embedder_shape_and_finite():
    wav = np.random.default_rng(0).standard_normal(48000).astype("float32")
    emb = TorchPannsEmbedder(device="cpu").embed([wav])
    assert emb.shape == (1, 2048)
    assert emb.dtype == np.float32
    assert np.all(np.isfinite(emb))


@pytest.mark.skipif(_NO_CKPT, reason=_CKPT_REASON)
def test_torch_embedder_deterministic():
    wav = np.random.default_rng(1).standard_normal(48000).astype("float32")
    embedder = TorchPannsEmbedder(device="cpu")
    first = embedder.embed([wav])
    second = embedder.embed([wav])
    assert np.array_equal(first, second)


@pytest.mark.skipif(_NO_CKPT, reason=_CKPT_REASON)
def test_get_embedder_torch_roundtrip():
    wav = np.zeros(48000, dtype=np.float32)
    embedder = get_embedder("torch", device="cpu")
    emb = embedder.embed([wav])
    assert emb.shape == (1, 2048)
    assert emb.dtype == np.float32
