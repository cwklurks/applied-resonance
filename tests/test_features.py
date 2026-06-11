import numpy as np
import pytest
import soundfile as sf

from engine.common import get_device
from engine.features import extract_features
from engine.features.logmel import logmel
from engine.features.panns import CHECKPOINT_PATH

SR = 16000


def _sine(duration_s: float, freq: float = 440.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(duration_s * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------- logmel ----


def test_logmel_one_second_sine_shape():
    wav = _sine(1.0)
    expected_T = 1 + 16000 // 512  # 32
    out = logmel(wav)
    assert out.shape == (64, expected_T)
    assert out.dtype == np.float32


def test_logmel_ten_seconds_zeros_shape():
    wav = np.zeros(160000, dtype=np.float32)
    out = logmel(wav)
    assert out.shape == (64, 313)


def test_logmel_deterministic_repeat():
    wav = _sine(1.0)
    first = logmel(wav)
    second = logmel(wav)
    assert np.array_equal(first, second)


def test_logmel_rejects_2d_input():
    wav = np.zeros((2, 16000), dtype=np.float32)
    with pytest.raises(ValueError):
        logmel(wav)


# -------------------------------------------------------- dispatch ----------


def test_extract_features_dispatch_logmel():
    wav = _sine(1.0)
    out = extract_features(wav, "logmel")
    assert out.shape == (64, 32)


def test_extract_features_rejects_bogus_kind():
    wav = _sine(1.0)
    with pytest.raises(ValueError):
        extract_features(wav, "bogus")


# ----------------------------------------------------------- panns ----------

_NO_CKPT = not CHECKPOINT_PATH.exists()
_CKPT_REASON = "checkpoint not downloaded"


@pytest.mark.skipif(_NO_CKPT, reason=_CKPT_REASON)
def test_panns_embedding_shape_and_finite():
    wav = np.random.default_rng(0).standard_normal(160000).astype("float32")
    emb = extract_features(wav, "panns", device=get_device("cpu"))
    assert emb.shape == (2048,)
    assert emb.dtype == np.float32
    assert np.all(np.isfinite(emb))


@pytest.mark.skipif(_NO_CKPT, reason=_CKPT_REASON)
def test_panns_embedding_deterministic():
    wav = np.random.default_rng(1).standard_normal(160000).astype("float32")
    device = get_device("cpu")
    first = extract_features(wav, "panns", device=device)
    second = extract_features(wav, "panns", device=device)
    assert np.array_equal(first, second)


@pytest.mark.skipif(_NO_CKPT, reason=_CKPT_REASON)
def test_panns_batch_matches_single():
    from engine.features.panns import panns_embedding, panns_embeddings

    device = get_device("cpu")
    rng = np.random.default_rng(2)
    wavs = [rng.standard_normal(160000).astype("float32") for _ in range(3)]
    batched = panns_embeddings(wavs, device=device, batch_size=2)
    assert batched.shape == (3, 2048)
    for i, w in enumerate(wavs):
        single = panns_embedding(w, device=device)
        assert np.allclose(batched[i], single, atol=1e-5)


# ----------------------------------------------------------- cache ----------


def test_cached_features_writes_and_reuses(tmp_path, monkeypatch):
    import engine.features.cache as cache_mod

    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(cache_mod, "CACHE_DIR", cache_dir)

    wav = _sine(1.0)
    wav_path = tmp_path / "tone.wav"
    sf.write(str(wav_path), wav, SR)

    first = cache_mod.cached_features(wav_path, "logmel")
    cache_file = cache_mod._cache_path(wav_path, "logmel")
    assert cache_file.exists()

    second = cache_mod.cached_features(wav_path, "logmel")
    assert np.array_equal(first, second)
