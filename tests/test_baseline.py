"""Tests for engine.baseline (synthetic embedder only)."""

import hashlib

import numpy as np
import pytest

from engine.baseline import Baseline, BaselineManager


# -------------------------------------------------------- test doubles ----


class FakeEmbedder:
    """Deterministic hash-of-bytes -> 2048-d vector. Records call order."""

    def __init__(self):
        self.calls: list[list[bytes]] = []

    def embed(self, wavs):
        self.calls.append([np.asarray(w, dtype=np.float32).tobytes() for w in wavs])
        out = np.empty((len(wavs), 2048), dtype=np.float32)
        for i, w in enumerate(wavs):
            digest = hashlib.sha256(np.asarray(w, dtype=np.float32).tobytes()).digest()
            seed = int.from_bytes(digest[:8], "little")
            rng = np.random.default_rng(seed)
            out[i] = rng.standard_normal(2048).astype(np.float32)
        return out


def _windows(n, length=48000, seed=0):
    rng = np.random.default_rng(seed)
    # Distinct per-window content so embeddings differ.
    return [
        (rng.standard_normal(length).astype(np.float32) + i) for i in range(n)
    ]


# --------------------------------------------------- fit / save / load ----


def test_fit_save_load_roundtrip(tmp_path):
    mgr = BaselineManager(root=tmp_path)
    bl = mgr.fit(_windows(28), tag="m1", embedder=FakeEmbedder(), rpm=1450.0)
    mgr.save(bl)
    loaded = mgr.load("m1")

    assert np.array_equal(loaded.train_embeddings, bl.train_embeddings)
    assert np.array_equal(loaded.calib_scores, bl.calib_scores)
    assert np.array_equal(loaded.mel_mean, bl.mel_mean)
    assert np.array_equal(loaded.timestamps, bl.timestamps)
    assert loaded.tag == "m1"
    assert loaded.rpm == 1450.0


def test_load_preserves_none_rpm(tmp_path):
    mgr = BaselineManager(root=tmp_path)
    bl = mgr.fit(_windows(12), tag="norpm", embedder=FakeEmbedder())
    mgr.save(bl)
    assert mgr.load("norpm").rpm is None


def test_exists_and_list_tags(tmp_path):
    mgr = BaselineManager(root=tmp_path)
    assert mgr.list_tags() == []
    assert not mgr.exists("z")
    mgr.save(mgr.fit(_windows(12), tag="b", embedder=FakeEmbedder()))
    mgr.save(mgr.fit(_windows(12), tag="a", embedder=FakeEmbedder()))
    assert mgr.exists("a")
    assert mgr.list_tags() == ["a", "b"]


# ----------------------------------------------------- holdout split ----


def test_holdout_determinism_and_indices():
    # 28 windows -> calib at indices 4,9,14,19,24 (5 calib), 23 train.
    embedder = FakeEmbedder()
    mgr = BaselineManager()
    windows = _windows(28)
    bl = mgr.fit(windows, tag="m", embedder=embedder)

    assert bl.calib_scores.shape[0] == 5
    assert bl.train_embeddings.shape[0] == 23

    # First embed call covers train (23) then holdout (5) in that order.
    first_call = embedder.calls[0]
    assert len(first_call) == 28
    calib_idx = [4, 9, 14, 19, 24]
    train_idx = [i for i in range(28) if i not in calib_idx]
    expected_order = (
        [windows[i].tobytes() for i in train_idx]
        + [windows[i].tobytes() for i in calib_idx]
    )
    assert first_call == expected_order

    # Two fits -> identical baselines (no RNG in the split).
    bl2 = mgr.fit(windows, tag="m", embedder=FakeEmbedder())
    assert np.array_equal(bl.train_embeddings, bl2.train_embeddings)
    assert np.array_equal(bl.calib_scores, bl2.calib_scores)
    assert np.array_equal(bl.timestamps, bl2.timestamps)


def test_default_timestamps_align_with_train_rows():
    bl = BaselineManager().fit(_windows(28), tag="m", embedder=FakeEmbedder())
    train_idx = [i for i in range(28) if i % 5 != 4]
    assert np.array_equal(bl.timestamps, np.array(train_idx, dtype=np.float64))


# --------------------------------------------------------- percentile ----


def test_percentile_mapping():
    bl = Baseline(
        tag="t",
        train_embeddings=np.zeros((2, 2048), dtype=np.float32),
        calib_scores=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64),
        mel_mean=np.zeros(64, dtype=np.float32),
        timestamps=np.zeros(2, dtype=np.float64),
    )
    assert bl.percentile(0.0) == 0.0
    assert bl.percentile(3.0) == 60.0
    assert bl.percentile(10.0) == 100.0


# ------------------------------------------------------- augmentation ----


def test_augmentation_doubles_train_and_preserves_calib():
    embedder_plain = FakeEmbedder()
    mgr = BaselineManager()
    windows = _windows(28)
    plain = mgr.fit(windows, tag="m", embedder=embedder_plain)

    n_calls = {"n": 0}

    def halve(w):
        n_calls["n"] += 1
        return (w * 0.5).astype(np.float32)

    augmented = mgr.fit(
        windows, tag="m", embedder=FakeEmbedder(), augmentations=[halve]
    )

    n_train = plain.train_embeddings.shape[0]  # 23
    assert augmented.train_embeddings.shape[0] == 2 * n_train
    assert augmented.timestamps.shape[0] == 2 * n_train
    # Augmented rows repeat the source window's timestamp.
    assert np.array_equal(
        augmented.timestamps[:n_train], augmented.timestamps[n_train:]
    )
    # Callable invoked exactly once per train window.
    assert n_calls["n"] == n_train
    # Calibration is NEVER augmented -> identical to plain fit.
    assert np.array_equal(augmented.calib_scores, plain.calib_scores)


# ----------------------------------------------------------- guards ----


def test_library_stub_raises_and_is_documented():
    with pytest.raises(NotImplementedError):
        BaselineManager().load_library("fan")
    assert "stub" in (BaselineManager.load_library.__doc__ or "").lower() or (
        "not yet implemented" in (BaselineManager.load_library.__doc__ or "").lower()
    )


def test_bad_tag_rejected():
    with pytest.raises(ValueError):
        BaselineManager().fit(_windows(12), tag="..//evil", embedder=FakeEmbedder())


def test_too_few_windows_rejected():
    with pytest.raises(ValueError):
        BaselineManager().fit(_windows(9), tag="m", embedder=FakeEmbedder())
