"""Tests for engine.labeling.write_labeled_clip (no PANNs, no network)."""

import json

import numpy as np
import pytest
import soundfile as sf

from engine.labeling import write_labeled_clip


def _audio(n=8000, seed=0):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32) * 0.1


# ----------------------------------------------------------- roundtrip ----


def test_roundtrip_writes_both_files(tmp_path):
    audio = _audio(8000, seed=1)
    meta = {"machine_type": "fan", "suspected_fault": "bearing", "note": "hi"}
    wav_path, json_path = write_labeled_clip(audio, 16000, meta, out_dir=tmp_path)

    assert wav_path.exists()
    assert json_path.exists()
    assert wav_path.suffix == ".wav"
    assert json_path.suffix == ".json"
    # Same stem.
    assert wav_path.stem == json_path.stem


def test_wav_loads_back_quantized_equal(tmp_path):
    audio = _audio(8000, seed=2)
    wav_path, _ = write_labeled_clip(audio, 16000, {}, out_dir=tmp_path)

    back, sr = sf.read(str(wav_path), dtype="float32")
    assert sr == 16000
    assert back.shape[0] == audio.shape[0]
    # PCM_16 quantization tolerance (1 LSB = 1/32768).
    assert np.max(np.abs(back - audio)) < 1.5 / 32768.0


def test_json_has_meta_plus_fields(tmp_path):
    audio = _audio(5000, seed=3)
    meta = {"machine_type": "pump", "site_tag": "siteA"}
    _, json_path = write_labeled_clip(audio, 16000, meta, out_dir=tmp_path)

    data = json.loads(json_path.read_text())
    # Original meta preserved.
    assert data["machine_type"] == "pump"
    assert data["site_tag"] == "siteA"
    # Injected fields.
    assert data["sr"] == 16000
    assert data["n_samples"] == 5000
    assert "created_at" in data
    # created_at is an ISO-ish UTC string.
    assert "T" in data["created_at"]


def test_stem_uses_site_tag(tmp_path):
    audio = _audio(3000)
    wav_path, _ = write_labeled_clip(
        audio, 16000, {"site_tag": "plant7"}, out_dir=tmp_path
    )
    assert wav_path.stem.endswith("_plant7")


def test_stem_unlabeled_when_no_site_tag(tmp_path):
    audio = _audio(3000)
    wav_path, _ = write_labeled_clip(audio, 16000, {}, out_dir=tmp_path)
    assert wav_path.stem.endswith("_unlabeled")


# ------------------------------------------------------------ collisions ----


def test_collision_suffix(tmp_path):
    audio = _audio(2000)
    meta = {"site_tag": "x"}
    p1, _ = write_labeled_clip(audio, 16000, meta, out_dir=tmp_path)
    p2, _ = write_labeled_clip(audio, 16000, meta, out_dir=tmp_path)
    p3, _ = write_labeled_clip(audio, 16000, meta, out_dir=tmp_path)

    stems = {p1.stem, p2.stem, p3.stem}
    assert len(stems) == 3  # all distinct
    # Suffixed variants exist.
    assert any(s.endswith("_1") for s in stems)
    assert any(s.endswith("_2") for s in stems)


# --------------------------------------------------------- sanitization ----


def test_sanitizes_unsafe_site_tag(tmp_path):
    audio = _audio(2000)
    wav_path, json_path = write_labeled_clip(
        audio, 16000, {"site_tag": "../evil site"}, out_dir=tmp_path
    )
    # No path traversal: file lands directly under out_dir.
    assert wav_path.parent == tmp_path
    assert json_path.parent == tmp_path
    # Stem chars restricted to the safe alphabet.
    for ch in wav_path.stem:
        assert ch in "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ._-"
    # Slashes and spaces gone.
    assert "/" not in wav_path.stem
    assert " " not in wav_path.stem


def test_empty_after_sanitize_falls_back_to_unlabeled(tmp_path):
    audio = _audio(2000)
    wav_path, _ = write_labeled_clip(
        audio, 16000, {"site_tag": "///"}, out_dir=tmp_path
    )
    assert wav_path.stem.endswith("_unlabeled")


# ----------------------------------------------------------- validation ----


def test_rejects_non_1d_audio(tmp_path):
    bad = np.zeros((10, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        write_labeled_clip(bad, 16000, {}, out_dir=tmp_path)


def test_rejects_empty_audio(tmp_path):
    with pytest.raises(ValueError):
        write_labeled_clip(np.zeros(0, dtype=np.float32), 16000, {}, out_dir=tmp_path)


def test_rejects_bad_sr(tmp_path):
    with pytest.raises(ValueError):
        write_labeled_clip(_audio(1000), 0, {}, out_dir=tmp_path)
