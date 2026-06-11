import time

import numpy as np
import pytest
import soundfile as sf

from shared.audio_source import (
    SR,
    AudioSource,
    FileSource,
    MicSource,
    list_input_devices,
)


def _write_wav(path, wav, sr=SR):
    sf.write(str(path), wav, sr)
    return path


def _ramp(n_samples):
    # Deterministic, full-scale-ish content that survives PCM round-trip.
    return (0.5 * np.sin(np.linspace(0, 50 * np.pi, n_samples))).astype(np.float32)


# ------------------------------------------------------------- FileSource ----


def test_file_source_chunk_lengths_one_second(tmp_path):
    wav = _ramp(int(3.5 * SR))
    path = _write_wav(tmp_path / "a.wav", wav)
    chunks = list(FileSource(path, chunk_s=1.0).chunks())
    lengths = [c.shape[0] for c in chunks]
    assert lengths == [16000, 16000, 16000, 8000]
    for c in chunks:
        assert c.dtype == np.float32
        assert c.ndim == 1


def test_file_source_half_second_yields_seven_chunks(tmp_path):
    wav = _ramp(int(3.5 * SR))
    path = _write_wav(tmp_path / "b.wav", wav)
    chunks = list(FileSource(path, chunk_s=0.5).chunks())
    assert len(chunks) == 7
    assert [c.shape[0] for c in chunks] == [8000] * 7


def test_file_source_content_round_trips(tmp_path):
    wav = _ramp(int(3.5 * SR))
    path = _write_wav(tmp_path / "c.wav", wav)
    chunks = list(FileSource(path, chunk_s=1.0).chunks())
    joined = np.concatenate(chunks)
    assert joined.shape[0] == wav.shape[0]
    # 16-bit PCM round-trip tolerance.
    assert np.allclose(joined, wav, atol=1e-3)


def test_file_source_rejects_nonpositive_chunk_s(tmp_path):
    wav = _ramp(SR)
    path = _write_wav(tmp_path / "d.wav", wav)
    with pytest.raises(ValueError):
        FileSource(path, chunk_s=0)
    with pytest.raises(ValueError):
        FileSource(path, chunk_s=-1.0)


def test_file_source_takes_channel_zero(tmp_path):
    left = _ramp(int(2.0 * SR))
    stereo = np.stack([left, np.zeros_like(left)], axis=1)
    path = _write_wav(tmp_path / "stereo.wav", stereo)
    chunks = list(FileSource(path, chunk_s=1.0).chunks())
    joined = np.concatenate(chunks)
    assert joined.shape[0] == left.shape[0]
    assert np.allclose(joined, left, atol=1e-3)


def test_file_source_rejects_wrong_samplerate(tmp_path):
    wav = _ramp(8000)
    path = _write_wav(tmp_path / "wrong_sr.wav", wav, sr=8000)
    with pytest.raises(ValueError):
        list(FileSource(path, chunk_s=1.0).chunks())


def test_file_source_realtime_false_does_not_sleep(tmp_path):
    wav = _ramp(int(3.5 * SR))
    path = _write_wav(tmp_path / "rt.wav", wav)
    start = time.perf_counter()
    list(FileSource(path, chunk_s=1.0, realtime=False).chunks())
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0


# -------------------------------------------------------------- MicSource ----


def test_mic_source_construction_does_not_open_stream():
    # Bogus device; lazy open means construction must not raise.
    src = MicSource(device="no-such-device-xyz", chunk_s=1.0)
    assert src.device == "no-such-device-xyz"


def test_mic_source_rejects_nonpositive_chunk_s():
    with pytest.raises(ValueError):
        MicSource(chunk_s=0)


def test_list_input_devices_structure(monkeypatch):
    fake = [
        {"name": "Built-in Mic", "max_input_channels": 2, "default_samplerate": 48000.0},
        {"name": "Speakers", "max_input_channels": 0, "default_samplerate": 44100.0},
        {"name": "USB Mic", "max_input_channels": 1, "default_samplerate": 16000.0},
    ]
    import shared.audio_source as mod

    monkeypatch.setattr("sounddevice.query_devices", lambda: fake)
    devices = list_input_devices()
    # Only input-capable devices (channels > 0).
    assert len(devices) == 2
    for d in devices:
        assert set(d.keys()) == {
            "index",
            "name",
            "max_input_channels",
            "default_samplerate",
        }
    assert devices[0]["index"] == 0
    assert devices[1]["index"] == 2  # original positional index preserved


def test_list_input_devices_returns_list_of_dicts():
    # On a real (possibly headless) box: structure only.
    devices = list_input_devices()
    assert isinstance(devices, list)
    for d in devices:
        assert set(d.keys()) == {
            "index",
            "name",
            "max_input_channels",
            "default_samplerate",
        }


# ------------------------------------------------------------ AudioSource ----


def test_audio_source_is_abstract():
    with pytest.raises(TypeError):
        AudioSource()
