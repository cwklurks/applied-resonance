"""Tests for engine.evidence.build_evidence (all synthetic, seeded)."""

import json

import numpy as np
import pytest

from engine.evidence import build_evidence

SR = 16000
N_MELS = 64


# --------------------------------------------------------------- stub baseline


class StubBaseline:
    """Minimal duck-typed baseline: only the three read attributes."""

    def __init__(self, mel_mean=None, train_embeddings=None, timestamps=None):
        self.mel_mean = (
            np.zeros(N_MELS, dtype=np.float32) if mel_mean is None else mel_mean
        )
        self.train_embeddings = train_embeddings
        self.timestamps = timestamps


# ----------------------------------------------------------------- synthesis -


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f-shaped noise via rFFT of white noise, modest level."""
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / SR)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    pink = np.fft.irfft(spectrum * scale, n=n)
    pink = pink / (np.max(np.abs(pink)) + 1e-12)
    return 0.05 * pink


def _impulse_train(
    n: int, rate_hz: float, burst_freq: float, rng: np.random.Generator
) -> np.ndarray:
    """Train of 5 ms exponential-decay tone bursts repeating at rate_hz."""
    sig = np.zeros(n, dtype=np.float64)
    period = int(round(SR / rate_hz))
    burst_len = int(0.005 * SR)
    t = np.arange(burst_len) / SR
    burst = np.exp(-t / 0.0015) * np.sin(2 * np.pi * burst_freq * t)
    for start in range(0, n - burst_len, period):
        sig[start : start + burst_len] += burst
    return sig


def _make_signal(rate_hz: float, seed: int = 0) -> np.ndarray:
    """3 s pink noise + a loud 2.5 kHz burst train at rate_hz."""
    rng = np.random.default_rng(seed)
    n = 3 * SR
    noise = _pink_noise(n, rng)
    bursts = _impulse_train(n, rate_hz, burst_freq=2500.0, rng=rng)
    return (noise + 0.8 * bursts).astype(np.float32)


def _peak_labels(result: dict) -> list[str]:
    return [p["label"] for p in result["envelope_peaks"]]


def _peak_near(result: dict, target: float, tol: float):
    for p in result["envelope_peaks"]:
        if abs(p["freq_hz"] - target) <= tol:
            return p
    return None


# --------------------------------------------------------------------- tests -


def test_criterion_impulse_train_surfaced():
    """90 Hz burst train -> an 'impulse train' peak near 90 Hz is returned."""
    sig = _make_signal(90.0, seed=42)
    result = build_evidence(sig, StubBaseline())

    peak = _peak_near(result, 90.0, tol=5.0)
    assert peak is not None, f"no peak near 90 Hz: {result['envelope_peaks']}"
    assert peak["label"].startswith("impulse train ~")


def test_line_hum_120hz_labeled():
    sig = _make_signal(120.0, seed=7)
    result = build_evidence(sig, StubBaseline())

    peak = _peak_near(result, 120.0, tol=5.0)
    assert peak is not None, f"no peak near 120 Hz: {result['envelope_peaks']}"
    assert peak["label"] == "line hum ~120 Hz"


def test_rpm_one_x_rotation_labeled():
    sig = _make_signal(30.0, seed=11)
    result = build_evidence(sig, StubBaseline(), rpm=1800.0)

    peak = _peak_near(result, 30.0, tol=5.0)
    assert peak is not None, f"no peak near 30 Hz: {result['envelope_peaks']}"
    assert peak["label"] == "1x rotation ~30 Hz"


def test_hud_always_within_60_chars():
    # All three constructions plus a pathological extreme-delta case.
    for rate, rpm in [(90.0, None), (120.0, None), (30.0, 1800.0)]:
        result = build_evidence(_make_signal(rate, seed=int(rate)), StubBaseline(), rpm=rpm)
        assert 0 < len(result["hud"]) <= 60

    # Pathological: extreme baseline mean -> huge deltas, long numbers.
    mel_mean = np.full(N_MELS, -1e6, dtype=np.float32)
    result = build_evidence(_make_signal(90.0, seed=3), StubBaseline(mel_mean=mel_mean))
    assert 0 < len(result["hud"]) <= 60


def test_mel_bands_concentrated_noise():
    """Band-limited noise at 1-2 kHz -> top band overlaps 1000-2000 Hz, delta>0.

    The baseline mean is the silent log-mel floor (as a real baseline of
    quiet bands would be), so the 1-2 kHz energy is the dominant deviation
    rather than the always-empty high bands.
    """
    rng = np.random.default_rng(99)
    n = 3 * SR
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / SR)
    mask = (freqs >= 1000.0) & (freqs <= 2000.0)
    spectrum[~mask] = 0.0
    band_noise = np.fft.irfft(spectrum, n=n)
    band_noise = (band_noise / (np.max(np.abs(band_noise)) + 1e-12)).astype(np.float32)

    # Baseline floor near librosa's silent power_to_db floor (~ -98 dB).
    mel_mean = np.full(N_MELS, -98.0, dtype=np.float32)
    result = build_evidence(band_noise, StubBaseline(mel_mean=mel_mean))
    bands = result["mel_bands"]
    assert len(bands) == 3
    for b in bands:
        assert b["lo_hz"] < b["hi_hz"]

    top = bands[0]
    assert top["delta_db"] > 0.0
    # Top band interval overlaps the 1000-2000 Hz energy region.
    assert top["lo_hz"] < 2000.0 and top["hi_hz"] > 1000.0


def test_nearest_baseline_exact_match_and_empty():
    rng = np.random.default_rng(5)
    train = rng.standard_normal((5, 8)).astype(np.float32)
    timestamps = np.arange(5, dtype=np.float64)
    baseline = StubBaseline(train_embeddings=train, timestamps=timestamps)

    sig = _make_signal(90.0, seed=1)
    result = build_evidence(sig, baseline, window_embedding=train[3].copy())
    assert result["nearest_baseline_s"][0] == 3.0
    assert len(result["nearest_baseline_s"]) == 3

    no_emb = build_evidence(sig, baseline)
    assert no_emb["nearest_baseline_s"] == []


def test_result_is_json_serializable():
    rng = np.random.default_rng(8)
    train = rng.standard_normal((5, 8)).astype(np.float32)
    baseline = StubBaseline(
        train_embeddings=train, timestamps=np.arange(5, dtype=np.float64)
    )
    result = build_evidence(
        _make_signal(90.0, seed=2), baseline, window_embedding=train[0].copy(), rpm=1800.0
    )
    dumped = json.dumps(result)
    assert isinstance(dumped, str)


def test_rejects_short_window():
    short = np.zeros(SR // 2, dtype=np.float32)
    with pytest.raises(ValueError):
        build_evidence(short, StubBaseline())


def test_rejects_2d_input():
    bad = np.zeros((2, 3 * SR), dtype=np.float32)
    with pytest.raises(ValueError):
        build_evidence(bad, StubBaseline())
