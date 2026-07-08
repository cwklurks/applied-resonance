from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from killtest.sweep import analyze_transfer, write_sweep

SR = 16000


def test_write_sweep_wav_and_sidecar(tmp_path: Path) -> None:
    out_wav = tmp_path / "sweep.wav"

    params = write_sweep(out_wav, duration_s=1.25)

    data, sr = sf.read(str(out_wav), dtype="float32")
    sidecar = json.loads(Path(f"{out_wav}.json").read_text())
    assert sr == SR
    assert len(data) == int(round(SR * 1.25))
    assert sidecar == params
    assert sidecar["sr"] == SR
    assert sidecar["duration_s"] == 1.25


def test_analyze_transfer_roundtrip_flat_with_leading_silence(tmp_path: Path) -> None:
    reference = tmp_path / "sweep.wav"
    write_sweep(reference)
    sweep, _ = sf.read(str(reference), dtype="float32")
    lead_n = int(round(0.35 * SR))
    recording = np.concatenate([np.zeros(lead_n, dtype=np.float32), sweep])
    recorded = tmp_path / "recorded.wav"
    sf.write(str(recorded), recording, SR, subtype="PCM_16")

    result = analyze_transfer(recorded, reference)

    assert abs(result["align_lag_samples"] - lead_n) <= 2
    gains = np.asarray([band["gain_db"] for band in result["bands"]])
    assert np.max(np.abs(gains)) < 1.0
    assert result["flatness_db"] < 2.0


def test_analyze_transfer_recovers_known_lowpass(tmp_path: Path) -> None:
    reference = tmp_path / "sweep.wav"
    write_sweep(reference)
    sweep, _ = sf.read(str(reference), dtype="float32")
    sos = butter(4, 2000, "lowpass", fs=SR, output="sos")
    filtered = sosfiltfilt(sos, sweep).astype(np.float32)
    recorded = tmp_path / "lowpass.wav"
    sf.write(str(recorded), filtered, SR, subtype="PCM_16")

    result = analyze_transfer(recorded, reference)

    low_bands = [
        band
        for band in result["bands"]
        if np.sqrt(band["lo_hz"] * band["hi_hz"]) < 1000.0
    ]
    high_bands = [band for band in result["bands"] if band["lo_hz"] > 4000.0]
    assert low_bands
    assert high_bands
    assert all(abs(band["gain_db"]) <= 2.0 for band in low_bands)
    assert all(band["gain_db"] < -12.0 for band in high_bands)
    assert result["flatness_db"] > 12.0


def test_analyze_transfer_noise_robust_flat(tmp_path: Path) -> None:
    reference = tmp_path / "sweep.wav"
    write_sweep(reference)
    sweep, _ = sf.read(str(reference), dtype="float32")
    lead_n = int(round(0.35 * SR))
    clean = np.concatenate([np.zeros(lead_n, dtype=np.float32), sweep])
    rng = np.random.default_rng(42)
    sweep_power = float(np.mean(np.square(sweep, dtype=np.float64)))
    noise_power = sweep_power / (10.0 ** (30.0 / 10.0))
    noisy = clean + (
        np.sqrt(noise_power) * rng.standard_normal(clean.size)
    ).astype(np.float32)
    recorded = tmp_path / "noisy.wav"
    sf.write(str(recorded), noisy, SR, subtype="PCM_16")

    result = analyze_transfer(recorded, reference)

    gains = np.asarray([band["gain_db"] for band in result["bands"]])
    assert abs(result["align_lag_samples"] - lead_n) <= 2
    assert np.max(np.abs(gains)) < 2.0


def test_analyze_transfer_rejects_absent_sweep(tmp_path: Path) -> None:
    reference = tmp_path / "sweep.wav"
    write_sweep(reference)
    rng = np.random.default_rng(7)
    noise = (0.2 * rng.standard_normal(SR * 16)).astype(np.float32)
    recorded = tmp_path / "noise.wav"
    sf.write(str(recorded), noise, SR, subtype="PCM_16")

    with pytest.raises(ValueError, match="sweep not found in recording"):
        analyze_transfer(recorded, reference)
