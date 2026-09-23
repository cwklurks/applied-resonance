"""Fully synthetic tests for the re-record kill-test kit.

No MIMII data is touched; every WAV is generated in tmp_path. Fast (< 30 s).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from engine.dataset import Clip
from engine.eval.splits import make_anomaly_splits
from killtest.noise import mix_at_snr, overlay_tree, stable_hash
from killtest.playlist import make_chirp, make_playlist, select_split_test_clips
from killtest.segment import find_offset, segment_recording

SR = 16000
PCM16_ATOL = 2.0 / 32768.0


def _write_burst(path: Path, seed: int, n: int = SR, amp: float = 0.3) -> np.ndarray:
    """Write a seeded random noise burst as a PCM_16 WAV and return what was
    written (after the PCM round-trip), so callers compare against on-disk bytes."""
    rng = np.random.default_rng(seed)
    data = (amp * rng.standard_normal(n)).astype(np.float32)
    sf.write(str(path), data, SR, subtype="PCM_16")
    return sf.read(str(path), dtype="float32")[0]


# --------------------------------------------------------------------------- #
# Playlist
# --------------------------------------------------------------------------- #


def test_make_playlist_length_and_alignment(tmp_path):
    clips = []
    for i in range(3):
        p = tmp_path / f"clip_{i}.wav"
        _write_burst(p, seed=100 + i)
        clips.append(p)

    out_wav = tmp_path / "playlist.wav"
    out_manifest = tmp_path / "manifest.json"
    silence_s, chirp_s = 2.0, 2.0
    manifest = make_playlist(
        clips, out_wav, out_manifest, sr=SR, silence_s=silence_s, chirp_s=chirp_s
    )

    written, sr = sf.read(str(out_wav), dtype="float32")
    assert sr == SR

    chirp_n = int(round(SR * chirp_s))
    silence_n = int(round(SR * silence_s))
    clip_n = SR  # each synthetic clip is 1 s
    # chirp + 4 silences (one after chirp, one after each of 3 clips) + 3 clips.
    expected = chirp_n + 4 * silence_n + 3 * clip_n
    assert len(written) == expected

    gain = manifest["normalization_gain"]
    assert manifest["sr"] == SR
    assert len(manifest["entries"]) == 3

    for i, entry in enumerate(manifest["entries"]):
        start = entry["start_sample"]
        n = entry["n_samples"]
        assert n == clip_n
        original = sf.read(str(clips[i]), dtype="float32")[0]
        recovered = written[start : start + n]
        # The playlist was scaled by `gain`; undo it to compare to the original.
        np.testing.assert_allclose(
            recovered / gain, original, atol=PCM16_ATOL + 1e-3
        )


def test_make_playlist_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        make_playlist([], tmp_path / "a.wav", tmp_path / "m.json")


def _build_split_tree(root: Path) -> list[Clip]:
    clips: list[Clip] = []
    seed = 0
    for machine in ("fan", "pump"):
        for machine_id in ("id_00", "id_02", "id_06"):
            for label, n in (("normal", 5), ("abnormal", 2)):
                for i in range(n):
                    seed += 1
                    path = (
                        root / "0_dB" / machine / machine_id / label / f"{i:08d}.wav"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    _write_burst(path, seed=seed, n=SR // 10)
                    clips.append(Clip(path, machine, machine_id, label))
    return clips


def test_select_split_test_clips_from_split_exact_ids(tmp_path):
    clips = _build_split_tree(tmp_path / "mimii")
    ids = (("fan", "id_06"), ("pump", "id_02"))

    selected = select_split_test_clips(clips, ids=ids, seed=1337)

    expected: list[Path] = []
    for split in make_anomaly_splits(clips, seed=1337):
        if (split.machine, split.machine_id) in set(ids):
            expected.extend(clip.path for clip in split.test_normal)
            expected.extend(clip.path for clip in split.test_abnormal)

    assert selected == expected


# --------------------------------------------------------------------------- #
# find_offset
# --------------------------------------------------------------------------- #


def test_find_offset_known_shift():
    rng = np.random.default_rng(7)
    ref = rng.standard_normal(2000).astype(np.float64)
    recording = np.concatenate([np.zeros(1000), ref, np.zeros(500)])
    lag, peak = find_offset(recording, ref)
    assert lag == 1000
    assert peak > 0.99


# --------------------------------------------------------------------------- #
# Round-trip segmentation
# --------------------------------------------------------------------------- #


def test_segment_roundtrip_alignment(tmp_path):
    clips = []
    for i in range(3):
        p = tmp_path / f"src_{i}.wav"
        _write_burst(p, seed=200 + i)
        clips.append(p)

    out_wav = tmp_path / "playlist.wav"
    out_manifest = tmp_path / "manifest.json"
    make_playlist(clips, out_wav, out_manifest, sr=SR, silence_s=2.0, chirp_s=2.0)

    playlist, _ = sf.read(str(out_wav), dtype="float32")

    # Simulate a re-recording: leading 0.731 s of low-amplitude noise, scaled by
    # 0.5, with seeded broadband noise at -30 dB added on top.
    rng = np.random.default_rng(999)
    lead_n = int(round(0.731 * SR))
    lead = (0.01 * rng.standard_normal(lead_n)).astype(np.float32)
    body = (0.5 * playlist).astype(np.float32)
    recording = np.concatenate([lead, body]).astype(np.float32)
    body_rms = float(np.sqrt(np.mean(body**2)))
    noise_amp = body_rms * (10 ** (-30 / 20))
    recording = recording + (noise_amp * rng.standard_normal(len(recording))).astype(
        np.float32
    )

    rec_path = tmp_path / "recording.wav"
    sf.write(str(rec_path), recording, SR, subtype="PCM_16")

    seg_dir = tmp_path / "segmented"
    segments = segment_recording(rec_path, out_manifest, seg_dir)

    assert len(segments) == 3
    for i, seg in enumerate(segments):
        assert seg["refined"] is True
        assert seg["corr_peak"] > 0.5
        # Reconstructed clip aligns with the original within +/-1 sample.
        original = sf.read(clips[i], dtype="float32")[0]
        recovered = sf.read(seg["out_path"], dtype="float32")[0]
        lag, peak = find_offset(recovered, original)
        assert abs(lag) <= 1
        assert peak > 0.5


def test_segment_chirp_not_found(tmp_path):
    # Build a valid manifest, but feed pure noise as the recording.
    clip = tmp_path / "src.wav"
    _write_burst(clip, seed=5)
    out_manifest = tmp_path / "manifest.json"
    make_playlist([clip], tmp_path / "pl.wav", out_manifest, sr=SR)

    rng = np.random.default_rng(1)
    noise = (0.2 * rng.standard_normal(SR * 8)).astype(np.float32)
    rec = tmp_path / "noise.wav"
    sf.write(str(rec), noise, SR, subtype="PCM_16")

    with pytest.raises(ValueError, match="chirp not found"):
        segment_recording(rec, out_manifest, tmp_path / "out")


# --------------------------------------------------------------------------- #
# Noise overlay
# --------------------------------------------------------------------------- #


def _measure_snr(clean: np.ndarray, mix: np.ndarray) -> float:
    residual = mix - clean
    p_clean = float(np.mean(clean.astype(np.float64) ** 2))
    p_noise = float(np.mean(residual.astype(np.float64) ** 2))
    return 10.0 * np.log10(p_clean / p_noise)


def test_mix_at_snr_targets():
    # Use float64 throughout the measurement so float32 PCM-free rounding of the
    # sum does not pollute the residual at low SNR (0 dB).
    # Low amplitudes so the no-clip path holds even at 0 dB (where added noise
    # has the same RMS as the tone); then residual == scaled noise exactly.
    t = np.arange(SR * 2) / SR
    clean = 0.1 * np.sin(2 * np.pi * 440 * t)
    noise_rng = np.random.default_rng(42)
    noise = 0.5 * noise_rng.standard_normal(SR * 5)

    for target in (0.0, 5.0, 10.0):
        rng = np.random.default_rng(7)
        mix = mix_at_snr(clean, noise, target, rng).astype(np.float64)
        assert np.max(np.abs(mix)) <= 1.0  # confirm no rescale fired
        measured = _measure_snr(clean, mix)
        assert abs(measured - target) < 0.5


def test_mix_at_snr_zero_power_noise():
    clean = (0.3 * np.random.default_rng(0).standard_normal(SR)).astype(np.float32)
    silent = np.zeros(SR, dtype=np.float32)
    with pytest.raises(ValueError):
        mix_at_snr(clean, silent, 5.0, np.random.default_rng(0))


def test_mix_at_snr_zero_power_clean():
    silent = np.zeros(SR, dtype=np.float32)
    noise = (0.3 * np.random.default_rng(0).standard_normal(SR)).astype(np.float32)
    with pytest.raises(ValueError):
        mix_at_snr(silent, noise, 5.0, np.random.default_rng(0))


def test_overlay_tree_determinism(tmp_path):
    clip_dir = tmp_path / "clips"
    clip_dir.mkdir()
    clips = []
    for i in range(2):
        p = clip_dir / f"c_{i}.wav"
        _write_burst(p, seed=300 + i)
        clips.append(p)

    noise_dir = tmp_path / "noise"
    noise_dir.mkdir()
    for i in range(3):
        _write_burst(noise_dir / f"n_{i}.wav", seed=400 + i, n=SR * 4)

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    overlay_tree(clips, noise_dir, out_a, snrs_db=(0.0, 5.0, 10.0), seed=1337)
    overlay_tree(clips, noise_dir, out_b, snrs_db=(0.0, 5.0, 10.0), seed=1337)

    for snr in (0.0, 5.0, 10.0):
        for clip in clips:
            rel = f"snr_{snr:g}dB/{clip.name}"
            a = sf.read(str(out_a / rel), dtype="float32")[0]
            b = sf.read(str(out_b / rel), dtype="float32")[0]
            assert np.array_equal(a, b)


def test_overlay_tree_rejects_non_16k(tmp_path):
    clip_dir = tmp_path / "clips"
    clip_dir.mkdir()
    p = clip_dir / "c.wav"
    _write_burst(p, seed=1)

    noise_dir = tmp_path / "noise"
    noise_dir.mkdir()
    bad = (0.3 * np.random.default_rng(0).standard_normal(8000)).astype(np.float32)
    sf.write(str(noise_dir / "bad.wav"), bad, 8000, subtype="PCM_16")

    with pytest.raises(ValueError):
        overlay_tree([p], noise_dir, tmp_path / "out")


def test_stable_hash_is_deterministic():
    assert stable_hash("fan/id_00/normal/x.wav", 5.0) == stable_hash(
        "fan/id_00/normal/x.wav", 5.0
    )
    assert stable_hash("a.wav", 0.0) != stable_hash("b.wav", 0.0)
    assert stable_hash("a.wav", 0.0) != stable_hash("a.wav", 5.0)


def test_make_chirp_shape_and_fades():
    chirp = make_chirp(sr=SR, duration_s=2.0)
    assert chirp.dtype == np.float32
    assert len(chirp) == SR * 2
    assert abs(chirp[0]) < 1e-6  # fade-in starts at zero
    assert abs(chirp[-1]) < 1e-2  # fade-out ends near zero
    assert np.max(np.abs(chirp)) <= 0.8 + 1e-6
