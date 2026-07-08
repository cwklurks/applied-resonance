"""Generate log sweeps and estimate a re-recorded transfer function."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import csd, welch

from engine.audio import load_wav
from killtest.playlist import make_chirp
from killtest.segment import find_offset

logger = logging.getLogger("killtest.sweep")

_SR = 16000
_F0 = 20.0
_F1 = 7800.0
_NPERSEG = 4096


def make_sweep(
    sr: int = _SR,
    duration_s: float = 15.0,
    f0: float = _F0,
    f1: float = _F1,
    amplitude: float = 0.8,
) -> np.ndarray:
    """Return a Farina log sine sweep using the kill-test chirp generator."""
    return make_chirp(
        sr=sr,
        duration_s=duration_s,
        f0=f0,
        f1=f1,
        amplitude=amplitude,
    )


def write_sweep(
    out_wav: Path,
    sr: int = _SR,
    duration_s: float = 15.0,
    f0: float = _F0,
    f1: float = _F1,
    amplitude: float = 0.8,
) -> dict:
    """Write a PCM_16 sweep WAV and a ``{out_wav}.json`` parameter sidecar."""
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    sweep = make_sweep(
        sr=sr,
        duration_s=duration_s,
        f0=f0,
        f1=f1,
        amplitude=amplitude,
    )
    sf.write(str(out_wav), sweep, sr, subtype="PCM_16")

    params = {
        "out_wav": str(out_wav),
        "sr": int(sr),
        "duration_s": float(duration_s),
        "f0": float(f0),
        "f1": float(f1),
        "amplitude": float(amplitude),
        "n_samples": int(len(sweep)),
    }
    sidecar = Path(f"{out_wav}.json")
    sidecar.write_text(json.dumps(params, indent=2))
    logger.info("wrote sweep -> %s", out_wav)
    return params


def analyze_transfer(
    recorded_wav: Path,
    reference_wav: Path,
    n_bands: int = 24,
) -> dict:
    """Estimate normalized transfer-function band gains from a recorded sweep."""
    if n_bands <= 0:
        raise ValueError(f"n_bands must be positive, got {n_bands}")

    recorded = load_wav(recorded_wav, expected_sr=_SR)
    reference = load_wav(reference_wav, expected_sr=_SR)

    lag, peak = find_offset(recorded, reference)
    if peak <= 0.3:
        raise ValueError("sweep not found in recording")

    aligned = _cut_aligned(recorded, lag, len(reference))
    freqs, pxy = csd(
        reference,
        aligned,
        fs=_SR,
        nperseg=min(_NPERSEG, len(reference)),
    )
    _, pxx = welch(
        reference,
        fs=_SR,
        nperseg=min(_NPERSEG, len(reference)),
    )

    valid = pxx > np.finfo(np.float64).tiny
    gain_db = np.full(freqs.shape, np.nan, dtype=np.float64)
    gain_db[valid] = 20.0 * np.log10(np.maximum(np.abs(pxy[valid] / pxx[valid]), 1e-12))

    bands = _band_gains(freqs, gain_db, n_bands=n_bands, f0=_F0, f1=_F1)
    gains = np.asarray([band["gain_db"] for band in bands], dtype=np.float64)
    median_gain = float(np.median(gains))
    for band in bands:
        band["gain_db"] = float(band["gain_db"] - median_gain)

    normalized = np.asarray([band["gain_db"] for band in bands], dtype=np.float64)
    flatness = float(np.max(normalized) - np.min(normalized))
    return {
        "bands": bands,
        "align_lag_samples": int(lag),
        "align_corr": float(peak),
        "flatness_db": flatness,
    }


def _cut_aligned(recorded: np.ndarray, lag: int, n_samples: int) -> np.ndarray:
    aligned = np.zeros(n_samples, dtype=np.float32)
    src_start = max(int(lag), 0)
    dst_start = max(-int(lag), 0)
    n = min(n_samples - dst_start, len(recorded) - src_start)
    if n > 0:
        aligned[dst_start : dst_start + n] = recorded[src_start : src_start + n]
    return aligned


def _band_gains(
    freqs: np.ndarray,
    gain_db: np.ndarray,
    n_bands: int,
    f0: float,
    f1: float,
) -> list[dict]:
    edges = np.geomspace(f0, f1, n_bands + 1)
    bands: list[dict] = []
    for index in range(n_bands):
        lo = float(edges[index])
        hi = float(edges[index + 1])
        if index == n_bands - 1:
            mask = (freqs >= lo) & (freqs <= hi)
        else:
            mask = (freqs >= lo) & (freqs < hi)
        values = gain_db[mask & np.isfinite(gain_db)]
        mean_gain = float(np.mean(values)) if values.size else float("nan")
        bands.append(
            {
                "lo_hz": lo,
                "hi_hz": hi,
                "gain_db": mean_gain,
            }
        )

    if any(not np.isfinite(band["gain_db"]) for band in bands):
        raise ValueError("not enough spectral data to fill requested bands")
    return bands


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a log sweep or analyze a re-recorded transfer function."
    )
    parser.add_argument("--out", type=str, default="killtest_out/sweep.wav")
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--sr", type=int, default=_SR)
    parser.add_argument("--f0", type=float, default=_F0)
    parser.add_argument("--f1", type=float, default=_F1)
    parser.add_argument("--amplitude", type=float, default=0.8)
    parser.add_argument("--analyze", type=str, default=None)
    parser.add_argument("--reference", type=str, default=None)
    parser.add_argument("--out-json", type=str, default=None)
    parser.add_argument("--n-bands", type=int, default=24)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)

    if args.analyze:
        if not args.reference:
            raise ValueError("--reference is required with --analyze")
        result = analyze_transfer(
            recorded_wav=Path(args.analyze),
            reference_wav=Path(args.reference),
            n_bands=args.n_bands,
        )
        out_json = (
            Path(args.out_json)
            if args.out_json
            else Path(args.analyze).with_name(f"{Path(args.analyze).stem}_tf.json")
        )
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, indent=2))
        print(f"aligned lag: {result['align_lag_samples']} samples")
        print(f"align corr: {result['align_corr']:.3f}")
        print("band_hz,gain_db")
        for band in result["bands"]:
            print(f"{band['lo_hz']:7.1f}-{band['hi_hz']:7.1f},{band['gain_db']:7.2f}")
        print(f"wrote {out_json}")
        return

    params = write_sweep(
        out_wav=Path(args.out),
        sr=args.sr,
        duration_s=args.duration,
        f0=args.f0,
        f1=args.f1,
        amplitude=args.amplitude,
    )
    print(params["out_wav"])


if __name__ == "__main__":
    main()
