"""Cut a single re-recorded WAV back into aligned per-clip files.

The re-recording starts with the playlist's sync chirp. We find the chirp in the
recording by cross-correlation, which fixes the global offset (playback latency,
leading silence the user captured, etc.). Each manifest clip then sits at a known
nominal position; we refine that per-clip by cross-correlating the original clip
against a short window of the recording, so even small clock drift between
playback and capture is absorbed.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate

from engine.audio import load_wav
from engine.paths import MIMII_DIR
from killtest.playlist import make_chirp

logger = logging.getLogger("killtest.segment")


def find_offset(recording: np.ndarray, reference: np.ndarray) -> tuple[int, float]:
    """Locate ``reference`` inside ``recording`` by cross-correlation.

    Returns ``(best_lag_samples, peak)`` where ``best_lag_samples`` is the offset
    at which ``reference`` best aligns in ``recording`` (so ``recording[lag:]``
    lines up with ``reference[0:]``), and ``peak`` is the normalized peak
    correlation in ``[0, 1]`` -- it is ~1.0 when a scaled copy of ``reference``
    is present at that lag, regardless of amplitude.
    """
    rec = np.asarray(recording, dtype=np.float64).ravel()
    ref = np.asarray(reference, dtype=np.float64).ravel()
    if ref.size == 0 or rec.size == 0:
        raise ValueError("recording and reference must both be non-empty")

    corr = correlate(rec, ref, mode="full", method="fft")
    # In "full" mode the lag axis runs from -(len(ref)-1) .. (len(rec)-1).
    best_idx = int(np.argmax(corr))
    lag = best_idx - (ref.size - 1)

    # Normalize by the energy of the reference and of the overlapping recording
    # window so the peak is a correlation coefficient in [0, 1].
    ref_energy = float(np.dot(ref, ref))
    start = lag
    end = lag + ref.size
    seg_start = max(start, 0)
    seg_end = min(end, rec.size)
    if seg_end <= seg_start or ref_energy <= 0:
        return lag, 0.0
    window = rec[seg_start:seg_end]
    window_energy = float(np.dot(window, window))
    denom = np.sqrt(ref_energy * window_energy)
    peak = float(corr[best_idx] / denom) if denom > 0 else 0.0
    peak = max(0.0, min(1.0, abs(peak)))
    return lag, peak


def _relative_to_mimii(path: Path) -> Path | None:
    """Return ``path`` relative to the MIMII root, or ``None`` if outside it."""
    try:
        return path.resolve().relative_to(MIMII_DIR.resolve())
    except ValueError:
        return None


def segment_recording(
    recording_wav: Path,
    manifest_path: Path,
    out_dir: Path,
    refine_slack_s: float = 0.5,
    min_chirp_corr: float = 0.3,
) -> list[dict]:
    """Segment ``recording_wav`` into per-clip WAVs using the playlist manifest.

    Raises a clear error if the sync chirp cannot be located (peak correlation
    below ``min_chirp_corr``). Each clip is refined against the original (when its
    path is available) within ``nominal ± refine_slack_s``; otherwise the nominal
    cut is used and ``"refined"`` is False. Writes per-clip PCM_16 WAVs plus a
    ``segments.json`` sidecar, and returns the list of segment records.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    sr = int(manifest["sr"])
    chirp_s = float(manifest["chirp_s"])
    entries = manifest["entries"]

    recording = load_wav(recording_wav, expected_sr=sr)

    chirp = make_chirp(sr=sr, duration_s=chirp_s)
    chirp_lag, chirp_peak = find_offset(recording, chirp)
    if chirp_peak < min_chirp_corr:
        raise ValueError(
            f"chirp not found (peak corr {chirp_peak:.3f} < {min_chirp_corr}) in "
            f"{recording_wav} -- wrong file or too noisy?"
        )
    logger.info("chirp located at sample %d (corr %.3f)", chirp_lag, chirp_peak)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slack = int(round(sr * refine_slack_s))

    segments: list[dict] = []
    for index, entry in enumerate(entries):
        n_samples = int(entry["n_samples"])
        nominal_start = chirp_lag + int(entry["start_sample"])
        source = entry["path"]
        source_path = Path(source)

        refined_start = nominal_start
        refined = False
        corr_peak = chirp_peak
        if source_path.is_file():
            original = load_wav(source_path, expected_sr=sr)
            win_start = max(0, nominal_start - slack)
            win_end = min(len(recording), nominal_start + n_samples + slack)
            window = recording[win_start:win_end]
            if len(window) >= len(original):
                local_lag, corr_peak = find_offset(window, original)
                refined_start = win_start + local_lag
                refined = True
        else:
            logger.warning("original missing, nominal cut: %s", source)

        clip_start = max(0, refined_start)
        clip = recording[clip_start : clip_start + n_samples]
        if len(clip) < n_samples:
            clip = np.pad(clip, (0, n_samples - len(clip)))

        rel = _relative_to_mimii(source_path)
        if rel is not None:
            out_path = out_dir / rel
        else:
            out_path = out_dir / f"clip_{index:04d}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), clip.astype(np.float32), sr, subtype="PCM_16")

        segments.append(
            {
                "source": source,
                "out_path": str(out_path),
                "nominal_start": int(nominal_start),
                "refined_start": int(refined_start),
                "corr_peak": float(corr_peak),
                "refined": bool(refined),
            }
        )

    sidecar = out_dir / "segments.json"
    sidecar.write_text(json.dumps(segments, indent=2))
    logger.info("wrote %d segments -> %s", len(segments), out_dir)
    return segments


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment a re-recorded playlist WAV into aligned clips."
    )
    parser.add_argument("--recording", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="killtest_out/segmented")
    parser.add_argument("--refine-slack-s", type=float, default=0.5)
    parser.add_argument("--min-chirp-corr", type=float, default=0.3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    segment_recording(
        recording_wav=Path(args.recording),
        manifest_path=Path(args.manifest),
        out_dir=Path(args.out_dir),
        refine_slack_s=args.refine_slack_s,
        min_chirp_corr=args.min_chirp_corr,
    )


if __name__ == "__main__":
    main()
