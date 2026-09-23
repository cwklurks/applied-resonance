"""Playback playlist generation for the re-record kill test.

Builds a single long WAV that the user physically plays through a speaker and
re-records with a consumer mic. The layout is::

    [chirp][silence][clip_0][silence][clip_1]...[clip_n-1][silence]

A leading log sine sweep (the chirp) gives the segmenter a strong, easy-to-find
sync anchor. The accompanying manifest records the exact sample position of each
clip in the written WAV so :mod:`killtest.segment` can cut the re-recording back
into aligned per-clip files.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf

from engine.audio import load_wav
from engine.dataset import Clip
from engine.eval.splits import make_anomaly_splits
from killtest.ids import IdKey, format_id_key, parse_id_tokens

logger = logging.getLogger("killtest.playlist")

_FADE_MS = 10.0


def make_chirp(
    sr: int = 16000,
    duration_s: float = 2.0,
    f0: float = 100.0,
    f1: float = 7000.0,
    amplitude: float = 0.8,
) -> np.ndarray:
    """Log (exponential) sine sweep from ``f0`` to ``f1`` with fades.

    A 10 ms linear fade-in and fade-out suppress click transients at the edges
    so the sweep cross-correlates cleanly. Returns a 1-D float32 array of length
    ``round(sr * duration_s)``.
    """
    if sr <= 0:
        raise ValueError(f"sr must be positive, got {sr}")
    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    if not 0 < f0 < f1 < sr / 2:
        raise ValueError(
            f"require 0 < f0 < f1 < Nyquist; got f0={f0}, f1={f1}, sr={sr}"
        )

    n = int(round(sr * duration_s))
    t = np.arange(n, dtype=np.float64) / sr
    total = n / sr
    # Exponential sweep instantaneous phase (Farina's log sweep).
    ratio = f1 / f0
    k = total / np.log(ratio)
    phase = 2.0 * np.pi * f0 * k * (np.expm1(t / k))
    sweep = amplitude * np.sin(phase)

    fade_n = min(int(round(sr * _FADE_MS / 1000.0)), n // 2)
    if fade_n > 0:
        ramp = np.linspace(0.0, 1.0, fade_n, endpoint=False)
        sweep[:fade_n] *= ramp
        sweep[-fade_n:] *= ramp[::-1]

    return sweep.astype(np.float32)


def make_playlist(
    clip_paths: Sequence[Path],
    out_wav: Path,
    out_manifest: Path,
    sr: int = 16000,
    silence_s: float = 2.0,
    chirp_s: float = 2.0,
) -> dict:
    """Assemble clips into a single playback WAV plus an alignment manifest.

    Layout: ``[chirp][silence][clip_0][silence]...[clip_n-1][silence]``. The full
    playlist is peak-normalized to 0.9 and written as PCM_16. ``start_sample`` in
    each manifest entry is the exact sample offset of that clip in the written
    WAV. Returns the manifest dict (also written to ``out_manifest`` as JSON).
    """
    if not clip_paths:
        raise ValueError("clip_paths is empty; nothing to build")
    if silence_s < 0:
        raise ValueError(f"silence_s must be non-negative, got {silence_s}")

    chirp = make_chirp(sr=sr, duration_s=chirp_s)
    silence_n = int(round(sr * silence_s))
    silence = np.zeros(silence_n, dtype=np.float32)

    segments: list[np.ndarray] = [chirp, silence]
    entries: list[dict] = []
    cursor = len(chirp) + silence_n
    for path in clip_paths:
        clip = load_wav(path, expected_sr=sr)
        entries.append(
            {
                "path": str(path),
                "start_sample": int(cursor),
                "n_samples": int(len(clip)),
            }
        )
        segments.append(clip)
        segments.append(silence)
        cursor += len(clip) + silence_n

    playlist = np.concatenate(segments).astype(np.float32)

    peak = float(np.max(np.abs(playlist)))
    gain = 0.9 / peak if peak > 0 else 1.0
    playlist = (playlist * gain).astype(np.float32)

    out_wav = Path(out_wav)
    out_manifest = Path(out_manifest)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    sf.write(str(out_wav), playlist, sr, subtype="PCM_16")

    manifest = {
        "sr": int(sr),
        "chirp_s": float(chirp_s),
        "silence_s": float(silence_s),
        "normalization_gain": float(gain),
        "total_samples": int(len(playlist)),
        "duration_s": float(len(playlist) / sr),
        "entries": entries,
    }
    out_manifest.write_text(json.dumps(manifest, indent=2))
    logger.info(
        "wrote playlist: %d clips, %d samples (%.1fs) -> %s",
        len(entries),
        len(playlist),
        len(playlist) / sr,
        out_wav,
    )
    return manifest


def select_clips(
    snr: str,
    machines: Sequence[str],
    per_id: int,
    label: str,
) -> list[Path]:
    """Deterministically pick ``per_id`` clips per (machine_id, label).

    ``label`` is one of ``"normal"``, ``"abnormal"``, ``"both"``. Clips come from
    :func:`engine.dataset.list_clips` (already sorted) and the first ``per_id``
    per group are taken, so the selection is fully reproducible.
    """
    from engine.dataset import list_clips

    wanted = ("normal", "abnormal") if label == "both" else (label,)
    clips = list_clips(snr=snr, machines=tuple(machines))

    chosen: list[Path] = []
    groups: dict[tuple[str, str, str], int] = {}
    for clip in clips:
        if clip.label not in wanted:
            continue
        key = (clip.machine, clip.machine_id, clip.label)
        count = groups.get(key, 0)
        if count < per_id:
            chosen.append(clip.path)
            groups[key] = count + 1
    return chosen


def select_split_test_clips(
    clips: Sequence[Clip],
    ids: Sequence[IdKey],
    seed: int = 1337,
) -> list[Path]:
    """Select every split test clip for the requested machine ids.

    Splits are built from the full clip list first, then filtered, so the split
    seed behavior and per-id RNG offsets remain identical to the baseline eval.
    """
    if not ids:
        raise ValueError("--from-split requires --ids, e.g. --ids fan:id_06")

    wanted = set(ids)
    splits = make_anomaly_splits(clips, seed=seed)
    found = {(split.machine, split.machine_id) for split in splits}
    missing = [format_id_key(key) for key in ids if key not in found]
    if missing:
        raise ValueError(f"requested ids not present in split set: {missing}")

    selected: list[Path] = []
    for split in splits:
        if (split.machine, split.machine_id) not in wanted:
            continue
        selected.extend(clip.path for clip in split.test_normal)
        selected.extend(clip.path for clip in split.test_abnormal)
    return selected


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a re-record playback playlist.")
    parser.add_argument("--snr", type=str, default="0_dB")
    parser.add_argument("--machines", type=str, nargs="+", default=["fan", "pump"])
    parser.add_argument("--per-id", type=int, default=5)
    parser.add_argument(
        "--label",
        type=str,
        choices=["abnormal", "normal", "both"],
        default="both",
    )
    parser.add_argument(
        "--from-split",
        action="store_true",
        help="select test_normal + test_abnormal from the anomaly eval split",
    )
    parser.add_argument(
        "--ids",
        type=str,
        nargs="+",
        default=None,
        help='machine ids for --from-split, e.g. --ids fan:id_06 pump:id_02',
    )
    parser.add_argument("--out-dir", type=str, default="killtest_out")
    parser.add_argument("--silence-s", type=float, default=2.0)
    parser.add_argument("--chirp-s", type=float, default=2.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    if args.from_split:
        from engine.dataset import list_clips

        ids = parse_id_tokens(args.ids, option_name="--ids")
        clip_paths = select_split_test_clips(
            clips=list_clips(snr=args.snr),
            ids=ids,
            seed=1337,
        )
    else:
        clip_paths = select_clips(
            snr=args.snr,
            machines=args.machines,
            per_id=args.per_id,
            label=args.label,
        )
    if not clip_paths:
        raise ValueError(
            f"no clips selected for snr={args.snr} machines={args.machines} "
            f"label={args.label}"
        )
    out_dir = Path(args.out_dir)
    manifest = make_playlist(
        clip_paths=clip_paths,
        out_wav=out_dir / "playlist.wav",
        out_manifest=out_dir / "playlist_manifest.json",
        silence_s=args.silence_s,
        chirp_s=args.chirp_s,
    )
    print(
        "total playback duration: "
        f"{manifest['duration_s']:.1f}s ({len(manifest['entries'])} clips)"
    )


if __name__ == "__main__":
    main()
