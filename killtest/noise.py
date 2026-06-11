"""Overlay shop noise onto clips at fixed SNRs.

Used to simulate the noisy-environment leg of the kill test without a physical
re-recording: take clean clips, mix in user-provided shop-noise WAVs at 0/5/10 dB
SNR, and emit a parallel directory tree. Noise selection and cropping are seeded
per (clip, snr) with a stable hash so the same inputs always yield bit-identical
outputs across processes.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf

from engine.audio import load_wav

logger = logging.getLogger("killtest.noise")

_SR = 16000


def _rms_power(x: np.ndarray) -> float:
    return float(np.mean(np.square(x, dtype=np.float64)))


def mix_at_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mix ``noise`` into ``clean`` at a target ``snr_db`` (RMS-power based).

    A random crop of ``noise`` is taken (tiled first if ``noise`` is shorter than
    ``clean``). The noise is scaled so ``10*log10(P_clean / P_noise) == snr_db``.
    Raises ``ValueError`` if either signal has zero power. The returned mix is
    float32 and NOT clipped; if it exceeds unit peak it is rescaled as a whole so
    its peak is 0.99 (preserving the clean/noise ratio).
    """
    clean = np.asarray(clean, dtype=np.float64).ravel()
    noise = np.asarray(noise, dtype=np.float64).ravel()
    if clean.size == 0 or noise.size == 0:
        raise ValueError("clean and noise must both be non-empty")

    clean_power = _rms_power(clean)
    if clean_power <= 0:
        raise ValueError("clean signal has zero power")

    if noise.size < clean.size:
        reps = int(np.ceil(clean.size / noise.size))
        noise = np.tile(noise, reps)
    max_start = noise.size - clean.size
    start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
    noise_crop = noise[start : start + clean.size]

    noise_power = _rms_power(noise_crop)
    if noise_power <= 0:
        raise ValueError("noise signal has zero power")

    target_noise_power = clean_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)
    mix = clean + scale * noise_crop

    peak = float(np.max(np.abs(mix)))
    if peak > 1.0:
        mix = mix * (0.99 / peak)

    return mix.astype(np.float32)


def stable_hash(relpath: str, snr_db: float) -> int:
    """Deterministic non-negative hash of ``(relpath, snr)`` via blake2b.

    Python's built-in ``hash`` is salted per-process; this uses hashlib so seeds
    are stable across runs and machines.
    """
    key = f"{relpath}|{snr_db:g}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _load_noise_bank(noise_dir: Path) -> list[np.ndarray]:
    noise_dir = Path(noise_dir)
    paths = sorted(noise_dir.glob("*.wav"))
    if not paths:
        raise ValueError(f"no .wav files found in noise_dir {noise_dir}")
    bank: list[np.ndarray] = []
    for path in paths:
        info = sf.info(str(path))
        if info.samplerate != _SR:
            raise ValueError(
                f"noise file {path} has samplerate {info.samplerate}, expected {_SR}"
            )
        bank.append(load_wav(path, expected_sr=_SR))
    return bank


def overlay_tree(
    clip_paths: Sequence[Path],
    noise_dir: Path,
    out_root: Path,
    snrs_db: Sequence[float] = (0.0, 5.0, 10.0),
    seed: int = 1337,
) -> None:
    """Write a noise-overlaid copy of each clip at every SNR.

    For each clip x snr, a per-pair seeded RNG (``seed + stable_hash(relpath,
    snr)``) deterministically picks a noise file from ``noise_dir`` and crops it.
    Outputs go to ``out_root/snr_{snr:g}dB/<clip relpath>`` as PCM_16. The clip
    relpath is taken relative to its anchor directory when discoverable, else the
    clip filename. Noise files must be 16 kHz.
    """
    bank = _load_noise_bank(noise_dir)
    out_root = Path(out_root)

    for clip_path in clip_paths:
        clip_path = Path(clip_path)
        clean = load_wav(clip_path, expected_sr=_SR)
        relpath = _clip_relpath(clip_path)
        for snr in snrs_db:
            pair_seed = (seed + stable_hash(relpath, float(snr))) % (2**63)
            rng = np.random.Generator(np.random.PCG64(pair_seed))
            noise = bank[int(rng.integers(0, len(bank)))]
            mixed = mix_at_snr(clean, noise, float(snr), rng)
            out_path = out_root / f"snr_{snr:g}dB" / relpath
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), mixed, _SR, subtype="PCM_16")
    logger.info(
        "overlaid %d clips x %d snrs -> %s",
        len(clip_paths),
        len(snrs_db),
        out_root,
    )


def _clip_relpath(clip_path: Path) -> str:
    """Best-effort stable relative path for a clip (relative to MIMII root)."""
    from engine.paths import MIMII_DIR

    try:
        return str(clip_path.resolve().relative_to(MIMII_DIR.resolve()))
    except ValueError:
        return clip_path.name


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay shop noise onto clips at fixed SNRs."
    )
    parser.add_argument("--noise-dir", type=str, required=True)
    parser.add_argument("--snr", type=float, nargs="+", default=[0.0, 5.0, 10.0])
    parser.add_argument("--out-dir", type=str, default="killtest_out/noisy")
    parser.add_argument("--snr-source", type=str, default="0_dB")
    parser.add_argument("--machines", type=str, nargs="+", default=["fan", "pump"])
    parser.add_argument("--per-id", type=int, default=10)
    parser.add_argument(
        "--label",
        type=str,
        choices=["abnormal", "normal", "both"],
        default="both",
    )
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    from killtest.playlist import select_clips

    clip_paths = select_clips(
        snr=args.snr_source,
        machines=args.machines,
        per_id=args.per_id,
        label=args.label,
    )
    if not clip_paths:
        raise ValueError(
            f"no clips selected for snr={args.snr_source} machines={args.machines}"
        )
    overlay_tree(
        clip_paths=clip_paths,
        noise_dir=Path(args.noise_dir),
        out_root=Path(args.out_dir),
        snrs_db=tuple(args.snr),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
