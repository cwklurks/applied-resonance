"""Disk-cached feature extraction keyed by wav path + feature kind."""

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from engine.audio import load_wav
from engine.features import extract_features
from engine.features.panns import panns_embeddings
from engine.paths import CACHE_DIR, MIMII_DIR


def _cache_path(wav_path: Path, kind: str) -> Path:
    """Return the .npy cache path for a given wav and feature kind.

    Mirrors the wav's path relative to MIMII_DIR; for wavs outside MIMII_DIR,
    uses a sha1 of the absolute path as the filename.
    """
    wav_path = Path(wav_path)
    base = CACHE_DIR / kind
    try:
        rel = wav_path.resolve().relative_to(MIMII_DIR.resolve())
        return base / rel.with_suffix(".npy")
    except ValueError:
        digest = hashlib.sha1(str(wav_path.resolve()).encode()).hexdigest()
        return base / f"{digest}.npy"


def _atomic_save(target: Path, array: np.ndarray) -> None:
    """Save array to target atomically. np.save appends .npy, so write to a
    temp path with a .npy suffix first, then os.replace onto target."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp.npy")
    np.save(tmp, array)  # tmp already ends in .npy; np.save leaves it as-is
    os.replace(tmp, target)


def cached_features(
    wav_path: Path, kind: str, device: torch.device | None = None
) -> np.ndarray:
    """Load cached features for a wav, computing and caching them if absent."""
    cache_file = _cache_path(wav_path, kind)
    if cache_file.exists():
        return np.load(cache_file)

    wav = load_wav(wav_path)
    feats = extract_features(wav, kind, device=device)
    _atomic_save(cache_file, feats)
    return feats


def cached_features_many(
    wav_paths: Sequence[Path],
    kind: str,
    device: torch.device | None = None,
    batch_size: int = 8,
) -> np.ndarray:
    """Load/compute features for many wavs, returned in input order.

    Cached entries are read from disk; missing ones are computed (batched for
    'panns', looped for 'logmel'), cached, and slotted back into order.
    """
    paths = [Path(p) for p in wav_paths]
    results: list[np.ndarray | None] = [None] * len(paths)
    missing_idx: list[int] = []

    for i, p in enumerate(paths):
        cache_file = _cache_path(p, kind)
        if cache_file.exists():
            results[i] = np.load(cache_file)
        else:
            missing_idx.append(i)

    if missing_idx:
        if kind == "panns":
            wavs = [load_wav(paths[i]) for i in tqdm(missing_idx, desc="load (panns)")]
            embeds = panns_embeddings(wavs, device=device, batch_size=batch_size)
            for slot, i in enumerate(missing_idx):
                feats = embeds[slot]
                _atomic_save(_cache_path(paths[i], kind), feats)
                results[i] = feats
        else:
            for i in tqdm(missing_idx, desc=f"compute ({kind})"):
                wav = load_wav(paths[i])
                feats = extract_features(wav, kind, device=device)
                _atomic_save(_cache_path(paths[i], kind), feats)
                results[i] = feats

    return np.stack([r for r in results])
