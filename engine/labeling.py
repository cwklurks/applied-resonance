"""Write a labeled audio clip to the datakit: a PCM-16 WAV plus a JSON sidecar.

A technician (or the serving layer) hands us a captured window, a sample rate,
and a free-form metadata dict; we drop a timestamped, collision-proof pair of
files into the datakit so the clip can later be folded into training data.

The filename stem is ``{UTC timestamp}_{site_tag}`` so clips sort chronologically
and group by site at a glance. ``site_tag`` is sanitized to a filesystem-safe
alphabet (path-traversal guard) and falls back to ``unlabeled`` when missing or
empty after sanitization.
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from engine.paths import REPO_ROOT

logger = logging.getLogger(__name__)

_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]")
_DEFAULT_OUT_DIR = REPO_ROOT / "datakit" / "data"


def _sanitize_site_tag(site_tag: str) -> str:
    """Reduce ``site_tag`` to [A-Za-z0-9._-]; fall back to 'unlabeled'."""
    safe = _SAFE_RE.sub("", site_tag or "")
    return safe or "unlabeled"


def _unique_stem(out_dir: Path, base: str) -> str:
    """``base`` if free, else ``base_1``, ``base_2`` ... until both extensions free."""
    candidate = base
    n = 1
    while (out_dir / f"{candidate}.wav").exists() or (
        out_dir / f"{candidate}.json"
    ).exists():
        candidate = f"{base}_{n}"
        n += 1
    return candidate


def write_labeled_clip(
    audio: np.ndarray,
    sr: int,
    meta: dict,
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Write ``audio`` as a PCM-16 WAV plus a JSON metadata sidecar.

    Args:
        audio: 1-D, non-empty waveform (float, [-1, 1)).
        sr: sample rate, must be > 0.
        meta: free-form metadata; ``meta.get("site_tag")`` names the file.
        out_dir: target directory (created if needed). Defaults to
            ``REPO_ROOT / "datakit" / "data"``.

    Returns:
        ``(wav_path, json_path)`` for the written pair.

    Raises:
        ValueError: if ``audio`` is not 1-D / is empty, or ``sr`` <= 0.
    """
    audio = np.asarray(audio)
    if audio.ndim != 1:
        raise ValueError(f"audio must be 1-D, got shape {audio.shape}.")
    if audio.shape[0] == 0:
        raise ValueError("audio must be non-empty.")
    if sr <= 0:
        raise ValueError(f"sr must be > 0, got {sr}.")

    out_dir = Path(out_dir) if out_dir is not None else _DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    site = _sanitize_site_tag(str(meta.get("site_tag", "")))
    base = f"{now.strftime('%Y%m%d_%H%M%S')}_{site}"
    stem = _unique_stem(out_dir, base)

    wav_path = out_dir / f"{stem}.wav"
    json_path = out_dir / f"{stem}.json"

    sf.write(str(wav_path), audio, sr, subtype="PCM_16")

    sidecar = {
        **meta,
        "sr": int(sr),
        "n_samples": int(audio.shape[0]),
        "created_at": now.isoformat(),
    }
    json_path.write_text(json.dumps(sidecar, indent=2))

    logger.info("wrote labeled clip %s (+ .json)", wav_path)
    return wav_path, json_path
