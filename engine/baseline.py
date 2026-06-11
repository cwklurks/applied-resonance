"""Baseline fitting, persistence, and percentile calibration.

A ``Baseline`` is the per-machine reference against which streaming windows
are scored: a kNN fit set of normal embeddings, a sorted calibration set of
raw kNN scores (for mapping raw distances onto a 0-100 percentile), and a
mean log-mel fingerprint for HUD/diagnostics.

``BaselineManager`` owns fitting (with a deterministic holdout split and an
augmentation hook) plus save/load to ``{root}/{tag}.npz`` + ``{tag}.json``.
"""

import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from engine.embedder import Embedder
from engine.eval.scorers import KNNScorer
from engine.features.logmel import logmel
from engine.paths import DATA_DIR

logger = logging.getLogger(__name__)

# Path-traversal guard: tags become filenames, so disallow slashes/dots-runs.
_TAG_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_MIN_WINDOWS = 10  # need enough to carve off a calibration holdout
_CALIB_MODULO = 5  # i % 5 == 4 -> calibration (~20%), else train (~80%)


def _validate_tag(tag: str) -> str:
    if not _TAG_RE.match(tag):
        raise ValueError(
            f"Invalid baseline tag {tag!r}: must match {_TAG_RE.pattern} "
            "(path-traversal guard)."
        )
    return tag


@dataclass(frozen=True)
class Baseline:
    """Frozen per-machine reference. Arrays are exact; scorer is cached lazily."""

    tag: str
    train_embeddings: np.ndarray  # (N, 2048) float32 -- kNN fit set
    calib_scores: np.ndarray  # 1-D float64, SORTED ascending -- holdout raw scores
    mel_mean: np.ndarray  # (64,) float32 -- mean log-mel over ALL baseline windows
    timestamps: np.ndarray  # (N,) float64 -- seconds offset of each TRAIN window
    rpm: float | None = None

    def scorer(self) -> KNNScorer:
        """KNNScorer(k=2) fit on ``train_embeddings``, cached on the instance."""
        cached = self.__dict__.get("_scorer")
        if cached is None:
            cached = KNNScorer(k=2).fit(self.train_embeddings)
            object.__setattr__(self, "_scorer", cached)
        return cached

    def percentile(self, raw: float) -> float:
        """Map a raw kNN score onto 0-100 via the calibration distribution."""
        n = len(self.calib_scores)
        if n == 0:
            raise ValueError("calib_scores is empty; cannot compute percentile.")
        rank = int(np.searchsorted(self.calib_scores, raw, side="right"))
        return 100.0 * rank / n


class BaselineManager:
    """Fits, saves, and loads baselines under a root directory."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else DATA_DIR / "baselines"

    # ----------------------------------------------------------------- fit ----

    def fit(
        self,
        windows: Sequence[np.ndarray],
        tag: str,
        embedder: Embedder,
        timestamps: Sequence[float] | None = None,
        rpm: float | None = None,
        augmentations: Sequence[Callable[[np.ndarray], np.ndarray]] | None = None,
    ) -> Baseline:
        """Fit a baseline from normal windows.

        Holdout is deterministic: indices where ``i % 5 == 4`` form the
        calibration set, the rest are train. Calibration windows are never
        augmented. ``augmentations`` apply each callable to each raw TRAIN
        window, embedding the augmented copies and concatenating them onto the
        train set (the deliverable hook for the re-record kill test; intended
        for impulse-response convolution and noise mixing once that test
        reports its verdict). No built-in augmentations ship here.
        """
        _validate_tag(tag)
        windows = list(windows)
        for i, w in enumerate(windows):
            if np.asarray(w).ndim != 1:
                raise ValueError(
                    f"window {i} must be 1-D, got shape {np.asarray(w).shape}."
                )
        if len(windows) < _MIN_WINDOWS:
            raise ValueError(
                f"need >= {_MIN_WINDOWS} windows for a calibration holdout, "
                f"got {len(windows)}."
            )

        if timestamps is None:
            timestamps = [float(i) * 1.0 for i in range(len(windows))]
        else:
            timestamps = [float(t) for t in timestamps]
            if len(timestamps) != len(windows):
                raise ValueError(
                    f"timestamps length {len(timestamps)} != windows length "
                    f"{len(windows)}."
                )

        train_idx = [i for i in range(len(windows)) if i % _CALIB_MODULO != 4]
        calib_idx = [i for i in range(len(windows)) if i % _CALIB_MODULO == 4]
        if not calib_idx:
            raise ValueError("no calibration windows after holdout split.")

        train_windows = [np.asarray(windows[i], dtype=np.float32) for i in train_idx]
        calib_windows = [np.asarray(windows[i], dtype=np.float32) for i in calib_idx]

        # Single embed call over train then holdout (equal-length in practice).
        all_raw = train_windows + calib_windows
        all_emb = np.asarray(embedder.embed(all_raw), dtype=np.float32)
        n_train = len(train_windows)
        train_emb = all_emb[:n_train]
        calib_emb = all_emb[n_train:]

        train_ts = [timestamps[i] for i in train_idx]

        # Calibration scores are NEVER augmented: fit on the ORIGINAL train
        # embeddings, score the holdout. (Augmentation only extends the stored
        # kNN fit set used at inference; it must not move the calibration
        # distribution -- otherwise re-record augmentation would silently
        # re-calibrate the alarm thresholds.)
        calib_scores = (
            KNNScorer(k=2).fit(train_emb).score(calib_emb).astype(np.float64)
        )
        calib_scores = np.sort(calib_scores)

        if augmentations:
            aug_embeds: list[np.ndarray] = []
            aug_ts: list[float] = []
            for fn in augmentations:
                aug_windows = [
                    np.asarray(fn(w), dtype=np.float32) for w in train_windows
                ]
                aug_embeds.append(
                    np.asarray(embedder.embed(aug_windows), dtype=np.float32)
                )
                aug_ts.extend(train_ts)
            train_emb = np.concatenate([train_emb, *aug_embeds], axis=0)
            train_ts = train_ts + aug_ts

        # mel_mean over ALL ORIGINAL (un-augmented) windows: time-avg then window-avg.
        per_window = np.stack(
            [logmel(np.asarray(w, dtype=np.float32)).mean(axis=1) for w in windows],
            axis=0,
        )
        mel_mean = per_window.mean(axis=0).astype(np.float32)

        return Baseline(
            tag=tag,
            train_embeddings=train_emb,
            calib_scores=calib_scores,
            mel_mean=mel_mean,
            timestamps=np.asarray(train_ts, dtype=np.float64),
            rpm=rpm,
        )

    # ------------------------------------------------------- persistence ----

    def _npz_path(self, tag: str) -> Path:
        return self.root / f"{_validate_tag(tag)}.npz"

    def _json_path(self, tag: str) -> Path:
        return self.root / f"{_validate_tag(tag)}.json"

    def save(self, baseline: Baseline) -> Path:
        """Write ``{root}/{tag}.npz`` plus a ``{tag}.json`` metadata sidecar."""
        _validate_tag(baseline.tag)
        self.root.mkdir(parents=True, exist_ok=True)
        npz_path = self._npz_path(baseline.tag)
        np.savez(
            npz_path,
            train_embeddings=baseline.train_embeddings,
            calib_scores=baseline.calib_scores,
            mel_mean=baseline.mel_mean,
            timestamps=baseline.timestamps,
            # np.nan stands in for "no rpm" inside the array store.
            rpm=np.float64(np.nan if baseline.rpm is None else baseline.rpm),
        )
        sidecar = {
            "tag": baseline.tag,
            "rpm": baseline.rpm,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_train": int(baseline.train_embeddings.shape[0]),
            "n_calib": int(baseline.calib_scores.shape[0]),
        }
        self._json_path(baseline.tag).write_text(json.dumps(sidecar, indent=2))
        logger.info("saved baseline %s -> %s", baseline.tag, npz_path)
        return npz_path

    def load(self, tag: str) -> Baseline:
        """Load a baseline previously written by :meth:`save`."""
        npz_path = self._npz_path(tag)
        if not npz_path.exists():
            raise FileNotFoundError(f"no baseline at {npz_path}")
        with np.load(npz_path) as data:
            rpm_val = float(data["rpm"])
            rpm = None if np.isnan(rpm_val) else rpm_val
            return Baseline(
                tag=tag,
                train_embeddings=data["train_embeddings"],
                calib_scores=data["calib_scores"],
                mel_mean=data["mel_mean"],
                timestamps=data["timestamps"],
                rpm=rpm,
            )

    def exists(self, tag: str) -> bool:
        return self._npz_path(tag).exists()

    def list_tags(self) -> list[str]:
        """Tags of all ``.npz`` baselines under root (sorted)."""
        if not self.root.exists():
            return []
        return sorted(p.stem for p in self.root.glob("*.npz"))

    def load_library(self, machine_type: str) -> Baseline:
        """Library mode: score against a pre-built fleet-wide baseline for a
        machine type (e.g. 'fan'). Not yet implemented -- the landing zone for
        baselines distilled from MIMII + field data."""
        raise NotImplementedError(
            "library baselines (fleet-wide, machine-type) are not yet "
            f"implemented; requested {machine_type!r}."
        )
