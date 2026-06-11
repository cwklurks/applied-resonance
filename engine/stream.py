"""Streaming scorer: sliding windows over live audio -> scored results.

A ``StreamScorer`` buffers incoming chunks, emits a fixed-cadence sliding
window (default 3 s window, 1 s hop), embeds each window, scores it against a
``Baseline``, smooths with an EMA, maps to a percentile, asks the ``policy``
for a state, and -- only when the state is SUSPECT/ALERT -- attaches an
evidence bundle.

Decoupling: ``policy`` is injected (any object with
``update(percentile, t) -> state`` where ``state.name`` is a string), and
evidence is built via a lazy import wrapped in module-level ``_build_evidence``
so this module never imports ``engine.policy`` / ``engine.evidence`` at import
time (those land as concurrent work items, and tests monkeypatch the wrapper).
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

SR = 16000
_EVIDENCE_STATES = ("SUSPECT", "ALERT")


def _build_evidence(**kwargs) -> dict:
    """Lazy seam to engine.evidence.build_evidence.

    Imported here (not at module level) so engine.stream stays importable
    while engine.evidence is being written concurrently. Tests monkeypatch
    ``engine.stream._build_evidence`` instead of exercising the real builder.
    """
    from engine.evidence import build_evidence

    return build_evidence(**kwargs)


@dataclass
class WindowResult:
    t: float  # seconds at WINDOW END since stream start
    raw: float  # raw kNN distance of this window
    ema: float  # EMA-smoothed score (alpha on raw; first window: ema = raw)
    percentile: float  # baseline.percentile(ema)
    state: object  # whatever policy.update returned
    evidence: dict | None  # only when state.name in (SUSPECT, ALERT), else None
    window: np.ndarray  # the 3 s window audio (for evidence/labeling)


class StreamScorer:
    """Slides a window over incoming audio and scores each hop."""

    def __init__(
        self,
        baseline,
        embedder,
        policy,
        window_s: float = 3.0,
        hop_s: float = 1.0,
        ema_alpha: float = 0.3,
        rpm: float | None = None,
    ) -> None:
        self.baseline = baseline
        self.embedder = embedder
        self.policy = policy
        self.window_s = window_s
        self.hop_s = hop_s
        self.ema_alpha = ema_alpha
        self.rpm = rpm

        self.window_n = round(window_s * SR)  # 48000
        self.hop_n = round(hop_s * SR)  # 16000

        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start = 0  # absolute index of buffer[0]
        self._total = 0  # absolute count of samples ever seen
        self._next_end = self.window_n  # absolute sample count of next window end
        self._prev_ema: float | None = None

    def process(self, chunk: np.ndarray) -> list[WindowResult]:
        """Ingest a chunk; return any windows that completed on this call."""
        chunk = np.asarray(chunk)
        if chunk.ndim != 1:
            raise ValueError(f"chunk must be 1-D, got shape {chunk.shape}.")
        chunk = chunk.astype(np.float32, copy=False)

        self._buffer = np.concatenate([self._buffer, chunk])
        self._total += chunk.shape[0]

        results: list[WindowResult] = []
        while self._total >= self._next_end:
            end = self._next_end  # absolute sample index of window end
            start = end - self.window_n
            lo = start - self._buffer_start
            hi = end - self._buffer_start
            window = self._buffer[lo:hi].copy()
            t = end / SR
            results.append(self._score_window(window, t))
            self._next_end += self.hop_n

        self._trim_buffer()
        return results

    def run(self, source) -> Iterator[WindowResult]:
        """Yield results for every chunk produced by ``source.chunks()``."""
        for chunk in source.chunks():
            yield from self.process(chunk)

    # ----------------------------------------------------------- internals ----

    def _score_window(self, window: np.ndarray, t: float) -> WindowResult:
        emb = np.asarray(self.embedder.embed([window]))[0]
        raw = float(self.baseline.scorer().score(emb.reshape(1, -1))[0])

        if self._prev_ema is None:
            ema = raw
        else:
            ema = self.ema_alpha * raw + (1.0 - self.ema_alpha) * self._prev_ema
        self._prev_ema = ema

        pct = self.baseline.percentile(ema)
        state = self.policy.update(pct, t)

        evidence = None
        if getattr(state, "name", None) in _EVIDENCE_STATES:
            evidence = _build_evidence(
                window=window,
                baseline=self.baseline,
                window_embedding=emb,
                rpm=self.rpm,
            )

        return WindowResult(
            t=t,
            raw=raw,
            ema=ema,
            percentile=pct,
            state=state,
            evidence=evidence,
            window=window,
        )

    def _trim_buffer(self) -> None:
        """Drop samples no future window needs; keep memory bounded."""
        next_start = self._next_end - self.window_n  # earliest sample still needed
        drop = next_start - self._buffer_start
        if drop > 0:
            self._buffer = self._buffer[drop:]
            self._buffer_start += drop
