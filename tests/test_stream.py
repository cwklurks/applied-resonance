"""Tests for engine.stream (synthetic embedder + duck-typed baseline)."""

import hashlib
import types

import numpy as np

from engine.stream import StreamScorer, WindowResult

SR = 16000


# -------------------------------------------------------- test doubles ----


class FakeEmbedder:
    """Deterministic hash-of-bytes -> 2048-d vector."""

    def embed(self, wavs):
        out = np.empty((len(wavs), 2048), dtype=np.float32)
        for i, w in enumerate(wavs):
            digest = hashlib.sha256(np.asarray(w, dtype=np.float32).tobytes()).digest()
            seed = int.from_bytes(digest[:8], "little")
            out[i] = np.random.default_rng(seed).standard_normal(2048).astype(
                np.float32
            )
        return out


class _ContentScorer:
    """Raw score = mean of the embedding (deterministic per window content)."""

    def score(self, x):
        return np.asarray(x, dtype=np.float64).mean(axis=1)


class ContentBaseline:
    """Duck-typed baseline: raw score derives only from window content."""

    def __init__(self):
        self._scorer = _ContentScorer()

    def scorer(self):
        return self._scorer

    def percentile(self, raw):
        return 50.0


class ScriptedScorer:
    def __init__(self, raws):
        self._raws = list(raws)
        self._i = 0

    def score(self, x):
        val = self._raws[self._i]
        self._i += 1
        return np.array([val], dtype=np.float64)


class ScriptedBaseline:
    """Returns scripted raw scores in call order; percentile is identity."""

    def __init__(self, raws):
        self._scorer = ScriptedScorer(raws)

    def scorer(self):
        return self._scorer

    def percentile(self, raw):
        return float(raw)


def _state(name):
    return types.SimpleNamespace(name=name)


class StubPolicy:
    """Returns a namespace with .name; either constant or a scripted list."""

    def __init__(self, name=None, names=None):
        self._name = name
        self._names = list(names) if names is not None else None
        self._i = 0

    def update(self, percentile, t):
        if self._names is not None:
            name = self._names[self._i]
            self._i += 1
            return _state(name)
        return _state(self._name)


def _audio(n_samples, seed=0):
    return np.random.default_rng(seed).standard_normal(n_samples).astype(np.float32)


# ---------------------------------------------------------- hop schedule ----


def test_hop_schedule_matches_across_chunkings():
    audio = _audio(10 * SR, seed=7)

    one = StreamScorer(ContentBaseline(), FakeEmbedder(), StubPolicy(name="LISTENING"))
    big = one.process(audio)
    assert [r.t for r in big] == [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    # Same audio dribbled as 0.25 s chunks -> identical t sequence and raws.
    dribble = StreamScorer(
        ContentBaseline(), FakeEmbedder(), StubPolicy(name="LISTENING")
    )
    quarter = round(0.25 * SR)
    drib_results: list[WindowResult] = []
    for start in range(0, audio.shape[0], quarter):
        drib_results.extend(dribble.process(audio[start : start + quarter]))

    assert [r.t for r in drib_results] == [r.t for r in big]
    assert [r.raw for r in drib_results] == [r.raw for r in big]


def test_first_window_needs_full_window():
    scorer = StreamScorer(
        ContentBaseline(), FakeEmbedder(), StubPolicy(name="LISTENING")
    )
    # 2.99 s -> nothing yet.
    assert scorer.process(_audio(round(2.99 * SR))) == []
    # Cross 3.0 s -> exactly one window at t=3.0.
    out = scorer.process(_audio(round(0.01 * SR) + 1))
    assert len(out) == 1
    assert out[0].t == 3.0


# ------------------------------------------------------------- EMA math ----


def test_ema_recurrence_exact():
    raws = [10.0, 20.0, 0.0, 5.0]
    baseline = ScriptedBaseline(raws)
    scorer = StreamScorer(
        baseline, FakeEmbedder(), StubPolicy(name="LISTENING"), ema_alpha=0.3
    )
    results = scorer.process(_audio(6 * SR))  # t = 3,4,5,6 -> 4 windows
    assert len(results) == 4

    alpha = 0.3
    expected = []
    prev = None
    for r in raws:
        prev = r if prev is None else alpha * r + (1 - alpha) * prev
        expected.append(prev)

    got = [r.ema for r in results]
    assert got == expected
    # First window: ema == raw.
    assert results[0].ema == raws[0]


# -------------------------------------------------------- evidence gating ----


def test_evidence_gated_on_state(monkeypatch):
    captured = {}

    def fake_build(**kwargs):
        captured["window_embedding"] = kwargs["window_embedding"]
        return {"hud": "x"}

    monkeypatch.setattr("engine.stream._build_evidence", fake_build)

    names = ["LISTENING", "LISTENING", "SUSPECT", "ALERT"]
    scorer = StreamScorer(
        ContentBaseline(), FakeEmbedder(), StubPolicy(names=names)
    )
    results = scorer.process(_audio(6 * SR))  # 4 windows
    assert len(results) == 4

    assert results[0].evidence is None
    assert results[1].evidence is None
    assert results[2].evidence == {"hud": "x"}
    assert results[3].evidence == {"hud": "x"}

    # _build_evidence received a (2048,) embedding.
    assert captured["window_embedding"].shape == (2048,)


def test_evidence_not_built_when_listening(monkeypatch):
    called = {"n": 0}

    def fake_build(**kwargs):
        called["n"] += 1
        return {}

    monkeypatch.setattr("engine.stream._build_evidence", fake_build)
    scorer = StreamScorer(
        ContentBaseline(), FakeEmbedder(), StubPolicy(name="LISTENING")
    )
    scorer.process(_audio(6 * SR))
    assert called["n"] == 0


# ----------------------------------------------------------------- run ----


def test_run_over_audio_source():
    class FakeSource:
        def chunks(self):
            for _ in range(4):
                yield _audio(SR, seed=1)

    scorer = StreamScorer(
        ContentBaseline(), FakeEmbedder(), StubPolicy(name="LISTENING")
    )
    results = list(scorer.run(FakeSource()))
    assert [r.t for r in results] == [3.0, 4.0]
