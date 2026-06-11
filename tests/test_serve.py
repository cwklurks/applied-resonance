"""Tests for engine.serve (FastAPI scoring service).

All exercised through TestClient with an injected FakeEmbedder; the real PANNs
backend is never loaded. Baselines persist to a tmp dir so restart-survival can
be checked by building a second app over the same root.
"""

import base64
import hashlib
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from engine.serve import create_app

SR = 16000


# -------------------------------------------------------- test doubles ----


class FakeEmbedder:
    """Deterministic hash -> 2048-d vector.

    In normal mode, embeddings are near-orthogonal standard-normal vectors and
    the running mean of everything embedded is accumulated. When ``.far`` is
    flipped True, every window is mapped onto the direction *opposite* that
    running mean. In cosine space that is reliably farther from the baseline
    cloud than any in-distribution window (which sit ~1.0 apart by chance), so
    the kNN score clears the calibration band -> percentile 100 -> SUSPECT.
    """

    def __init__(self):
        self.far = False
        self._sum = np.zeros(2048, dtype=np.float64)

    def embed(self, wavs):
        out = np.empty((len(wavs), 2048), dtype=np.float32)
        for i, w in enumerate(wavs):
            digest = hashlib.sha256(
                np.asarray(w, dtype=np.float32).tobytes()
            ).digest()
            seed = int.from_bytes(digest[:8], "little")
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(2048).astype(np.float32)
            if self.far:
                # Anti-aligned with the accumulated normal cloud -> max cosine
                # distance, plus a hair of jitter so windows are not identical.
                out[i] = (-self._sum).astype(np.float32) + vec * 1e-3
            else:
                self._sum += vec
                out[i] = vec
        return out


def _pcm_b64(audio: np.ndarray) -> str:
    """float32 [-1,1) -> int16 LE -> base64, matching the service codec."""
    i16 = np.clip(audio * 32768.0, -32768, 32767).astype("<i2")
    return base64.b64encode(i16.tobytes()).decode("ascii")


def _noise(n, seed=0):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32) * 0.05


def _client(tmp_path, embedder=None):
    embedder = embedder if embedder is not None else FakeEmbedder()
    app = create_app(
        embedder=embedder,
        baseline_root=tmp_path / "b",
        label_dir=tmp_path / "d",
    )
    return TestClient(app), embedder


# ------------------------------------------------------------- /health ----


def test_health_ok_empty_baselines(tmp_path):
    client, _ = _client(tmp_path)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["baselines"] == []
    assert body["sessions"] == 0
    assert "backend" in body


# ----------------------------------------------- session capture flow ----


def _run_capture(client, tag="m1", rpm=None, seed=0):
    """Start a session and post 30 one-second chunks; return final response."""
    r = client.post(
        "/session/start", json={"mode": "session", "tag": tag, "rpm": rpm}
    )
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]
    assert r.json()["state"] == "CAPTURING_BASELINE"

    last = None
    for i in range(30):
        chunk = _pcm_b64(_noise(SR, seed=seed + i))
        last = client.post("/score", json={"session_id": sid, "pcm_b64": chunk})
        assert last.status_code == 200, last.text
    return sid, last


def test_session_capture_then_listen(tmp_path):
    client, _ = _client(tmp_path)

    r = client.post(
        "/session/start", json={"mode": "session", "tag": "m1", "rpm": 1450.0}
    )
    sid = r.json()["session_id"]
    assert r.json()["state"] == "CAPTURING_BASELINE"
    assert r.json()["capture_remaining_s"] == 30.0

    # 29 chunks -> still capturing, remaining decreasing.
    remainings = []
    for i in range(29):
        resp = client.post(
            "/score",
            json={"session_id": sid, "pcm_b64": _pcm_b64(_noise(SR, seed=i))},
        ).json()
        assert resp["state"] == "CAPTURING_BASELINE"
        assert resp["score"] is None
        assert resp["percentile"] is None
        assert resp["evidence_line"] is None
        remainings.append(resp["capture_remaining_s"])

    assert remainings[0] == 29.0
    assert remainings[-1] == 1.0
    assert remainings == sorted(remainings, reverse=True)

    # 30th chunk completes capture -> LISTENING.
    final = client.post(
        "/score",
        json={"session_id": sid, "pcm_b64": _pcm_b64(_noise(SR, seed=29))},
    ).json()
    assert final["state"] == "LISTENING"
    assert final["capture_remaining_s"] == 0.0

    # Baseline now persisted and visible in /health.
    assert "m1" in client.get("/health").json()["baselines"]

    # Subsequent /score chunks eventually produce a numeric score+percentile.
    got_numeric = False
    for i in range(5):
        resp = client.post(
            "/score",
            json={"session_id": sid, "pcm_b64": _pcm_b64(_noise(SR, seed=100 + i))},
        ).json()
        assert resp["state"] in {"LISTENING", "SUSPECT", "ALERT"}
        if resp["score"] is not None:
            got_numeric = True
            assert resp["percentile"] is not None
    assert got_numeric


# ------------------------------------------------ saved / restart survival ----


def test_saved_mode_survives_restart(tmp_path):
    # First app: capture a baseline to disk.
    client1, _ = _client(tmp_path)
    _run_capture(client1, tag="rig")
    assert "rig" in client1.get("/health").json()["baselines"]

    # Second app over the SAME baseline root: no re-capture needed.
    client2, _ = _client(tmp_path)
    assert "rig" in client2.get("/health").json()["baselines"]

    r = client2.post(
        "/session/start", json={"mode": "saved", "tag": "rig", "rpm": None}
    )
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "LISTENING"
    sid = r.json()["session_id"]

    # /score works immediately against the loaded baseline.
    resp = client2.post(
        "/score", json={"session_id": sid, "pcm_b64": _pcm_b64(_noise(SR, seed=3))}
    )
    assert resp.status_code == 200, resp.text


def test_saved_mode_missing_tag_404(tmp_path):
    client, _ = _client(tmp_path)
    r = client.post(
        "/session/start", json={"mode": "saved", "tag": "nope", "rpm": None}
    )
    assert r.status_code == 404


# ----------------------------------------------------------- library 501 ----


def test_library_mode_501(tmp_path):
    client, _ = _client(tmp_path)
    r = client.post(
        "/session/start", json={"mode": "library", "tag": "fan", "rpm": None}
    )
    assert r.status_code == 501
    assert "not yet implemented" in r.json()["detail"].lower()


def test_invalid_mode_422(tmp_path):
    client, _ = _client(tmp_path)
    r = client.post(
        "/session/start", json={"mode": "bogus", "tag": "x", "rpm": None}
    )
    assert r.status_code == 422


# --------------------------------------------------------- /score errors ----


def test_score_unknown_session_404(tmp_path):
    client, _ = _client(tmp_path)
    r = client.post(
        "/score", json={"session_id": "deadbeef", "pcm_b64": _pcm_b64(_noise(100))}
    )
    assert r.status_code == 404


def test_score_bad_base64_400(tmp_path):
    client, _ = _client(tmp_path)
    sid = client.post(
        "/session/start", json={"mode": "session", "tag": "m1", "rpm": None}
    ).json()["session_id"]
    r = client.post(
        "/score", json={"session_id": sid, "pcm_b64": "not!!base64!!"}
    )
    assert r.status_code == 400


def test_score_odd_byte_count_400(tmp_path):
    client, _ = _client(tmp_path)
    sid = client.post(
        "/session/start", json={"mode": "session", "tag": "m1", "rpm": None}
    ).json()["session_id"]
    # 3 raw bytes -> odd, cannot be int16 LE.
    odd = base64.b64encode(b"\x01\x02\x03").decode("ascii")
    r = client.post("/score", json={"session_id": sid, "pcm_b64": odd})
    assert r.status_code == 400


# ------------------------------------------------------------- /label ----


def test_label_writes_both_files(tmp_path):
    client, _ = _client(tmp_path)
    body = {
        "session_id": None,
        "pcm_b64": _pcm_b64(_noise(3 * SR, seed=11)),
        "machine_type": "fan",
        "suspected_fault": "imbalance",
        "contains_speech": False,
        "note": "loud",
        "site_tag": "plantA",
    }
    r = client.post("/label", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    wav = tmp_path / "d"  # label_dir
    # Paths are repo-relative strings; resolve and confirm existence.
    from engine.paths import REPO_ROOT

    wav_path = REPO_ROOT / out["wav_path"]
    json_path = REPO_ROOT / out["json_path"]
    assert wav_path.exists()
    assert json_path.exists()
    assert wav_path.parent == wav
    meta = json.loads(json_path.read_text())
    assert meta["machine_type"] == "fan"
    assert meta["suspected_fault"] == "imbalance"
    assert meta["sr"] == SR


# ------------------------------------------------------- evidence line ----


def test_evidence_line_surfaced_on_suspect(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "engine.stream._build_evidence", lambda **kw: {"hud": "test line"}
    )

    embedder = FakeEmbedder()
    client, _ = _client(tmp_path, embedder=embedder)

    # Capture a baseline on near-normal content.
    sid, _ = _run_capture(client, tag="ev", seed=0)

    # Flip the embedder far away so listening windows score anomalously high.
    embedder.far = True

    seen = None
    for i in range(6):
        resp = client.post(
            "/score",
            json={"session_id": sid, "pcm_b64": _pcm_b64(_noise(SR, seed=200 + i))},
        ).json()
        if resp["evidence_line"] is not None:
            seen = resp
            break

    assert seen is not None
    assert seen["state"] in {"SUSPECT", "ALERT"}
    assert seen["evidence_line"] == "test line"
