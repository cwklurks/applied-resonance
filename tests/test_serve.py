"""Tests for engine.serve (FastAPI scoring service).

All exercised through TestClient with an injected FakeEmbedder; the real PANNs
backend is never loaded. Baselines persist to a tmp dir so restart-survival can
be checked by building a second app over the same root.
"""

import base64
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from engine.serve import (
    MAX_REQUEST_BYTES,
    MAX_SCORE_PCM_BYTES,
    ServiceSettings,
    create_app,
)

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


class FakeClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def _client(tmp_path, embedder=None, *, settings=None, clock=None, client_host="127.0.0.1"):
    embedder = embedder if embedder is not None else FakeEmbedder()
    app = create_app(
        embedder=embedder,
        baseline_root=tmp_path / "b",
        label_dir=tmp_path / "d",
        capture_root=tmp_path / "c",
        settings=settings,
        clock=clock,
    )
    return TestClient(app, client=(client_host, 50000)), embedder


# ------------------------------------------------------------- /health ----


def test_health_ok_empty_baselines(tmp_path):
    client, _ = _client(tmp_path)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_production_embedder_is_warmed_before_health_is_available(
    tmp_path, monkeypatch
):
    class StartupTrackingEmbedder(FakeEmbedder):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def embed(self, wavs):
            self.batch_sizes.append(len(wavs))
            return super().embed(wavs)

    embedder = StartupTrackingEmbedder()
    constructions = []

    def build_embedder(backend, device):
        constructions.append((backend, device))
        return embedder

    monkeypatch.setenv("EARSIGHT_BACKEND", "torch")
    monkeypatch.setenv("EARSIGHT_DEVICE", "cpu")
    monkeypatch.setattr("engine.serve.get_embedder", build_embedder)
    app = create_app(
        baseline_root=tmp_path / "b",
        label_dir=tmp_path / "d",
        capture_root=tmp_path / "c",
    )

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert constructions == [("torch", "cpu")]
        assert embedder.batch_sizes == [1]

        _, final = _run_capture(client, tag="startup-warm")
        assert final.json()["state"] == "LISTENING"
        assert embedder.batch_sizes == [1, 28]


def test_model_dependent_request_fails_fast_without_lifespan_startup(
    tmp_path, monkeypatch
):
    constructions = []

    def build_embedder(backend, device):
        constructions.append((backend, device))
        return FakeEmbedder()

    monkeypatch.setattr("engine.serve.get_embedder", build_embedder)
    app = create_app(
        baseline_root=tmp_path / "b",
        label_dir=tmp_path / "d",
        capture_root=tmp_path / "c",
    )
    client = TestClient(app, client=("127.0.0.1", 50000))

    response = client.post(
        "/session/start", json={"mode": "session", "tag": "not-ready"}
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "engine model is not ready"}
    assert constructions == []


def test_loopback_mode_rejects_non_loopback_clients(tmp_path):
    client, _ = _client(tmp_path, client_host="192.0.2.10")
    assert client.get("/health").status_code == 200
    r = client.post("/session/start", json={"mode": "session", "tag": "x"})
    assert r.status_code == 403


def test_remote_mode_requires_token_and_explicit_cors(monkeypatch):
    monkeypatch.setenv("EARSIGHT_REMOTE_ACCESS", "1")
    monkeypatch.delenv("EARSIGHT_API_TOKEN", raising=False)
    monkeypatch.delenv("EARSIGHT_CORS_ORIGINS", raising=False)
    with pytest.raises(RuntimeError, match="EARSIGHT_API_TOKEN"):
        ServiceSettings.from_env()

    monkeypatch.setenv("EARSIGHT_API_TOKEN", "t" * 32)
    with pytest.raises(RuntimeError, match="EARSIGHT_CORS_ORIGINS"):
        ServiceSettings.from_env()

    monkeypatch.setenv("EARSIGHT_CORS_ORIGINS", "https://app.example.test")
    monkeypatch.setenv("EARSIGHT_BIND_HOST", "0.0.0.0")
    with pytest.raises(RuntimeError, match="HTTPS gateway"):
        ServiceSettings.from_env()


def test_remote_auth_covers_every_non_health_route(tmp_path):
    token = "t" * 32
    settings = replace(
        ServiceSettings(),
        remote_access=True,
        api_token=token,
        cors_origins=("https://app.example.test",),
    )
    client, _ = _client(
        tmp_path,
        settings=settings,
        client_host="192.0.2.10",
    )
    routes = [
        ("GET", "/baselines/pump", None),
        ("POST", "/session/start", {}),
        ("POST", "/score", {}),
        ("POST", "/label", {}),
        ("POST", "/capture/start", {}),
        ("POST", "/capture/append", {}),
        ("POST", "/capture/stop", {}),
    ]

    assert client.get("/health").json() == {"status": "ok"}
    for method, path, body in routes:
        request = getattr(client, method.lower())
        kwargs = {} if body is None else {"json": body}
        missing = request(path, **kwargs)
        assert missing.status_code == 401, (method, path, missing.text)
        assert missing.headers["www-authenticate"] == "Bearer"

        wrong = request(
            path,
            headers={"Authorization": "Bearer wrong"},
            **kwargs,
        )
        assert wrong.status_code == 401, (method, path, wrong.text)

        authenticated = request(
            path,
            headers={"Authorization": f"Bearer {token}"},
            **kwargs,
        )
        assert authenticated.status_code != 401, (method, path, authenticated.text)


def test_docs_and_openapi_are_not_exposed(tmp_path):
    client, _ = _client(tmp_path)
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404


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

    # Baseline now persisted and visible only through the protected lookup.
    assert client.get("/baselines/m1").json() == {"exists": True}

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
    assert client1.get("/baselines/rig").json() == {"exists": True}

    # Second app over the SAME baseline root: no re-capture needed.
    client2, _ = _client(tmp_path)
    assert client2.get("/baselines/rig").json() == {"exists": True}

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


def test_protected_baseline_existence_replaces_health_disclosure(tmp_path):
    client, _ = _client(tmp_path)
    assert client.get("/baselines/missing").json() == {"exists": False}


def test_scoring_session_capacity_and_idle_expiry(tmp_path):
    clock = FakeClock()
    settings = replace(ServiceSettings(), max_scoring_sessions=1, session_idle_s=5.0)
    client, _ = _client(tmp_path, settings=settings, clock=clock)

    first = client.post("/session/start", json={"mode": "session", "tag": "one"})
    assert first.status_code == 200
    first_id = first.json()["session_id"]
    full = client.post("/session/start", json={"mode": "session", "tag": "two"})
    assert full.status_code == 429

    clock.advance(6.0)
    replacement = client.post(
        "/session/start", json={"mode": "session", "tag": "two"}
    )
    assert replacement.status_code == 200
    expired = client.post(
        "/score", json={"session_id": first_id, "pcm_b64": _pcm_b64(_noise(1))}
    )
    assert expired.status_code == 404


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


def test_score_decoded_pcm_and_request_body_limits(tmp_path):
    client, _ = _client(tmp_path)
    sid = client.post(
        "/session/start", json={"mode": "session", "tag": "m1", "rpm": None}
    ).json()["session_id"]

    exact = base64.b64encode(b"\0" * MAX_SCORE_PCM_BYTES).decode("ascii")
    assert client.post(
        "/score", json={"session_id": sid, "pcm_b64": exact}
    ).status_code == 200

    too_much = base64.b64encode(b"\0" * (MAX_SCORE_PCM_BYTES + 2)).decode("ascii")
    r = client.post("/score", json={"session_id": sid, "pcm_b64": too_much})
    assert r.status_code == 413

    r = client.post(
        "/score",
        content=b"x" * (MAX_REQUEST_BYTES + 1),
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 413


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


def test_label_accepts_field_recorder_condition_and_stats(tmp_path):
    client, _ = _client(tmp_path)
    audio = _noise(2 * SR, seed=12)
    body = {
        "session_id": None,
        "pcm_b64": _pcm_b64(audio),
        "machine_type": "compressor",
        "condition": "suspected issue",
        "contains_speech": True,
        "note": "north wall",
        "site_tag": "shop-7-comp-2",
        "client_recorded_at": "2026-06-11T20:00:00.000Z",
        "client_duration_s": 2.0,
        "client_peak_abs": 0.98,
        "client_clipped": True,
    }
    r = client.post("/label", json=body)
    assert r.status_code == 200, r.text

    from engine.paths import REPO_ROOT

    meta = json.loads((REPO_ROOT / r.json()["json_path"]).read_text())
    assert meta["machine_type"] == "compressor"
    assert meta["condition"] == "suspected issue"
    assert meta["site_tag"] == "shop-7-comp-2"
    assert meta["contains_speech"] is True
    assert meta["public_dataset_default_excluded"] is True
    assert meta["duration_s"] == pytest.approx(2.0)
    assert meta["client_duration_s"] == 2.0
    assert meta["client_peak_abs"] == 0.98
    assert meta["client_clipped"] is True
    assert meta["clipped"] is True


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


def test_cors_uses_an_explicit_allowlist(tmp_path):
    settings = replace(
        ServiceSettings(),
        cors_origins=("https://app.example.test",),
    )
    client, _ = _client(tmp_path, embedder=object(), settings=settings)

    r = client.get("/health", headers={"Origin": "https://app.example.test"})
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "https://app.example.test"

    preflight = client.options(
        "/score",
        headers={
            "Origin": "https://app.example.test",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert preflight.status_code == 200
    assert "POST" in preflight.headers.get("access-control-allow-methods", "")

    denied = client.options(
        "/score",
        headers={
            "Origin": "https://evil.example.test",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert denied.status_code == 400
    assert "access-control-allow-origin" not in denied.headers


def test_cors_rejects_wildcards():
    with pytest.raises(RuntimeError, match="wildcard"):
        replace(ServiceSettings(), cors_origins=("*",)).validate()

    with pytest.raises(RuntimeError, match="path"):
        replace(
            ServiceSettings(), cors_origins=("https://app.example.test/",)
        ).validate()

    for noncanonical in (
        "https://APP.example.test",
        "https://app.example.test:443",
    ):
        with pytest.raises(RuntimeError, match="canonical"):
            replace(ServiceSettings(), cors_origins=(noncanonical,)).validate()
