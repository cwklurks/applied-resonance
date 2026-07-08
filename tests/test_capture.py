"""Tests for continuous raw PCM capture endpoints."""

import base64
import re
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from engine.capture import CaptureStore
from engine.paths import REPO_ROOT
from engine.serve import create_app

SR = 16000


def _client(tmp_path):
    app = create_app(
        embedder=object(),
        baseline_root=tmp_path / "baselines",
        capture_root=tmp_path / "captures",
        label_dir=tmp_path / "labels",
    )
    return TestClient(app)


def _pcm_b64(samples: np.ndarray) -> str:
    i16 = np.asarray(samples, dtype="<i2")
    return base64.b64encode(i16.tobytes()).decode("ascii")


def _resolve_response_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def test_capture_roundtrip_writes_exact_wav(tmp_path):
    client = _client(tmp_path)
    chunks = [
        np.array([0, 1, -1, 32767], dtype="<i2"),
        np.array([-32768, 42, -42], dtype="<i2"),
        np.arange(100, dtype="<i2"),
    ]
    expected_bytes = b"".join(chunk.tobytes() for chunk in chunks)

    r = client.post("/capture/start", json={"tag": "pump.A"})
    assert r.status_code == 200, r.text
    capture_id = r.json()["capture_id"]

    bytes_total = 0
    samples_total = 0
    for chunk in chunks:
        bytes_total += chunk.nbytes
        samples_total += chunk.size
        r = client.post(
            "/capture/append",
            json={"capture_id": capture_id, "pcm_b64": _pcm_b64(chunk)},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["bytes_total"] == bytes_total
        assert body["seconds_total"] == pytest.approx(samples_total / SR)

    r = client.post("/capture/stop", json={"capture_id": capture_id})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["duration_s"] == pytest.approx(samples_total / SR)

    wav_path = _resolve_response_path(body["wav_path"])
    assert wav_path.exists()
    with wave.open(str(wav_path), "rb") as wav:
        assert wav.getframerate() == SR
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getnframes() == samples_total
        assert wav.readframes(samples_total) == expected_bytes


def test_capture_http_errors_and_stop_twice(tmp_path):
    client = _client(tmp_path)

    r = client.post(
        "/capture/append",
        json={
            "capture_id": "missing",
            "pcm_b64": _pcm_b64(np.array([1], dtype="<i2")),
        },
    )
    assert r.status_code == 404

    r = client.post("/capture/stop", json={"capture_id": "missing"})
    assert r.status_code == 404

    r = client.post(
        "/capture/append",
        json={"capture_id": "missing", "pcm_b64": "not!!base64!!"},
    )
    assert r.status_code == 400

    capture_id = client.post("/capture/start", json={"tag": "err"}).json()[
        "capture_id"
    ]
    odd = base64.b64encode(b"\x01").decode("ascii")
    r = client.post(
        "/capture/append", json={"capture_id": capture_id, "pcm_b64": odd}
    )
    assert r.status_code == 400

    r = client.post(
        "/capture/append",
        json={
            "capture_id": capture_id,
            "pcm_b64": _pcm_b64(np.array([7], dtype="<i2")),
        },
    )
    assert r.status_code == 200, r.text
    r = client.post("/capture/stop", json={"capture_id": capture_id})
    assert r.status_code == 200, r.text
    r = client.post("/capture/stop", json={"capture_id": capture_id})
    assert r.status_code == 404


def test_capture_tag_sanitization_stays_in_capture_root(tmp_path):
    client = _client(tmp_path)
    capture_id = client.post("/capture/start", json={"tag": "../evil tag"}).json()[
        "capture_id"
    ]
    r = client.post(
        "/capture/append",
        json={
            "capture_id": capture_id,
            "pcm_b64": _pcm_b64(np.array([1], dtype="<i2")),
        },
    )
    assert r.status_code == 200, r.text
    r = client.post("/capture/stop", json={"capture_id": capture_id})
    assert r.status_code == 200, r.text

    wav_path = _resolve_response_path(r.json()["wav_path"])
    assert wav_path.parent == tmp_path / "captures"
    assert " " not in wav_path.name
    assert "/" not in wav_path.name
    assert re.fullmatch(r"\d{8}_\d{6}_[A-Za-z0-9._-]+\.wav", wav_path.name)


def test_capture_store_unit_roundtrip_and_validation(tmp_path):
    store = CaptureStore(root=tmp_path)
    capture_id = store.start("")
    raw = np.array([10, -10, 0, 32767], dtype="<i2").tobytes()

    assert store.append(capture_id, raw[:4]) == 4
    assert store.append(capture_id, raw[4:]) == len(raw)
    with pytest.raises(ValueError):
        store.append(capture_id, b"\x00")

    wav_path, duration_s = store.stop(capture_id)
    assert wav_path.exists()
    assert wav_path.parent == tmp_path
    assert wav_path.name.endswith("_capture.wav")
    assert duration_s == pytest.approx(4 / SR)
    with wave.open(str(wav_path), "rb") as wav:
        assert wav.getframerate() == SR
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.readframes(4) == raw

    with pytest.raises(KeyError):
        store.stop(capture_id)
