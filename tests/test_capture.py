"""Tests for continuous raw PCM capture endpoints."""

import base64
import io
import re
import time
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from engine.capture import (
    COPY_CHUNK_BYTES,
    CaptureCapacityError,
    CaptureLimitError,
    CaptureStore,
    _copy_pcm_frames,
)
from engine.paths import REPO_ROOT
from engine.serve import ServiceSettings, create_app

SR = 16000


def _client(tmp_path, *, settings=None, clock=None):
    app = create_app(
        embedder=object(),
        baseline_root=tmp_path / "baselines",
        capture_root=tmp_path / "captures",
        label_dir=tmp_path / "labels",
        settings=settings,
        clock=clock,
    )
    return TestClient(app, client=("127.0.0.1", 50000))


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
    assert re.fullmatch(
        r"\d{8}_\d{6}_[A-Za-z0-9._-]+_[0-9a-f]{8}\.wav", wav_path.name
    )


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
    assert re.fullmatch(r"\d{8}_\d{6}_capture_[0-9a-f]{8}\.wav", wav_path.name)
    assert duration_s == pytest.approx(4 / SR)
    with wave.open(str(wav_path), "rb") as wav:
        assert wav.getframerate() == SR
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.readframes(4) == raw

    with pytest.raises(KeyError):
        store.stop(capture_id)


class FakeClock:
    def __init__(self):
        self.value = 50.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def test_capture_store_bounds_active_bytes_duration_and_idle(tmp_path):
    clock = FakeClock()
    store = CaptureStore(
        root=tmp_path,
        max_active=1,
        max_bytes=8,
        max_duration_s=1.0,
        idle_s=5.0,
        clock=clock,
    )
    first = store.start("one")
    raw_path = tmp_path / f"{first}.raw"
    with pytest.raises(CaptureCapacityError):
        store.start("two")

    assert store.append(first, b"\0" * 8) == 8
    with pytest.raises(CaptureLimitError, match="byte"):
        store.append(first, b"\0\0")
    assert raw_path.stat().st_size == 8

    clock.advance(6.0)
    second = store.start("two")
    assert second != first
    assert not raw_path.exists()
    with pytest.raises(KeyError):
        store.append(first, b"\0\0")

    duration_store = CaptureStore(
        root=tmp_path / "duration",
        max_active=1,
        max_bytes=100_000,
        max_duration_s=0.000125,
        idle_s=5.0,
        clock=clock,
    )
    capture_id = duration_store.start("duration")
    assert duration_store.append(capture_id, b"\0" * 4) == 4
    with pytest.raises(CaptureLimitError, match="duration"):
        duration_store.append(capture_id, b"\0\0")


def test_capture_api_maps_capacity_and_size_limits(tmp_path):
    settings = replace(
        ServiceSettings(),
        max_capture_sessions=1,
        capture_max_bytes=4,
        capture_max_duration_s=10.0,
    )
    client = _client(tmp_path, settings=settings)
    capture_id = client.post("/capture/start", json={"tag": "one"}).json()[
        "capture_id"
    ]
    assert client.post("/capture/start", json={"tag": "two"}).status_code == 429
    assert client.post(
        "/capture/append",
        json={"capture_id": capture_id, "pcm_b64": _pcm_b64(np.array([1, 2], dtype="<i2"))},
    ).status_code == 200
    assert client.post(
        "/capture/append",
        json={"capture_id": capture_id, "pcm_b64": _pcm_b64(np.array([3], dtype="<i2"))},
    ).status_code == 413


def test_capture_store_cleans_orphan_raw_and_part_files_on_startup(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    raw = tmp_path / "abandoned.raw"
    part = tmp_path / "abandoned.wav.part"
    raw.write_bytes(b"raw")
    part.write_bytes(b"partial")

    CaptureStore(root=tmp_path)

    assert not raw.exists()
    assert not part.exists()


def test_wav_copy_reads_in_bounded_chunks():
    payload = b"x" * (COPY_CHUNK_BYTES * 2 + 7)

    class GuardedReader(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= COPY_CHUNK_BYTES
            return super().read(size)

    class Writer:
        def __init__(self):
            self.parts = []

        def writeframesraw(self, data):
            self.parts.append(data)

    writer = Writer()
    _copy_pcm_frames(GuardedReader(payload), writer)
    assert b"".join(writer.parts) == payload
    assert len(writer.parts) == 3


def test_app_reaps_idle_capture_and_cleans_live_raw_on_shutdown(tmp_path):
    settings = replace(ServiceSettings(), capture_idle_s=0.05)
    capture_root = tmp_path / "captures"
    app = create_app(
        embedder=object(),
        baseline_root=tmp_path / "baselines",
        capture_root=capture_root,
        label_dir=tmp_path / "labels",
        settings=settings,
    )

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        first = client.post("/capture/start", json={"tag": "idle"}).json()[
            "capture_id"
        ]
        deadline = time.monotonic() + 1.0
        while list(capture_root.glob("*.raw")) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not list(capture_root.glob("*.raw"))
        assert client.post(
            "/capture/append",
            json={"capture_id": first, "pcm_b64": _pcm_b64(np.array([1], dtype="<i2"))},
        ).status_code == 404

        client.post("/capture/start", json={"tag": "shutdown"})
        assert list(capture_root.glob("*.raw"))

    assert not list(capture_root.glob("*.raw"))
