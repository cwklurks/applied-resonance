"""In-memory raw PCM capture sessions finalized as WAV files."""

import logging
import re
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import BinaryIO
from uuid import uuid4

from engine.paths import DATA_DIR

logger = logging.getLogger(__name__)

SR = 16000
SAMPLE_WIDTH_BYTES = 2
CHANNELS = 1

_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


@dataclass
class _CaptureSession:
    tag: str
    started_at: str
    bytes: int
    raw_path: Path
    raw_file: BinaryIO


def _sanitize_tag(tag: str) -> str:
    """Reduce ``tag`` to [A-Za-z0-9._-]; fall back to 'capture'."""
    safe = _SAFE_RE.sub("", tag or "")
    return safe or "capture"


def _unique_wav_path(root: Path, base: str) -> Path:
    candidate = base
    n = 1
    while (root / f"{candidate}.wav").exists():
        candidate = f"{base}_{n}"
        n += 1
    return root / f"{candidate}.wav"


class CaptureStore:
    """Manage ephemeral raw PCM capture sessions backed by on-disk raw files."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else DATA_DIR / "captures"
        self._sessions: dict[str, _CaptureSession] = {}
        self._lock = Lock()

    def start(self, tag: str) -> str:
        capture_id = uuid4().hex
        safe_tag = _sanitize_tag(tag)
        self.root.mkdir(parents=True, exist_ok=True)
        raw_path = self.root / f"{capture_id}.raw"
        raw_file = raw_path.open("ab")
        started_at = datetime.now(timezone.utc).isoformat()

        with self._lock:
            self._sessions[capture_id] = _CaptureSession(
                tag=safe_tag,
                started_at=started_at,
                bytes=0,
                raw_path=raw_path,
                raw_file=raw_file,
            )
        logger.info("started capture %s tag=%s", capture_id, safe_tag)
        return capture_id

    def append(self, capture_id: str, pcm: bytes) -> int:
        if len(pcm) % SAMPLE_WIDTH_BYTES != 0:
            raise ValueError("pcm byte count must be even (int16 LE)")

        with self._lock:
            session = self._sessions[capture_id]
            session.raw_file.write(pcm)
            session.raw_file.flush()
            session.bytes += len(pcm)
            return session.bytes

    def stop(self, capture_id: str) -> tuple[Path, float]:
        with self._lock:
            session = self._sessions[capture_id]
            session.raw_file.flush()
            session.raw_file.close()

            started = datetime.fromisoformat(session.started_at)
            base = f"{started.strftime('%Y%m%d_%H%M%S')}_{session.tag}"
            wav_path = _unique_wav_path(self.root, base)
            duration_s = session.bytes / SAMPLE_WIDTH_BYTES / SR

            with session.raw_path.open("rb") as raw_file:
                raw = raw_file.read()
            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(CHANNELS)
                wav.setsampwidth(SAMPLE_WIDTH_BYTES)
                wav.setframerate(SR)
                wav.writeframes(raw)

            session.raw_path.unlink()
            del self._sessions[capture_id]

        logger.info("stopped capture %s wrote %s", capture_id, wav_path)
        return wav_path, duration_s
