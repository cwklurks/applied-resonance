"""Bounded raw PCM capture sessions finalized to WAV files incrementally."""

import logging
import os
import re
import time
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import BinaryIO, Callable
from uuid import uuid4

from engine.paths import DATA_DIR

logger = logging.getLogger(__name__)

SR = 16000
SAMPLE_WIDTH_BYTES = 2
CHANNELS = 1
COPY_CHUNK_BYTES = 64 * 1024
MAX_TAG_CHARS = 48

_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


@dataclass
class _CaptureSession:
    tag: str
    started_at: str
    bytes: int
    raw_path: Path
    raw_file: BinaryIO
    last_activity: float


class CaptureCapacityError(RuntimeError):
    """Raised when the active capture-session bound has been reached."""


class CaptureLimitError(ValueError):
    """Raised before an append would exceed a capture byte/duration bound."""


def _sanitize_tag(tag: str) -> str:
    """Reduce ``tag`` to [A-Za-z0-9._-]; fall back to 'capture'."""
    safe = _SAFE_RE.sub("", tag or "")[:MAX_TAG_CHARS]
    return safe or "capture"


def _copy_pcm_frames(raw_file: BinaryIO, wav_file) -> None:
    """Copy PCM in fixed chunks so finalization memory stays constant."""
    while True:
        chunk = raw_file.read(COPY_CHUNK_BYTES)
        if not chunk:
            return
        wav_file.writeframesraw(chunk)


class CaptureStore:
    """Manage ephemeral raw PCM capture sessions backed by on-disk raw files."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        max_active: int = 2,
        max_bytes: int = 16 * 1024 * 1024,
        max_duration_s: float = 300.0,
        idle_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        if max_active < 1 or max_bytes < SAMPLE_WIDTH_BYTES:
            raise ValueError("capture bounds must be positive")
        if max_duration_s <= 0 or idle_s <= 0:
            raise ValueError("capture time bounds must be positive")
        self.root = Path(root) if root is not None else DATA_DIR / "captures"
        self.max_active = max_active
        self.max_bytes = max_bytes
        self.max_duration_s = max_duration_s
        self.idle_s = idle_s
        self._clock = clock
        self._sessions: dict[str, _CaptureSession] = {}
        self._lock = Lock()
        self._cleanup_orphans()

    def _cleanup_orphans(self) -> None:
        """Raw/partial files cannot be resumed after a process restart."""
        if not self.root.exists():
            return
        for pattern in ("*.raw", "*.wav.part"):
            for path in self.root.glob(pattern):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

    def _expire_locked(self, now: float) -> int:
        expired = [
            capture_id
            for capture_id, session in self._sessions.items()
            if now - session.last_activity > self.idle_s
        ]
        for capture_id in expired:
            session = self._sessions.pop(capture_id)
            try:
                session.raw_file.close()
            finally:
                session.raw_path.unlink(missing_ok=True)
            logger.info("expired capture %s", capture_id)
        return len(expired)

    def cleanup_expired(self) -> int:
        with self._lock:
            return self._expire_locked(self._clock())

    def start(self, tag: str) -> str:
        with self._lock:
            now = self._clock()
            self._expire_locked(now)
            if len(self._sessions) >= self.max_active:
                raise CaptureCapacityError("active capture session limit reached")

            capture_id = uuid4().hex
            safe_tag = _sanitize_tag(tag)
            self.root.mkdir(parents=True, exist_ok=True)
            raw_path = self.root / f"{capture_id}.raw"
            raw_file = raw_path.open("xb")
            started_at = datetime.now(timezone.utc).isoformat()
            self._sessions[capture_id] = _CaptureSession(
                tag=safe_tag,
                started_at=started_at,
                bytes=0,
                raw_path=raw_path,
                raw_file=raw_file,
                last_activity=now,
            )
        logger.info("started capture %s tag=%s", capture_id, safe_tag)
        return capture_id

    def append(self, capture_id: str, pcm: bytes) -> int:
        if len(pcm) % SAMPLE_WIDTH_BYTES != 0:
            raise ValueError("pcm byte count must be even (int16 LE)")

        with self._lock:
            now = self._clock()
            self._expire_locked(now)
            session = self._sessions[capture_id]
            next_bytes = session.bytes + len(pcm)
            if next_bytes > self.max_bytes:
                raise CaptureLimitError("capture byte limit exceeded")
            if next_bytes / SAMPLE_WIDTH_BYTES / SR > self.max_duration_s:
                raise CaptureLimitError("capture duration limit exceeded")
            session.raw_file.write(pcm)
            session.raw_file.flush()
            session.bytes = next_bytes
            session.last_activity = now
            return session.bytes

    def stop(self, capture_id: str) -> tuple[Path, float]:
        with self._lock:
            self._expire_locked(self._clock())
            session = self._sessions.pop(capture_id)
            session.raw_file.flush()
            session.raw_file.close()

            started = datetime.fromisoformat(session.started_at)
            base = (
                f"{started.strftime('%Y%m%d_%H%M%S')}_"
                f"{session.tag}_{capture_id[:8]}"
            )
            wav_path = self.root / f"{base}.wav"
            duration_s = session.bytes / SAMPLE_WIDTH_BYTES / SR

        part_path = wav_path.with_suffix(".wav.part")
        try:
            with session.raw_path.open("rb") as raw_file:
                with wave.open(str(part_path), "wb") as wav:
                    wav.setnchannels(CHANNELS)
                    wav.setsampwidth(SAMPLE_WIDTH_BYTES)
                    wav.setframerate(SR)
                    _copy_pcm_frames(raw_file, wav)
            os.replace(part_path, wav_path)
        finally:
            part_path.unlink(missing_ok=True)
            session.raw_path.unlink(missing_ok=True)

        logger.info("stopped capture %s wrote %s", capture_id, wav_path)
        return wav_path, duration_s

    def close(self) -> None:
        """Close and remove every unfinished raw file (application shutdown)."""
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            try:
                session.raw_file.close()
            finally:
                session.raw_path.unlink(missing_ok=True)
