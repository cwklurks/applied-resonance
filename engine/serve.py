"""FastAPI scoring service: live mic chunks in, anomaly verdicts out.

Phone / glasses shells POST base64 PCM-16 mic chunks here. A session is
ephemeral (held in memory): it either captures ~30 s of baseline audio and then
listens, or loads a saved/library baseline and listens straight away. Baselines
persist on disk via :class:`engine.baseline.BaselineManager`.

Run with validated environment settings:

    uv run python -m engine.serve

The production embedder is constructed and warmed during application startup,
before the service becomes ready. ``create_app`` itself (and ``TestClient``
with an injected fake) stays cheap.
"""

import asyncio
import base64
import binascii
import ipaddress
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Callable, Literal
from urllib.parse import urlsplit
from uuid import uuid4

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import JSONResponse

from engine.baseline import BaselineManager
from engine.capture import CaptureCapacityError, CaptureLimitError, CaptureStore
from engine.embedder import get_embedder
from engine.labeling import write_labeled_clip
from engine.paths import REPO_ROOT
from engine.policy import AlertPolicy
from engine.stream import StreamScorer

logger = logging.getLogger(__name__)

SR = 16000
CAPTURE_S = 30.0
WINDOW_S = 3.0
HOP_S = 1.0
LABEL_SR = 16000

MAX_REQUEST_BYTES = 3 * 1024 * 1024
MAX_SCORE_PCM_BYTES = 2 * SR * 2
MAX_CAPTURE_CHUNK_BYTES = 2 * SR * 2
MAX_LABEL_PCM_BYTES = 60 * LABEL_SR * 2

DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)

_CAPTURE_SAMPLES = round(CAPTURE_S * SR)
_WINDOW_N = round(WINDOW_S * SR)
_HOP_N = round(HOP_S * SR)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be 0/1 or true/false")


def _validate_origin(origin: str, *, require_https: bool) -> str:
    if origin == "*":
        raise RuntimeError("CORS wildcard origins are not allowed")
    parsed = urlsplit(origin)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RuntimeError(f"invalid CORS origin: {origin!r}")
    if require_https and parsed.scheme != "https":
        raise RuntimeError("remote CORS origins must use https")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise RuntimeError(f"invalid CORS origin: {origin!r}")
    if parsed.path:
        raise RuntimeError(f"CORS origins may not contain a path: {origin!r}")
    try:
        port = parsed.port
    except ValueError as exc:
        raise RuntimeError(f"invalid CORS origin: {origin!r}") from exc
    if (parsed.scheme, port) in {("http", 80), ("https", 443)}:
        port = None
    hostname = parsed.hostname
    if ":" in hostname:
        hostname = f"[{hostname}]"
    canonical = f"{parsed.scheme}://{hostname}"
    if port is not None:
        canonical = f"{canonical}:{port}"
    if origin != canonical:
        raise RuntimeError(f"CORS origin must be canonical: {canonical!r}")
    return canonical


@dataclass(frozen=True)
class ServiceSettings:
    """Security and resource limits for one service process."""

    remote_access: bool = False
    api_token: str | None = field(default=None, repr=False)
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    bind_host: str = "127.0.0.1"
    max_request_bytes: int = MAX_REQUEST_BYTES
    max_score_pcm_bytes: int = MAX_SCORE_PCM_BYTES
    max_label_pcm_bytes: int = MAX_LABEL_PCM_BYTES
    max_capture_chunk_bytes: int = MAX_CAPTURE_CHUNK_BYTES
    max_scoring_sessions: int = 8
    session_idle_s: float = 120.0
    max_capture_sessions: int = 2
    capture_max_bytes: int = 16 * 1024 * 1024
    capture_max_duration_s: float = 300.0
    capture_idle_s: float = 30.0

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        remote = _env_flag("EARSIGHT_REMOTE_ACCESS")
        token = os.environ.get("EARSIGHT_API_TOKEN")
        cors_raw = os.environ.get("EARSIGHT_CORS_ORIGINS")
        if remote and not token:
            raise RuntimeError("EARSIGHT_API_TOKEN is required for remote access")
        if remote and not cors_raw:
            raise RuntimeError("EARSIGHT_CORS_ORIGINS is required for remote access")
        origins = (
            tuple(part.strip() for part in cors_raw.split(",") if part.strip())
            if cors_raw is not None
            else DEFAULT_CORS_ORIGINS
        )
        bind_host = os.environ.get("EARSIGHT_BIND_HOST", "127.0.0.1").strip()
        return cls(
            remote_access=remote,
            api_token=token,
            cors_origins=origins,
            bind_host=bind_host,
        ).validate()

    def validate(self) -> "ServiceSettings":
        if self.remote_access and (self.api_token is None or len(self.api_token) < 32):
            raise RuntimeError("EARSIGHT_API_TOKEN must contain at least 32 characters")
        if self.remote_access and not self.cors_origins:
            raise RuntimeError("EARSIGHT_CORS_ORIGINS is required for remote access")
        for origin in self.cors_origins:
            _validate_origin(origin, require_https=self.remote_access)
        try:
            if not ipaddress.ip_address(self.bind_host).is_loopback:
                raise RuntimeError(
                    "EARSIGHT_BIND_HOST must stay loopback; use a local HTTPS gateway"
                )
        except ValueError as exc:
            raise RuntimeError("EARSIGHT_BIND_HOST must be a loopback IP") from exc
        limits = (
            self.max_request_bytes,
            self.max_score_pcm_bytes,
            self.max_label_pcm_bytes,
            self.max_capture_chunk_bytes,
            self.max_scoring_sessions,
            self.session_idle_s,
            self.max_capture_sessions,
            self.capture_max_bytes,
            self.capture_max_duration_s,
            self.capture_idle_s,
        )
        if any(limit <= 0 for limit in limits):
            raise RuntimeError("service resource limits must be positive")
        return self


class RequestBodyLimitMiddleware:
    """Reject declared and streamed request bodies above a fixed byte bound."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                if int(declared) > self.max_bytes:
                    await self._reject(scope, receive, send)
                    return
            except ValueError:
                pass

        seen = 0

        async def limited_receive():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    raise _RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestBodyTooLarge:
            await self._reject(scope, receive, send)

    @staticmethod
    async def _reject(scope, receive, send):
        response = JSONResponse(
            {"detail": "request body too large"},
            status_code=413,
        )
        await response(scope, receive, send)


class _RequestBodyTooLarge(Exception):
    pass


class AccessControlMiddleware:
    """Keep health public and gate every other request before body parsing."""

    def __init__(self, app, settings: ServiceSettings):
        self.app = app
        self.settings = settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if scope.get("path") == "/health" or scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return

        if self.settings.remote_access:
            headers = {key.lower(): value for key, value in scope.get("headers", [])}
            supplied = headers.get(b"authorization", b"").decode("latin-1")
            prefix = "Bearer "
            expected = self.settings.api_token or ""
            valid = supplied.startswith(prefix) and secrets.compare_digest(
                supplied[len(prefix) :], expected
            )
            if not valid:
                response = JSONResponse(
                    {"detail": "invalid or missing bearer token"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                await response(scope, receive, send)
                return
        else:
            client = scope.get("client")
            try:
                is_loopback = bool(client) and ipaddress.ip_address(client[0]).is_loopback
            except ValueError:
                is_loopback = False
            if not is_loopback:
                response = JSONResponse(
                    {"detail": "remote access is disabled"},
                    status_code=403,
                )
                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)


# --------------------------------------------------------- request models ----


class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["session", "saved", "library"]
    tag: str = Field(max_length=64)
    rpm: float | None = Field(default=None, gt=0, le=1_000_000)


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=64)
    pcm_b64: str = Field(max_length=((MAX_SCORE_PCM_BYTES + 2) // 3) * 4)


class LabelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str | None = None
    pcm_b64: str = Field(max_length=((MAX_LABEL_PCM_BYTES + 2) // 3) * 4)
    machine_type: str = Field(min_length=1, max_length=64)
    condition: str = Field(default="", max_length=128)
    suspected_fault: str = Field(default="", max_length=128)
    contains_speech: bool = False
    note: str = Field(default="", max_length=1024)
    site_tag: str = Field(default="", max_length=64)
    client_recorded_at: str | None = Field(default=None, max_length=64)
    client_duration_s: float | None = Field(default=None, ge=0, le=60.0)
    client_peak_abs: float | None = Field(default=None, ge=0, le=1.0)
    client_clipped: bool | None = None


class CaptureStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tag: str = Field(max_length=64)


class CaptureAppendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capture_id: str = Field(min_length=1, max_length=64)
    pcm_b64: str = Field(max_length=((MAX_CAPTURE_CHUNK_BYTES + 2) // 3) * 4)


class CaptureStopRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capture_id: str = Field(min_length=1, max_length=64)


# -------------------------------------------------------------- session ----


@dataclass
class Session:
    mode: str
    tag: str
    rpm: float | None
    status: str  # "capturing" | "listening"
    scorer: StreamScorer | None = None
    capture_buffer: np.ndarray | None = None
    capture_samples: int = 0
    created_at: float = 0.0
    last_activity: float = 0.0
    lock: Lock = field(default_factory=Lock, repr=False)


class SessionCapacityError(RuntimeError):
    pass


class SessionRegistry:
    """Small monotonic-TTL registry for stateful scoring sessions."""

    def __init__(self, max_active: int, idle_s: float, clock: Callable[[], float]):
        self.max_active = max_active
        self.idle_s = idle_s
        self.clock = clock
        self._sessions: dict[str, Session] = {}
        self._lock = Lock()

    def _expire_locked(self, now: float) -> None:
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.last_activity > self.idle_s
        ]
        for session_id in expired:
            del self._sessions[session_id]

    def add(self, session: Session) -> str:
        with self._lock:
            now = self.clock()
            self._expire_locked(now)
            if len(self._sessions) >= self.max_active:
                raise SessionCapacityError("active scoring session limit reached")
            session_id = uuid4().hex
            session.created_at = now
            session.last_activity = now
            self._sessions[session_id] = session
            return session_id

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            now = self.clock()
            self._expire_locked(now)
            session = self._sessions.get(session_id)
            if session is not None:
                session.last_activity = now
            return session

    def cleanup_expired(self) -> None:
        with self._lock:
            self._expire_locked(self.clock())


# ---------------------------------------------------------------- codec ----


def _decode_pcm(pcm_b64: str, *, max_bytes: int) -> np.ndarray:
    """base64 -> int16 LE -> float32 in [-1, 1). 400 on bad input."""
    raw = _decode_pcm_bytes(pcm_b64, max_bytes=max_bytes)
    arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return arr


def _decode_pcm_bytes(pcm_b64: str, *, max_bytes: int) -> bytes:
    """base64 -> raw int16 LE bytes. 400 on bad input."""
    if len(pcm_b64) > ((max_bytes + 2) // 3) * 4:
        raise HTTPException(status_code=413, detail="decoded pcm limit exceeded")
    try:
        raw = base64.b64decode(pcm_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="invalid base64 pcm") from exc
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail="decoded pcm limit exceeded")
    if len(raw) % 2 != 0:
        raise HTTPException(
            status_code=400, detail="pcm byte count must be even (int16 LE)"
        )
    return raw


def _slice_windows(audio: np.ndarray) -> list[np.ndarray]:
    """3 s windows hopped at 1 s over ``audio`` (28 windows from 30 s)."""
    windows: list[np.ndarray] = []
    start = 0
    while start + _WINDOW_N <= audio.shape[0]:
        windows.append(audio[start : start + _WINDOW_N].copy())
        start += _HOP_N
    return windows


def _build_ready_embedder():
    """Construct the configured backend and materialize it on one bounded window."""
    backend = os.environ.get("EARSIGHT_BACKEND", "torch")
    device = os.environ.get("EARSIGHT_DEVICE")
    logger.info("warming embedder backend=%s device=%s", backend, device)
    started = time.monotonic()
    embedder = get_embedder(backend, device)
    warmup = np.asarray(
        embedder.embed([np.zeros(_WINDOW_N, dtype=np.float32)]),
        dtype=np.float32,
    )
    if warmup.shape != (1, 2048) or not np.isfinite(warmup).all():
        raise RuntimeError(
            "embedder warmup must return one finite 2048-dimensional embedding"
        )
    logger.info("embedder ready after %.3fs", time.monotonic() - started)
    return embedder


# ------------------------------------------------------------- the app ----


def create_app(
    embedder=None,
    baseline_root: Path | None = None,
    label_dir: Path | None = None,
    capture_root: Path | None = None,
    settings: ServiceSettings | None = None,
    clock: Callable[[], float] | None = None,
) -> FastAPI:
    """Build the scoring app. ``embedder`` may be injected (tests); otherwise it
    is warmed during lifespan startup from ``EARSIGHT_BACKEND``/``EARSIGHT_DEVICE``.
    """
    settings = (settings or ServiceSettings.from_env()).validate()
    clock = clock or time.monotonic
    session_registry = SessionRegistry(
        settings.max_scoring_sessions,
        settings.session_idle_s,
        clock,
    )
    capture_store = CaptureStore(
        root=capture_root,
        max_active=settings.max_capture_sessions,
        max_bytes=settings.capture_max_bytes,
        max_duration_s=settings.capture_max_duration_s,
        idle_s=settings.capture_idle_s,
        clock=clock,
    )

    @asynccontextmanager
    async def lifespan(_app):
        if embedder is None:
            _app.state._embedder = await asyncio.to_thread(_build_ready_embedder)

        interval = max(0.1, min(settings.session_idle_s, settings.capture_idle_s) / 2)

        async def reap_idle():
            while True:
                await asyncio.sleep(interval)
                session_registry.cleanup_expired()
                capture_store.cleanup_expired()

        reaper = asyncio.create_task(reap_idle())
        try:
            yield
        finally:
            reaper.cancel()
            with suppress(asyncio.CancelledError):
                await reaper
            capture_store.close()

    app = FastAPI(
        title="earsight scoring service",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_bytes=settings.max_request_bytes,
    )
    app.add_middleware(AccessControlMiddleware, settings=settings)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )
    app.state.settings = settings
    app.state.sessions = session_registry
    app.state.manager = BaselineManager(root=baseline_root)
    app.state.label_dir = label_dir
    app.state.captures = capture_store
    app.state._embedder = embedder

    def get_embedder_dep(request: Request):
        """Return only a lifespan-prepared or explicitly injected embedder."""
        ready = request.app.state._embedder
        if ready is None:
            raise HTTPException(status_code=503, detail="engine model is not ready")
        return ready

    # ------------------------------------------------------------ health ----

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/baselines/{tag:path}")
    def baseline_exists(tag: str, request: Request):
        if len(tag) > 64:
            raise HTTPException(status_code=422, detail="tag too long")
        return {"exists": tag in request.app.state.manager.list_tags()}

    # ----------------------------------------------------- session/start ----

    @app.post("/session/start")
    def session_start(req: StartRequest, request: Request, embedder=Depends(get_embedder_dep)):
        state = request.app.state

        if req.mode == "session":
            session = Session(
                mode=req.mode,
                tag=req.tag,
                rpm=req.rpm,
                status="capturing",
                capture_buffer=np.empty(_CAPTURE_SAMPLES, dtype=np.float32),
            )
            try:
                session_id = state.sessions.add(session)
            except SessionCapacityError as exc:
                raise HTTPException(status_code=429, detail=str(exc)) from exc
            return {
                "session_id": session_id,
                "state": "CAPTURING_BASELINE",
                "capture_remaining_s": CAPTURE_S,
            }

        if req.mode == "saved":
            try:
                baseline = state.manager.load(req.tag)
            except FileNotFoundError as exc:
                raise HTTPException(
                    status_code=404, detail=f"no saved baseline for tag {req.tag!r}"
                ) from exc
            scorer = StreamScorer(baseline, embedder, AlertPolicy(), rpm=baseline.rpm)
            try:
                session_id = state.sessions.add(
                    Session(
                        mode=req.mode,
                        tag=req.tag,
                        rpm=baseline.rpm,
                        status="listening",
                        scorer=scorer,
                    )
                )
            except SessionCapacityError as exc:
                raise HTTPException(status_code=429, detail=str(exc)) from exc
            return {"session_id": session_id, "state": "LISTENING"}

        # mode == "library"
        try:
            state.manager.load_library(req.tag)
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        # Unreachable today (load_library always raises), kept for when it lands.
        return {"state": "LISTENING"}

    # ------------------------------------------------------------- score ----

    @app.post("/score")
    def score(req: ScoreRequest, request: Request, embedder=Depends(get_embedder_dep)):
        state = request.app.state
        session = state.sessions.get(req.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session_id")

        chunk = _decode_pcm(
            req.pcm_b64,
            max_bytes=state.settings.max_score_pcm_bytes,
        )

        with session.lock:
            if session.status == "capturing":
                return _handle_capture(state, session, chunk, embedder)
            return _handle_listen(session, chunk)

    # ------------------------------------------------------------- label ----

    @app.post("/label")
    def label(req: LabelRequest, request: Request):
        state = request.app.state
        audio = _decode_pcm(
            req.pcm_b64,
            max_bytes=state.settings.max_label_pcm_bytes,
        )
        duration_s = float(audio.shape[0] / LABEL_SR)
        peak_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
        clipped = bool(peak_abs >= 0.999 or req.client_clipped is True)
        meta = {
            "machine_type": req.machine_type,
            "condition": req.condition,
            "suspected_fault": req.suspected_fault,
            "contains_speech": req.contains_speech,
            "note": req.note,
            "site_tag": req.site_tag,
            "session_id": req.session_id,
            "duration_s": duration_s,
            "peak_abs": peak_abs,
            "clipped": clipped,
            "client_recorded_at": req.client_recorded_at,
            "client_duration_s": req.client_duration_s,
            "client_peak_abs": req.client_peak_abs,
            "client_clipped": req.client_clipped,
            "public_dataset_default_excluded": bool(req.contains_speech),
        }
        wav_path, json_path = write_labeled_clip(
            audio, LABEL_SR, meta, out_dir=state.label_dir
        )
        return {
            "wav_path": _repo_relative(wav_path),
            "json_path": _repo_relative(json_path),
        }

    # ----------------------------------------------------------- capture ----

    @app.post("/capture/start")
    def capture_start(req: CaptureStartRequest, request: Request):
        try:
            capture_id = request.app.state.captures.start(req.tag)
        except CaptureCapacityError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        return {"capture_id": capture_id}

    @app.post("/capture/append")
    def capture_append(req: CaptureAppendRequest, request: Request):
        pcm = _decode_pcm_bytes(
            req.pcm_b64,
            max_bytes=request.app.state.settings.max_capture_chunk_bytes,
        )
        try:
            bytes_total = request.app.state.captures.append(req.capture_id, pcm)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown capture_id") from exc
        except CaptureLimitError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "bytes_total": bytes_total,
            "seconds_total": bytes_total / 2 / LABEL_SR,
        }

    @app.post("/capture/stop")
    def capture_stop(req: CaptureStopRequest, request: Request):
        try:
            wav_path, duration_s = request.app.state.captures.stop(req.capture_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown capture_id") from exc
        return {"wav_path": _repo_relative(wav_path), "duration_s": duration_s}

    return app


def _repo_relative(path: Path) -> str:
    """Path as a repo-relative string when under REPO_ROOT, else absolute."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------- score helpers ----


def _handle_capture(state, session: Session, chunk: np.ndarray, embedder) -> dict:
    """Buffer a chunk during baseline capture; fit + flip to listening at 30 s."""
    remaining_samples = _CAPTURE_SAMPLES - session.capture_samples
    accepted = min(remaining_samples, chunk.shape[0])
    if accepted:
        start = session.capture_samples
        end = start + accepted
        session.capture_buffer[start:end] = chunk[:accepted]
        session.capture_samples = end

    if session.capture_samples < _CAPTURE_SAMPLES:
        remaining = (_CAPTURE_SAMPLES - session.capture_samples) / SR
        return _capturing_response(round(remaining, 6))

    audio = session.capture_buffer
    windows = _slice_windows(audio)
    baseline = state.manager.fit(windows, session.tag, embedder, rpm=session.rpm)
    state.manager.save(baseline)

    session.scorer = StreamScorer(
        baseline, embedder, AlertPolicy(), rpm=session.rpm
    )
    session.status = "listening"
    session.capture_buffer = None

    return {
        "state": "LISTENING",
        "score": None,
        "percentile": None,
        "evidence_line": None,
        "capture_remaining_s": 0.0,
    }


def _capturing_response(remaining_s: float) -> dict:
    return {
        "state": "CAPTURING_BASELINE",
        "score": None,
        "percentile": None,
        "evidence_line": None,
        "capture_remaining_s": remaining_s,
    }


def _handle_listen(session: Session, chunk: np.ndarray) -> dict:
    """Score a chunk against the session's baseline; report the last window."""
    results = session.scorer.process(chunk)
    if not results:
        return {
            "state": session.scorer.policy.state.name,
            "score": None,
            "percentile": None,
            "evidence_line": None,
        }
    result = results[-1]
    evidence_line = (result.evidence or {}).get("hud")
    return {
        "state": result.state.name,
        "score": result.ema,
        "percentile": result.percentile,
        "evidence_line": evidence_line,
    }


def main() -> None:
    """Run with a validated loopback-by-default service configuration."""
    import uvicorn

    settings = ServiceSettings.from_env()
    try:
        port = int(os.environ.get("EARSIGHT_PORT", "8000"))
    except ValueError as exc:
        raise RuntimeError("EARSIGHT_PORT must be an integer") from exc
    uvicorn.run(
        "engine.serve:create_app",
        factory=True,
        host=settings.bind_host,
        port=port,
        proxy_headers=False,
    )


if __name__ == "__main__":
    main()
