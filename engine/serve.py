"""FastAPI scoring service: live mic chunks in, anomaly verdicts out.

Phone / glasses shells POST base64 PCM-16 mic chunks here. A session is
ephemeral (held in memory): it either captures ~30 s of baseline audio and then
listens, or loads a saved/library baseline and listens straight away. Baselines
persist on disk via :class:`engine.baseline.BaselineManager`.

Run (production):

    uv run uvicorn engine.serve:create_app --factory

The embedder is constructed lazily on the first request that needs it, so
``create_app`` (and ``TestClient`` with an injected fake) stays cheap and never
triggers a model load just to answer ``/health``.
"""

import base64
import binascii
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine.baseline import BaselineManager
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

_CAPTURE_SAMPLES = round(CAPTURE_S * SR)
_WINDOW_N = round(WINDOW_S * SR)
_HOP_N = round(HOP_S * SR)


# --------------------------------------------------------- request models ----


class StartRequest(BaseModel):
    mode: Literal["session", "saved", "library"]
    tag: str
    rpm: float | None = None


class ScoreRequest(BaseModel):
    session_id: str
    pcm_b64: str


class LabelRequest(BaseModel):
    session_id: str | None = None
    pcm_b64: str
    machine_type: str
    suspected_fault: str = ""
    contains_speech: bool = False
    note: str = ""
    site_tag: str = ""


# -------------------------------------------------------------- session ----


@dataclass
class Session:
    mode: str
    tag: str
    rpm: float | None
    status: str  # "capturing" | "listening"
    scorer: StreamScorer | None = None
    capture_chunks: list[np.ndarray] = field(default_factory=list)
    capture_samples: int = 0
    created_at: float = 0.0


# ---------------------------------------------------------------- codec ----


def _decode_pcm(pcm_b64: str) -> np.ndarray:
    """base64 -> int16 LE -> float32 in [-1, 1). 400 on bad input."""
    try:
        raw = base64.b64decode(pcm_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="invalid base64 pcm") from exc
    if len(raw) % 2 != 0:
        raise HTTPException(
            status_code=400, detail="pcm byte count must be even (int16 LE)"
        )
    arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return arr


def _slice_windows(audio: np.ndarray) -> list[np.ndarray]:
    """3 s windows hopped at 1 s over ``audio`` (28 windows from 30 s)."""
    windows: list[np.ndarray] = []
    start = 0
    while start + _WINDOW_N <= audio.shape[0]:
        windows.append(audio[start : start + _WINDOW_N].copy())
        start += _HOP_N
    return windows


# ------------------------------------------------------------- the app ----


def create_app(
    embedder=None,
    baseline_root: Path | None = None,
    label_dir: Path | None = None,
) -> FastAPI:
    """Build the scoring app. ``embedder`` may be injected (tests); otherwise it
    is constructed lazily on first use from ``EARSIGHT_BACKEND``/``EARSIGHT_DEVICE``.
    """
    app = FastAPI(title="earsight scoring service")
    # Browser-based shells (Even Hub WebView, desktop dev servers) call from a
    # different origin. Default is permissive for local dev; pin origins in
    # production via EARSIGHT_CORS_ORIGINS (comma-separated).
    origins = os.environ.get("EARSIGHT_CORS_ORIGINS", "*")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins.split(",")],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.sessions = {}
    app.state.manager = BaselineManager(root=baseline_root)
    app.state.label_dir = label_dir
    app.state._embedder = embedder

    def get_embedder_dep(request: Request):
        """Lazily build the embedder on first request that needs it."""
        if request.app.state._embedder is None:
            backend = os.environ.get("EARSIGHT_BACKEND", "torch")
            device = os.environ.get("EARSIGHT_DEVICE")
            logger.info("constructing embedder backend=%s device=%s", backend, device)
            request.app.state._embedder = get_embedder(backend, device)
        return request.app.state._embedder

    # ------------------------------------------------------------ health ----

    @app.get("/health")
    def health(request: Request):
        state = request.app.state
        return {
            "status": "ok",
            "backend": os.environ.get("EARSIGHT_BACKEND", "torch"),
            "baselines": state.manager.list_tags(),
            "sessions": len(state.sessions),
        }

    # ----------------------------------------------------- session/start ----

    @app.post("/session/start")
    def session_start(req: StartRequest, request: Request, embedder=Depends(get_embedder_dep)):
        state = request.app.state
        session_id = uuid4().hex

        if req.mode == "session":
            session = Session(
                mode=req.mode,
                tag=req.tag,
                rpm=req.rpm,
                status="capturing",
            )
            state.sessions[session_id] = session
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
            state.sessions[session_id] = Session(
                mode=req.mode,
                tag=req.tag,
                rpm=baseline.rpm,
                status="listening",
                scorer=scorer,
            )
            return {"session_id": session_id, "state": "LISTENING"}

        # mode == "library"
        try:
            state.manager.load_library(req.tag)
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        # Unreachable today (load_library always raises), kept for when it lands.
        return {"session_id": session_id, "state": "LISTENING"}

    # ------------------------------------------------------------- score ----

    @app.post("/score")
    def score(req: ScoreRequest, request: Request, embedder=Depends(get_embedder_dep)):
        state = request.app.state
        session = state.sessions.get(req.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session_id")

        chunk = _decode_pcm(req.pcm_b64)

        if session.status == "capturing":
            return _handle_capture(state, session, chunk, embedder)
        return _handle_listen(session, chunk)

    # ------------------------------------------------------------- label ----

    @app.post("/label")
    def label(req: LabelRequest, request: Request):
        state = request.app.state
        audio = _decode_pcm(req.pcm_b64)
        meta = {
            "machine_type": req.machine_type,
            "suspected_fault": req.suspected_fault,
            "contains_speech": req.contains_speech,
            "note": req.note,
            "site_tag": req.site_tag,
            "session_id": req.session_id,
        }
        wav_path, json_path = write_labeled_clip(
            audio, LABEL_SR, meta, out_dir=state.label_dir
        )
        return {
            "wav_path": _repo_relative(wav_path),
            "json_path": _repo_relative(json_path),
        }

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
    session.capture_chunks.append(chunk)
    session.capture_samples += chunk.shape[0]

    if session.capture_samples < _CAPTURE_SAMPLES:
        remaining = (_CAPTURE_SAMPLES - session.capture_samples) / SR
        return _capturing_response(round(remaining, 6))

    audio = np.concatenate(session.capture_chunks).astype(np.float32)
    windows = _slice_windows(audio)
    baseline = state.manager.fit(windows, session.tag, embedder, rpm=session.rpm)
    state.manager.save(baseline)

    session.scorer = StreamScorer(
        baseline, embedder, AlertPolicy(), rpm=session.rpm
    )
    session.status = "listening"
    session.capture_chunks = []

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
