"""EarSight desktop demo -- live acoustic-anomaly monitor + data collector.

Run with::

    uv run streamlit run desktop/app.py

Architecture (the proven Streamlit live-audio pattern):
  * A background thread reads ``MicSource.chunks()`` into a bounded ring buffer
    (a ``deque`` of ``(seq, chunk)`` guarded by a lock) with a monotonic
    counter. The thread, buffer, and stop Event live in ``st.session_state``
    and are created once (sentinel-guarded). Changing the device tears the
    thread down and starts a fresh one.
  * Scoring happens in the UI refresh cycle, NOT the capture thread: a
    ``@st.fragment(run_every=1.0)`` fragment drains new chunks from the ring
    buffer into the ``StreamScorer`` and renders the live panels.

Everything that pulls torch (the embedder, the baseline fit, the scorer) is
imported lazily inside functions so the script imports fast and headless boots
without a model load or a mic.
"""

import threading
import time
from collections import deque

import numpy as np
import streamlit as st

from desktop.ui_helpers import pcm_concat, slice_windows, spectrogram_image

SR = 16000
RING_SECONDS = 30  # ring buffer holds the last ~30 one-second chunks
BASELINE_CHUNKS = 30  # 30 s of audio -> 28 windows after slicing
CLIP_SECONDS = 10  # record-and-label clip length
DEFAULT_TAG = "bench-unit"

MACHINE_TYPES = ["fan", "pump", "compressor", "furnace", "fridge", "other"]

_STATE_STYLE = {
    "LISTENING": ("#1b8a3a", "🟢", "LISTENING"),
    "SUSPECT": ("#d98c00", "🟡", "SUSPECT"),
    "ALERT": ("#c62828", "🔴", "ALERT"),
}


# ======================================================== capture thread ====


class _Capture:
    """Owns the background mic thread and its ring buffer.

    Constructing this does NOT open the mic; :meth:`start` spawns the thread
    which opens the stream lazily via ``MicSource.chunks()``.
    """

    def __init__(self, device, backend_label: str):
        self.device = device
        self.backend_label = backend_label
        self.lock = threading.Lock()
        self.buffer: deque = deque(maxlen=RING_SECONDS)  # (seq, chunk)
        self.seq = 0  # monotonic count of chunks ever captured
        self.stop_event = threading.Event()
        self.error: str | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        from shared.audio_source import MicSource

        try:
            source = MicSource(device=self.device, chunk_s=1.0)
            for chunk in source.chunks():
                if self.stop_event.is_set():
                    break
                with self.lock:
                    self.seq += 1
                    self.buffer.append((self.seq, np.asarray(chunk, dtype=np.float32)))
        except Exception as exc:  # mic permission / device errors
            self.error = str(exc)

    def stop(self) -> None:
        """Signal the thread to stop and briefly wait for it to release the
        stream, shrinking the window in which an old stream and a freshly
        started one are open on the same device. The thread is parked in
        ``q.get()`` until the next chunk arrives, so the join is bounded by the
        ~1 s chunk cadence; we cap it so the UI never blocks."""
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)

    def snapshot(self) -> tuple[int, list[np.ndarray]]:
        """Return (latest seq, ordered list of buffered chunks). ``seq`` is read
        under the lock so it is consistent with the returned buffer contents."""
        with self.lock:
            items = list(self.buffer)
            latest = self.seq
        return (latest, [c for _, c in items])

    def chunks_since(self, after_seq: int) -> tuple[int, list[np.ndarray]]:
        """Return (latest returned seq, chunks with seq > ``after_seq``) in order.

        ``latest`` is the max seq actually present in the returned chunks (or
        ``after_seq`` when none are new), NOT ``self.seq`` -- so a caller that
        advances its cursor to ``latest`` never skips a chunk that is still in
        the ring. Chunks already evicted from the bounded buffer are
        unrecoverable, but they are not silently skipped over.
        """
        with self.lock:
            new = [(s, c) for s, c in self.buffer if s > after_seq]
        latest = new[-1][0] if new else after_seq
        return latest, [c for _, c in new]


# ============================================================ session glue ===


def _init_state() -> None:
    ss = st.session_state
    if ss.get("_earsight_init"):
        return
    ss._earsight_init = True
    ss.capture = None  # type: _Capture | None
    ss.scorer = None  # StreamScorer once a baseline exists
    ss.consumed_seq = 0  # last chunk seq drained into the scorer
    ss.windows_scored = 0
    ss.latest_result = None  # last WindowResult with non-None evidence kept sticky
    ss.last_evidence = None
    ss.last_t = 0.0
    ss.state_name = "LISTENING"
    ss.percentile = 0.0
    ss.baseline_tag = DEFAULT_TAG
    ss.capturing_baseline = False
    ss.baseline_start_seq = 0


def _stop_capture() -> None:
    cap = st.session_state.capture
    if cap is not None:
        cap.stop()
    st.session_state.capture = None
    st.session_state.consumed_seq = 0
    st.session_state.capturing_baseline = False


def _start_capture(device, backend_label: str) -> None:
    _stop_capture()
    cap = _Capture(device=device, backend_label=backend_label)
    cap.start()
    st.session_state.capture = cap
    st.session_state.consumed_seq = 0


@st.cache_resource(show_spinner="Loading embedder (first run downloads the model)...")
def _get_embedder(backend: str):
    """Cache the embedder across reruns (one model load per backend)."""
    from engine.embedder import get_embedder

    return get_embedder(backend)


# ============================================================== baseline =====


def _fit_baseline(chunks: list[np.ndarray], tag: str, backend: str, rpm) -> None:
    """Slice 30 s of audio into windows, fit + save a baseline, build a scorer."""
    from engine.baseline import BaselineManager
    from engine.policy import AlertPolicy
    from engine.stream import StreamScorer

    windows = slice_windows(chunks)
    if len(windows) < 10:
        st.error(
            f"Captured only {len(windows)} windows (<10). Keep monitoring on and "
            "try again with the machine running."
        )
        return

    embedder = _get_embedder(backend)
    manager = BaselineManager()
    baseline = manager.fit(windows, tag, embedder, rpm=rpm)
    manager.save(baseline)

    st.session_state.scorer = StreamScorer(baseline, embedder, AlertPolicy(), rpm=rpm)
    # Start scoring from "now" so baseline audio isn't re-scored against itself.
    st.session_state.consumed_seq = st.session_state.capture.snapshot()[0]
    st.session_state.windows_scored = 0
    st.toast(f"Baseline '{tag}' fitted from {len(windows)} windows.", icon="✅")


def _load_baseline(tag: str, backend: str) -> None:
    from engine.baseline import BaselineManager
    from engine.policy import AlertPolicy
    from engine.stream import StreamScorer

    embedder = _get_embedder(backend)
    baseline = BaselineManager().load(tag)
    st.session_state.scorer = StreamScorer(
        baseline, embedder, AlertPolicy(), rpm=baseline.rpm
    )
    cap = st.session_state.capture
    st.session_state.consumed_seq = cap.snapshot()[0] if cap is not None else 0
    st.session_state.windows_scored = 0
    st.session_state.baseline_tag = tag
    st.toast(f"Loaded saved baseline '{tag}'.", icon="📂")


# =============================================================== sidebar =====


def _device_options() -> list:
    try:
        from shared.audio_source import list_input_devices

        return list_input_devices()
    except Exception as exc:  # no audio backend available
        st.sidebar.warning(f"Could not list input devices: {exc}")
        return []


def _saved_tags() -> list[str]:
    from engine.baseline import BaselineManager

    try:
        return BaselineManager().list_tags()
    except Exception:
        return []


def _render_sidebar() -> dict:
    ss = st.session_state
    st.sidebar.title("EarSight setup")

    devices = _device_options()
    device_labels = [f"{d['index']}: {d['name']}" for d in devices]
    device_choice = st.sidebar.selectbox(
        "Input device",
        options=range(len(devices)) if devices else [0],
        format_func=lambda i: device_labels[i] if devices else "(no devices found)",
        disabled=not devices,
    )
    device_index = devices[device_choice]["index"] if devices else None

    backend = st.sidebar.selectbox("Embedder backend", ["torch", "onnx"], index=0)

    ss.baseline_tag = st.sidebar.text_input("Baseline tag", value=ss.baseline_tag)
    rpm_in = st.sidebar.number_input(
        "Machine RPM (0 = unknown)", min_value=0, max_value=100000, value=0, step=10
    )
    rpm = float(rpm_in) if rpm_in > 0 else None

    saved = _saved_tags()
    st.sidebar.divider()
    load_tag = st.sidebar.selectbox(
        "Load saved baseline",
        options=saved if saved else ["(none saved)"],
        disabled=not saved,
    )
    if st.sidebar.button("Load baseline", disabled=not saved):
        _load_baseline(load_tag, backend)

    st.sidebar.divider()
    monitoring = st.sidebar.toggle("Monitor (open mic)", value=ss.capture is not None)

    # Reconcile the toggle / device with the live capture thread.
    if monitoring:
        cap = ss.capture
        if cap is None or cap.device != device_index:
            _start_capture(device_index, f"{device_index}")
    else:
        if ss.capture is not None:
            _stop_capture()

    return {
        "backend": backend,
        "rpm": rpm,
        "device_label": device_labels[device_choice] if devices else "none",
        "monitoring": monitoring,
    }


# ============================================================== rendering ====


def _state_badge(name: str) -> None:
    color, emoji, text = _STATE_STYLE.get(name, _STATE_STYLE["LISTENING"])
    st.markdown(
        f"<div style='font-size:2.2rem;font-weight:700;color:{color};'>"
        f"{emoji} {text}</div>",
        unsafe_allow_html=True,
    )


def _drain_and_score() -> None:
    """Pull new chunks from the ring buffer into the active scorer."""
    ss = st.session_state
    cap = ss.capture
    if cap is None or ss.scorer is None:
        return
    latest, new_chunks = cap.chunks_since(ss.consumed_seq)
    if not new_chunks:
        return
    ss.consumed_seq = latest
    for chunk in new_chunks:
        for result in ss.scorer.process(chunk):
            ss.windows_scored += 1
            ss.last_t = result.t
            ss.state_name = result.state.name
            ss.percentile = result.percentile
            if result.evidence is not None:
                ss.last_evidence = result.evidence
                ss.latest_result = result


def _render_baseline_capture(cfg: dict) -> None:
    ss = st.session_state
    cap = ss.capture

    if ss.capturing_baseline:
        # One consistent snapshot: count and chunks come from the same instant.
        latest, chunks = cap.snapshot() if cap is not None else (0, [])
        captured = latest - ss.baseline_start_seq
        remaining = max(0, BASELINE_CHUNKS - captured)
        st.progress(
            min(captured, BASELINE_CHUNKS) / BASELINE_CHUNKS,
            text=f"{remaining}s remaining -- keep the machine sounding NORMAL",
        )
        if cap is not None and captured >= BASELINE_CHUNKS:
            ss.capturing_baseline = False
            with st.spinner("Fitting baseline..."):
                _fit_baseline(
                    chunks[-BASELINE_CHUNKS:], ss.baseline_tag, cfg["backend"], cfg["rpm"]
                )
        return

    disabled = not cfg["monitoring"]
    if st.button("Capture baseline (30 s)", disabled=disabled, type="primary"):
        ss.baseline_start_seq = cap.snapshot()[0] if cap is not None else 0
        ss.capturing_baseline = True
    if disabled:
        st.caption("Turn on **Monitor** in the sidebar to capture a baseline.")


def _render_live_row() -> None:
    ss = st.session_state
    cap = ss.capture
    col_state, col_score, col_spec = st.columns([1, 1, 2])

    with col_state:
        _state_badge(ss.state_name)
    with col_score:
        st.progress(min(ss.percentile, 100.0) / 100.0)
        st.metric("anomaly percentile", f"{ss.percentile:.0f}")
    with col_spec:
        if cap is not None:
            _, chunks = cap.snapshot()
            img = spectrogram_image(chunks[-CLIP_SECONDS:])
            st.image(img, use_container_width=True, caption="rolling 10 s log-mel")
        else:
            st.caption("Spectrogram appears once monitoring starts.")


def _render_evidence() -> None:
    ss = st.session_state
    evidence = ss.last_evidence
    if evidence and ss.state_name in ("SUSPECT", "ALERT"):
        st.subheader(f"⚠ {evidence.get('hud', 'anomaly vs baseline')}")
        with st.expander("Full evidence"):
            st.json(evidence)
    else:
        st.caption("No anomaly evidence yet -- the meter stays quiet while normal.")


def _render_record_panel(cfg: dict) -> None:
    ss = st.session_state
    cap = ss.capture
    st.subheader("Record & label")

    buffered = 0
    if cap is not None:
        _, chunks = cap.snapshot()
        buffered = len(chunks)

    if buffered < CLIP_SECONDS:
        st.info(
            f"Buffer has {buffered}s; need {CLIP_SECONDS}s. Keep monitoring to enable "
            "recording."
        )
        return

    with st.form("record_label", clear_on_submit=False):
        machine_type = st.selectbox("Machine type", MACHINE_TYPES)
        suspected_fault = st.text_input("Suspected fault", value="")
        contains_speech = st.checkbox("Contains speech (privacy flag)", value=False)
        note = st.text_area("Note", value="")
        site_tag = st.text_input("Site tag", value=ss.baseline_tag)
        submitted = st.form_submit_button("Save last 10 s clip", type="primary")

    if submitted:
        _, chunks = cap.snapshot()
        audio = pcm_concat(chunks[-CLIP_SECONDS:])
        meta = {
            "machine_type": machine_type,
            "suspected_fault": suspected_fault,
            "contains_speech": contains_speech,
            "note": note,
            "site_tag": site_tag,
        }
        try:
            from engine.labeling import write_labeled_clip  # lazy: lands concurrently

            wav_path, _ = write_labeled_clip(audio, SR, meta)
            st.success(f"Saved clip -> {wav_path}")
        except Exception as exc:
            st.error(f"Could not save clip: {exc}")


@st.fragment(run_every=1.0)
def _live_fragment(cfg: dict) -> None:
    """Runs every second: drain new audio, score it, redraw the live panels."""
    ss = st.session_state
    cap = ss.capture

    if cap is not None and cap.error:
        st.error(
            f"Microphone error: {cap.error}\n\n"
            "On macOS, grant Terminal/your IDE microphone access in "
            "System Settings -> Privacy & Security -> Microphone, then toggle "
            "Monitor off and on."
        )
        return

    _drain_and_score()
    _render_baseline_capture(cfg)
    st.divider()
    _render_live_row()
    st.divider()
    _render_evidence()
    st.divider()
    _render_record_panel(cfg)


# ================================================================== main =====


def main() -> None:
    st.set_page_config(page_title="EarSight monitor", page_icon="🎧", layout="wide")
    _init_state()

    st.title("🎧 EarSight -- live acoustic-anomaly monitor")
    st.caption(
        "Point the mic at a machine, capture a 30 s healthy baseline, then watch "
        "the anomaly meter. On SUSPECT/ALERT the evidence line explains why."
    )

    cfg = _render_sidebar()

    if not cfg["monitoring"] and st.session_state.scorer is None:
        st.info(
            "Turn on **Monitor** in the sidebar to open the mic, then capture a "
            "baseline. Nothing opens the microphone until you do."
        )

    _live_fragment(cfg)

    ss = st.session_state
    st.divider()
    st.caption(
        f"device: {cfg['device_label']} · backend: {cfg['backend']} · "
        f"windows scored: {ss.windows_scored} · last window t: {ss.last_t:.0f}s"
    )


if __name__ == "__main__":
    main()
