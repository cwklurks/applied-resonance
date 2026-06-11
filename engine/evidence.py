"""Evidence module: explain WHY an audio window looks anomalous.

Produces a compact, physical, technician-facing explanation built from three
signals:

  (a) per-mel-band dB deviations from the baseline mean spectrum,
  (b) envelope-spectrum peaks (impulse / modulation periodicities) in the
      500-5000 Hz band, labeled by physical cause where recognisable,
  (c) the nearest baseline training windows by embedding cosine distance.

The ``baseline`` argument is duck-typed (see :func:`build_evidence`) so this
module stays decoupled from the concurrently-written ``engine.baseline``.

Pure functions, no I/O, no torch.
"""

from typing import Protocol

import librosa
import numpy as np
import scipy.signal

from engine.features.logmel import logmel

SR = 16000
N_MELS = 64

# Envelope-spectrum analysis band and search range.
_BP_LO_HZ = 500.0
_BP_HI_HZ = 5000.0
_PEAK_LO_HZ = 5.0
_PEAK_HI_HZ = 300.0
_PROMINENCE_FRAC = 0.1  # peak prominence threshold as a fraction of max magnitude

_TOP_K = 3
_HUD_MAX = 60
_MAINS_2X_HZ = 120.0
_MAINS_TOL_HZ = 3.0


class _Baseline(Protocol):
    """Structural type for the baseline argument.

    Only these three attributes are read:
      mel_mean:         (64,) float, mean log-mel spectrum of the baseline.
      train_embeddings: (N, D) float, per-window embeddings of training data.
      timestamps:       (N,) float, seconds, one per training window.
    """

    mel_mean: np.ndarray
    train_embeddings: np.ndarray
    timestamps: np.ndarray


def build_evidence(
    window: np.ndarray,
    baseline: object,
    window_embedding: np.ndarray | None = None,
    rpm: float | None = None,
    sr: int = SR,
) -> dict:
    """Explain why ``window`` looks anomalous against ``baseline``.

    Args:
        window: 1-D waveform, at least ~1 s long at ``sr``.
        baseline: object exposing ``.mel_mean`` (64,), ``.train_embeddings``
            (N, D) and ``.timestamps`` (N,). Duck-typed; not imported.
        window_embedding: optional (D,) embedding of ``window``; when omitted
            ``nearest_baseline_s`` is empty.
        rpm: optional machine rotation speed, used to label a 1x-rotation peak.
        sr: sample rate of ``window``.

    Returns a json-safe dict (see module docstring / spec).
    """
    if window.ndim != 1:
        raise ValueError(f"build_evidence expects a 1-D window, got shape {window.shape}")
    if window.shape[0] < sr:
        raise ValueError(
            f"window too short: {window.shape[0]} samples < {sr} (~1 s) at sr={sr}"
        )

    wav = window.astype(np.float64)

    mel_bands = _mel_band_deltas(wav, np.asarray(baseline.mel_mean, dtype=np.float64), sr)
    envelope_peaks = _envelope_peaks(wav, rpm, sr)
    nearest = _nearest_baseline(window_embedding, baseline)
    hud = _compose_hud(mel_bands, envelope_peaks)

    return {
        "mel_bands": mel_bands,
        "envelope_peaks": envelope_peaks,
        "nearest_baseline_s": nearest,
        "hud": hud,
    }


# --------------------------------------------------------------- mel bands ---


def _mel_band_deltas(wav: np.ndarray, mel_mean: np.ndarray, sr: int) -> list[dict]:
    """Top-3 per-band dB deviations from the baseline mean spectrum."""
    band_db = logmel(wav.astype(np.float32), sr=sr).mean(axis=1).astype(np.float64)
    delta = band_db - mel_mean

    edges = librosa.mel_frequencies(N_MELS + 2, fmin=0.0, fmax=sr / 2)
    order = np.argsort(np.abs(delta))[::-1][:_TOP_K]

    return [
        {
            "lo_hz": float(edges[i]),
            "hi_hz": float(edges[i + 2]),
            "delta_db": float(delta[i]),
        }
        for i in order
    ]


# ----------------------------------------------------------- envelope peaks --


def _envelope_peaks(wav: np.ndarray, rpm: float | None, sr: int) -> list[dict]:
    """Top-3 envelope-spectrum peaks in 5-300 Hz, labeled by physical cause."""
    sos = scipy.signal.butter(
        4, [_BP_LO_HZ, _BP_HI_HZ], btype="bandpass", fs=sr, output="sos"
    )
    filtered = scipy.signal.sosfiltfilt(sos, wav)

    envelope = np.abs(scipy.signal.hilbert(filtered))
    envelope = envelope - envelope.mean()
    envelope = envelope * np.hanning(envelope.shape[0])

    spectrum = np.abs(np.fft.rfft(envelope))
    freqs = np.fft.rfftfreq(envelope.shape[0], d=1.0 / sr)

    in_range = (freqs >= _PEAK_LO_HZ) & (freqs <= _PEAK_HI_HZ)
    if not np.any(in_range) or spectrum.max() <= 0.0:
        return []

    threshold = _PROMINENCE_FRAC * float(spectrum.max())
    peak_idx, props = scipy.signal.find_peaks(spectrum, prominence=threshold)
    if peak_idx.size == 0:
        return []

    keep = in_range[peak_idx]
    peak_idx = peak_idx[keep]
    prominences = props["prominences"][keep]
    if peak_idx.size == 0:
        return []

    order = np.argsort(prominences)[::-1][:_TOP_K]
    peaks = []
    for j in order:
        f = float(freqs[peak_idx[j]])
        peaks.append(
            {
                "freq_hz": f,
                "label": _label_peak(f, rpm),
                "prominence": float(prominences[j]),
            }
        )
    return peaks


def _label_peak(f: float, rpm: float | None) -> str:
    """Physical label for an envelope-spectrum peak at ``f`` Hz."""
    if abs(f - _MAINS_2X_HZ) <= _MAINS_TOL_HZ:
        return "line hum ~120 Hz"
    if rpm is not None:
        rps = rpm / 60.0
        if abs(f - rps) <= max(3.0, 0.05 * rps):
            return f"1x rotation ~{f:.0f} Hz"
    return f"impulse train ~{f:.0f} Hz"


# -------------------------------------------------------- nearest baseline ---


def _nearest_baseline(window_embedding: np.ndarray | None, baseline: object) -> list[float]:
    """Timestamps of the 3 nearest baseline windows by cosine distance."""
    if window_embedding is None:
        return []

    train = np.asarray(baseline.train_embeddings, dtype=np.float64)
    query = np.asarray(window_embedding, dtype=np.float64).ravel()

    train_norm = train / (np.linalg.norm(train, axis=1, keepdims=True) + 1e-12)
    query_norm = query / (np.linalg.norm(query) + 1e-12)

    cosine_sim = train_norm @ query_norm
    distance = 1.0 - cosine_sim

    k = min(_TOP_K, distance.shape[0])
    nearest_idx = np.argsort(distance)[:k]
    timestamps = np.asarray(baseline.timestamps, dtype=np.float64)
    return [float(timestamps[i]) for i in nearest_idx]


# ----------------------------------------------------------------- the HUD ---


def _band_phrase(top_band: dict) -> str:
    """Human phrase for the dominant mel band, e.g. 'high-band energy up'."""
    center = 0.5 * (top_band["lo_hz"] + top_band["hi_hz"])
    if center < 500.0:
        where = "low-band"
    elif center <= 2000.0:
        where = "mid-band"
    else:
        where = "high-band"
    direction = "energy up" if top_band["delta_db"] >= 0.0 else "energy down"
    return f"{where} {direction}"


def _compose_hud(mel_bands: list[dict], envelope_peaks: list[dict]) -> str:
    """≤60 char physical summary; peak label first, then dominant band."""
    band_phrase = _band_phrase(mel_bands[0]) if mel_bands else "anomaly vs baseline"

    if envelope_peaks:
        hud = f"{envelope_peaks[0]['label']}, {band_phrase}"
    else:
        hud = f"{band_phrase} vs baseline"

    if len(hud) > _HUD_MAX:
        hud = hud[:_HUD_MAX].rstrip(", ")
    return hud
