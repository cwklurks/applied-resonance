"""PANNs Cnn14 (16 kHz) embedding extraction.

Downloads the official pretrained checkpoint, loads it with strict=True, and
exposes helpers to compute 2048-d clip embeddings from raw waveforms.
"""

import json
import logging
import os
import time
from collections.abc import Sequence

import numpy as np
import numpy.core.multiarray as _np_multiarray
import requests
import torch

from engine.models.panns_cnn14 import Cnn14
from engine.paths import CHECKPOINT_DIR, ensure_dirs

logger = logging.getLogger(__name__)

# The official checkpoint pickles numpy arrays (e.g. STFT/mel buffers). These
# are the only non-tensor globals it contains, so we allowlist exactly them to
# keep weights_only=True (safe) loading. The _reconstruct entry is registered
# under the legacy pickle name because numpy 2.x reports the function under
# numpy._core.multiarray, which would not match the stored name otherwise.
_SAFE_GLOBALS = [
    (_np_multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
    np.ndarray,
    np.dtype,
    np.dtypes.Int64DType,
]

CHECKPOINT_URL = (
    "https://zenodo.org/api/records/3987831/files/"
    "Cnn14_16k_mAP%3D0.438.pth/content"
)
CHECKPOINT_PATH = CHECKPOINT_DIR / "Cnn14_16k_mAP=0.438.pth"
_SIDECAR_PATH = CHECKPOINT_DIR / "Cnn14_16k_mAP=0.438.pth.size.json"

EMBEDDING_DIM = 2048
_MAX_RETRIES = 5
_CHUNK_SIZE = 1 << 20  # 1 MiB

# Documented Cnn14_16k config from the PANNs repo.
_MODEL_CONFIG = dict(
    sample_rate=16000,
    window_size=512,
    hop_size=160,
    mel_bins=64,
    fmin=50,
    fmax=8000,
    classes_num=527,
)

# Memoized singletons keyed by device string.
_MODELS: dict[str, torch.nn.Module] = {}


def _remote_content_length() -> int | None:
    """Fetch the remote file size, caching it in a sidecar .json."""
    if _SIDECAR_PATH.exists():
        try:
            return int(json.loads(_SIDECAR_PATH.read_text())["content_length"])
        except (ValueError, KeyError, OSError):
            pass

    try:
        resp = requests.head(CHECKPOINT_URL, allow_redirects=True, timeout=30)
        resp.raise_for_status()
        length = resp.headers.get("Content-Length")
        if length is None:
            return None
        size = int(length)
    except (requests.RequestException, ValueError):
        return None

    try:
        _SIDECAR_PATH.write_text(json.dumps({"content_length": size}))
    except OSError:
        pass
    return size


def download_checkpoint() -> str | os.PathLike:
    """Download the pretrained checkpoint if missing.

    Skips the download when the file already exists and its size matches the
    remote Content-Length. Otherwise streams to a .part file with Range resume,
    retrying with exponential backoff, then atomically replaces the target.
    Returns the checkpoint path.
    """
    ensure_dirs()
    expected_size = _remote_content_length()

    if CHECKPOINT_PATH.exists():
        actual = CHECKPOINT_PATH.stat().st_size
        if expected_size is None or actual == expected_size:
            logger.info("Checkpoint already present at %s (%d bytes)", CHECKPOINT_PATH, actual)
            return CHECKPOINT_PATH
        logger.warning(
            "Checkpoint size mismatch (%d != %d); re-downloading", actual, expected_size
        )

    part_path = CHECKPOINT_PATH.with_suffix(CHECKPOINT_PATH.suffix + ".part")

    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resume_from = part_path.stat().st_size if part_path.exists() else 0
            headers = {}
            mode = "wb"
            if resume_from > 0:
                headers["Range"] = f"bytes={resume_from}-"
                mode = "ab"

            with requests.get(
                CHECKPOINT_URL, headers=headers, stream=True, timeout=60
            ) as resp:
                # If server ignores Range (200 instead of 206), restart from scratch.
                if resume_from > 0 and resp.status_code == 200:
                    resume_from = 0
                    mode = "wb"
                resp.raise_for_status()
                logger.info(
                    "Downloading checkpoint (resume_from=%d) -> %s", resume_from, part_path
                )
                with open(part_path, mode) as fh:
                    for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                        if chunk:
                            fh.write(chunk)

            downloaded = part_path.stat().st_size
            if expected_size is not None and downloaded != expected_size:
                raise OSError(
                    f"Incomplete download: {downloaded} != {expected_size} bytes"
                )

            os.replace(part_path, CHECKPOINT_PATH)
            logger.info(
                "Checkpoint downloaded to %s (%d bytes)", CHECKPOINT_PATH, downloaded
            )
            return CHECKPOINT_PATH
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            backoff = 2**attempt
            logger.warning(
                "Download attempt %d/%d failed: %s (retrying in %ds)",
                attempt + 1,
                _MAX_RETRIES,
                exc,
                backoff,
            )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(backoff)

    raise RuntimeError(
        f"Failed to download checkpoint after {_MAX_RETRIES} attempts"
    ) from last_error


def get_panns_model(device: torch.device) -> torch.nn.Module:
    """Construct and cache the Cnn14 model for the given device.

    Loads the pretrained state dict with strict=True, sets eval mode, and moves
    the model to the requested device. Memoized per device.
    """
    key = str(device)
    cached = _MODELS.get(key)
    if cached is not None:
        return cached

    if not CHECKPOINT_PATH.exists():
        download_checkpoint()

    model = Cnn14(**_MODEL_CONFIG)
    with torch.serialization.safe_globals(_SAFE_GLOBALS):
        state = torch.load(
            CHECKPOINT_PATH, map_location="cpu", weights_only=True
        )["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)

    _MODELS[key] = model
    return model


def _resolve_device(device: torch.device | None) -> torch.device:
    if device is not None:
        return device
    from engine.common import get_device

    return get_device()


def panns_embedding(wav: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    """Compute the 2048-d Cnn14 embedding for a single clip, shape (2048,)."""
    if wav.ndim != 1:
        raise ValueError(f"panns_embedding expects a 1-D waveform, got shape {wav.shape}")
    device = _resolve_device(device)
    model = get_panns_model(device)

    tensor = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
    tensor = tensor.unsqueeze(0).to(device)  # (1, samples)
    with torch.no_grad():
        out = model(tensor)["embedding"]
    return out.squeeze(0).cpu().numpy().astype(np.float32)


def panns_embeddings(
    wavs: Sequence[np.ndarray],
    device: torch.device | None = None,
    batch_size: int = 8,
) -> np.ndarray:
    """Compute embeddings for a sequence of equal-length clips, shape (N, 2048)."""
    if len(wavs) == 0:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    length = wavs[0].shape[0]
    for i, w in enumerate(wavs):
        if w.ndim != 1:
            raise ValueError(f"wav {i} is not 1-D (shape {w.shape})")
        assert w.shape[0] == length, (
            f"panns_embeddings requires equal-length clips: wav {i} has "
            f"{w.shape[0]} samples, expected {length}"
        )

    device = _resolve_device(device)
    model = get_panns_model(device)

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(wavs), batch_size):
            chunk = wavs[start : start + batch_size]
            batch = np.stack(
                [np.ascontiguousarray(w, dtype=np.float32) for w in chunk]
            )
            tensor = torch.from_numpy(batch).to(device)
            emb = model(tensor)["embedding"].cpu().numpy().astype(np.float32)
            outputs.append(emb)

    return np.concatenate(outputs, axis=0)
