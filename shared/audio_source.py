"""Audio-input seam: stream 16 kHz mono float32 chunks from any source.

Everything downstream consumes an ``AudioSource``. Adding a new source
(browser, phone, glasses) means writing a new subclass here; nothing else
in the system changes.
"""

import abc
import logging
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from engine.audio import load_wav

logger = logging.getLogger(__name__)

SR = 16000


class AudioSource(abc.ABC):
    """Iterator of 16 kHz mono float32 chunks. The ONLY audio-I/O seam:
    everything downstream consumes an AudioSource; future sources
    (browser, phone, glasses) are new subclasses, nothing else changes."""

    sr: int = SR

    @abc.abstractmethod
    def chunks(self) -> Iterator[np.ndarray]:  # 1-D float32 arrays
        ...


def _validate_chunk_s(chunk_s: float) -> None:
    if chunk_s <= 0:
        raise ValueError(f"chunk_s must be > 0, got {chunk_s}")


class FileSource(AudioSource):
    """Yields successive chunk_s-second chunks of a WAV file (channel 0,
    asserts 16 kHz via engine.audio.load_wav). realtime=True sleeps to
    simulate live capture (default False). Last partial chunk IS yielded."""

    def __init__(self, path: str | Path, chunk_s: float = 1.0, realtime: bool = False):
        _validate_chunk_s(chunk_s)
        self.path = Path(path)
        self.chunk_s = chunk_s
        self.realtime = realtime

    def chunks(self) -> Iterator[np.ndarray]:
        wav = load_wav(self.path, expected_sr=SR)
        n = int(round(self.chunk_s * SR))
        for start in range(0, wav.shape[0], n):
            chunk = wav[start : start + n]
            if self.realtime:
                time.sleep(chunk.shape[0] / SR)
            yield np.ascontiguousarray(chunk, dtype=np.float32)


class MicSource(AudioSource):
    """sounddevice.InputStream -> queue -> chunks(). device may be an index,
    a name substring, or None (system default). Constructing MUST NOT open
    the stream; the stream opens lazily inside chunks() and closes on
    generator close/GC (use try/finally). dtype float32, mono; if the device
    captures at a different default samplerate, request samplerate=SR from
    sounddevice (PortAudio resamples or errors clearly - surface that error
    with the device name)."""

    def __init__(self, device=None, chunk_s: float = 1.0):
        _validate_chunk_s(chunk_s)
        self.device = device
        self.chunk_s = chunk_s

    def chunks(self) -> Iterator[np.ndarray]:
        import queue

        import sounddevice as sd

        n = int(round(self.chunk_s * SR))
        q: "queue.Queue[np.ndarray]" = queue.Queue()

        def _callback(indata, frames, time_info, status):
            if status:
                logger.warning("MicSource stream status: %s", status)
            q.put(indata[:, 0].copy())

        try:
            stream = sd.InputStream(
                device=self.device,
                channels=1,
                samplerate=SR,
                dtype="float32",
                blocksize=n,
                callback=_callback,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open input stream for device {self.device!r}: {exc}"
            ) from exc

        buffer = np.empty(0, dtype=np.float32)
        try:
            stream.start()
            while True:
                buffer = np.concatenate([buffer, q.get()])
                while buffer.shape[0] >= n:
                    yield np.ascontiguousarray(buffer[:n], dtype=np.float32)
                    buffer = buffer[n:]
        finally:
            stream.stop()
            stream.close()


def list_input_devices() -> list[dict]:
    """[{index, name, max_input_channels, default_samplerate}] for devices
    with max_input_channels > 0, via sounddevice.query_devices()."""
    import sounddevice as sd

    devices = sd.query_devices()
    return [
        {
            "index": i,
            "name": d["name"],
            "max_input_channels": d["max_input_channels"],
            "default_samplerate": d["default_samplerate"],
        }
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]
