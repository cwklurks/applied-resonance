"""Log-mel spectrogram feature extraction via librosa."""

import librosa
import numpy as np

N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64


def logmel(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Log-mel spectrogram, shape (64, T), float32.

    For a 10 s 16 kHz clip (160000 samples), T == 313 (librosa center=True).
    """
    if wav.ndim != 1:
        raise ValueError(f"logmel expects a 1-D waveform, got shape {wav.shape}")

    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=1.0, top_db=None)
    return log_mel.astype(np.float32)
