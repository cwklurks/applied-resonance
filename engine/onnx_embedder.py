"""onnxruntime CPU backend for the exported Cnn14 embedding model.

Mirrors ``TorchPannsEmbedder.embed`` (equal-length validation, batches of 8,
(N, 2048) float32 output) so the two backends are interchangeable behind
``get_embedder``. CPUExecutionProvider only for now; the ``device`` kwarg is
accepted for interface parity and ignored.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort

from engine.paths import ARTIFACT_DIR

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = ARTIFACT_DIR / "cnn14_16k.onnx"
EMBEDDING_DIM = 2048
_BATCH_SIZE = 8


class OnnxPannsEmbedder:
    """onnxruntime CPU inference of the exported CNN14 embedding model.

    device kwarg accepted for interface parity with get_embedder(); ignored
    (CPUExecutionProvider only for now).
    """

    def __init__(self, model_path: Path | None = None, device=None):
        path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        if not path.exists():
            logger.info("ONNX model missing at %s; exporting now", path)
            from engine.export_onnx import export

            path = export(out_path=path)

        so = ort.SessionOptions()
        # Single-threaded, fully deterministic graph for bit-stable embeddings.
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            str(path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name

    def embed(self, wavs: Sequence[np.ndarray]) -> np.ndarray:  # (N, 2048) float32
        if len(wavs) == 0:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        length = wavs[0].shape[0]
        for i, w in enumerate(wavs):
            if w.ndim != 1:
                raise ValueError(f"wav {i} is not 1-D (shape {w.shape})")
            if w.shape[0] != length:
                raise ValueError(
                    f"OnnxPannsEmbedder requires equal-length clips: wav {i} has "
                    f"{w.shape[0]} samples, expected {length}"
                )

        outputs: list[np.ndarray] = []
        for start in range(0, len(wavs), _BATCH_SIZE):
            chunk = wavs[start : start + _BATCH_SIZE]
            batch = np.stack(
                [np.ascontiguousarray(w, dtype=np.float32) for w in chunk]
            )
            emb = self.session.run(
                [self._output_name], {self._input_name: batch}
            )[0]
            outputs.append(np.asarray(emb, dtype=np.float32))

        return np.concatenate(outputs, axis=0)
