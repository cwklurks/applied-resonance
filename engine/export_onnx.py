"""Export the PANNs Cnn14 embedding model to ONNX + RTF benchmark.

The ONNX artifact is the pivot: the same model runs in Python (onnxruntime),
the browser (onnxruntime-web), and on phone. This module wraps the loaded
Cnn14 so its sole output is the 2048-d embedding, exports it with dynamic
batch + sample axes, validates with onnx.checker, and offers a real-time
factor (RTF) benchmark for the streaming schedule.
"""

import argparse
import logging
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn

from engine.embedder import Embedder
from engine.features.panns import get_panns_model
from engine.paths import ARTIFACT_DIR, ensure_dirs

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT = ARTIFACT_DIR / "cnn14_16k.onnx"
DEFAULT_OPSET = 17
_DUMMY_SAMPLES = 48000  # 3 s at 16 kHz


class EmbeddingWrapper(nn.Module):
    """Wrap a Cnn14 so forward returns ONLY the 2048-d embedding tensor."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (B, samples) float32 -> (B, 2048)
        return self.model(waveform)["embedding"]


def export(out_path: Path = DEFAULT_ARTIFACT, opset: int = DEFAULT_OPSET) -> Path:
    """Export the pretrained Cnn14 embedding model to ONNX.

    Loads the model on CPU, wraps it to emit only the embedding, and exports
    with dynamic batch + sample axes. Tries the legacy TorchScript exporter
    first; if that errors on this graph (conv1d-STFT with reflect padding),
    retries the torch 2.x dynamo exporter. Validates the result with
    onnx.checker.check_model. Returns the artifact path.
    """
    ensure_dirs()
    out_path = Path(out_path)

    model = get_panns_model(torch.device("cpu"))
    wrapper = EmbeddingWrapper(model).eval()

    dummy = torch.zeros(1, _DUMMY_SAMPLES, dtype=torch.float32)
    dynamic_axes = {
        "waveform": {0: "batch", 1: "samples"},
        "embedding": {0: "batch"},
    }

    last_error: Exception | None = None
    for dynamo in (False, True):
        path_name = "dynamo" if dynamo else "torchscript"
        try:
            with torch.no_grad():
                torch.onnx.export(
                    wrapper,
                    (dummy,),
                    str(out_path),
                    input_names=["waveform"],
                    output_names=["embedding"],
                    dynamic_axes=dynamic_axes,
                    opset_version=opset,
                    dynamo=dynamo,
                )
            logger.info("ONNX export succeeded via %s exporter", path_name)
            break
        except Exception as exc:  # noqa: BLE001 - retry the other exporter path
            last_error = exc
            logger.warning("ONNX export via %s exporter failed: %s", path_name, exc)
    else:
        raise RuntimeError(
            "ONNX export failed on both torchscript and dynamo exporters"
        ) from last_error

    onnx.checker.check_model(onnx.load(str(out_path)))
    logger.info("ONNX model validated: %s", out_path)
    return out_path


def rtf_benchmark(
    embedders: Mapping[str, Embedder],
    window_s: float = 3.0,
    hop_s: float = 1.0,
    reps: int = 20,
) -> dict[str, float]:
    """Real-time factor per embedder for the streaming schedule.

    RTF = (mean wall-clock seconds to embed one window) / hop_s. A 3 s window
    embedded faster than the 1 s hop (RTF < 1) means the stream keeps up; the
    target is RTF < 0.5 on CPU. Each embedder is warmed up once before timing.
    """
    sample_rate = 16000
    samples = int(round(window_s * sample_rate))
    window = (
        np.random.default_rng(0)
        .standard_normal(samples)
        .astype(np.float32)
    )

    results: dict[str, float] = {}
    for name, embedder in embedders.items():
        embedder.embed([window])  # warm up (lazy export / graph optimization)
        start = time.perf_counter()
        for _ in range(reps):
            embedder.embed([window])
        elapsed = time.perf_counter() - start
        mean_time = elapsed / reps
        results[name] = mean_time / hop_s
    return results


def _cli() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Export PANNs Cnn14 embedder to ONNX.")
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT, help="output .onnx path")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset version")
    parser.add_argument(
        "--bench",
        action="store_true",
        help="benchmark RTF for torch + onnx embedders after export",
    )
    args = parser.parse_args()

    path = export(out_path=args.out, opset=args.opset)
    size_mb = path.stat().st_size / (1 << 20)
    print(f"Exported ONNX artifact: {path} ({size_mb:.1f} MB)")

    if args.bench:
        from engine.embedder import get_embedder

        embedders = {
            "torch": get_embedder("torch", device="cpu"),
            "onnx": get_embedder("onnx", device="cpu"),
        }
        rtfs = rtf_benchmark(embedders)
        for name, rtf in rtfs.items():
            print(f"RTF[{name}]={rtf:.3f}")


if __name__ == "__main__":
    _cli()
