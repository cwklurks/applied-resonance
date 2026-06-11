"""Train the machine-type classifier on normal MIMII log-mels.

Run as a module::

    uv run python -m engine.train_classifier --epochs 5

or call :func:`run_training` directly from Python (the eval/smoke scripts do).
The split is by machine id (id-disjoint), so validation measures generalisation
to unseen hardware. Only NORMAL clips are used here; abnormal clips are reserved
for the downstream anomaly task.
"""

import argparse
import json
import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from engine.common import get_device, seed_everything
from engine.dataset import list_clips, split_by_machine_id
from engine.models.classifier import MACHINE_CLASSES
from engine.models.classifier import MachineTypeCNN
from engine.paths import ARTIFACT_DIR, ensure_dirs

logger = logging.getLogger(__name__)


def _build_tensors(clips, class_to_idx: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Load log-mel features for ``clips`` into ``(N, 1, 64, T)`` X and label Y.

    The ``engine.features.cache`` import is local so unit tests never need it.
    """
    from engine.features.cache import cached_features

    feats: list[np.ndarray] = []
    labels: list[int] = []
    for clip in tqdm(clips, desc="features", unit="clip"):
        logmel = cached_features(clip.path, "logmel")  # (64, T) float32
        feats.append(np.asarray(logmel, dtype=np.float32))
        labels.append(class_to_idx[clip.machine])

    x = torch.from_numpy(np.stack(feats)).unsqueeze(1)  # (N, 1, 64, T)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


@torch.no_grad()
def _accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Fraction of correct predictions for ``x`` against labels ``y``."""
    if x.numel() == 0:
        return 0.0
    model.eval()
    preds = model(x).argmax(dim=1)
    return (preds == y).float().mean().item()


def run_training(
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str | None = None,
    snr: str = "0_dB",
    val_ids: tuple[str, ...] = ("id_06",),
    seed: int = 1337,
) -> dict:
    """Train the classifier and persist the checkpoint + metrics.

    Returns the metrics dict (also written to ``classifier_metrics.json``).
    """
    seed_everything(seed)
    ensure_dirs()
    torch_device = get_device(device)

    clips = list_clips(snr=snr)
    normal_clips = [c for c in clips if c.label == "normal"]
    train_clips, val_clips = split_by_machine_id(normal_clips, val_ids=val_ids)
    logger.info(
        "clips: %d normal (%d train / %d val), val_ids=%s",
        len(normal_clips),
        len(train_clips),
        len(val_clips),
        list(val_ids),
    )

    class_to_idx = {name: i for i, name in enumerate(MACHINE_CLASSES)}
    x_train, y_train = _build_tensors(train_clips, class_to_idx)
    x_val, y_val = _build_tensors(val_clips, class_to_idx)

    # Standardize using TRAIN-set statistics only; apply the same shift to val.
    mean = x_train.mean().item()
    std = x_train.std().item()
    std = std if std > 1e-6 else 1.0
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    x_train = x_train.to(torch_device)
    y_train = y_train.to(torch_device)
    x_val = x_val.to(torch_device)
    y_val = y_val.to(torch_device)

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    model = MachineTypeCNN(n_classes=len(MACHINE_CLASSES)).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= max(len(x_train), 1)
        val_acc = _accuracy(model, x_val, y_val)
        logger.info("epoch %d/%d: train_loss=%.4f val_acc=%.4f", epoch, epochs, epoch_loss, val_acc)

    train_acc = _accuracy(model, x_train, y_train)
    val_acc = _accuracy(model, x_val, y_val)

    checkpoint = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "classes": MACHINE_CLASSES,
        "mean": mean,
        "std": std,
        "val_ids": list(val_ids),
        "seed": seed,
        "epochs": epochs,
    }
    torch.save(checkpoint, ARTIFACT_DIR / "classifier.pt")

    metrics = {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "n_train": len(train_clips),
        "n_val": len(val_clips),
        "val_ids": list(val_ids),
        "epochs": epochs,
        "seed": seed,
    }
    with open(ARTIFACT_DIR / "classifier_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the machine-type classifier.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--snr", type=str, default="0_dB")
    parser.add_argument("--val-ids", type=str, nargs="+", default=["id_06"])
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        snr=args.snr,
        val_ids=tuple(args.val_ids),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
