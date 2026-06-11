"""Synthetic tests for the machine-type classifier stack.

Everything here is fast and self-contained: no downloaded data, no audio, and
no import of ``engine.features`` (the feature cache is only touched inside the
training function, which these tests never call).
"""

from pathlib import Path

import pytest
import torch
from torch import nn

from engine.dataset import Clip, list_clips, split_by_machine_id
from engine.models.classifier import MachineTypeCNN

IDS = ("id_00", "id_02", "id_04", "id_06")
MACHINES = ("fan", "pump")


def _fake_clips() -> list[Clip]:
    return [
        Clip(Path(f"/tmp/{m}/{i}/normal/x.wav"), m, i, "normal")
        for m in MACHINES
        for i in IDS
    ]


# --- split_by_machine_id ----------------------------------------------------


def test_split_isolates_val_ids():
    train, val = split_by_machine_id(_fake_clips(), val_ids=("id_06",))

    assert {c.machine_id for c in val} == {"id_06"}
    train_ids = {c.machine_id for c in train}
    assert train_ids == {"id_00", "id_02", "id_04"}
    assert train_ids.isdisjoint({c.machine_id for c in val})
    assert train and val


def test_split_raises_when_val_covers_everything():
    with pytest.raises((AssertionError, ValueError)):
        split_by_machine_id(_fake_clips(), val_ids=IDS)


# --- list_clips -------------------------------------------------------------


def _make_tree(root: Path) -> None:
    for machine in ("fan", "pump"):
        for machine_id in ("id_00", "id_06"):
            for label in ("normal", "abnormal"):
                d = root / "0_dB" / machine / machine_id / label
                d.mkdir(parents=True, exist_ok=True)
                for name in ("00000001.wav", "00000000.wav"):
                    (d / name).touch()


def test_list_clips_is_deterministic_and_parsed(tmp_path):
    _make_tree(tmp_path)
    clips = list_clips(snr="0_dB", mimii_dir=tmp_path)

    # 2 machines * 2 ids * 2 labels * 2 files
    assert len(clips) == 16

    keys = [(c.machine, c.machine_id, c.label, c.path.name) for c in clips]
    assert keys == sorted(keys)

    first = clips[0]
    assert first.machine == "fan"
    assert first.machine_id == "id_00"
    assert first.label == "abnormal"
    assert first.path.name == "00000000.wav"


def test_list_clips_missing_snr_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_clips(snr="99_dB", mimii_dir=tmp_path)


# --- model ------------------------------------------------------------------


def test_forward_shape_default():
    model = MachineTypeCNN()
    out = model(torch.randn(4, 1, 64, 313))
    assert out.shape == (4, 2)


def test_forward_handles_short_time_dim():
    model = MachineTypeCNN()
    out = model(torch.randn(4, 1, 64, 32))
    assert out.shape == (4, 2)


def test_param_count_under_500k():
    model = MachineTypeCNN()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 500_000


# --- training loop sanity ---------------------------------------------------


def test_training_loop_learns_two_gaussian_classes():
    torch.manual_seed(0)
    per_class = 20
    shape = (per_class, 1, 64, 64)
    class0 = torch.randn(shape) - 1.0
    class1 = torch.randn(shape) + 1.0
    x = torch.cat([class0, class1], dim=0)
    y = torch.cat([torch.zeros(per_class), torch.ones(per_class)]).long()

    model = MachineTypeCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(3):
        for _ in range(4):  # a few mini-batches per epoch
            idx = torch.randperm(x.size(0))[:16]
            optimizer.zero_grad()
            loss = criterion(model(x[idx]), y[idx])
            loss.backward()
            optimizer.step()

    assert torch.isfinite(loss)
    model.eval()
    with torch.no_grad():
        acc = (model(x).argmax(dim=1) == y).float().mean().item()
    assert acc > 0.8
