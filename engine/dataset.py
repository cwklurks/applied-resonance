"""MIMII clip discovery and id-disjoint train/val splitting.

A :class:`Clip` is a single 10 s recording on disk. Discovery is deterministic
and the machine-id split keeps train and validation machine ids disjoint so the
classifier is evaluated on hardware it never saw during training.
"""

from pathlib import Path
from typing import NamedTuple, Sequence

from engine.paths import MIMII_DIR

LABELS = ("normal", "abnormal")


class Clip(NamedTuple):
    path: Path
    machine: str  # "fan" | "pump"
    machine_id: str  # "id_00" ...
    label: str  # "normal" | "abnormal"


def list_clips(
    snr: str = "0_dB",
    machines: Sequence[str] = ("fan", "pump"),
    mimii_dir: Path | None = None,
) -> list[Clip]:
    """List clips under ``{mimii_dir or MIMII_DIR}/{snr}/{machine}/id_*/{label}/*.wav``.

    Walks every requested machine, every ``id_*`` directory, and the
    ``normal`` / ``abnormal`` subdirectories. The result is sorted by
    ``(machine, machine_id, label, filename)`` so repeated calls are stable.

    Raises:
        FileNotFoundError: if the snr directory does not exist, with a hint to
            run ``python -m engine.download``.
    """
    root = (mimii_dir or MIMII_DIR) / snr
    if not root.is_dir():
        raise FileNotFoundError(
            f"MIMII snr directory not found: {root}. "
            "Download the dataset first with `python -m engine.download`."
        )

    clips: list[Clip] = []
    for machine in machines:
        machine_dir = root / machine
        if not machine_dir.is_dir():
            continue
        for id_dir in sorted(p for p in machine_dir.iterdir() if p.is_dir()):
            machine_id = id_dir.name
            for label in LABELS:
                label_dir = id_dir / label
                if not label_dir.is_dir():
                    continue
                for wav in label_dir.glob("*.wav"):
                    clips.append(Clip(wav, machine, machine_id, label))

    return sorted(clips, key=lambda c: (c.machine, c.machine_id, c.label, c.path.name))


def split_by_machine_id(
    clips: Sequence[Clip],
    val_ids: Sequence[str] = ("id_06",),
) -> tuple[list[Clip], list[Clip]]:
    """Split ``clips`` into (train, val) by machine id.

    Train holds every clip whose ``machine_id`` is NOT in ``val_ids``; val holds
    the rest. Asserts both splits are non-empty and that their machine-id sets
    are disjoint.
    """
    val_id_set = set(val_ids)
    train = [c for c in clips if c.machine_id not in val_id_set]
    val = [c for c in clips if c.machine_id in val_id_set]

    train_ids = {c.machine_id for c in train}
    val_ids_seen = {c.machine_id for c in val}
    if not train_ids or not val_ids_seen:
        raise ValueError(
            "split_by_machine_id produced an empty split: "
            f"train ids={sorted(train_ids)}, val ids={sorted(val_ids_seen)}. "
            "Check that val_ids leaves at least one machine id for training."
        )
    assert train_ids.isdisjoint(val_ids_seen), (
        f"train/val machine ids overlap: {sorted(train_ids & val_ids_seen)}"
    )

    return train, val
