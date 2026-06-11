"""Anomaly-detection train/test splits per (machine, machine_id).

This implements the MIMII-paper protocol: for each machine id, every abnormal
clip goes into the test set, an equal number of normal clips are held out as
test-normal, and the rest of the normals form the (normal-only) training set
the scorer is fit on.

The held-out normals are sampled WITHOUT replacement from the sorted normal
list. Determinism comes from a fresh ``np.random.default_rng`` per id, seeded
with ``seed`` plus a stable per-id offset: the unique ``(machine, machine_id)``
keys are sorted and enumerated, and the key's index in that sorted order is the
offset. So two ids never share a generator state, yet the same clips + seed
always yield identical splits regardless of input order.
"""

from typing import NamedTuple, Sequence

import numpy as np

from engine.dataset import Clip


class IdSplit(NamedTuple):
    machine: str
    machine_id: str
    train_normal: list[Clip]
    test_normal: list[Clip]
    test_abnormal: list[Clip]


def make_anomaly_splits(clips: Sequence[Clip], seed: int = 1337) -> list[IdSplit]:
    """Build one :class:`IdSplit` per ``(machine, machine_id)`` group.

    For each id:
      * ``test_abnormal`` = ALL abnormal clips for that id.
      * ``test_normal`` = ``len(test_abnormal)`` normal clips sampled without
        replacement from the sorted normal list, via a fresh
        ``np.random.default_rng(seed + offset)`` where ``offset`` is the id's
        index in the sorted list of unique ``(machine, machine_id)`` keys.
      * ``train_normal`` = the remaining normals (sorted order preserved).

    Args:
        clips: any iterable of :class:`Clip` (order-independent; grouped here).
        seed: base seed for the per-id generators.

    Returns:
        Splits ordered by ``(machine, machine_id)``.

    Raises:
        ValueError: if any id has fewer normals than ``abnormals + 1`` (there
            would be no normal clip left for training).
    """
    # Group by (machine, machine_id); keep label sub-lists sorted by filename so
    # sampling is reproducible regardless of input iteration order.
    groups: dict[tuple[str, str], dict[str, list[Clip]]] = {}
    for clip in clips:
        key = (clip.machine, clip.machine_id)
        bucket = groups.setdefault(key, {"normal": [], "abnormal": []})
        bucket[clip.label].append(clip)

    sorted_keys = sorted(groups)
    key_offset = {key: i for i, key in enumerate(sorted_keys)}

    splits: list[IdSplit] = []
    for key in sorted_keys:
        machine, machine_id = key
        normals = sorted(groups[key]["normal"], key=lambda c: c.path.name)
        abnormals = sorted(groups[key]["abnormal"], key=lambda c: c.path.name)

        n_abnormal = len(abnormals)
        if len(normals) < n_abnormal + 1:
            raise ValueError(
                f"{machine}/{machine_id}: need at least {n_abnormal + 1} normal "
                f"clips (abnormal={n_abnormal} + 1 for training), "
                f"got {len(normals)}."
            )

        rng = np.random.default_rng(seed + key_offset[key])
        test_normal_pos = rng.choice(len(normals), size=n_abnormal, replace=False)
        test_normal_set = set(test_normal_pos.tolist())

        test_normal = [normals[i] for i in sorted(test_normal_set)]
        train_normal = [n for i, n in enumerate(normals) if i not in test_normal_set]

        assert all(c.label == "normal" for c in train_normal), (
            f"{machine}/{machine_id}: non-normal clip leaked into train_normal."
        )

        splits.append(
            IdSplit(
                machine=machine,
                machine_id=machine_id,
                train_normal=train_normal,
                test_normal=test_normal,
                test_abnormal=abnormals,
            )
        )

    return splits
