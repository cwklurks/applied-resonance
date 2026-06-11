"""Synthetic tests for the anomaly eval pipeline.

No real MIMII data, no PANNs, no audio. Splits are tested with fake clips,
pAUC against a direct sklearn call, and the report renderer against a
hand-built results structure.
"""

from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from engine.dataset import Clip
from engine.eval.run_eval import (
    ScorerResult,
    SplitResult,
    aggregate_by_type,
    compute_pauc,
    render_report,
)
from engine.eval.splits import make_anomaly_splits

IDS = ("id_00", "id_02")
MACHINES = ("fan", "pump")
N_NORMAL = 8
N_ABNORMAL = 3


def _fake_clips(n_normal: int = N_NORMAL, n_abnormal: int = N_ABNORMAL) -> list[Clip]:
    clips: list[Clip] = []
    for machine in MACHINES:
        for machine_id in IDS:
            for i in range(n_normal):
                clips.append(
                    Clip(
                        Path(f"/tmp/{machine}/{machine_id}/normal/{i:04d}.wav"),
                        machine,
                        machine_id,
                        "normal",
                    )
                )
            for i in range(n_abnormal):
                clips.append(
                    Clip(
                        Path(f"/tmp/{machine}/{machine_id}/abnormal/{i:04d}.wav"),
                        machine,
                        machine_id,
                        "abnormal",
                    )
                )
    return clips


# --- make_anomaly_splits ----------------------------------------------------


def test_splits_one_per_id():
    splits = make_anomaly_splits(_fake_clips())
    assert len(splits) == len(MACHINES) * len(IDS)
    keys = [(s.machine, s.machine_id) for s in splits]
    assert keys == sorted(keys)


def test_test_abnormal_is_all_abnormal():
    splits = make_anomaly_splits(_fake_clips())
    for s in splits:
        assert len(s.test_abnormal) == N_ABNORMAL
        assert all(c.label == "abnormal" for c in s.test_abnormal)


def test_test_normal_matches_abnormal_count():
    splits = make_anomaly_splits(_fake_clips())
    for s in splits:
        assert len(s.test_normal) == len(s.test_abnormal)


def test_train_and_test_normal_disjoint():
    splits = make_anomaly_splits(_fake_clips())
    for s in splits:
        train_paths = {c.path for c in s.train_normal}
        test_paths = {c.path for c in s.test_normal}
        assert train_paths.isdisjoint(test_paths)
        # No clip lost: train + test_normal == all normals for this id.
        assert len(s.train_normal) + len(s.test_normal) == N_NORMAL


def test_all_train_clips_are_normal():
    splits = make_anomaly_splits(_fake_clips())
    for s in splits:
        assert all(c.label == "normal" for c in s.train_normal)


def test_splits_deterministic():
    clips = _fake_clips()
    a = make_anomaly_splits(clips, seed=1337)
    b = make_anomaly_splits(clips, seed=1337)
    for sa, sb in zip(a, b, strict=True):
        assert [c.path for c in sa.test_normal] == [c.path for c in sb.test_normal]
        assert [c.path for c in sa.train_normal] == [c.path for c in sb.train_normal]


def test_splits_order_independent():
    """Shuffling input clips must not change the resulting splits."""
    clips = _fake_clips()
    shuffled = list(reversed(clips))
    a = make_anomaly_splits(clips, seed=1337)
    b = make_anomaly_splits(shuffled, seed=1337)
    for sa, sb in zip(a, b, strict=True):
        assert [c.path for c in sa.test_normal] == [c.path for c in sb.test_normal]
        assert [c.path for c in sa.train_normal] == [c.path for c in sb.train_normal]


def test_raises_when_abnormal_geq_normal():
    # normals == abnormals leaves nothing for training -> ValueError.
    clips = _fake_clips(n_normal=3, n_abnormal=3)
    with pytest.raises(ValueError):
        make_anomaly_splits(clips)


# --- compute_pauc -----------------------------------------------------------


def test_pauc_matches_sklearn_directly():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
    assert compute_pauc(y, s) == roc_auc_score(y, s, max_fpr=0.1)


# --- render_report ----------------------------------------------------------


def _fake_results() -> list[SplitResult]:
    def sr(auc, pauc):
        return ScorerResult(auc=auc, pauc=pauc)

    return [
        SplitResult(
            machine="fan",
            machine_id="id_00",
            n_train=20,
            n_test_normal=15,
            n_test_abnormal=15,
            scorers={"knn": sr(0.72, 0.61), "mahalanobis": sr(0.80, 0.65)},
        ),
        SplitResult(
            machine="fan",
            machine_id="id_02",
            n_train=20,
            n_test_normal=15,
            n_test_abnormal=15,
            scorers={"knn": sr(0.78, 0.63), "mahalanobis": sr(0.82, 0.67)},
        ),
        SplitResult(
            machine="pump",
            machine_id="id_00",
            n_train=20,
            n_test_normal=15,
            n_test_abnormal=15,
            scorers={"knn": sr(0.70, 0.60), "mahalanobis": sr(0.75, 0.62)},
        ),
    ]


def test_render_report_contains_tables_and_type_average():
    results = _fake_results()
    averages = aggregate_by_type(results)
    report = render_report(
        results=results,
        averages=averages,
        n_clips=400,
        seed=1337,
        device="cpu",
        classifier_metrics={
            "val_acc": 0.95,
            "train_acc": 0.99,
            "n_train": 105,
            "n_val": 35,
            "val_ids": ["id_06"],
        },
        determinism="PASS — example",
        sanity_ok=True,
    )

    # Header / table headers present.
    assert "# EarSight Baseline Report" in report
    assert "## Anomaly detection — AUC / pAUC (p=0.1)" in report
    assert "| machine | id | n_train | n_test (norm/abn) | kNN AUC | kNN pAUC |" in report
    # A type-average row.
    assert "type average" in report
    # Sanity table.
    assert "## Sanity check vs published baselines" in report
    assert "DCASE2020 Task2 baseline" in report
    assert "this run (kNN, subset)" in report
    # Classifier section.
    assert "## Machine-type classifier" in report
    # Determinism section.
    assert "## Determinism" in report


def test_aggregate_by_type_means():
    results = _fake_results()
    averages = aggregate_by_type(results)
    # fan kNN AUC = mean(0.72, 0.78) = 0.75
    assert averages["fan"]["knn"].auc == pytest.approx(0.75)
    # pump has a single id -> equals that id's value.
    assert averages["pump"]["knn"].auc == pytest.approx(0.70)


def test_render_report_skipped_classifier():
    results = _fake_results()
    averages = aggregate_by_type(results)
    report = render_report(
        results=results,
        averages=averages,
        n_clips=400,
        seed=1337,
        device="cpu",
        classifier_metrics=None,
        determinism="Not run",
        sanity_ok=False,
    )
    assert "skipped" in report
    assert "SUSPECT" in report
