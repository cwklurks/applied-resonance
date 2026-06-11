"""Synthetic, fast tests for killtest.eval_rerun.

No PANNs, no real MIMII data. A tiny fake MIMII tree is built in tmp_path and
embeddings are injected via deterministic per-path hash functions, so the whole
eval-rerun flow (splits -> substitution -> scoring -> accumulation -> report)
runs in milliseconds.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import killtest.eval_rerun as er
from engine.dataset import Clip

SR = 16000
DIM = 2048
N_NORMAL = 8
N_ABNORMAL = 3
MACHINE = "fan"
IDS = ("id_00", "id_02")


# --------------------------------------------------------------------------- #
# Fixtures: fake MIMII tree + clips + injectable embeddings
# --------------------------------------------------------------------------- #


def _write_wav(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    data = (0.1 * rng.standard_normal(int(0.1 * SR))).astype(np.float32)
    sf.write(str(path), data, SR, subtype="PCM_16")


def _build_tree(root: Path) -> list[Clip]:
    """Build {root}/0_dB/fan/id_*/{normal,abnormal}/*.wav and return clips."""
    clips: list[Clip] = []
    seed = 0
    for mid in IDS:
        for label, n in (("normal", N_NORMAL), ("abnormal", N_ABNORMAL)):
            for i in range(n):
                seed += 1
                p = root / "0_dB" / MACHINE / mid / label / f"{i:08d}.wav"
                _write_wav(p, seed=seed)
                clips.append(Clip(p, MACHINE, mid, label))
    return clips


def _hash_vec(path: Path, offset: float = 0.0) -> np.ndarray:
    """Deterministic (2048,) vector from a clip's resolved path."""
    key = str(Path(path).resolve()).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "big") % (2**32))
    return (rng.standard_normal(DIM).astype(np.float32) + offset)


def _make_embedders():
    """Return (embed_clean, embed_fresh, calls) where calls records fresh paths.

    embed_clean: hash(resolved original path) -> vector.
    embed_fresh: hash(resolved original path) + constant offset, BUT we key the
        fresh hash off the *substituted* file path so substituted clips shift.
    """
    fresh_calls: list[Path] = []

    def embed_clean(paths):
        return np.stack([_hash_vec(p) for p in paths]) if paths else np.empty(
            (0, DIM), dtype=np.float32
        )

    def embed_fresh(paths):
        fresh_calls.extend(Path(p) for p in paths)
        return np.stack([_hash_vec(p, offset=5.0) for p in paths]) if paths else (
            np.empty((0, DIM), dtype=np.float32)
        )

    return embed_clean, embed_fresh, fresh_calls


@pytest.fixture
def fake_mimii(tmp_path):
    root = tmp_path / "mimii"
    clips = _build_tree(root)
    return root, clips


@pytest.fixture(autouse=True)
def _redirect_outputs(tmp_path, monkeypatch):
    """Point results.json / REPORT.md at tmp_path so tests never touch the repo."""
    monkeypatch.setattr(er, "RESULTS_PATH", tmp_path / "results.json")
    monkeypatch.setattr(er, "REPORT_PATH", tmp_path / "REPORT.md")


# --------------------------------------------------------------------------- #
# (a) clip_root=None -> clean baseline pass
# --------------------------------------------------------------------------- #


def test_clean_pass_no_replacement(fake_mimii):
    root, clips = fake_mimii
    embed_clean, embed_fresh, fresh_calls = _make_embedders()

    result = er.rerun_eval(
        clips=clips,
        clip_root=None,
        embed_clean=embed_clean,
        embed_fresh=embed_fresh,
        mimii_root=root,
    )

    assert result["n_replaced"] == 0
    assert result["n_test"] == len(IDS) * (N_ABNORMAL + N_ABNORMAL)  # norm+abn
    assert fresh_calls == []  # embed_fresh never called on a clean pass
    # Structure complete.
    assert set(result.keys()) == {"n_replaced", "n_test", "per_id", "averages"}
    assert MACHINE in result["averages"]
    for scorer in ("knn", "mahalanobis"):
        assert "auc" in result["averages"][MACHINE][scorer]
        assert "pauc" in result["averages"][MACHINE][scorer]
    # Two ids x two scorers = 4 per_id rows.
    assert len(result["per_id"]) == len(IDS) * 2


def test_clean_pass_deterministic(fake_mimii):
    root, clips = fake_mimii
    ec1, ef1, _ = _make_embedders()
    ec2, ef2, _ = _make_embedders()
    r1 = er.rerun_eval(clips, None, embed_clean=ec1, embed_fresh=ef1, mimii_root=root)
    r2 = er.rerun_eval(clips, None, embed_clean=ec2, embed_fresh=ef2, mimii_root=root)
    assert r1 == r2


# --------------------------------------------------------------------------- #
# (b) clip_root with SOME test clips substituted
# --------------------------------------------------------------------------- #


def test_partial_substitution(fake_mimii, tmp_path):
    root, clips = fake_mimii
    embed_clean, embed_fresh, fresh_calls = _make_embedders()

    # Determine which clips are test clips (so we substitute real test clips).
    from engine.eval.splits import make_anomaly_splits

    splits = make_anomaly_splits(clips, seed=1337)
    test_clips = []
    for s in splits:
        test_clips.extend(s.test_normal)
        test_clips.extend(s.test_abnormal)

    # Copy only the first 3 test clips into a substituted tree.
    to_sub = test_clips[:3]
    clip_root = tmp_path / "subtree"
    sub_paths = []
    for clip in to_sub:
        rel = clip.path.resolve().relative_to(root.resolve())
        dest = clip_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(clip.path, dest)
        sub_paths.append(dest.resolve())

    result = er.rerun_eval(
        clips=clips,
        clip_root=clip_root,
        embed_clean=embed_clean,
        embed_fresh=embed_fresh,
        mimii_root=root,
    )

    assert result["n_replaced"] == 3
    # embed_fresh was called ONLY with the substituted file paths (never train).
    assert sorted(p.resolve() for p in fresh_calls) == sorted(sub_paths)


def test_train_embeddings_never_fresh(fake_mimii, tmp_path):
    """Substitute every test clip; embed_fresh must still never see a train clip."""
    root, clips = fake_mimii
    embed_clean, embed_fresh, fresh_calls = _make_embedders()

    from engine.eval.splits import make_anomaly_splits

    splits = make_anomaly_splits(clips, seed=1337)
    train_paths = set()
    test_clips = []
    for s in splits:
        train_paths.update(c.path.resolve() for c in s.train_normal)
        test_clips.extend(s.test_normal)
        test_clips.extend(s.test_abnormal)

    clip_root = tmp_path / "subtree"
    for clip in test_clips:
        rel = clip.path.resolve().relative_to(root.resolve())
        dest = clip_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(clip.path, dest)

    result = er.rerun_eval(
        clips, clip_root, embed_clean=embed_clean, embed_fresh=embed_fresh,
        mimii_root=root,
    )
    assert result["n_replaced"] == len(test_clips)
    # No fresh-embedded path corresponds to a clean train clip's relpath.
    for fp in fresh_calls:
        rel = fp.resolve().relative_to(clip_root.resolve())
        assert (root / rel).resolve() not in train_paths


# --------------------------------------------------------------------------- #
# (c) empty clip_root raises
# --------------------------------------------------------------------------- #


def test_empty_clip_root_raises(fake_mimii, tmp_path):
    root, clips = fake_mimii
    embed_clean, embed_fresh, _ = _make_embedders()
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no test clips found"):
        er.rerun_eval(
            clips, empty, embed_clean=embed_clean, embed_fresh=embed_fresh,
            mimii_root=root,
        )


# --------------------------------------------------------------------------- #
# (d) results.json accumulation
# --------------------------------------------------------------------------- #


def test_results_accumulation(fake_mimii):
    root, clips = fake_mimii
    ec, ef, _ = _make_embedders()
    r_clean = er.rerun_eval(clips, None, embed_clean=ec, embed_fresh=ef, mimii_root=root)

    er.accumulate_results("clean", r_clean, None)
    er.accumulate_results("noisy_snr0", r_clean, Path("/tmp/x"))

    data = json.loads(er.RESULTS_PATH.read_text())
    assert set(data["conditions"]) == {"clean", "noisy_snr0"}
    assert data["conditions"]["clean"]["clip_root"] is None
    assert data["conditions"]["noisy_snr0"]["clip_root"] == "/tmp/x"

    # Rerunning a label REPLACES (does not duplicate).
    er.accumulate_results("clean", r_clean, None)
    data2 = json.loads(er.RESULTS_PATH.read_text())
    assert set(data2["conditions"]) == {"clean", "noisy_snr0"}


# --------------------------------------------------------------------------- #
# (e) REPORT.md content
# --------------------------------------------------------------------------- #


def test_report_table(fake_mimii):
    root, clips = fake_mimii
    ec, ef, _ = _make_embedders()
    r = er.rerun_eval(clips, None, embed_clean=ec, embed_fresh=ef, mimii_root=root)

    data = er.accumulate_results("clean", r, None)
    er.accumulate_results("noisy_snr0", r, Path("/tmp/x"))
    data = json.loads(er.RESULTS_PATH.read_text())
    er.write_report(data)

    report = er.REPORT_PATH.read_text()
    assert "| condition | n_replaced | fan kNN AUC |" in report
    # One body row per condition, clean first.
    body = [ln for ln in report.splitlines() if ln.startswith("| ")]
    # header + separator + 2 condition rows = 4 table lines starting with "| ".
    table_rows = [ln for ln in body if "---" not in ln and "condition" not in ln]
    assert len(table_rows) == 2
    assert table_rows[0].startswith("| clean ")
    assert table_rows[1].startswith("| noisy_snr0 ")
