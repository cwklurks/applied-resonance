"""Re-run the IDENTICAL anomaly eval on a substituted clip tree.

After the user re-records the playlist through a speaker -> mic chain (then
segments it with :mod:`killtest.segment`) or overlays shop noise with
:mod:`killtest.noise`, the degraded TEST clips live in a parallel directory that
mirrors the MIMII relpath layout (e.g. ``0_dB/pump/id_06/abnormal/00000003.wav``).
This module re-runs the exact same per-id anomaly eval on those degraded test
clips and accumulates the result against the clean baseline.

The scorers stay fit on the ORIGINAL (clean) normal training clips; only the
test clips are substituted. That isolates the capture-chain degradation. Reuses
:func:`engine.eval.splits.make_anomaly_splits` and
:func:`engine.eval.run_eval.evaluate_split` so the protocol, seed, and scoring
machinery are bit-identical to the baseline.

Runnable as::

    uv run python -m killtest.eval_rerun --label clean --device cpu
    uv run python -m killtest.eval_rerun --label noisy_snr0 \
        --clip-root /tmp/noisy_tree/snr_0dB --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from engine.common import get_device, seed_everything
from engine.dataset import Clip, list_clips
from engine.eval.run_eval import aggregate_by_type, evaluate_split
from engine.eval.splits import IdSplit, make_anomaly_splits
from engine.paths import MIMII_DIR, REPO_ROOT
from killtest.ids import IdKey, format_id_key, parse_id_tokens

logger = logging.getLogger("killtest.eval_rerun")

RESULTS_PATH = REPO_ROOT / "killtest" / "results.json"
REPORT_PATH = REPO_ROOT / "killtest" / "REPORT.md"
SCORER_NAMES = ("knn", "mahalanobis")

EmbedFn = Callable[[Sequence[Path]], np.ndarray]


def _default_embed_clean(device: object | None) -> EmbedFn:
    """Clean originals via the disk-cached PANNs features."""

    def embed(paths: Sequence[Path]) -> np.ndarray:
        from engine.features.cache import cached_features_many

        if not paths:
            return np.empty((0, 2048), dtype=np.float32)
        return cached_features_many(list(paths), kind="panns", device=device)

    return embed


def _default_embed_fresh(device: object | None) -> EmbedFn:
    """Substituted clips: load wavs + compute PANNs embeddings fresh, NO cache.

    Substituted trees are arbitrary (re-recorded / noise-overlaid), so caching
    keyed by relpath would collide with the clean cache. Always compute fresh.
    """

    def embed(paths: Sequence[Path]) -> np.ndarray:
        from engine.audio import load_wav
        from engine.features.panns import panns_embeddings

        if not paths:
            return np.empty((0, 2048), dtype=np.float32)
        wavs = [load_wav(p) for p in paths]
        return panns_embeddings(wavs, device=device, batch_size=8)

    return embed


def _relpath(clip: Clip, mimii_root: Path) -> Path:
    """Path of ``clip`` relative to the MIMII root (e.g. 0_dB/fan/id_06/...)."""
    return clip.path.resolve().relative_to(mimii_root.resolve())


def _split_key(split: IdSplit) -> IdKey:
    return (split.machine, split.machine_id)


def _split_test_clips(split: IdSplit) -> list[Clip]:
    return [*split.test_normal, *split.test_abnormal]


def _restrict_splits(
    splits: Sequence[IdSplit],
    restrict_ids: Sequence[IdKey] | None,
) -> list[IdSplit]:
    if not restrict_ids:
        return list(splits)

    wanted = set(restrict_ids)
    found = {_split_key(split) for split in splits}
    missing = [format_id_key(key) for key in restrict_ids if key not in found]
    if missing:
        raise ValueError(
            f"requested restrict ids not present in split set: {missing}"
        )

    return [split for split in splits if _split_key(split) in wanted]


def _assert_restricted_coverage(
    splits: Sequence[IdSplit],
    clip_root: Path,
    mimii_root: Path,
) -> None:
    missing_by_id: dict[IdKey, list[Path]] = {}
    for split in splits:
        key = _split_key(split)
        for clip in _split_test_clips(split):
            rel = _relpath(clip, mimii_root)
            if not (clip_root / rel).exists():
                missing_by_id.setdefault(key, []).append(rel)

    if not missing_by_id:
        return

    lines = [
        "restricted clip_root coverage is incomplete; missing relpaths:",
    ]
    for key in sorted(missing_by_id):
        lines.append(f"{format_id_key(key)}:")
        lines.extend(f"  {rel.as_posix()}" for rel in missing_by_id[key])
    raise ValueError("\n".join(lines))


def rerun_eval(
    clips: Sequence[Clip],
    clip_root: Path | None,
    seed: int = 1337,
    restrict_ids: Sequence[IdKey] | None = None,
    embed_clean: EmbedFn | None = None,
    embed_fresh: EmbedFn | None = None,
    mimii_root: Path = MIMII_DIR,
) -> dict:
    """Score the anomaly eval with TEST clips optionally substituted.

    Builds the same per-id splits as ``run_eval`` via
    :func:`make_anomaly_splits`. TRAIN embeddings always come from the clean
    originals (``embed_clean``). For each TEST clip, its relpath under
    ``mimii_root`` is computed; if ``clip_root / relpath`` exists, that file is
    the substituted version and is embedded via ``embed_fresh``; otherwise the
    clean embedding is used.

    Args:
        clips: clean MIMII clips (their paths are the canonical identity).
        clip_root: directory holding substituted test clips mirroring the MIMII
            relpath layout, or ``None`` for a pure clean baseline pass.
        seed: split seed (must match the baseline; default 1337).
        restrict_ids: optional ``(machine, machine_id)`` keys to evaluate. Splits
            are built from all clips first, then filtered, preserving split seed
            behavior. If ``clip_root`` is given, every test clip for each
            restricted id must exist under that root.
        embed_clean: ``(paths) -> (N, 2048)``; defaults to cached PANNs features.
        embed_fresh: ``(paths) -> (N, 2048)``; defaults to fresh PANNs (no cache).
        mimii_root: MIMII root for relpath computation (default ``MIMII_DIR``).

    Returns:
        ``{"n_replaced", "n_test", "per_id": [...], "averages": {...}}``.

    Raises:
        ValueError: if ``clip_root`` is given but no test clip is found under it,
            or if restricted ids are missing substituted test coverage.
    """
    if embed_clean is None:
        embed_clean = _default_embed_clean(None)
    if embed_fresh is None:
        embed_fresh = _default_embed_fresh(None)

    splits = _restrict_splits(make_anomaly_splits(clips, seed=seed), restrict_ids)

    if restrict_ids and clip_root is not None:
        _assert_restricted_coverage(splits, Path(clip_root), mimii_root)

    # Every clip that participates: train clips + test clips across all splits.
    train_clips: list[Clip] = []
    test_clips: list[Clip] = []
    for split in splits:
        train_clips.extend(split.train_normal)
        test_clips.extend(_split_test_clips(split))

    # TRAIN embeddings: always clean. Keyed by the clip's resolved path string,
    # matching evaluate_split's lookup.
    embeddings: dict[str, np.ndarray] = {}
    if train_clips:
        train_embeds = embed_clean([c.path for c in train_clips])
        for clip, emb in zip(train_clips, train_embeds):
            embeddings[str(clip.path.resolve())] = emb

    # TEST embeddings: substitute where a file exists under clip_root.
    substituted_paths: list[Path] = []
    substituted_clips: list[Clip] = []
    clean_test_clips: list[Clip] = []
    for clip in test_clips:
        sub_path = None
        if clip_root is not None:
            candidate = Path(clip_root) / _relpath(clip, mimii_root)
            if candidate.exists():
                sub_path = candidate
        if sub_path is not None:
            substituted_clips.append(clip)
            substituted_paths.append(sub_path)
        else:
            clean_test_clips.append(clip)

    n_replaced = len(substituted_clips)
    n_test = len(test_clips)

    if clip_root is not None and n_replaced == 0:
        raise ValueError(
            "no test clips found under clip_root — wrong directory?"
        )

    if clean_test_clips:
        clean_test_embeds = embed_clean([c.path for c in clean_test_clips])
        for clip, emb in zip(clean_test_clips, clean_test_embeds):
            embeddings[str(clip.path.resolve())] = emb

    if substituted_clips:
        fresh_embeds = embed_fresh(substituted_paths)
        for clip, emb in zip(substituted_clips, fresh_embeds):
            # Keyed by the ORIGINAL clip path so evaluate_split finds it; the
            # VALUE is the substituted (degraded) embedding.
            embeddings[str(clip.path.resolve())] = emb

    results = [evaluate_split(split, embeddings) for split in splits]
    averages = aggregate_by_type(results)

    per_id = [
        {
            "machine": r.machine,
            "machine_id": r.machine_id,
            "scorer": name,
            "auc": r.scorers[name].auc,
            "pauc": r.scorers[name].pauc,
        }
        for r in results
        for name in SCORER_NAMES
    ]

    averages_out = {
        machine: {
            name: {"auc": scorers[name].auc, "pauc": scorers[name].pauc}
            for name in SCORER_NAMES
        }
        for machine, scorers in averages.items()
    }

    return {
        "n_replaced": n_replaced,
        "n_test": n_test,
        "per_id": per_id,
        "averages": averages_out,
    }


def _load_results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return {"conditions": {}}


def accumulate_results(
    label: str, result: dict, clip_root: Path | None
) -> dict:
    """Load results.json, REPLACE the entry for ``label``, write it back.

    Returns the full results dict (``{"conditions": {label: {...}}}``).
    """
    data = _load_results()
    conditions = data.setdefault("conditions", {})
    conditions[label] = {
        **result,
        "clip_root": str(clip_root) if clip_root is not None else None,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n"
    )
    return data


def _ordered_conditions(conditions: dict) -> list[str]:
    """Condition labels with ``clean`` first (if present), then sorted rest."""
    rest = sorted(k for k in conditions if k != "clean")
    return (["clean"] if "clean" in conditions else []) + rest


def _avg_auc(condition: dict, machine: str, scorer: str) -> str:
    avg = condition.get("averages", {}).get(machine, {}).get(scorer)
    return f"{avg['auc']:.3f}" if avg else "n/a"


def render_report(data: dict) -> str:
    """Render REPORT.md from the accumulated results dict."""
    conditions = data.get("conditions", {})
    lines: list[str] = []
    lines.append("# Kill-test report — degraded-capture eval vs clean baseline")
    lines.append("(auto-generated by `python -m killtest.eval_rerun`)")
    lines.append("")
    lines.append(
        "| condition | n_replaced | coverage | fan kNN AUC | pump kNN AUC | "
        "fan Mahal. AUC | pump Mahal. AUC |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for label in _ordered_conditions(conditions):
        cond = conditions[label]
        n_replaced = cond.get("n_replaced", 0)
        n_test = cond.get("n_test", 0)
        lines.append(
            f"| {label} | {n_replaced} | {n_replaced}/{n_test} | "
            f"{_avg_auc(cond, 'fan', 'knn')} | "
            f"{_avg_auc(cond, 'pump', 'knn')} | "
            f"{_avg_auc(cond, 'fan', 'mahalanobis')} | "
            f"{_avg_auc(cond, 'pump', 'mahalanobis')} |"
        )
    lines.append("")
    lines.append("## Pass/fail guide (from the build spec)")
    lines.append(
        "PASS: re-recorded AUC retains >= 80% of the clean margin above chance "
        "(clean 0.90 -> >= 0.78)."
    )
    lines.append(
        "SOFT FAIL (0.65-0.78 of clean ~0.85-0.90): mic chain hurts; plan "
        "augmentation. HARD FAIL (< 0.65): pivot capture path."
    )
    lines.append("")
    return "\n".join(lines)


def write_report(data: dict) -> Path:
    """Regenerate REPORT.md from ``data`` and return its path."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(data))
    return REPORT_PATH


def _print_summary(label: str, result: dict, report_path: Path) -> None:
    print(f"\n=== eval_rerun: {label} ===")
    print(
        f"  test clips: {result['n_test']}  replaced: {result['n_replaced']}  "
        f"coverage: {result['n_replaced']}/{result['n_test']}"
    )
    for machine in sorted(result["averages"]):
        knn = result["averages"][machine]["knn"]
        mah = result["averages"][machine]["mahalanobis"]
        print(
            f"  {machine:5s}  kNN AUC={knn['auc']:.3f} pAUC={knn['pauc']:.3f}  |  "
            f"Mahalanobis AUC={mah['auc']:.3f} pAUC={mah['pauc']:.3f}"
        )
    print(f"  report: {report_path}")


def run(args: argparse.Namespace) -> int:
    seed_everything(args.seed)
    device = get_device(args.device)

    clip_root = Path(args.clip_root) if args.clip_root else None
    clips = list_clips(snr=args.snr)
    logger.info("listed %d clips for snr=%s", len(clips), args.snr)

    result = rerun_eval(
        clips=clips,
        clip_root=clip_root,
        seed=args.seed,
        restrict_ids=parse_id_tokens(
            getattr(args, "restrict_ids", None),
            option_name="--restrict-ids",
        ),
        embed_clean=_default_embed_clean(device),
        embed_fresh=_default_embed_fresh(device),
    )

    data = accumulate_results(args.label, result, clip_root)
    report_path = write_report(data)
    _print_summary(args.label, result, report_path)
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run the anomaly eval on a substituted clip tree."
    )
    parser.add_argument(
        "--label",
        required=True,
        help='condition label, e.g. "clean", "noisy_snr0", "rerecorded_macmic"',
    )
    parser.add_argument(
        "--clip-root",
        default=None,
        help="dir of substituted test clips (omit for the clean baseline)",
    )
    parser.add_argument("--snr", default="0_dB")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--restrict-ids",
        nargs="+",
        default=None,
        help=(
            "only evaluate these machine ids, e.g. "
            "--restrict-ids fan:id_06 pump:id_02"
        ),
    )
    parser.add_argument("--device", default=None, help="cpu|cuda|mps")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
