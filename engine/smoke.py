"""One-command smoke test for the EarSight Phase 0 pipeline.

Runs the full baseline end-to-end, in-process, on CPU::

    uv run python -m engine.smoke

Stages: download (idempotent) -> eval (features -> fit -> score -> classifier
-> report -> determinism double-pass) -> verify (assert REPORT_BASELINE.md was
written with the expected sections). Prints a stage-timing table and asserts the
whole thing finishes under the budget. Exit 0 = PASS, exit 1 = FAIL.
"""

import argparse
import time

import engine.download
import engine.eval.run_eval
from engine.paths import REPO_ROOT

REPORT_PATH = REPO_ROOT / "engine" / "REPORT_BASELINE.md"
REQUIRED_REPORT_STRINGS = (
    "Anomaly detection",
    "type average",
    "Determinism",
    "PASS",
)


def verify_report(text: str) -> None:
    """Assert the rendered report contains every required marker string."""
    missing = [s for s in REQUIRED_REPORT_STRINGS if s not in text]
    if missing:
        raise AssertionError(f"REPORT_BASELINE.md missing strings: {missing}")


def _as_exit_code(result: object) -> int:
    """Normalise a stage return (int or SystemExit) to an exit code."""
    if isinstance(result, SystemExit):
        return 0 if result.code is None else int(result.code)
    return int(result)


def _run_eval(subset: int, device: str, seed: int) -> int:
    argv = ["--device", device, "--seed", str(seed), "--check-determinism"]
    try:
        return _as_exit_code(engine.eval.run_eval.main(argv))
    except SystemExit as exc:  # main() may sys.exit(); treat code as the result
        return _as_exit_code(exc)


def run(subset: int, device: str, budget_s: float, seed: int) -> int:
    """Execute all stages; return the process exit code (0 PASS / 1 FAIL)."""
    timings: dict[str, float] = {}
    failure: str | None = None

    # 1) download — idempotent; 0 requests when the subset is already present.
    t0 = time.perf_counter()
    dl = engine.download.main(
        ["--subset", str(subset), "--snr", "0", "--machines", "fan", "pump"]
    )
    timings["download"] = time.perf_counter() - t0
    if int(dl) != 0:
        failure = f"download stage exited {dl}"

    # 2) eval — features -> fit -> score -> classifier -> report -> determinism.
    if failure is None:
        t0 = time.perf_counter()
        code = _run_eval(subset, device, seed)
        timings["eval"] = time.perf_counter() - t0
        if code == 2:
            failure = "sanity SUSPECT — investigate REPORT_BASELINE.md"
        elif code != 0:
            failure = f"eval stage exited {code}"

    # 3) verify — the report exists and carries the expected sections.
    if failure is None:
        t0 = time.perf_counter()
        assert REPORT_PATH.exists(), f"missing report: {REPORT_PATH}"
        verify_report(REPORT_PATH.read_text())
        timings["verify"] = time.perf_counter() - t0

    total = sum(timings.values())
    _print_timings(timings, total)

    over_budget = total > budget_s
    if failure is None and not over_budget:
        print(f"SMOKE: PASS (total {total:.0f}s < budget {budget_s:.0f}s)")
        return 0

    reason = failure or f"total {total:.0f}s > budget {budget_s:.0f}s"
    print(f"SMOKE: FAIL ({reason})")
    return 1


def _print_timings(timings: dict[str, float], total: float) -> None:
    print("\n=== EarSight smoke — stage timings ===")
    print(f"  {'stage':<10} {'seconds':>10}")
    for stage in ("download", "eval", "verify"):
        if stage in timings:
            print(f"  {stage:<10} {timings[stage]:>10.1f}")
    print(f"  {'TOTAL':<10} {total:>10.1f}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the EarSight Phase 0 pipeline end-to-end as a smoke test."
    )
    parser.add_argument("--subset", type=int, default=50)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu|cuda|mps (default cpu — the smoke test proves CPU viability)",
    )
    parser.add_argument("--budget-s", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(
        subset=args.subset,
        device=args.device,
        budget_s=args.budget_s,
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
