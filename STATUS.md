# EarSight status

Last updated: 2026-07-21

This status records the verified evidence and open gates from
[`EARSIGHT_CONTEXT.md`](EARSIGHT_CONTEXT.md) §2. “Implemented,” “verified,” and
“deployed” are distinct readiness levels.

## Verified evidence

- On the full 9,755-clip MIMII 0 dB dataset, the engine’s fan cosine-kNN AUC is
  **0.693** and pump cosine-kNN AUC is **0.880**. The DCASE2020 baselines are
  **0.658** and **0.729**, respectively; determinism is enforced to three
  decimals. See `engine/REPORT_FULL.md`.
- The implemented pipeline uses log-mel + PANNs CNN14, cosine-kNN as the primary
  scorer, Ledoit-Wolf Mahalanobis as secondary, and machine-ID-disjoint splits.
- Streaming inference includes an ONNX numerical-parity check and a physics-based
  evidence layer using Hilbert-envelope peak-picking for mains hum, shaft
  rotation, and bearing impulses.
- The FastAPI serve path has fail-closed bearer authentication, exact CORS, and
  queue bounds. Model warmup was moved to startup; recorded readiness was about
  3.03 s with a 734.5 ms baseline transition.
- The Even Realities G2 EvenHub app was verified in the official simulator and
  a valid `.ehpk` was built.
- Honest negative: the fan/pump classifier reached **0.577** validation accuracy
  on machine-ID-disjoint validation. The generalization gap is structural and is
  documented rather than hidden.

## Open hardware and deployment gates

- No physical G2 hardware test.
- No deployed HTTPS gateway.
- No store listing live.
- No re-record kill test on hardware.
- No accelerometer path yet.
- Zero real users, pilots installed, and revenue.

## Wave 1 checklist

- [ ] P0 Repo hygiene + STATUS.md
- [ ] P1 VibrationSource + fused mic/accel scoring
- [ ] P2 Pi pilot image (one-command headless install)
- [ ] P3 Alerts + weekly report generator
- [ ] P4 Phone status page (token auth, no glasses)
- [ ] P5 Noise-robustness eval harness (published DCASE 2026 benchmark; conditional 2027 entry)
- [ ] P6 Datakit consent/feature-only hardening
- [ ] P7 Integration + verification (Connor personally)
