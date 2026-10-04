# Applied Resonance status

Last updated: 2026-10-04

What is verified, and what is still open. “Implemented,” “verified,” and
“deployed” are distinct readiness levels; see
[docs/OPERATIONS.md](docs/OPERATIONS.md) for the definitions.

## Verified

- **Benchmark.** On the full 9,755-clip MIMII 0 dB dataset, the engine’s fan
  cosine-kNN AUC is **0.693** and pump cosine-kNN AUC is **0.880**. The
  DCASE2020 baselines are **0.658** and **0.729**. Determinism is enforced to
  three decimals. See [`engine/REPORT_FULL.md`](engine/REPORT_FULL.md).
- **Pipeline.** Log-mel + PANNs CNN14 embeddings, cosine-kNN as the primary
  scorer, Ledoit-Wolf Mahalanobis as secondary, machine-ID-disjoint splits.
- **Streaming.** ONNX numerical-parity check, and an evidence layer that uses
  Hilbert-envelope peak-picking for mains hum, shaft rotation and bearing
  impulses.
- **Service.** The FastAPI scoring service has fail-closed bearer
  authentication, exact CORS and queue bounds. Model warmup runs at startup;
  recorded readiness was about 3.03 s, with a 734.5 ms baseline transition.
- **G2 app in the simulator.** The Even Hub app was verified in the official
  simulator and builds a valid `.ehpk` package.
- **Full loop on physical G2 glasses (controlled playback).** The real G2
  microphone captured a 30-second healthy pump baseline. Playback then switched
  to an abnormal recording from the same pump ID at the same volume, and the
  persistent change raised an alert on the lens. One temple tap saved the
  previous 10 seconds as evidence, and the state returned to `LISTENING` when
  normal sound came back. Scoring ran on a laptop. Recorded in the
  [89-second demo](https://www.connork.com/applied-resonance).
- **Honest negative.** A separate fan/pump classifier reached **0.577**
  validation accuracy on machine-ID-disjoint validation. The generalization gap
  is structural, so the classifier was removed from the live runtime.

## Open

- **Detailed G2 hardware checks.** The 60-second byte-cadence check, 1 kHz tone
  check, 10-minute locked-phone capture and 30-minute battery measurement have
  not been run. See step 6 of the verification ladder in
  [`apps/evenhub/README.md`](apps/evenhub/README.md).
- **Re-record kill test through the G2 microphone.** Not run yet. The current
  kill-test numbers use synthetic noise overlays only
  ([`killtest/REPORT.md`](killtest/REPORT.md)).
- **Deployment.** No deployed HTTPS gateway. Inference runs on a laptop, not on
  the glasses or phone.
- **Distribution.** No Even Hub store listing yet.
- **Field use.** No field pilots yet. All results above come from public data
  or controlled playback.
