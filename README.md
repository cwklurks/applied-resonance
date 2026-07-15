# EarSight — Phase 0: MIMII acoustic-anomaly baseline

EarSight asks whether machine-fault audio classification survives a consumer-microphone capture chain. Phase 0 builds the baseline half of that kill test: anomaly detection on MIMII fan and pump recordings using PANNs CNN14 embeddings scored with per-id kNN and Mahalanobis distance, plus a small log-mel CNN that classifies machine type, evaluated with AUC / pAUC against the published DCASE2020 baselines. It also ships a re-record kit (`killtest/`) that generates a playback playlist, segments a single re-recorded WAV back into clips, and overlays shop noise at fixed SNRs, so the same eval can later run on consumer-mic audio.

This repository is `uv`-managed and pinned to Python 3.12.

## Setup

Install dependencies and create the virtual environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

This creates `.venv` and resolves the lockfile (`uv.lock`). Torch installs CPU/MPS wheels on a Mac; on a CUDA box (e.g. an RTX 5090) `engine.common.get_device()` auto-detects `cuda`. Prefix every project command with `uv run`.

## Smoke test (one command)

```bash
uv run python -m engine.smoke
```

Runs the full pipeline in-process on CPU — download (idempotent) → PANNs + log-mel features → fit scorers → eval → classifier → `REPORT_BASELINE.md` → determinism double-pass — then prints a stage-timing table and asserts the whole run finishes under 10 minutes (600 s). Exit 0 = PASS, exit 1 = FAIL (budget exceeded or a stage failed); a `sanity SUSPECT` eval (exit 2) also fails the smoke test.

On a fresh clone the first run additionally downloads the ~755 MB MIMII subset and the 359 MB PANNs checkpoint (network-dependent, so the first run is slower); every rerun reads from the on-disk caches and is much faster. Flags: `--subset 50`, `--device cpu` (pass `cuda`/`mps` to prove a non-CPU path), `--budget-s 600`, `--seed 1337`.

## Download data

Subset mode (default for the smoke test): 50 clips per machine id — 35 normal + 15 abnormal:

```bash
uv run python -m engine.download --subset 50 --snr 0 --machines fan pump
```

Full download — documented separately because of the disk footprint:

```bash
uv run python -m engine.download --snr 0 --machines fan pump
```

> **Disk warning:** the full 0 dB fan + pump zips are ~18.3 GB. Downloads stream to resumable `.part` files, so an interrupted run picks up where it left off.

Layout written under `data/mimii/0_dB/{machine}/id_XX/{normal,abnormal}/`. Each clip is a 16 kHz, 10 s, 8-channel WAV (channel 0 is used downstream). Reruns are idempotent — already-present clips are skipped with 0 network requests. Other SNRs are available via `--snr -6 | 0 | 6`.

## Feature extraction

Two feature kinds, both reached through one dispatch function:

```python
from engine.audio import load_wav
from engine.features import extract_features

wav = load_wav("data/mimii/0_dB/fan/id_00/normal/00000000.wav")  # 1-D float32, 16 kHz
logmel = extract_features(wav, "logmel")  # (64, 313) log-mel spectrogram
panns  = extract_features(wav, "panns")   # (2048,)  PANNs Cnn14_16k embedding
```

`kind="logmel"` returns a `(64, 313)` array; `kind="panns"` returns a `(2048,)` embedding. The PANNs Cnn14_16k checkpoint auto-downloads to `data/checkpoints/` on first use. For batches, use `engine.features.cache.cached_features(wav_path, kind)` (single) or `cached_features_many(paths, kind)` (batched) — results are cached as `.npy` under `data/cache/{kind}/`, mirroring the MIMII path layout, so features are computed once and reused.

## Train classifier

```bash
uv run python -m engine.train_classifier --epochs 5 --device cpu
```

Trains the log-mel machine-type CNN (fan vs pump) on normal clips only. The train/val split is **id-disjoint** (`--val-ids id_06` by default), so validation measures generalisation to unseen hardware.

> **Honest subset caveat:** on the 400-clip subset val accuracy sits at ≈ 0.50 (chance) while train accuracy is ≈ 0.995 — an id-disjoint generalisation gap, not a bug. With only 3 training ids per machine the CNN latches onto id-specific spectral signatures that don't transfer to the held-out id. The per-id anomaly scorers are unaffected. Full diagnosis is in `engine/REPORT_BASELINE.md`; expect this to recover with more ids/clips on the full dataset.

## Evaluate

```bash
uv run python -m engine.eval.run_eval --device cpu --check-determinism
```

Computes PANNs embeddings (cached), fits a kNN and a Mahalanobis scorer **per machine id on normal clips only** (MIMII protocol), scores the held-out normal + abnormal test clips, and reports **AUC** and **pAUC** (DCASE2020 convention, `max_fpr=0.1`) per id and per machine type. It also trains the classifier and runs a determinism double-pass, then writes the full table to `engine/REPORT_BASELINE.md` with a sanity check against the published DCASE2020 baselines. Exit 0 = OK; **exit 2 = sanity SUSPECT** (report still written, but a type-average kNN AUC fell outside the plausible band — investigate before trusting).

> **Full-dataset results** (all 9,755 0 dB fan+pump clips, run on CUDA): fan kNN AUC **0.693**, pump **0.880** — both above the DCASE2020 AE baselines (0.658 / 0.729), determinism PASS. Snapshot in `engine/REPORT_FULL.md`. Note: with the full dataset on disk, the smoke test's eval stage exceeds its 10-minute CPU budget — the smoke budget is specified for the dev subset.

## Live streaming (Phase 1)

The streaming stack scores 3 s windows hopped every 1 s against a per-machine
baseline: PANNs embedding → kNN distance → EMA (α=0.3) → percentile vs the
baseline's own calibration holdout → a conservative LISTENING → SUSPECT →
ALERT policy (5 consecutive windows ≥ P99 to alert, P95 hysteresis exit,
30 s cooldown). On SUSPECT/ALERT an evidence bundle explains why (deviating
mel bands in Hz, envelope-spectrum peaks labeled line-hum / 1x-rotation /
impulse-train, nearest baseline windows) plus a ≤60-char HUD line.

All audio I/O flows through `shared.audio_source.AudioSource` (file, mic —
glasses later). The machine-type CNN is **not** in the runtime path
(`tests/test_runtime_imports.py` enforces this); it remains for offline eval.

### Scoring service (FastAPI)

Local development is loopback-only and does not require a token:

```bash
EARSIGHT_DEVICE=cpu uv run python -m engine.serve
```

`python -m engine.serve` is the supported launcher and always binds a loopback
IP. Remote clients must arrive through the local HTTPS gateway; the service
rejects a non-loopback `EARSIGHT_BIND_HOST`. Do not expose a raw HTTP listener
to the LAN.

For a phone or glasses session, put a trusted HTTPS gateway in front of the
loopback listener, then enable the remote policy. The CORS value is the exact
origin hosting the companion WebView, not the engine origin:

```bash
export EARSIGHT_REMOTE_ACCESS=1
export EARSIGHT_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export EARSIGHT_CORS_ORIGINS=https://companion.example.com
export EARSIGHT_BIND_HOST=127.0.0.1
EARSIGHT_DEVICE=cpu uv run python -m engine.serve
```

Configure the HTTPS gateway to forward `https://engine.example.com` to
`http://127.0.0.1:8000`. Keep the generated token only in the engine process
environment and enter it into the Even Hub companion prompt at runtime; never
place it in a URL, manifest, Vite variable, source file, or checked-in `.env`.

- `POST /session/start` `{mode: "session"|"saved"|"library", tag, rpm?}` —
  session mode captures the first 30 s as the baseline; saved mode loads a
  persisted baseline by tag; library mode is a documented stub (501).
- `POST /score` `{session_id, pcm_b64}` (16 kHz mono int16 LE base64) →
  `{state, score, percentile, evidence_line}`.
- `POST /label` — writes WAV + sidecar JSON to `datakit/data/`.
- `POST /capture/start|append|stop` — bounded raw PCM capture finalized as WAV.
- `GET /baselines/{tag}` — authenticated baseline-existence lookup.
- `GET /health` — public liveness only: exactly `{"status":"ok"}`.

Every route except `/health` requires `Authorization: Bearer …` in remote mode.
CORS permits only `GET`/`POST` and `Authorization`/`Content-Type` from the exact
configured origins. Defaults are bounded to a 3 MiB HTTP body, 2 s score/capture
chunks, 60 s labels, 8 scoring sessions (120 s idle), and 2 raw captures
(300 s/16 MiB total, 30 s idle). Expired captures are closed and deleted;
startup removes non-resumable `.raw`/`.wav.part` files, and WAV finalization
copies in 64 KiB chunks.

Baselines persist across restarts (`data/baselines/`); sessions do not.
Backend/device via `EARSIGHT_BACKEND` (`torch`|`onnx`) and `EARSIGHT_DEVICE`.

### Desktop demo (Streamlit)

```bash
uv run streamlit run desktop/app.py
```

Input-device picker, rolling 10 s spectrogram, capture-baseline (30 s) with
countdown, state badge + percentile gauge, evidence panel, and a
record-and-label form writing to `datakit/data/`. macOS prompts for mic
permission on first use. The 60-second demo video script is in
`desktop/DEMO.md`.

### ONNX export

```bash
uv run python -m engine.export_onnx          # writes data/artifacts/cnn14_16k.onnx
uv run python -m engine.export_onnx --bench  # prints torch + onnx real-time factors
```

Parity vs PyTorch: max |Δ| ≈ 2.1e-6 on 20 real clips (tolerance 1e-4).
Measured real-time factor on this machine (CPU, 3 s window / 1 s hop):
**RTF[onnx] ≈ 0.02, RTF[torch] ≈ 0.015** — ~25× under the 0.5 target, so the
default 1 s hop is comfortable (a 2 s hop remains available via the
`hop_s` parameter if ever needed on weaker hardware).

## Re-record kit (`killtest/`)

The kill test's consumer-mic half. Three CLIs:

```bash
# 1) Build a playback playlist (clips + 2 s silences + leading sync chirp) and a manifest.
uv run python -m killtest.playlist --per-id 5 --label both --out-dir killtest_out

# 2) Segment one long re-recorded WAV back into per-clip files using chirp + cross-correlation.
uv run python -m killtest.segment --recording rec.wav \
  --manifest killtest_out/playlist_manifest.json --out-dir killtest_out/segmented

# 3) Overlay shop-noise WAVs onto clips at fixed SNRs for a synthetic noise sweep.
uv run python -m killtest.noise --noise-dir <noise_wavs> --snr 0 5 10 --out-dir killtest_out/noisy
```

Physical protocol (your ~1 afternoon of hands-on work):

- Play `killtest_out/playlist.wav` through a decent speaker (bookshelf/studio monitor, not a laptop), 30–50 cm from the mic, at realistic volume.
- Record **one long WAV per mic chain** (Mac internal, phone, cheap MEMS/earbud, plus a good-mic control to separate speaker coloration from mic degradation).
- Run `killtest.segment` to slice that recording back into aligned clips; use `killtest.noise` for synthetic SNR sweeps without re-recording.

Re-run the identical eval on segmented or noise-overlaid clips (scorers stay
fit on clean normals; results accumulate in `killtest/REPORT.md`):

```bash
uv run python -m killtest.eval_rerun --label rerecorded_macmic \
  --clip-root killtest_out/segmented --device cpu
```

## Glasses app (Even Hub)

The first glasses shell lives in [`apps/evenhub/`](apps/evenhub/README.md) — an
Even Hub WebView app (official asr template lineage) that streams glasses-mic
PCM to the FastAPI service and renders the two-line HUD contract (≤1 push per
2 s). Shared lens formatting + the typed engine client live in
`shared/display-card/` (TypeScript) so the upcoming MentraOS shell cannot
drift. See the app README for the dev loop, simulator, packaging, mock mode,
and the hardware-unknowns test plan.

```bash
cd shared/display-card && npm install && npm test   # HUD contract suite
cd apps/evenhub && npm install && npm test          # app + integration suite
```

## Run tests

```bash
uv run pytest
```

## Repo layout

```
earsight/
  shared/audio_source.py   # AudioSource seam: FileSource, MicSource (glasses later)
  engine/
    download.py            # MIMII download (full + subset modes)
    audio.py               # WAV I/O (channel-0, 16 kHz)
    features/              # logmel + PANNs extraction, disk cache
    models/                # MachineTypeCNN, PANNs Cnn14
    embedder.py            # torch | onnx embedding backends
    stream.py              # 3 s / 1 s streaming scorer with EMA
    baseline.py            # session/saved baselines, percentile calibration
    policy.py              # LISTENING -> SUSPECT -> ALERT state machine
    evidence.py            # mel-band + envelope-spectrum explanations
    export_onnx.py         # ONNX export + RTF benchmark
    serve.py               # FastAPI scoring service
    labeling.py            # labeled-clip writer (datakit/data/)
    eval/run_eval.py       # AUC/pAUC anomaly eval + report
    train_classifier.py    # machine-type CNN (offline only, not in runtime)
    smoke.py               # one-command end-to-end smoke test
    REPORT_BASELINE.md     # generated eval report
  desktop/                 # Streamlit live demo app + DEMO.md video script
  killtest/                # playlist / segment / noise / eval_rerun CLIs
  datakit/data/            # labeled field clips (WAV + JSON sidecars)
  tests/                   # pytest suite (one CI command: uv run pytest)
  data/                    # mimii, checkpoints, cache, artifacts (gitignored)
```

## Determinism

Seeds are fixed at `1337` across `random`, NumPy, and Torch (`engine.common.seed_everything`); torch deterministic algorithms and cuDNN-deterministic flags are enabled. Features are cached so identical inputs reuse identical arrays. With `--check-determinism`, the eval runs the whole pipeline twice and asserts every AUC / pAUC matches to 3 decimals before writing the report.
