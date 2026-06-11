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

> The glue that re-runs the baseline eval directly on segmented clips is coming next.

## Run tests

```bash
uv run pytest
```

## Repo layout

```
earsight/
  engine/
    download.py            # MIMII download (full + subset modes)
    audio.py               # WAV I/O (channel-0, 16 kHz)
    features/              # logmel + PANNs extraction, disk cache
    models/                # MachineTypeCNN, PANNs Cnn14
    eval/run_eval.py       # AUC/pAUC anomaly eval + report
    train_classifier.py    # machine-type CNN (id-disjoint split)
    smoke.py               # one-command end-to-end smoke test
    REPORT_BASELINE.md     # generated eval report
  killtest/                # playlist / segment / noise CLIs
  tests/                   # pytest suite
  data/                    # mimii, checkpoints, cache, artifacts (gitignored)
```

## Determinism

Seeds are fixed at `1337` across `random`, NumPy, and Torch (`engine.common.seed_everything`); torch deterministic algorithms and cuDNN-deterministic flags are enabled. Features are cached so identical inputs reuse identical arrays. With `--check-determinism`, the eval runs the whole pipeline twice and asserts every AUC / pAUC matches to 3 decimals before writing the report.
