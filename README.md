# EarSight — Phase 0: MIMII acoustic-anomaly baseline

EarSight is an acoustic-anomaly-detection project built on the MIMII dataset. Phase 0 establishes a reproducible baseline: it downloads MIMII audio, extracts log-mel and related spectral features, trains a classifier to distinguish normal from anomalous machine sounds, and evaluates detection performance. This repository is `uv`-managed and pinned to Python 3.12.

## Setup

Install dependencies and create the virtual environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

This creates `.venv` and resolves the lockfile (`uv.lock`). Prefix project commands with `uv run` to use the managed environment.

## Download data

<!-- TODO -->

## Feature extraction

<!-- TODO -->

## Train classifier

<!-- TODO -->

## Evaluate

<!-- TODO -->

## Smoke test

<!-- TODO -->

## Run tests

<!-- TODO -->
