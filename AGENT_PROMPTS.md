# Wave 1 Agent Prompts — EarSight Pivot
*Paste these one at a time into Claude Code / Codex. Written in your standard contract style. Each prompt assumes EARSIGHT_CONTEXT.md is committed at repo root — commit it FIRST. File paths are from the June session; every prompt starts with recon so agents adapt to drift instead of guessing. Dispatch order: P0 → P1 → (P2 ∥ P3 ∥ P4 ∥ P5 ∥ P6) → P7 (you personally). Don't run two writers in the same files concurrently.*

---

## P0 — Repo hygiene + STATUS.md (run this before anything else)

```
Task: Repository hygiene pass and STATUS.md — no feature work.

Context: /Users/connork/code/earsight (git, branch main). Read EARSIGHT_CONTEXT.md at repo root first; it is ground truth. This repo has accumulated modified/untracked entries across sessions. Recon first: `git status --porcelain`, `git stash list`, and inventory what each dirty path is.

You own exactly: STATUS.md (new), README.md (updates only where stale vs reality), .gitignore. You may STAGE existing work into logical commits ONLY after presenting me a categorized plan (feature work / experiments / junk / unknown) and getting my approval per category. Do NOT delete anything. Do NOT touch engine internals.

Spec:
1. Categorize every dirty/untracked path: (a) belongs in a commit, (b) intentional local-only (add to .gitignore or document), (c) junk (list for my deletion approval), (d) unknown (ask).
2. Write STATUS.md: current verified state (pull exact numbers from EARSIGHT_CONTEXT.md §2 — do not restate claims more strongly), open hardware gates, Wave 1 checklist, last-updated date.
3. Fix README drift: glasses now demo-only, hybrid sensing incoming, pilot product = box + alerts + weekly report (per EARSIGHT_CONTEXT.md §3). Do not remove the honest-negatives sections.

DONE criteria: `git status` output explained line-by-line in your report; STATUS.md exists and agrees with EARSIGHT_CONTEXT.md; README contains no claim stronger than EARSIGHT_CONTEXT.md §2; `uv run pytest` still green (you changed no code — prove it).

Report: evidence bundle — files touched, commands run, exact test count, the categorized dirty-path table, residual unknowns.
```

## P1 — VibrationSource + fused scoring (product-critical, run alone)

```
Task: Accelerometer ingestion path and fused mic+accel anomaly scoring.

Context: /Users/connork/code/earsight. Read EARSIGHT_CONTEXT.md first (§3.2 explains WHY: airborne mics lose early bearing faults in noisy rooms; a contact MEMS accelerometer becomes the primary early-warning channel; the mic stays for non-contact/room-level context). Recon: shared/audio_source.py (AudioSource is the load-bearing seam — study its contract exactly), engine/embedder.py, engine/scorers, engine/stream (StreamScorer), engine/policy, tests/. Everything runs via `uv run`.

You own exactly: shared/vibration_source.py (new), engine/vib_features.py (new), engine/fusion.py (new), tests/test_vibration_source.py, tests/test_vib_features.py, tests/test_fusion.py, and dependency additions ONLY in pyproject.toml. Do NOT edit AudioSource, existing scorers, serve.py, or any other file. Do NOT commit.

Spec (build exactly):
1. VibrationSource: same seam pattern as AudioSource — swappable implementations: (a) file-based (CSV/npy of 3-axis samples, for tests and MIMII-era development), (b) I2C MEMS driver stub for MPU-6050/ADXL345 behind an interface, hardware-absent-safe (raises a typed, catchable error, never crashes the engine). Abstract over sample rate; expose achievable ODR as a property. Document honestly in a module docstring: cheap MEMS I2C tops out ~1-3.2 kHz ODR, which limits envelope analysis for early-stage bearing bands; that limitation is ACCEPTED for v1 and must be stated, not hidden.
2. vib_features: per-window (1 s, matching audio windows) classical features: RMS, kurtosis, crest factor, peak-to-peak, plus band energies from an envelope spectrum (Hilbert) at configurable bands. Validated inputs, deterministic, no torch dependency.
3. fusion: takes per-window audio anomaly score (existing scorer output contract — recon it, don't guess) + vibration feature vector scored by the same cosine-kNN approach against a vibration baseline; outputs fused score = configurable weighted max (default) with per-channel scores preserved for evidence. Baselines per-channel, never mixed.
4. Tests: synthetic signals with known properties (injected impulse train must raise kurtosis + envelope band energy; clean sine must not), file-source round-trip, fusion behavior when one channel is missing (degrades to single-channel, flagged in output, never silently).

DONE criteria: all three new test files pass; full `uv run pytest` green with zero regressions and zero skips; hardware-absent path proven by a test; docstring limitation present; no torch import in vib_features.

Report: evidence bundle — exact test counts before/after, commands, files touched, the fused-score output schema, residual blockers.
```

## P2 — Pi pilot image: one-command headless install

```
Task: Pilot-box provisioning — from fresh Raspberry Pi OS to scoring in under 30 minutes.

Context: /Users/connork/code/earsight. Read EARSIGHT_CONTEXT.md §3, §7 first: the pilot product is a quiet box on a wall in a brewery. Assume Raspberry Pi 4/Zero 2 W, Raspberry Pi OS Lite (arm64), USB mic, optional I2C accelerometer (P1's seam; degrade gracefully if absent), possibly no stable Wi-Fi. Recon engine/serve.py and engine/stream for the runtime entrypoints; do not modify them.

You own exactly: deploy/ (new directory: install.sh, earsight.service, config.example.toml, README.md), tests/test_deploy_config.py. Do NOT edit engine code. Do NOT commit.

Spec:
1. install.sh: idempotent; installs uv + pinned python, syncs deps, installs systemd unit, creates data dirs with bounded-size config, registers device identity (site name, machine name, token) from config.toml. Re-running never destroys local baselines or buffered data.
2. earsight.service: auto-restart, boots scoring on startup, logs to journald, memory-bounded.
3. Offline resilience: score buffering to disk with a hard cap (config), flush on reconnect; alert queue survives restart. If this requires engine changes, STOP and report — do not reach outside ownership.
4. deploy/README.md: exact copy-paste steps, fresh-flash → listening, with a timed checklist; honest section for what is NOT verified without physical hardware.
5. config validation test: bad/missing fields fail loudly with actionable messages.

DONE criteria: shellcheck-clean install.sh; unit file lints (systemd-analyze verify if available, else documented); config test passes; full suite green; a dry-run mode (`install.sh --dry-run`) prints the plan without touching the system — proven in your report by running it.

Report: evidence bundle + explicit list of steps that REQUIRE physical hardware to verify (per EARSIGHT_CONTEXT.md readiness language: this task delivers "implemented," not "verified-on-hardware").
```

## P3 — Alerts + weekly report generator (the report IS the product)

```
Task: SMS/email alerting with hysteresis, and the auto-generated weekly machine-health report.

Context: /Users/connork/code/earsight. Read EARSIGHT_CONTEXT.md §3, §5: the weekly one-pager is what a brewery owner forwards to a friend — treat its content and layout with UI-level care. Recon engine/policy (State/AlertPolicy), engine/evidence (explainer lines), engine/stream for the score/state stream contract. Everything via `uv run`.

You own exactly: engine/alerting.py (new), engine/reporting.py (new), templates/report.md.j2 (new), tests/test_alerting.py, tests/test_reporting.py, dependency additions ONLY in pyproject.toml (prefer stdlib + jinja2; alert transport behind an interface with a console/file implementation — do NOT add a hard Twilio dependency; provider adapter is a 20-line plugin documented for later). Do NOT edit policy/stream internals. Do NOT commit.

Spec:
1. Alerting: consumes the existing state stream (LISTENING/SUSPECT/ALERT — recon exact names). Rules: alert fires on entry to ALERT only; re-arm requires N consecutive healthy windows (config); max 1 alert per machine per H hours (config); every alert carries the evidence line (≤60 chars, existing contract) + timestamp + machine id. No flapping — prove with a test that oscillating scores at the threshold produce exactly one alert.
2. Reporting: weekly (and on-demand) Markdown report per machine from stored window scores: uptime %, baseline drift sparkline (unicode or embedded PNG via matplotlib-optional), event log, plain-English summary ("Normal all week" / "One suspect episode Tue 14:03, resolved"), data-coverage honesty (gaps shown, never interpolated away). Renders to .md always; .pdf optional if pandoc present (soft dependency, degrade gracefully).
3. Language rule from EARSIGHT_CONTEXT.md §8: no jargon in the summary line; "bearing-like anomaly" style, never model internals.

DONE criteria: hysteresis test (oscillation → exactly 1 alert) passes; report golden-file test from a fixture week of synthetic scores (include one gap + one anomaly) passes; missing-data rendering shows the gap; full suite green, zero skips.

Report: evidence bundle + a rendered sample report included verbatim so I can judge it as a product artifact.
```

## P4 — Phone status page (token auth, glasses-free)

```
Task: Minimal mobile status page per pilot machine.

Context: /Users/connork/code/earsight. Read EARSIGHT_CONTEXT.md §3.1: pilots have NO glasses; the host checks their phone. Recon engine/serve.py (FastAPI, fail-closed bearer auth, exact CORS, queue bounds — match its security posture exactly), engine/stream for live state.

You own exactly: engine/statuspage.py (new router, mounted via one import line you may add to serve.py — that one line is the ONLY serve.py edit permitted), static/status/ (new: single HTML file, inline CSS/JS, zero build step, zero external CDNs), tests/test_statuspage.py. Do NOT commit.

Spec:
1. GET /status/{machine_id}?token=… : token per machine (config), constant-time comparison, 404 on bad token (no existence leak — match existing auth style).
2. Page shows: machine name, current state (color-coded), last-24h score sparkline (inline SVG from JSON endpoint), last alert + evidence line, last-data-received timestamp with a stale-data warning if >5 min. Auto-refresh every 30 s. Loads in <2 s on a phone; total page weight <50 KB.
3. Read-only. No raw audio, no config, no controls exposed.

DONE criteria: TestClient tests — valid token 200 with all fields, bad token 404, stale-data flag flips on old timestamps; page passes a manual phone-width render check (screenshot in report if tooling allows, else the HTML included); full suite green; serve.py diff is exactly one import + one mount line.

Report: evidence bundle + the JSON contract documented so the future contractor dashboard can reuse it.
```

## P5 — Noise-robustness eval harness (published DCASE 2026 benchmark; conditional 2027 entry)

```
Task: Measure how mic, accelerometer, and fused scoring degrade under realistic background noise.

Context: /Users/connork/code/earsight. Read EARSIGHT_CONTEXT.md §3.4: this is the research heart of the pivot. DCASE 2026 Task 2 published a noise-aware benchmark. Measure against it now; aim for DCASE 2027 only if the machine-monitoring task returns, otherwise publish the same evaluation independently. Recon engine/eval (existing MIMII-protocol AUC/pAUC eval — reuse, don't fork), engine/dataset, P1's fusion module if merged (if not, mic-only first, fusion hook stubbed).

You own exactly: engine/noise_eval.py (new), engine/noise_beds.py (new), tests/test_noise_eval.py, scripts/run_noise_sweep.py (new). Dependency additions ONLY in pyproject.toml. Do NOT commit.

Spec:
1. noise_beds: deterministic (seeded) synthesis + file-based loading of interference: broadband HVAC-like noise, compressor hum (50/60 Hz + harmonics), music, voices (file-based only, from a documented free corpus — do not vendor audio into the repo; download script with checksums like the MIMII downloader).
2. noise_eval: mix noise into MIMII test clips at SNR ∈ {−5, 0, +5, +10, +20} dB with exact, logged gain math; run the EXISTING eval protocol per SNR; output one table: AUC per machine type × SNR × channel (mic / accel-when-available / fused). Determinism: same seed → identical table to 3 decimals (same assertion style as REPORT_FULL).
3. Honesty rules from EARSIGHT_CONTEXT.md §8: failed mixes stay missing and are reported; no silent clip skipping; cache intermediate mixes with bounded size.
4. scripts/run_noise_sweep.py: one command, resumable, writes engine/NOISE_REPORT.md with the table + methodology + limitations (synthetic beds ≠ real brewery; field recordings replace them later).

DONE criteria: unit tests for SNR gain math (mix at 0 dB SNR has measured power ratio ≈ 1 within tolerance) and determinism pass; a small smoke sweep (subset, documented size) completes locally and produces NOISE_REPORT.md; full suite green.

Report: evidence bundle + the smoke-sweep table itself + estimated runtime/cost for the full sweep so I can schedule it on the Spark.
```

## P6 — Datakit consent + feature-only field capture

```
Task: Make field data capture consent-clean for pilot sites.

Context: /Users/connork/code/earsight. Read EARSIGHT_CONTEXT.md §8.7: pilot capture is feature-only unless the host explicitly opts into raw audio; bounded storage; deletion on request. Recon datakit/ and the /label path in engine/serve.py + engine/labeling.py (existing WAV+JSON write behavior).

You own exactly: datakit/capture_policy.py (new), modifications to engine/labeling.py ONLY (extend, don't rewrite; preserve existing behavior behind the default policy), datakit/CONSENT.md (new: plain-English one-pager for pilot hosts), tests/test_capture_policy.py. Do NOT commit.

Spec:
1. Capture policy modes: (a) features-only — log-mel/embedding + vib features persisted, raw audio NEVER written to disk; (b) raw-with-consent — current WAV+JSON behavior, plus site-level consent flag required in config, refusing loudly if absent; (c) off. Default: features-only.
2. Retention: per-site max-bytes cap and max-age; oldest-first deletion; `earsight-datakit purge --site X` deletes everything for a site and prints a verifiable count (this backs the "deletion on request" promise).
3. CONSENT.md: what is collected in each mode, where it's stored, how deletion works, contact. Written for a brewery owner, not an engineer. ≤1 page.
4. Every persisted artifact tagged with site, machine, mode, and policy version.

DONE criteria: test proving features-only mode writes zero WAV bytes even when the /label tap fires; refusal test for raw mode without consent flag; purge round-trip test; existing label tests still green; full suite green.

Report: evidence bundle + confirmation that default behavior changed (features-only) and where that's documented.
```

## P7 — Integration + verification (YOU, not an agent)

```
Checklist, run personally after each wave merge:
1. `uv run pytest` — entire suite, one command, zero skips. Record exact count in STATUS.md.
2. End-to-end FileSource run: normal → abnormal MIMII pair through StreamScorer → states traverse correctly → alert fires once → evidence line present → weekly report renders from the session.
3. Boot serve + status page; open on your actual phone.
4. Read every new module top to bottom. If you can't explain a design decision out loud, reject or rewrite it — this is your unaided-competence evidence (assessment §measure-coding-skill).
5. Commit per task with evidence in the message; update STATUS.md + EARSIGHT_CONTEXT.md checkboxes; push.
6. Anything hardware-gated goes on the physical test list, labeled "implemented, not verified."
```

---

**Usage notes:** run P1 alone (it defines contracts P3/P5 consume). P2–P6 can run in parallel after P1 merges, in separate worktrees or serialized — they have disjoint ownership but P4 touches one line of serve.py, so merge it before or after P3, not simultaneously. Every agent reads EARSIGHT_CONTEXT.md first; if an agent proposes work outside its ownership list, that's your cue that the prompt drifted from the repo — fix the prompt, don't widen ownership mid-task.
