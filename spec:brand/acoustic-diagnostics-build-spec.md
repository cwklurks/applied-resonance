# Applied Resonance | Build Spec v2
**Hands-free acoustic diagnostics for field technicians. HVAC first. Glasses-native, but built glasses-free.**

## Changelog v1 → v2
1. **Phase 0 is DONE** (baselines beat published DCASE2020; killtest kit built). One bias patch + the physical session remain.
2. **Even Hub app now comes BEFORE MentraOS** — the Even Realities email is the nearest external deadline and it needs an Even Hub demo.
3. **Machine-type classifier removed from the runtime path** (chance-level on unseen hardware per Goal 1; the tech already knows what machine they're facing). Training script stays in-repo, unused at inference.
4. **Baseline modes are now explicit**: session / saved / library-stub. MIMII's per-id protocol doesn't exist in the field; this is the honest answer to "baseline against what?"
5. **Alert policy is a tested state machine** (dwell, hysteresis, cooldown) — "speaks rarely, right when it speaks" as code, not vibes.
6. **Evidence layer promoted into the engine** — every alert ships a why (deviating bands, labeled envelope-spectrum peaks, nearest-normal comparators). The wrench-verifiable trust feature.
7. **Mic augmentation is a hook, not a feature** — built only if the physical kill test reads SOFT FAIL.
8. **Datakit gains a privacy posture** (consent note, contains-speech flag).

## Design rules (apply to every phase)
- **The core engine never knows where audio comes from.** One `AudioSource` interface (file, browser mic, phone mic, glasses mic). The glasses "pivot" is swapping one class.
- **Speaks rarely, right when it speaks.** Precision over recall everywhere a threshold exists.
- **Every alert is explainable enough to verify with a wrench.**
- **Every demo session feeds the dataset.** Record-and-label is never optional UI.

---

## Phase 0 — The kill test (STATUS: baselines done; patch + physical session remain)

Question: does machine-fault audio classification survive consumer microphones?

### Done (Goal 1 + most of Goal 2)
- `engine/`: MIMII download (full + ranged-subset), log-mel + PANNs CNN14 features with disk cache, per-id kNN + Mahalanobis anomaly scorers (normal-only fitting), AUC/pAUC eval vs DCASE2020 convention, determinism double-pass, smoke test.
- **Results (0 dB subset, kNN type averages): fan 0.749, pump 0.893 vs published DCASE2020 AE baselines 0.658 / 0.729.** Sanity OK. Subset CIs are wide (15+15 test clips per id); full-set run pending.
- `killtest/`: playlist builder (sync chirp + 2 s separators), cross-correlation segmenter with per-clip refinement, SNR noise overlay, eval-rerun glue (train embeddings stay clean; only test clips substituted).
- Synthetic noise condition ran (fan 0.811 / pump 0.879 at 0 dB SNR). The fan increase is small-sample variance + loudness artifacts, not a finding.
- Classifier val acc 0.50 = id-disjoint generalization gap on 3 train ids. Diagnosed, deprioritized, and removed from the runtime path (see changelog #3).

### Remaining: the bias patch (run before recording anything)
The v1 playlist picks first-N clips by filename, but the eval's test split is all 15 abnormals + a random 15-of-35 normals. Result: ~33% of test abnormals vs ~14% of test normals get the mic-chain coloration, so the channel correlates with the label and the verdict is untrustworthy.

```
/goal Patch the killtest kit so physical re-recording is unbiased and practical.

DONE means ALL verified:

1. killtest/playlist.py gains a --from-split mode: when passed, clip selection
   comes from engine.eval.splits.make_anomaly_splits(list_clips(), seed=1337)
   instead of first-N-by-name. It takes --ids (e.g. --ids fan:id_06 pump:id_02)
   and includes EVERY test_normal and test_abnormal clip for those ids, and
   nothing else. Print total playback duration so the user can budget time.

2. killtest/eval_rerun.py gains --restrict-ids (same format). When passed, only
   splits for those ids are evaluated and reported, so partially re-recorded
   datasets never mix clean and degraded clips inside one AUC. Assert that for
   each restricted id, 100% of its test clips were found under clip-root, and
   fail loudly with the list of missing relpaths otherwise.

3. REPORT.md rows gain a coverage column (replaced/test per condition).

4. Tests: a synthetic tmp-path test proving --from-split selects exactly the
   split's test clips for the chosen ids, and a test proving --restrict-ids
   errors on incomplete coverage. Existing tests stay green.

Constraints: no new dependencies, no changes to split seeding or scorer logic.
Plan first.
```

### The physical session (~1 afternoon)
1. Playlist via `--from-split --ids fan:id_06 pump:id_02` (strongest clean signals → retention is measurable; ~60 clips ≈ 12 min playback per chain).
2. Decent speaker (bookshelf/monitor, not laptop), 30–50 cm, realistic volume. One long WAV per chain: Mac internal, phone, cheap MEMS/earbud, plus a good-mic control (separates speaker coloration from mic degradation — don't skip).
3. Segment → `eval_rerun --restrict-ids` per chain → read REPORT.md.
4. Also grab 10–15 min of real shop-ish noise (idling car, bathroom fan, traffic) for the overlay conditions.

### Pass/fail gates (computed from OUR clean numbers)
- **PASS**: ≥80% of the clean margin above chance retained → **pump kNN ≥ 0.814, fan kNN ≥ 0.699.** Proceed as planned.
- **SOFT FAIL** (0.65–0.78 absolute): signal exists, chain hurts → activate the augmentation hook in Goal 3 (impulse-response convolution + noise mix during baseline fitting, using the measured transfer functions). Budget one extra week.
- **HARD FAIL** (< 0.65): glasses-mic capture is dead for fine diagnostics → pivot capture to **phone-captures, glasses-display**. Company survives; only the capture path changes.

Physics constraint (unchanged): 16 kHz pipelines cap bandwidth at 8 kHz. Audible-band faults (bearing impulse trains, belt squeal, imbalance, knock, 120 Hz electrical) live below ~5 kHz, so acceptable. Ultrasonic earliest-stage wear is out of scope by design.

In parallel: start the FULL MIMII download (~18.3 GB, resumable) and rerun the eval for the publishable table.

---

## Phase 1 — Engine v1 + desktop demo (GOAL 3, revised)

```
/goal Productionize the earsight engine for streaming field use and build the desktop demo app, implementing baseline modes, a conservative alert policy, and an evidence layer.

DONE means ALL verified:

1. shared/audio_source.py: an AudioSource interface (iterator of 16 kHz mono float32 chunks) with FileSource and MicSource (sounddevice, device-selectable). Everything downstream consumes an AudioSource; nothing else touches audio I/O. This abstraction is load-bearing for the future glasses source.

2. Streaming core: 3.0 s analysis windows, 1.0 s hop, PANNs embedding per window, anomaly score = kNN distance to the ACTIVE BASELINE, EMA smoothing (alpha 0.3).

3. Baseline manager (engine/baseline.py): fit a scorer from a list of windows; persist/load baselines on disk keyed by a user tag (site/machine). Two modes wired into the API: "session" (fit on the first 30 s after the user marks the machine as sounding normal) and "saved" (load by tag). A "library" mode exists as a documented stub raising NotImplementedError. Calibration: raw scores map to percentiles against a held-out 20% of the baseline's own windows.

4. Alert policy (engine/policy.py): explicit state machine LISTENING -> SUSPECT -> ALERT. ALERT requires N consecutive windows above the P99 percentile (default N=5), exits below P95 (hysteresis), then a cooldown before re-alerting. All thresholds in one config dataclass. Unit tests drive the state machine with synthetic score sequences covering: brief spike (no alert), sustained anomaly (alert), flapping around threshold (no flapping alerts).

5. Evidence module (engine/evidence.py): on SUSPECT/ALERT, produce (a) top deviating mel bands vs the baseline mean, reported as Hz ranges; (b) envelope spectrum (band-pass 500-5000 Hz, Hilbert envelope, FFT) with peak picking, peaks labeled against candidates: 120 Hz line frequency (+-3 Hz), 1x rotation if the user supplied an RPM, otherwise "impulse train at {f} Hz"; (c) timestamps of the 3 nearest baseline windows. Emits a structured dict plus a single HUD-ready evidence string of at most 60 characters, e.g. "impulse train ~88 Hz, low-band energy up". Unit test: a synthetic signal with a known 90 Hz impulse train over pink noise must surface a labeled peak within +-5 Hz.

6. The machine-type CNN is removed from the runtime path entirely (training script remains, unused at inference).

7. ONNX export of the embedding model; parity test asserting ONNX vs PyTorch outputs within 1e-4 on 20 clips; onnxruntime inference path behind a flag; measure and print real-time factor, target < 0.5 on CPU for the 3 s/1 s schedule (if missed, document the measured RTF and add a 2 s hop fallback flag).

8. FastAPI service (engine/serve.py): POST /session/start {mode, tag, rpm?}, POST /score {session_id, base64 PCM chunk} -> {state, score, percentile, evidence_line}, POST /label (writes WAV + sidecar JSON to datakit/data/), GET /health. Baselines persist across restarts; sessions do not.

9. Streamlit app (desktop/app.py): input device picker, rolling 10 s spectrogram, a "capture baseline (30 s)" button with countdown, state badge + score gauge, evidence panel showing the structured evidence, and a RECORD & LABEL panel (last 10 s + machine type / suspected fault / contains-speech checkbox / free text / site tag).

10. desktop/DEMO.md: a shot-by-shot 60-second video script. Staging: box fan or fridge, capture baseline healthy, then induce a fault (tape a coin to one fan blade for imbalance, or loosen a panel screw for rattle), show SUSPECT -> ALERT with the evidence line, tap record-and-label.

11. An augmentation hook: BaselineManager.fit accepts an optional list of augmentation callables (impulse-response convolution, noise mix). Ship it empty; it is the landing zone for the kill-test verdict.

12. One CI command runs all tests including Goals 1-2 suites; everything green.

Constraints: no new heavy dependencies beyond sounddevice, onnx, onnxruntime, streamlit, fastapi, uvicorn. Plan first and show me the plan before building.
```

Manual after Goal 3: film the DEMO.md video (the coin-on-fan-blade staging doubles as the physics proof for X). Join Even Hub + Mentra Discords; ask the raw-PCM / sample-rate / background-mic questions; post the kill-test writeup.

---

## Phase 2 — Glasses apps in simulators (GOALS 4–5; Even Hub FIRST)

### GOAL 4 — Even Hub app (the email depends on this one)

```
/goal Build the Applied Resonance Even Hub app in earsight/apps/evenhub/, running in the Even Hub simulator, consuming the engine's FastAPI service.

DONE means ALL verified:

1. Scaffold from the even-realities evenhub-templates "asr" template via degit, since it already demonstrates the mic -> processing -> display loop. Replace the ASR call with POSTs to our /score endpoint; keep the template's mic-permission and lifecycle handling intact. Do NOT invent Even Hub SDK APIs; use only what the template source and hub.evenrealities.com docs show.

2. Defensive audio handling: detect the incoming sample rate and channel count at runtime, resample to 16 kHz mono, buffer 1 s chunks, attach a session id. Assume nothing about the stream format.

3. HUD contract (this is the product): maximum 2 lines, at most one push per 2 seconds. States: LISTENING ("Listening..."), SUSPECT/ALERT ("! Bearing-like anomaly · 81%" with line 2 = the engine's evidence_line + "tap to log"), ERROR (engine unreachable shows "engine offline", never a stack trace, never silence). Baseline capture is reachable from the app (start a session in "session" mode and show a 30 s countdown on the HUD).

4. Tap-to-log: on the template's available input event (temple tap or ring), POST /label with the last 10 s, confirm "logged" on the HUD for 2 s, return to listening.

5. Mock mode: a flag feeds a local WAV through the identical pipeline headlessly; an automated test runs a known-anomalous file and asserts the ALERT state reaches the display layer.

6. Simulator: try the official Even Hub simulator path from the docs first; if blocked, fall back to the BxNxM/even-dev environment. Document whichever works with exact commands. Package with @evenrealities/evenhub-cli and verify the packaged build loads in the simulator (publishing itself stays manual).

7. Shared code: extract HUD card formatting and the engine client into earsight/shared/display-card (TypeScript) so the upcoming MentraOS app cannot drift from this one.

8. README section "Hardware unknowns": raw PCM availability, true sample rate, AGC/beamforming behavior, background mic policy, battery cost; each with the planned on-hardware test.

Constraints: TypeScript, Node 20+. Plan first.
```

**Manual after Goal 4: record the simulator demo and SEND THE EVEN REALITIES EMAIL** (Variant 1 draft; attach the PDF brief; link a 60–90 s cut: desktop demo → simulated HUD). Post the same video in their Discord; apply to the Pilot Program. The earning sprint remains the fallback purchase path.

### GOAL 5 — MentraOS MiniApp (community + hackathon channel)

```
/goal Build the Applied Resonance MentraOS MiniApp in earsight/apps/mentraos/, running against Simulated Glasses mode, consuming the engine's FastAPI service and the shared display-card library.

DONE means ALL verified:

1. TypeScript app scaffolded from the Mentra-Community cloud example app using @mentra/sdk, README covering console.mentra.glass registration, ngrok, env vars, exact Simulated Glasses run commands. Do not guess SDK method names; consult the example source and docs.mentra.glass.

2. Audio path: subscribe to the glasses mic stream, defensively resample to 16 kHz mono, 1 s chunks, session id, POST to /score.

3. Display layer consumes earsight/shared/display-card UNCHANGED (same 2-line contract, same 3 states, same 2 s throttle, same tap-to-log, same baseline-capture flow). Any needed change goes into the shared lib with both apps' tests green.

4. Mock mode + the same headless ALERT-reaches-display test as the Even Hub app.

5. A 30-second screen-recording path documented for the simulated HUD demo.

Constraints: TypeScript, Node 20+. Plan first.
```

---

## Phase 3 — Field data + technicians (GOAL 6 + the human work)

```
/goal Build the mobile field-recording tool in earsight/datakit/ for collecting labeled machine audio at real job sites with a phone.

DONE means ALL verified:

1. Single-page mobile web app (plain TS + Vite, no framework): one giant record/stop button (MediaRecorder), live input-level meter (clipping visible), then the form: machine type (furnace / AC condenser / compressor / pump / fan / other), condition (sounds normal / suspected issue / confirmed issue), SITE/MACHINE TAG (free text; this key feeds the engine's saved-baseline registry), contains-speech checkbox, free-text note.

2. Saves 16 kHz WAV + sidecar JSON to the same datakit/data/ store via the engine's /label endpoint. Offline-tolerant: failed uploads persist to IndexedDB and retry with a visible queue count.

3. QR code generator script printing the LAN URL so a phone on the same WiFi opens the recorder in two seconds.

4. Ingest report command: dataset stats per machine type / condition / site tag, total minutes, flags for clips under 3 s or clipped, markdown output.

5. README "Field etiquette & privacy" section: ask the shop owner before recording, avoid capturing conversations, mark anything with voices via the contains-speech flag (those clips are excluded from any public dataset or demo by default).

6. Mobile-tested layout (375 px), big glove-friendly touch targets, verified in iOS Safari and Android Chrome (headless viewport test + documented manual check).

Constraints: brutally simple; it gets used standing in a mechanical room with gloves on. Plan first.
```

The human-only motion (unchanged, still the part that gets you into YC): $20 lav mic, Vancouver HVAC/refrigeration walk-ins offering a free "acoustic health snapshot," one shadow session with a tech, HVAC-Talk / r/HVAC discovery-first posts. **Targets by Week 6: 50–100 labeled clips, 5+ tech conversations, 1–2 shops trying it on real jobs.**

---

## Phase 4 — Hardware pivot checklist (the week the G2 arrives)

1. Confirm the mic access path (Even Hub SDK stream; MentraOS if G2 support has landed — verify, their pages are inconsistent).
2. **Characterize the G2 mic chain**: play the sweep + the kill-test playlist through the same speaker rig, record via the glasses, compute the transfer function. One evening.
3. Re-run the kill-test eval on G2-recorded audio. The FINAL gate, on real hardware.
4. If accuracy dropped: feed the measured G2 impulse response + noise into the Goal 3 augmentation hook, refit baselines, re-eval. (Also a great X writeup.)
5. Swap `AudioSource.phone` → `AudioSource.glasses`. Test glanceability outdoors, with gloves, head moving.
6. Battery/duty-cycle: continuous listen vs triggered 10 s captures via temple tap — both designed, pick from measured battery cost.
7. Publish to Even Hub (and Mentra) as a free beta.

Bonus deliverable from step 2: the G2 mic characterization data itself — offer it to Even Realities; it goes in the email.

---

## Repo layout (v2)
```
earsight/
  engine/
    baseline.py        # session/saved/library-stub baseline manager + calibration
    policy.py          # LISTENING -> SUSPECT -> ALERT state machine
    evidence.py        # deviating bands, envelope-spectrum labeling, comparators
    serve.py           # FastAPI: /session/start /score /label /health
    ...                # download, features, scorers, eval (Phase 0, done)
  killtest/            # playlist (--from-split), segment, noise, eval_rerun (--restrict-ids)
  desktop/             # Streamlit demo + DEMO.md
  apps/
    evenhub/           # FIRST: web app (sim -> hardware)
    mentraos/          # SECOND: TS MiniApp (sim -> hardware)
  datakit/             # mobile recorder (site tags, consent posture)
  shared/
    audio_source.py    # the load-bearing abstraction
    display-card/      # TS HUD contract shared by both apps
```

## Sequencing from today
1. Bias patch /goal → physical kill session (fan id_06 + pump id_02, four chains) → read the verdict. **Gates everything.**
2. Full MIMII download overnight in parallel → publishable eval table → X post.
3. Goal 3 (engine v1) → film the demo video.
4. Goal 4 (Even Hub sim) → **send the Even email** + Discord + Pilot Program application.
5. Goal 5 (MentraOS) and Goal 6 (datakit) while waiting on Even's reply → Vancouver shop walk-ins.
6. Glasses arrive (gifted or sprint-funded) → Phase 4 checklist, ~1 week.
7. Week 6+: field traction → public writeups → Z Fellows / 1517 / Emergent Ventures applications with real usage data.

## Honest unknowns (carry these, don't hide them)
- G2 mic AGC/beamforming may mangle machine audio in ways the earbud proxy doesn't predict. Phase 4 step 3 exists for exactly this.
- Continuous background mic access may be restricted on either platform; triggered-capture fallback is designed in.
- Cold-start baselines: session mode assumes the machine sounds healthy at capture time. If it's already degraded, the watchdog only catches FURTHER change until a saved/library baseline exists. Say this out loud to techs.
- Named-fault classification needs labeled field data that public datasets lack; anomaly-plus-evidence is the honest v1, named faults come as the dataset grows.
- A tech with 20 years of ears beats v1 on machines they know. The wedge is the junior tech, the unfamiliar machine, and the documentation trail. Position as "second opinion + logging," never "replaces your ears."
