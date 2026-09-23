# Applied Resonance | Build Spec v1
**Hands-free acoustic diagnostics for field technicians. HVAC first. Glasses-native, but built glasses-free.**

The single design rule that makes everything below work: **the core engine never knows where audio comes from.** One `AudioSource` interface with four implementations (file, browser mic, phone mic, glasses mic). Everything downstream (features, models, UI) is identical. The "pivot" when glasses arrive is swapping one class, not rebuilding anything.

---

## Phase 0 — The kill test (Days 1–7, zero hardware)

Question to answer: does machine-fault classification survive consumer microphones, or does it only work on the studio-grade recordings the public datasets were made with?

### Datasets (download Day 1)
- **MIMII** (Hitachi): fans, pumps, valves, slide rails. Normal + anomalous recordings WITH factory background noise mixed at multiple SNRs. The fans and pumps are your closest HVAC proxies. https://zenodo.org/records/3384388
- **DCASE Task 2** (anomalous sound detection challenge, any recent year): standardized train/test splits, published baselines to compare against, AUC/pAUC eval convention. https://dcase.community
- **ToyADMOS**: secondary, more machine types. https://zenodo.org/records/3351307
- **ESC-50**: general environmental sounds, useful later for sound-event awareness features. https://github.com/karolpiczak/ESC-50

Skip Case Western (vibration data, not audio). Skip scraping YouTube fault videos for now (label noise, save for Phase 3 as weak supervision if needed).

### Baseline (Days 1–3, on the 5090)
Two models, both simple on purpose:
1. **Anomaly detector** (the DCASE-style approach): pretrained audio embeddings (PANNs CNN14 or BEATs) → k-NN / Mahalanobis distance against normal-only training clips, per machine type. No fault labels needed. This matters because in the field, every machine is "unseen" and you won't have labeled faults for it. Pretrained PANNs: https://github.com/qiuqiangkong/audioset_tagging_cnn
2. **Machine-type classifier**: log-mel spectrogram → small CNN → "fan / pump / compressor / other." Trivial, but it gates the anomaly head (you score against the right normal-distribution).

Record baseline AUC and pAUC per machine type on clean test data. The published DCASE baselines tell you if your numbers are sane.

### The re-record protocol (Days 4–6)
This is the actual experiment. You're measuring how much signal survives a speaker → air → consumer-mic chain.

1. Play the MIMII test split through a decent speaker (bookshelf or studio monitor, NOT a laptop speaker) at realistic volume, 30–50 cm from the mic.
2. Record three mic chains: (a) M4 Mac internal mic at 48 kHz, (b) phone mic via the voice-memo path, (c) a cheap MEMS/earbud mic as the closest proxy for glasses-grade hardware.
3. Control run: re-record a subset with a GOOD mic too, so you can separate speaker coloration from mic degradation. Without this control the experiment is confounded.
4. Align re-recorded clips to originals by cross-correlation, then re-run the exact eval from Days 1–3.
5. Repeat condition (b) with overlaid shop noise (compressor hum, traffic, voices) at 0, 5, 10 dB SNR.

### Pass/fail gates
Using clean-test AUC as reference (say it lands ~0.85–0.90):
- **PASS**: re-recorded consumer-mic AUC retains ≥ 80% of the margin above chance (e.g., clean 0.90 → re-recorded ≥ 0.78). Proceed as planned.
- **SOFT FAIL** (0.65–0.78): the signal exists but the mic chain hurts. Standard fix: train-time augmentation with mic impulse responses, EQ tilt, and re-recorded data. This usually recovers most of the gap. Budget one extra week.
- **HARD FAIL** (< 0.65, near chance): glasses-mic capture is dead for fine diagnostics. Pivot the architecture to **phone-captures, glasses-display**: tech holds phone near the unit for 10 s, results stream to the HUD. Still hands-free during the actual work, still a real product. The company survives the kill test failing; only the capture path changes.

One known physics constraint to write down now: glasses/phone ASR pipelines often deliver 16 kHz audio, which caps you at 8 kHz of bandwidth. Most audible-band faults (bearing rumble, belt squeal, misfire periodicity, compressor knock) live below that, so this is acceptable. Ultrasonic early-bearing-wear detection (Senzoro's territory) is out of reach on this hardware, and that's fine. Different product.

### /goal prompt for Phase 0 (paste into Claude Code / Codex)
```
/goal Build and run an evaluation pipeline that answers: does machine-fault audio classification survive consumer microphones?

DONE means all of the following, self-verified:
1. A Python repo with: dataset download script for MIMII fans+pumps (Zenodo), feature extraction (log-mel + PANNs CNN14 embeddings), an anomaly scorer (k-NN on embeddings against normal-only training data, per machine type), and a machine-type CNN classifier.
2. Eval script computes AUC and pAUC per machine type on the clean MIMII test split, results within plausible range of published DCASE baselines (sanity-check table included).
3. A "re-record kit": script that (a) generates a playback playlist of the test split with 2s silence separators and a sync chirp at the start, (b) takes a single long re-recorded WAV, auto-segments it using the chirp + cross-correlation alignment, and (c) re-runs the identical eval on the segmented clips.
4. A noise-overlay eval mode: mixes shop-noise WAVs (user-provided folder) into test clips at 0/5/10 dB SNR and reports AUC at each.
5. REPORT.md with the full results table (clean vs re-recorded vs noisy), generated automatically.
6. All scripts run end-to-end with one documented command each; a smoke test on 20 clips passes.
Constraints: PyTorch, CUDA-ready for an RTX 5090 but CPU-fallback for dev; no cloud calls; minimal deps; plan first and show me the plan.
```
You do the physical playback/recording (an afternoon); the agent builds everything around it.

---

## Phase 1 — Core engine + desktop demo (Weeks 1–3, zero hardware)

Goal: a demo you can show a technician or tweet WITHOUT any glasses existing.

- **Training** (Python/PyTorch on the 5090): productionize the Phase 0 winner. Add temporal smoothing (rolling 3 s windows, exponential moving average on anomaly score) so the output is stable, not flickery.
- **Export**: ONNX. One artifact, runs everywhere (Python desktop, browser via onnxruntime-web, phone later). This export choice IS the smooth-pivot mechanism.
- **Desktop demo app** (Streamlit, your home turf): live mic in → rolling spectrogram → machine-type guess → anomaly meter → "closest known signatures" top-3. Add a record-and-label button (machine type, suspected fault, free-text note) that writes WAV + JSON. This button is the start of the data moat; every demo session feeds the dataset.
- **The 60-second video**: point the laptop at a running furnace/fridge/car, show the meter spike when you detune something (loosen a panel screw so it rattles, partially block a fan). Stage it honestly but visibly. This video does triple duty: X content, technician outreach asset, Even Realities pitch asset.

Deliverables: `engine/` repo, ONNX model, Streamlit demo, demo video.

---

## Phase 2 — Glasses integration in simulators (Weeks 2–4, parallel, zero hardware)

Build the actual glasses app against BOTH simulators. Same engine, two thin display shells.

- **MentraOS MiniApp** (TypeScript SDK, Simulated Glasses mode): mic stream → your local server running the ONNX engine → push glanceable cards to the simulated HUD. Template: https://github.com/Mentra-Community/MentraOS-Cloud-Example-App, docs: https://docs.mentra.glass
- **Even Hub app** (web app SDK + local simulator): port the display layer. Their `asr` starter template already demonstrates the mic → processing → display loop; you're swapping the ASR model for your acoustic engine. Templates: https://github.com/even-realities/evenhub-templates, sim env: https://github.com/BxNxM/even-dev
- **HUD UX spec** (this is where most glasses apps fail): max 2 lines. Line 1: `⚠ Bearing-like anomaly · 78%`. Line 2: trend arrow + `tap to log 10s`. No spectrograms on the HUD, no paragraphs. The phone holds the detail view.
- **Open question to resolve in their Discords (do this Week 2, it doubles as community visibility before your email):** does each platform give plugins raw mic PCM (what sample rate? what AGC/beamforming is baked in?), and can a background plugin hold the mic continuously, or only in foreground sessions? The answer changes duty-cycling design (continuous listen vs. 10 s triggered captures). Ask in Even Hub Discord and Mentra Discord; nobody has published this.

Deliverables: working sim demos on both platforms, screen recordings of each.

---

## Phase 3 — Field data + technicians (Weeks 3–6, zero glasses needed)

The moat and the traction motion, fully hardware-independent.

- **Field kit**: your phone + a $20 clip-on lav mic + the record-and-label flow from Phase 1 wrapped in a dead-simple mobile web page.
- **Finding techs**: local Vancouver HVAC/refrigeration/appliance shops (walk in, offer a free "acoustic health snapshot" of their bench units), HVAC-Talk forum, r/HVAC and r/hvacadvice (read each sub's self-promo rules first; lead by asking techs how they diagnose by ear, which is genuinely useful discovery, not growth hacking).
- **The Wizard-of-Oz validation**: shadow one tech for an afternoon. Record what they listen to, note what they concluded, run your engine on the clips that evening, compare. You learn whether your output categories even map to how techs think (they may say "compressor short-cycling," not "anomaly 0.78"). This conversation reshapes the product more than any model improvement.
- **Targets by Week 6**: 50–100 labeled field clips, 5+ real tech conversations, 1–2 shops willing to try the phone version on real jobs. That is teen-founder traction worth writing about.

---

## Phase 4 — Hardware pivot checklist (the week the G2 arrives)

Because of the AudioSource abstraction, this is a checklist, not a project:

1. Confirm mic access path (Even Hub SDK stream or MentraOS if G2 support has landed). You already know the answer from the Discord question in Phase 2.
2. **Characterize the G2 mic chain**: play a log sine sweep + your re-record playlist through the same speaker rig from Phase 0, record via the glasses, compute the transfer function. One evening of work.
3. Re-run the Phase 0 eval on G2-recorded audio. This is the FINAL kill-test gate, on real hardware.
4. If accuracy dropped: add G2 impulse-response + EQ augmentation to training, fine-tune, re-eval. (This is the standard fix and it's exactly the kind of writeup that does numbers on X: "I measured the G2's mic transfer function and fine-tuned around it.")
5. Swap `AudioSource.phone` → `AudioSource.glasses`. Ship the sim app to real HUD. Test glanceability outdoors in sunlight, with gloves, head moving.
6. Battery/duty-cycle: if continuous mic hold isn't allowed or drains too fast, fall back to triggered 10 s captures via temple tap. Design both now, pick later.
7. Publish to Even Hub / Mentra store as a free beta. Store presence + field techs = the story.

Bonus deliverable from step 2: the G2 mic characterization data itself. Even Realities' own team likely doesn't have third-party acoustic-sensing characterization of their hardware. Offering to share it is a genuinely valuable gift, and it goes in the email.

---

## Repo layout
```
earsight/
  engine/          # Python: training, eval, ONNX export (Phase 0–1)
  killtest/        # re-record kit + REPORT.md
  desktop/         # Streamlit live demo + labeling
  apps/
    mentraos/      # TS MiniApp (sim → hardware)
    evenhub/       # web app (sim → hardware)
  datakit/         # mobile web recorder for field collection
  shared/          # AudioSource interface, display-card schema
```

## Sequencing against everything else in flight
- **Week 1**: kill test runs WHILE the earning sprint continues (the /goal agent builds the pipeline; your hands-on time is ~1 afternoon of playback recording).
- **End of Week 2 or 3**: sim demos exist → send the Even Realities email (below) + post demos in their Discord + apply to the Pilot Program. The music-copilot app is now optional; this project is a stronger pitch.
- **Glasses arrive** (free unit, or bought with sprint earnings, whichever lands first): Phase 4 checklist, ~1 week.
- **Week 6+**: field traction → public writeups → Z Fellows / 1517 applications with real usage data.

## Honest unknowns (carry these, don't hide them)
- G2 mic AGC/beamforming may mangle machine audio in ways the cheap-earbud proxy doesn't predict. The Phase 4 re-test exists for exactly this.
- Continuous background mic access may be restricted on either platform. The triggered-capture fallback is designed in from the start.
- Named-fault classification (vs. anomaly scoring) needs labeled fault data that public datasets mostly lack. Field collection is the only real source, which is slow. Anomaly-first is the honest v1; named faults come as the dataset grows.
- A tech with 20 years of ears will beat v1 on machines they know. The wedge is the junior tech, the unfamiliar machine, and the documentation trail (logged acoustic evidence for the customer). Position it as "second opinion + logging," never "replaces your ears."
