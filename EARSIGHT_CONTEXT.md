# EARSIGHT / APPLIED RESONANCE — CONTEXT.md
**Ground truth for this repo. Read this before doing anything. Last updated: 2026-07-20.**
*Audience: me (Connor), and every human or agent working in this repo. If reality and this document disagree, fix this document in the same change.*

---

## 1. What this is

EarSight (venture name: Applied Resonance) is machine-health monitoring for small shops. A ~$60 box (Raspberry Pi + USB microphone + MEMS accelerometer) learns what a healthy pump/fan/compressor sounds and vibrates like, and sends a text before it fails. Sold eventually through HVAC/refrigeration service contractors; piloted free in Vancouver breweries and shops, fall 2026.

**One-liner:** *A $60 box that learns what your pump sounds like healthy and texts you before it fails.*

## 2. Honest current state (verified vs not)

**Verified (evidence exists, reproducible):**
- Engine beats DCASE2020 MIMII baselines: fan kNN AUC 0.693, pump 0.880 on the full 9,755-clip 0 dB dataset (baselines 0.658/0.729), determinism enforced to 3 decimals. See `engine/REPORT_FULL.md`.
- Log-mel + PANNs CNN14 pipeline, cosine-kNN primary scorer, Ledoit-Wolf Mahalanobis secondary; machine-id-disjoint splits (leakage designed out).
- Streaming inference with ONNX numerical-parity check; physics-based evidence layer (Hilbert-envelope peak-picking: mains hum / shaft rotation / bearing impulses).
- FastAPI serve path with fail-closed bearer auth, exact CORS, queue bounds; model warmup moved to startup (~3.03 s to readiness, 734.5 ms baseline transition).
- Working Even Realities G2 EvenHub app, verified in official simulator; valid `.ehpk` built. Test suites: 195 Python + 46 display-card + 20 app tests green (as of June 11 session).
- Known honest negative: fan/pump classifier at 0.577 val acc — id-disjoint generalization gap is structural; documented, not hidden.

**Not verified / open gates (do not claim these):**
- No physical G2 hardware test, no deployed HTTPS gateway, no store listing live, no re-record kill test on hardware.
- Zero real users. Zero pilots installed. Zero revenue.
- Accelerometer path does not exist yet (see Wave 1).
- Worktree has historical dirty entries — see §8 rules.

## 3. The strategic pivot (July 2026) — what changed and why

Deep market research (July 20, 2026) forced four changes. Full evidence in the execution playbook; condensed here because agents keep building the old vision.

1. **Glasses demoted.** HUD is demo-video + long-term enterprise vision ONLY. Pilot product = box + SMS/email alert + weekly report + phone status page. Industrial AR is a graveyard (Glass Enterprise dead '23, HoloLens dead '24, Vuzix −52% rev); small shops want texts, not eyewear.
2. **Hybrid sensing promoted.** Early bearing faults live at high frequencies where airborne mics have poor SNR in noisy rooms — the reason serious vendors (Augury et al.) use contact accelerometers/ultrasound, and the reason acoustic-only startups died (3DSignals pivoted off sound ~2020; Neuron Soundware ~14 people after a decade). A MEMS accelerometer joins the mic in every kit. Mic stays as differentiator: non-contact, one-per-room, catches cavitation/belt/airflow.
3. **Channel = service contractors, not direct SMB sales.** Amazon stopped accepting new Monitron customers in Oct 2024 despite having the strongest possible distribution — direct-to-SMB is unproven at best. The only working low-end model sells through service companies (Discovery Sound Technology via HVAC techs; Augury's original commercial-HVAC wedge). Pilot mix: 2 breweries + 1 contractor; **the contractor is the real business-model experiment.**
4. **Research target = noise robustness.** Beating the 2020 baseline is table stakes (40 teams cleared it in 2020). DCASE 2026 Task 2 explicitly targeted noise-aware detection — the unsolved part that matters for real deployments. With host consent, noisy field recordings from pilots can test the engine against the published 2026 results. Aim for a DCASE 2027 entry if the machine-monitoring task returns; 2027 tasks are not announced. Otherwise publish the same evaluation independently.

**Banned words in all materials: "revolutionary," any AI-first framing to customers.** Customer pitch is outcome-only: "I catch the failure before it costs you thousands."

## 4. Market reality (so nobody re-derives the old optimism)

- Demand is real: bearings cause ~41–51% of motor failures (IEEE/EPRI/ABB); ~61% of facilities still run-to-fail some assets (Plant Engineering); unplanned downtime ≈ $1.4T/yr across the Fortune G500 (Siemens/Senseye). But quantified pain lives upmarket; SMB pain is real, small-dollar (a brewery dumping 3 batches over a chiller failure ≈ "thousands").
- Named corpses/incumbents: Augury ($369M raised, enterprise-only, $1k+/point/yr), Tractian (mid-market), 3DSignals (pivoted off acoustics), Neuron Soundware (alive, tiny), Amazon Monitron (closed to new customers), UE Systems (ultrasound handhelds since 1973). The low end is open **partly because it's hard to monetize** — that's the thesis being tested, and it may fail. A documented failure is an acceptable outcome (see §7 December checkpoint).

## 5. The pitch, by audience

- **Brewery/shop:** "I built a device that learns what your pump sounds like healthy and texts you before it fails. One machine, 60 days, free — I'm a student researcher; worst case you get a free report."
- **HVAC/refrigeration contractor:** "You add 'continuous monitoring' to your service contracts; I supply the box, alerts route to you, you show up before the failure instead of after."
- **Funders (1517/EV/ZFellows):** "Enterprise tier is served (Augury). Amazon abandoned the low end (Monitron). Acoustic-only failed (3DSignals). The open ground: hybrid $60 sensing + noise-robust models (the DCASE 2026 problem) sold through service contractors. Benchmarked engine, three Vancouver pilots testing exactly that."
- **Never:** platform-speak, projections presented as traction, any number I couldn't defend to a skeptical engineer in person.

## 6. Funding ladder (status — update dates as they happen)

| Program | Status | Notes |
|---|---|---|
| Bagel Fund ($100–500) | Submitted 2026-07-__ | 48h response SLA |
| Hack Club Macondo (≤$1,000 hardware) | DUE AUG 15 | Funds 3 pilot kits + spares |
| Hack Club Stardance | Join | Ship Wave 1 through it |
| 1517 Medici ($1k+, no strings) | Loom video → partner email | Record after demo video exists |
| Emergent Ventures | Draft by Aug 31 | ~1,500 words, concrete budget |
| Z Fellows ($10k optional) | After first pilot live (Oct–Nov) | Reapplying after rejection is normal |
| Thiel Fellowship ($250k/2yr) | NOT NOW — late Grade 12 target | Bar = full-time-caliber; needs users/revenue |
| O'Shaughnessy Fellowship | Window opens Jan 2027 | Verify age minimum first |

## 7. Timeline & the December checkpoint

- **Aug 2026:** hardware gates (physical G2 test, re-record kill test), demo video, EvenHub submission, repo hygiene, Wave 1 engineering, grant applications.
- **Sept:** outreach begins (5 sends/wk; email+phone, not LinkedIn). First install: St. George's facilities or friendliest brewery.
- **Nov 30:** 3 machines live (2 breweries + 1 contractor). Weekly report to every host, never missed.
- **December checkpoint (pre-committed):** If pull exists (hosts engaged, contractor interested, anomaly caught) → double down; ZFellows; aim Thiel/YC conversations at 2027. If nobody cares → write the honest post-mortem, swap the vehicle to the agent-tooling thesis, keep the loop. Identity does not get a vote; evidence does.
- **Standing cadence during school (~10 h/wk):** Mon/Wed 2h pilot maintenance + reports · Sat 3h deep work · Sat 1h outreach + 1 public post · Sun 2h protected schoolwork. If a week breaks: cut Sat deep work, never grades, never pilot reports.

## 8. Rules of engagement (humans and agents)

1. **No new projects/repos until 3 pilots are live.** This repo's history includes a sprawl habit; the freeze is deliberate.
2. **Readiness language is defined:** *implemented* = code + tests green; *verified* = independently checked against spec on the target environment; *deployed* = running for a real user. Never substitute one for another in docs, commits, or pitches.
3. **Docs match reality in the same change** — a PR that changes behavior updates README/STATUS.md/this file or it isn't done.
4. **Evidence bundle per task:** files touched, exact commands, exact test counts, residual blockers. No "should work."
5. **Preserve dirty/user work.** Never commit, revert, or delete files outside declared ownership. Ask first.
6. **Honest numbers only.** Failed/missing observations stay missing. Caveats ship with results (see classifier 0.577 precedent — that's the house style).
7. **Privacy defaults:** pilot capture is feature-only mode unless the host explicitly consents to raw audio retention; bounded storage; deletion on request. Telemetry stays content-free.
8. Run everything via `uv run …` (py3.12, uv-managed). Small files, validated inputs, logger not prints, immutable where possible. Suite must be green in one command: `uv run pytest`.

## 9. Wave 1 engineering roadmap (current work — prompts in AGENT_PROMPTS.md)

- [ ] P0 Repo hygiene + STATUS.md
- [ ] P1 VibrationSource + fused mic/accel scoring  ← product-critical
- [ ] P2 Pi pilot image (one-command headless install)
- [ ] P3 Alerts + weekly report generator  ← the report IS the product
- [ ] P4 Phone status page (token auth, no glasses)
- [ ] P5 Noise-robustness eval harness (published DCASE 2026 benchmark; conditional 2027 entry)
- [ ] P6 Datakit consent/feature-only hardening
- [ ] P7 Integration + verification (Connor personally)

Nothing outside this list gets built until pilots are live.

## 10. Links

Repo: github.com/cwklurks/earsight (private) · Full eval: `engine/REPORT_FULL.md` · Portfolio: connork.com · Models: huggingface.co/connaaa · CodeSprint: codesprint.ca · Demo video: TBD (Aug) · EvenHub listing: TBD · Founder: Connor Klann, 16, Vancouver — connorklann@gmail.com
