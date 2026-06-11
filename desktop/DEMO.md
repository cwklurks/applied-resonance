# EarSight desktop demo — 60-second video script

A hands-free acoustic-diagnostics demo, staged honestly but visibly: capture a
healthy baseline on a running machine, induce a real, audible fault, and watch
the anomaly meter climb from **LISTENING** to **SUSPECT** to **ALERT** with a
plain-language evidence line explaining *why* — then log the clip with one tap.

App under test: `uv run streamlit run desktop/app.py`

---

## Prep checklist (do all of this before you hit record)

- [ ] **Mic permission granted beforehand.** First mic use on macOS pops a
      permission prompt — trigger it once before filming (toggle **Monitor**
      on, accept the prompt in System Settings → Privacy & Security →
      Microphone), then restart the app so the prompt never appears on camera.
- [ ] **Room reasonably quiet.** No HVAC of your own running, no music, close
      the door. Shop noise is realistic but it muddies a 60 s demo.
- [ ] **Box fan (or fridge) on medium.** A box fan is ideal: cheap, steady
      tone, easy to detune. Medium speed gives a clean steady baseline tone.
- [ ] **Fault kit ready and off-camera:** a coin + a strip of tape (imbalance),
      or a screwdriver to loosen one panel screw (rattle). Pre-test that your
      chosen fault actually moves the meter (see dry run).
- [ ] **Pick the embedder backend.** `torch` is the default; the **first** run
      downloads the PANNs model (can take a minute). Do this once before
      filming so there is no download spinner on camera.
- [ ] **Screen-record at 1080p.** QuickTime (⌘⇧5) or your recorder of choice,
      60 fps if available. Frame the browser window so the sidebar, the state
      badge, the meter, and the spectrogram are all visible.
- [ ] **Do a full dry run.** Capture a baseline, induce the fault, confirm you
      hit **ALERT** within ~10 s and that an evidence line appears. If it does
      not trigger, see the fallback note below *before* you film.

---

## Shot-by-shot (60 s)

| t (s) | shot | action | what's on screen |
|------:|------|--------|------------------|
| 0–4   | Title / wide | Laptop facing a running box fan; cursor on the **Monitor** toggle. | App title "🎧 EarSight — live acoustic-anomaly monitor"; sidebar setup; quiet placeholder ("Turn on Monitor…"). |
| 4–8   | Sidebar | Toggle **Monitor (open mic)** on. Spectrogram begins scrolling. | Rolling 10 s log-mel spectrogram comes alive; state badge reads **🟢 LISTENING**; percentile near 0. |
| 8–14  | Setup | Confirm **Baseline tag** = `bench-unit`, set **RPM** if known (else 0). Click **Capture baseline (30 s)**. | Progress bar: "30s remaining — keep the machine sounding NORMAL". |
| 14–24 | Baseline (sped up) | Let the 30 s baseline capture run on the **healthy** fan. (Speed this segment up 3–4× in edit; narrate "this is what healthy sounds like.") | Countdown ticks down; toast "Baseline 'bench-unit' fitted from 28 windows ✅". State settles at **🟢 LISTENING**, percentile low. |
| 24–30 | Hands / close-up | Induce the fault: **tape a coin to one fan blade** (imbalance) — or loosen one panel screw so it rattles. Step back. | Spectrogram texture visibly changes; low-band energy thickens. |
| 30–36 | Meter | Watch the percentile climb. | State badge flips **🟢 LISTENING → 🟡 SUSPECT**; anomaly percentile jumps into the 90s; progress gauge fills amber. |
| 36–42 | Meter (hold) | Keep the fault running; the streak builds. | Badge flips **🟡 SUSPECT → 🔴 ALERT** (red); gauge near full. |
| 42–48 | Evidence | Push in on the evidence line. | Big subheader, e.g. "⚠ impulse train ~88 Hz, low-band energy up"; expand "Full evidence" to flash the JSON (mel bands, envelope peaks, nearest baseline). |
| 48–56 | Record & label | Scroll to **Record & label**; pick machine type **fan**, type fault "blade imbalance", add a note, click **Save last 10 s clip**. | Form fills; success banner "Saved clip → …/datakit/data/…wav". (This is the data-moat moment — every demo feeds the dataset.) |
| 56–60 | End card | Remove the coin; meter relaxes back toward **LISTENING**. Cut to end card. | End card: "EarSight — hands-free acoustic diagnostics for techs. Baseline in 30 s. Anomalies with evidence. Glasses-native, built glasses-free." |

---

## Fallback note (if ALERT doesn't trigger in one take)

The default alert policy is deliberately conservative: it needs
`n_consecutive = 5` hot windows (≈5 s sustained anomaly) before it escalates
**SUSPECT → ALERT**, so a brief or marginal fault may stall at SUSPECT.

For the demo, make the alarm quicker to fire by lowering `n_consecutive` to
`3`. In `desktop/app.py`, the scorer is built with a default `AlertPolicy()`;
pass a tuned config instead. Both spots that construct the scorer
(`_fit_baseline` and `_load_baseline`) use:

```python
from engine.policy import AlertPolicy, PolicyConfig
policy = AlertPolicy(PolicyConfig(n_consecutive=3))
StreamScorer(baseline, embedder, policy, rpm=rpm)
```

Revert to `AlertPolicy()` after filming so the shipped app keeps its honest,
trust-preserving thresholds. If the meter still won't move, make the fault
*louder*: a larger coin, a looser panel, or move the mic closer (30–50 cm).
Keep it honest — stage a **real** fault the mic can actually hear, just make it
unambiguous on camera.
