# EarSight — Even Hub glasses app

The on-glasses shell for EarSight. The glasses are a **thin client**: they
capture mic audio, stream it to the EarSight engine for anomaly scoring, and
render the engine's verdict as a two-line HUD card on the G2 lens. All the
thinking (baseline modeling, scoring, evidence) lives in the engine — this app
owns only the lens lifecycle, the mic, the taps, and the companion phone view.

**HUD contract.** At most **2 glanceable lines**, each capped at 60 chars, and at
most **one push every 2 s** (a hard throttle — the BLE queue can't keep up with
faster writes). The card wording is owned by `@earsight/display-card`'s
`formatHudCard`, shared with the future MentraOS shell so the two never drift.

Sample cards:

```
Capturing baseline · 28s          Listening...          ! Bearing-like anomaly · 100%
keep the machine sounding normal                         high-band energy up vs baseline · tap to log
```

## Architecture

The heart of the app is a **bridge-agnostic `Pipeline`** (`src/pipeline.ts`) that
is deliberately decoupled from the Even Hub SDK, so the exact same code path
drives the live glasses, headless mock mode, and the tests:

```
AudioFeed → AudioNormalizer → 1 s frames → EngineClient.score → formatHudCard → HudThrottle → DisplayPort
```

| Feed | Display | Used by |
|---|---|---|
| `BridgeAudioFeed` (glasses mic) | `BridgeDisplay` (G2 lens) + companion UI | live app (`src/main.ts`) |
| `WavFileFeed` (a WAV) | companion UI / recorder | mock mode + integration test |

| File | Purpose |
|---|---|
| `src/main.ts` | Bridge wiring + lens lifecycle (container create, mic, taps, exit/cleanup). Mock-vs-bridge mode switch. |
| `src/pipeline.ts` | Bridge-agnostic monitoring loop: feed → normalize → score → throttle → render. Offline → `OFFLINE` card + retry, never raw text on the lens. |
| `src/audio.ts` | Defensive audio layer: s16le↔float, mix-to-mono, linear resampler, throughput-based rate detection, exact 1 s (32000-byte) framing. |
| `src/bridge_adapters.ts` | Binds `Pipeline` to the SDK: mic on/off, `audioPcm` coercion (Uint8Array / number[] / base64), lens text writes. |
| `src/mock.ts` | Minimal PCM-16 RIFF/WAVE parser + `WavFileFeed` + `runMockFromWav` for headless runs. |
| `src/ui.ts` | Companion WebView: settings panel, live card mirror, state badge, evidence panel, scrolling log. |
| `index.html` | WebView host, zoom-locked viewport. |
| `app.json` | Manifest: `g2-microphone` + `network` (whitelist `localhost:8000`) permissions, `package_id com.earsight.evenhub`. |

## Prereqs

- **Node 20+**.
- **The EarSight engine running** on `http://localhost:8000`. From the **repo
  root** (`/…/earsight`):

  ```bash
  EARSIGHT_DEVICE=cpu uv run uvicorn engine.serve:create_app --factory --port 8000
  ```

  CPU is fine for dev/sim. `GET /health` should return
  `{"status":"ok", "backend":"…", "baselines":[…], "sessions":N}`.

## Dev loop

```bash
cd apps/evenhub
npm install
npm run dev          # vite dev server on http://localhost:5173
```

Then drive it with the desktop simulator. **The simulator bin is not symlinked**
(`npx evenhub-simulator` and `npm run simulate` are broken), so launch it with
an explicit node path, from `apps/evenhub`:

```bash
node node_modules/@evenrealities/evenhub-simulator/bin/index.js \
  http://localhost:5173 --automation-port 9898
```

This opens a Tauri GUI window with a glasses-lens view and the companion
WebView side-by-side. (`--automation-port` is optional — it exposes an HTTP
control API; see "Automation API" below.)

> The simulator feeds your **desktop microphone** in as glasses audio, so a
> real (quiet-room) stream flows and the app runs a live session-mode baseline
> capture on ambient sound. List/choose the input device with
> `--list-audio-input-devices` and `--aid <id>`.

**On real glasses.** The `evenhub` CLI bin *is* symlinked, so QR-pairing works
via npx (point at your LAN IP, not localhost):

```bash
npx evenhub qr --url http://<your-lan-ip>:5173
```

## Test

```bash
npm test     # vitest run
```

20 tests across `audio`, `pipeline`, and one **real-engine integration test**
(`test/integration.test.ts`). The integration test is the criterion check: it
**spawns the engine itself** (`uv run uvicorn engine.serve:create_app --factory
--port 8731`, `EARSIGHT_DEVICE=cpu`), waits for `/health`, then feeds three
normal + two abnormal **MIMII pump** clips through the *same* `Pipeline` the
glasses use, and asserts the recorded card sequence: a `CAPTURING_BASELINE`
countdown → `LISTENING` → a `!`/`?` anomaly card with an evidence prefix and
`tap to log` → `logTap()` writing a labeled `.wav`+`.json` pair on disk. It
skips gracefully if `data/mimii/…` is absent.

## Package

The store artifact is an **`.ehpk`** binary (magic `EHPK`, *not* a zip).

```bash
npm run build                                              # tsc --noEmit && vite build → dist/
node node_modules/@evenrealities/evenhub-cli/main.js \
  pack app.json ./dist -o earsight.ehpk                    # ~38 KB
```

(`npx evenhub pack app.json ./dist -o earsight.ehpk` also works — the CLI bin is
symlinked, unlike the simulator.)

Notes:

- **Publishing is manual.** Submit the `.ehpk` to the store yourself after
  `evenhub login`. There is no CI publish step.
- **Local pack does NOT validate the `network` whitelist** — that check is
  store-side only (`pack --check` requires `evenhub login`; skipped here).
- Don't commit the `.ehpk`. It's a build output; regenerate it from `dist/`.

## Mock mode (browser dogfooding)

Append `?mock=<wav-url>` to the app URL and the identical `Pipeline` runs from a
WAV instead of the glasses bridge — no bridge required:

```
http://localhost:5173/?mock=/clips/pump-normal.wav
```

The companion UI shows a `MOCK` badge and renders the resulting cards; on a
fetch/engine failure it falls back to the `engine offline` card. The WAV must be
fetchable from the page origin and must be **16-bit PCM** RIFF/WAVE (the parser
also handles `WAVE_FORMAT_EXTENSIBLE` and multichannel, taking channel 0). The
vitest integration test exercises this same `runMockFromWav` path against real
MIMII audio, so the mock pipeline is covered by `npm test`.

## Settings

All runtime config lives in the **companion WebView settings panel** (no env
vars, no rebuild):

| Setting | Effect |
|---|---|
| **Engine URL** | Where to reach the engine. Default `http://localhost:8000`. |
| **Baseline tag** | Names a saved baseline; blank = fresh session baseline. |
| **RPM** | Optional machine RPM hint for the engine; blank = auto. |
| **Mode** | `auto` (use a saved baseline if the engine has the tag, else capture a session), `saved`, or `session`. `saved` falls back to `session` if the tag/engine is missing. |

Settings persist to `localStorage` (`earsight.settings.v1`); **Save & reload**
applies them. They survive reloads so a technician's config sticks.

> **`.env.example` is a stale template leftover and is unused.** It holds a
> `VITE_STT_API_KEY` placeholder from the speech-to-text scaffold this app was
> forked from. EarSight has no STT and reads no env vars — the engine URL and
> all other config come from the settings panel above. Ignore / delete the
> `.env*` files; they affect nothing.

## Controls

- **Single tap (temple)** → log the **last 10 s** of audio. The rolling ring
  buffer is POSTed to the engine's `/label`, and a `logged ✓` card shows for 2 s
  before live cards resume. Best-effort: a failed write never crashes the lens.
- **Double tap (temple)** → exit. Calls `shutDownPageContainer(1)`, which raises
  the system exit-confirmation dialog.

> **Footnote for future devs — the single-tap quirk.** `CLICK_EVENT` is `0` in
> the SDK's `OsEventTypeList`, and protobuf omits zero-valued fields on the
> wire, so a single tap arrives as a `sysEvent` with `eventType === undefined`.
> `src/main.ts` detects it with `sys.eventType ?? 0`. Don't "fix" that to a
> strict equality on a defined value — single taps will silently stop working.

## Troubleshooting

- **`engine offline` / `check phone connection` card.** The engine isn't
  reachable at the configured Engine URL. Start it (see Prereqs), confirm
  `GET /health` responds, and check the URL in the settings panel. The app
  retries the session every 5 s on its own — no restart needed.
- **No audio / mic.** Grant the OS mic permission to the simulator (or the
  glasses companion on device). In the simulator, list inputs with
  `--list-audio-input-devices` and pin one with `--aid <id>` if the wrong device
  is picked up.
- **`npx evenhub-simulator` / `npm run simulate` fail.** Expected — the
  simulator bin isn't symlinked. Use the explicit
  `node node_modules/@evenrealities/evenhub-simulator/bin/index.js …` form.
- **Rate-mismatch warnings in the companion log.** The audio normalizer detected
  the inbound stream's effective sample rate drifting >20% from 16 kHz and
  switched its working rate. Informational; audio is still normalized to the
  engine's 16 kHz mono target.

## Automation API (simulator)

With `--automation-port <port>`, the simulator exposes an HTTP control surface on
`127.0.0.1:<port>` — handy for scripted load/QA checks:

| Endpoint | Result |
|---|---|
| `GET /api/ping` | `pong` |
| `GET /api/screenshot/glasses` | 576×288 PNG of the lens |
| `GET /api/screenshot/webview` | PNG of the companion view |
| `GET /api/console` | `{entries:[{id,level,message,ts}]}` (captured webview console) |
| `POST /api/input` `{"action":"up"\|"down"\|"click"\|"double_click"}` | injects a temple input |

## G2 specifics

- Mic format (per the template): PCM s16le, 16 kHz, mono, via
  `event.audioEvent.audioPcm`. The normalizer assumes nothing and re-derives the
  rate from throughput — see Hardware unknowns below.
- The lens is a single 576×288 text container; the HUD writes `line1\nline2`.
- Glasses render is throttled to one push / 2 s (BLE queue limit).

## Hardware unknowns

The simulator proves the software path (boot → session → score → HUD → tap-to-log
→ exit) but feeds the **desktop mic**, not the real 4-mic G2 array. These
properties can only be confirmed on hardware, and each could materially change
anomaly-detection quality:

| Unknown | Why it matters | On-hardware test |
|---|---|---|
| **Raw PCM availability** via `audioControl` on real G2 | The whole app assumes a steady `audioPcm` byte stream; the sim only proves the desktop-mic path. | Log `audioPcm` byte counts + delivery cadence on hardware for 60 s; confirm steady ~32 KB/s. |
| **True sample rate & bit depth** (template claims 16 kHz s16le; `AudioNormalizer` detects mismatch via throughput) | A wrong rate/depth corrupts every frame's spectral features before scoring. | Play a known **1 kHz tone** from a speaker, FFT the captured stream, verify the peak lands at bin 1000. |
| **AGC / beamforming** (the 4-mic array may pre-process audio) | Automatic gain or beamforming can squash the stationary machine signature the engine keys on. | Re-record the MIMII playlist (`killtest/`) through the glasses, run `killtest.eval_rerun`, compare AUC against the other mic chains. |
| **Background mic policy** (can `audioControl` stay open with screen off / app backgrounded?) | Continuous monitoring is the product; if the OS suspends the mic when backgrounded, the use case breaks. | 10-min capture with the phone locked / app backgrounded; count delivered frames vs expected (~600). |
| **Battery cost** of continuous capture + BLE | Determines session length and whether all-shift monitoring is viable. | 30-min continuous session; record battery % before/after on both glasses and phone. |
