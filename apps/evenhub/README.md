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
| `app.json` | Loopback development manifest. `scripts/generate-manifest.mjs` adds the one validated HTTPS engine origin used by a packaged build. |

## Prereqs

- **Node 20+**.
- **The EarSight engine running** on `http://localhost:8000`. From the **repo
  root** (`/…/earsight`):

  ```bash
  EARSIGHT_DEVICE=cpu uv run python -m engine.serve
  ```

  CPU is fine for dev/sim. `GET /health` should return
  exactly `{"status":"ok"}`.

## Dev loop

```bash
cd apps/evenhub
npm install
npm run dev                       # loopback only: http://127.0.0.1:5173
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

**On a real phone/G2 session.** Do not pair to a laptop IP or plain-HTTP Vite
server. Deploy the built WebView at one stable HTTPS companion origin, expose
the engine through a separate stable HTTPS gateway, and configure both before
building. Example hostnames below are placeholders; use origins you control:

```bash
export EARSIGHT_ENGINE_ORIGIN=https://engine.example.com
npm run package:validate
npm run pack
npx evenhub qr --url https://companion.example.com
```

Start the engine with `EARSIGHT_REMOTE_ACCESS=1`, a generated
`EARSIGHT_API_TOKEN`, and
`EARSIGHT_CORS_ORIGINS=https://companion.example.com` as documented in the root
README. The companion shows a masked bearer-token form on every load and keeps
the value in memory only. Paste it there; never append it to the QR/engine URL
or put it in the manifest/build environment. The runtime engine origin is read-only and
cannot be overridden by a query string or stale `localStorage` value.

## Test

```bash
npm test     # vitest run
```

Unit suites cover audio, UI/origin validation, bounded pipeline queues, and the
shared HTTP client, plus one **real-engine integration test**
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
export EARSIGHT_ENGINE_ORIGIN=https://engine.example.com
npm run package:validate   # generated manifest and built runtime must match
npm run pack               # packs app.generated.json + dist/
```

`EARSIGHT_ENGINE_ORIGIN` must be an HTTPS hostname-only origin. The generator
rejects IP literals, HTTP, credentials, paths, query strings, and fragments.
`app.generated.json` is ignored build output; do not hand-edit or commit it.

Notes:

- **Publishing is manual.** Submit the `.ehpk` to the store yourself after
  `evenhub login`. There is no CI publish step.
- `package:validate` proves the built `dist/engine-origin.json` and generated
  manifest contain the same sole remote origin before the CLI is invoked.
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

Operational settings live in the companion WebView; the engine origin is fixed
at build time so it cannot drift from the manifest:

| Setting | Effect |
|---|---|
| **Engine origin** | Read-only. Loopback for local builds; `EARSIGHT_ENGINE_ORIGIN` for validated packaged builds. |
| **Baseline tag** | Names a saved baseline; blank = fresh session baseline. |
| **RPM** | Optional machine RPM hint for the engine; blank = auto. |
| **Mode** | `auto` (use a saved baseline if the engine has the tag, else capture a session), `saved`, or `session`. `saved` falls back to `session` if the tag/engine is missing. |

Baseline tag/RPM/mode persist to `localStorage` (`earsight.settings.v1`);
**Save & reload** applies them. The bearer token is explicitly excluded and is
requested again after reload.

Only `EARSIGHT_ENGINE_ORIGIN` is consumed during a packaged build. Never expose
`EARSIGHT_API_TOKEN` through a `VITE_*` variable or any frontend build input.

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
  reachable at the built engine origin, the token was omitted/rejected, or CORS
  does not include the exact companion origin. Confirm `GET /health`, then an
  authenticated `GET /baselines/<tag>`, and rebuild if the origin is wrong. The
  app retries the session every 5 s on its own — no restart needed.
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
- Engine requests have a 5 s deadline covering fetch and body parsing. Live
  scoring retains one active request plus the newest pending frame and drops
  frames older than 2.5 s. Raw capture has a two-frame pending bound and stops
  with a warning rather than recording a silently gapped stream.

## Real-G2 verification ladder

Software proof must be completed in order before claiming a hardware result:

1. Run `uv run pytest`, the display-card tests/build, EvenHub tests/build, and
   datakit tests/build. Run `package:validate` with the selected HTTPS engine
   origin and retain its exact matching-origin output.
2. Run the loopback simulator. Confirm minimal health, session creation,
   baseline countdown, LISTENING/anomaly cards, tap-to-log, raw capture stop,
   offline timeout, recovery, and clean exit.
3. Against the HTTPS gateway, verify: unauthenticated `/health` returns only
   `{"status":"ok"}`; missing/wrong bearer returns 401 on every other route;
   the correct bearer works; allowed CORS preflight succeeds; a foreign origin
   is denied; PCM/body/session/capture limits and idle cleanup return the
   documented 413/429/404 behavior.
4. Load the stable HTTPS companion URL on the phone, enter the token only in
   the runtime prompt, and confirm no token appears in the URL, storage,
   manifest, console, or built assets. Hold the engine request open to confirm
   an OFFLINE card within 5 s, bounded latest-frame recovery, and no queue burst.
5. Pair/install the generated package on a physical G2. Record the package
   hash, app/firmware versions, phone model/OS, companion and engine origins,
   and observed request `Origin`. Verify mic permission, steady PCM delivery,
   baseline/listening/anomaly HUD flow, tap-to-log, raw capture, and exit.
6. Run the 60 s byte-cadence check, 1 kHz tone check, 10 min background/lock
   check, re-record evaluation, and 30 min glasses/phone battery measurement in
   the table below.

Steps 1-4 are software/network evidence. Steps 5-6 are the remaining hardware
gate and must not be marked complete without the actual phone and G2 session.

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
