/**
 * Companion WebView (the phone holds the detail). Dependency-free vanilla
 * TS/DOM, matching the template's ui.ts style.
 *
 * Shows: a settings panel (engine URL, baseline tag, RPM, mode), a live mirror
 * of the current HUD card, a state badge, the last evidence line, and a scrolling
 * log of warnings (rate mismatch, offline events). Settings persist to
 * localStorage so a reload keeps the technician's configuration.
 */

import { ENGINE_ORIGIN } from './config'

export type Mode = 'auto' | 'saved' | 'session'

export interface Settings {
  tag: string
  rpm: number | null
  mode: Mode
}

export interface CaptureControlHandlers {
  start(tag: string): Promise<string | null>
  stop(): Promise<{ wavPath: string; durationS: number } | null>
  capturing(): boolean
  seconds(): number
}

const STORAGE_KEY = 'earsight.settings.v1'
const DEFAULT_CAPTURE_TAG = 'g2-characterization'

const DEFAULTS: Settings = {
  tag: '',
  rpm: null,
  mode: 'auto',
}

let line1El: HTMLDivElement
let line2El: HTMLDivElement
let badgeEl: HTMLDivElement
let evidenceEl: HTMLPreElement
let logEl: HTMLDivElement
let captureTagEl: HTMLInputElement
let captureButtonEl: HTMLButtonElement
let captureElapsedEl: HTMLDivElement
let capturePathEl: HTMLDivElement
let captureHandlers: CaptureControlHandlers | null = null
let captureTimer: number | null = null

/** Read persisted settings, falling back to defaults for any missing field. */
export function readSettings(): Settings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return { ...DEFAULTS }
    const parsed = JSON.parse(raw) as Partial<Settings>
    return {
      tag: typeof parsed.tag === 'string' ? parsed.tag : DEFAULTS.tag,
      rpm: typeof parsed.rpm === 'number' ? parsed.rpm : DEFAULTS.rpm,
      mode:
        parsed.mode === 'saved' || parsed.mode === 'session' || parsed.mode === 'auto'
          ? parsed.mode
          : DEFAULTS.mode,
    }
  } catch {
    return { ...DEFAULTS }
  }
}

function writeSettings(s: Settings): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(s))
  } catch {
    /* localStorage may be unavailable; settings are best-effort. */
  }
}

export function mountUi(): void {
  const app = document.querySelector<HTMLDivElement>('#app')
  if (!app) return
  const s = readSettings()

  app.innerHTML = `
    <main class="panel">
      <header>
        <h1>Applied Resonance</h1>
        <div id="badge" class="badge">idle</div>
      </header>

      <section class="card" aria-live="polite">
        <div id="line1" class="line1"></div>
        <div id="line2" class="line2"></div>
      </section>

      <section class="settings">
        <label>Engine origin
          <input id="engineUrl" type="url" value="${escapeAttr(ENGINE_ORIGIN)}" readonly />
        </label>
        <label>Baseline tag
          <input id="tag" type="text" value="${escapeAttr(s.tag)}" placeholder="(blank = fresh session)" />
        </label>
        <div class="row">
          <label>RPM
            <input id="rpm" type="number" min="0" step="1" value="${s.rpm ?? ''}" placeholder="auto" />
          </label>
          <label>Mode
            <select id="mode">
              <option value="auto"${s.mode === 'auto' ? ' selected' : ''}>auto</option>
              <option value="saved"${s.mode === 'saved' ? ' selected' : ''}>saved</option>
              <option value="session"${s.mode === 'session' ? ' selected' : ''}>session</option>
            </select>
          </label>
        </div>
        <button id="save">Save &amp; reload</button>
      </section>

      <section class="raw-capture">
        <h2>Raw capture</h2>
        <label>Tag
          <input id="captureTag" type="text" value="${DEFAULT_CAPTURE_TAG}" />
        </label>
        <div class="capture-row">
          <button id="captureToggle">Start</button>
          <div id="captureElapsed" class="capture-elapsed">0.0s</div>
        </div>
        <div id="capturePath" class="capture-path">—</div>
      </section>

      <section class="evidence">
        <h2>Last evidence</h2>
        <pre id="evidence">—</pre>
      </section>

      <section class="log">
        <h2>Log</h2>
        <div id="log"></div>
      </section>

      <footer>Tap temple = log clip · double-tap = exit</footer>
    </main>
  `

  line1El = app.querySelector<HTMLDivElement>('#line1')!
  line2El = app.querySelector<HTMLDivElement>('#line2')!
  badgeEl = app.querySelector<HTMLDivElement>('#badge')!
  evidenceEl = app.querySelector<HTMLPreElement>('#evidence')!
  logEl = app.querySelector<HTMLDivElement>('#log')!
  captureTagEl = app.querySelector<HTMLInputElement>('#captureTag')!
  captureButtonEl = app.querySelector<HTMLButtonElement>('#captureToggle')!
  captureElapsedEl = app.querySelector<HTMLDivElement>('#captureElapsed')!
  capturePathEl = app.querySelector<HTMLDivElement>('#capturePath')!

  app.querySelector<HTMLButtonElement>('#save')!.addEventListener('click', () => {
    const next: Settings = {
      tag: app.querySelector<HTMLInputElement>('#tag')!.value.trim(),
      rpm: parseRpm(app.querySelector<HTMLInputElement>('#rpm')!.value),
      mode: app.querySelector<HTMLSelectElement>('#mode')!.value as Mode,
    }
    writeSettings(next)
    location.reload()
  })

  captureButtonEl.addEventListener('click', () => {
    void toggleCapture()
  })

  injectStyles()
}

function parseRpm(value: string): number | null {
  const n = Number(value)
  return value.trim() !== '' && Number.isFinite(n) && n > 0 ? n : null
}

/** Mirror the current HUD card. Also lifts evidence onto the evidence panel. */
export function setCard(line1: string, line2: string): void {
  if (line1El) line1El.textContent = line1
  if (line2El) line2El.textContent = line2
}

export function setBadge(state: string): void {
  if (!badgeEl) return
  badgeEl.textContent = state
  badgeEl.className = `badge badge-${state.toLowerCase().replace(/[^a-z]/g, '')}`
}

export function setEvidence(json: string): void {
  if (evidenceEl) evidenceEl.textContent = json || '—'
}

/** Append a line to the scrolling log (warnings, offline/recovery, state). */
export function pushLog(message: string): void {
  if (!logEl) return
  const row = document.createElement('div')
  row.className = 'log-row'
  const time = new Date().toLocaleTimeString()
  row.textContent = `${time}  ${message}`
  logEl.appendChild(row)
  // Keep the log bounded so a long session doesn't grow the DOM unbounded.
  while (logEl.childElementCount > 200) logEl.removeChild(logEl.firstChild!)
  logEl.scrollTop = logEl.scrollHeight
}

export function bindCaptureControls(handlers: CaptureControlHandlers): void {
  captureHandlers = handlers
  refreshCaptureUi()
}

export function setCaptureStatus(capturing: boolean, seconds: number): void {
  if (captureButtonEl) captureButtonEl.textContent = capturing ? 'Stop' : 'Start'
  if (captureElapsedEl) captureElapsedEl.textContent = `${seconds.toFixed(1)}s`
  if (capturing) {
    startCaptureTimer()
  } else {
    stopCaptureTimer()
  }
}

export function setCaptureResult(wavPath: string, durationS: number): void {
  if (capturePathEl) capturePathEl.textContent = `${wavPath} (${durationS.toFixed(1)}s)`
}

async function toggleCapture(): Promise<void> {
  if (!captureHandlers) {
    pushLog('raw capture unavailable')
    return
  }

  captureButtonEl.disabled = true
  try {
    if (captureHandlers.capturing()) {
      const stopped = await captureHandlers.stop()
      if (stopped) {
        setCaptureResult(stopped.wavPath, stopped.durationS)
        setCaptureStatus(false, stopped.durationS)
      } else {
        refreshCaptureUi()
      }
      return
    }

    const tag = captureTagEl.value.trim() || DEFAULT_CAPTURE_TAG
    capturePathEl.textContent = '—'
    const captureId = await captureHandlers.start(tag)
    if (captureId) {
      setCaptureStatus(true, 0)
    } else {
      refreshCaptureUi()
    }
  } finally {
    captureButtonEl.disabled = false
  }
}

function refreshCaptureUi(): void {
  if (!captureHandlers) return
  setCaptureStatus(captureHandlers.capturing(), captureHandlers.seconds())
}

function startCaptureTimer(): void {
  if (captureTimer !== null) return
  captureTimer = window.setInterval(() => {
    if (!captureHandlers) return
    setCaptureStatus(captureHandlers.capturing(), captureHandlers.seconds())
  }, 1000)
}

function stopCaptureTimer(): void {
  if (captureTimer === null) return
  window.clearInterval(captureTimer)
  captureTimer = null
}

function escapeAttr(value: string): string {
  return value.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;')
}

function injectStyles(): void {
  // ER brand dark surfaces + OS green / signal red, mirroring the template.
  const css = `
    :root { color-scheme: dark; }
    html, body { margin: 0; height: 100%; background: #232323; color: #E5E5E5;
      font: 15px/1.4 -apple-system, BlinkMacSystemFont, 'Helvetica Neue', system-ui, sans-serif;
      touch-action: manipulation; -webkit-text-size-adjust: 100%; overscroll-behavior: none; }
    #app { display: flex; height: 100%; }
    .panel { display: flex; flex-direction: column; gap: 14px; width: 100%;
      max-width: 640px; margin: 0 auto; padding: 20px; box-sizing: border-box; overflow: auto; }
    header { display: flex; align-items: center; justify-content: space-between; }
    h1 { font-size: 18px; font-weight: 600; margin: 0; letter-spacing: 0.02em; }
    h2 { font-size: 11px; font-weight: 600; margin: 0 0 6px; letter-spacing: 0.08em;
      text-transform: uppercase; color: #8A8A8A; }
    .badge { font-size: 11px; padding: 4px 10px; border-radius: 999px; border: 1px solid #3E3E3E;
      color: #A7A7A7; letter-spacing: 0.06em; text-transform: uppercase; }
    .badge-listening { color: #3CFA44; border-color: #3CFA44; background: rgba(60,250,68,0.08); }
    .badge-capturingbaseline { color: #FFD60A; border-color: #FFD60A; background: rgba(255,214,10,0.08); }
    .badge-suspect { color: #FF9F0A; border-color: #FF9F0A; background: rgba(255,159,10,0.08); }
    .badge-alert { color: #FF453A; border-color: #FF453A; background: rgba(255,69,58,0.08); }
    .badge-offline { color: #FF453A; border-color: #FF453A; }
    .badge-logged { color: #3CFA44; border-color: #3CFA44; }
    .card { background: #2E2E2E; border: 1px solid #3E3E3E; border-radius: 12px; padding: 18px;
      min-height: 56px; }
    .line1 { font-size: 18px; font-weight: 600; }
    .line2 { font-size: 14px; color: #B5B5B5; margin-top: 4px; min-height: 18px; }
    .settings { display: flex; flex-direction: column; gap: 10px; background: #2A2A2A;
      border: 1px solid #3E3E3E; border-radius: 12px; padding: 14px; }
    .settings label { display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: #9A9A9A; }
    .settings .row { display: flex; gap: 12px; }
    .settings .row label { flex: 1; }
    .raw-capture { display: flex; flex-direction: column; gap: 10px; background: #2A2A2A;
      border: 1px solid #3E3E3E; border-radius: 12px; padding: 14px; }
    .raw-capture label { display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: #9A9A9A; }
    .capture-row { display: flex; align-items: center; gap: 12px; }
    .capture-row button { min-width: 96px; }
    .capture-elapsed { font: 13px/1.4 ui-monospace, 'SF Mono', Menlo, monospace; color: #C7C7C7; }
    .capture-path { min-height: 18px; font: 11px/1.5 ui-monospace, 'SF Mono', Menlo, monospace;
      color: #9A9A9A; word-break: break-word; }
    input, select { background: #1E1E1E; color: #E5E5E5; border: 1px solid #3E3E3E;
      border-radius: 8px; padding: 8px 10px; font: inherit; }
    button { background: #3CFA44; color: #0A0A0A; border: none; border-radius: 8px;
      padding: 10px; font: inherit; font-weight: 600; cursor: pointer; }
    .evidence pre { background: #1E1E1E; border: 1px solid #3E3E3E; border-radius: 8px;
      padding: 10px; margin: 0; font: 12px/1.4 ui-monospace, 'SF Mono', Menlo, monospace;
      white-space: pre-wrap; word-break: break-word; color: #C7C7C7; }
    .log { flex: 1; display: flex; flex-direction: column; min-height: 80px; }
    #log { flex: 1; overflow: auto; background: #1E1E1E; border: 1px solid #3E3E3E;
      border-radius: 8px; padding: 8px; font: 11px/1.5 ui-monospace, 'SF Mono', Menlo, monospace;
      color: #9A9A9A; max-height: 200px; }
    .log-row { white-space: pre-wrap; word-break: break-word; }
    footer { font-size: 11px; color: #7B7B7B; text-align: center; }
  `
  const style = document.createElement('style')
  style.textContent = css
  document.head.appendChild(style)
}
