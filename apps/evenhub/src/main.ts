/**
 * Applied Resonance Even Hub shell: a thin bridge wrapper around the bridge-agnostic
 * Pipeline. It owns only the lens lifecycle (container create, mic, taps,
 * exit/cleanup) and delegates all audio→engine→HUD logic to Pipeline.
 *
 * Lifecycle, kept from the template:
 *   - createStartUpPageContainer FIRST (single 576×288 text container),
 *   - single tap → pipeline.logTap(); double tap → shutDownPageContainer(1),
 *   - SYSTEM_EXIT/ABNORMAL_EXIT and beforeunload → cleanup (mic off, stop, unsub),
 *   - any thrown/rejected handler path → OFFLINE card, console.error (never raw
 *     text on the lens).
 */

import {
  waitForEvenAppBridge,
  TextContainerProperty,
  CreateStartUpPageContainer,
  OsEventTypeList,
} from '@evenrealities/even_hub_sdk'
import { EngineClient, formatHudCard } from '@earsight/display-card'
import { Pipeline } from './pipeline'
import { BridgeAudioFeed, BridgeDisplay, HUD_CONTAINER_ID } from './bridge_adapters'
import { runMockFromWav } from './mock'
import { ENGINE_ORIGIN, isLocalEngineOrigin } from './config'
import {
  bindCaptureControls,
  mountUi,
  setBadge,
  setCaptureResult,
  setCaptureStatus,
  setCard,
  setEvidence,
  pushLog,
  readSettings,
} from './ui'

/** Lift the evidence text (everything before ` · tap to log`) for the panel. */
function evidenceFrom(line1: string, line2: string): string | null {
  if (!(line1.startsWith('!') || line1.startsWith('?'))) return null
  const idx = line2.lastIndexOf(' · tap to log')
  return idx >= 0 ? line2.slice(0, idx) : line2
}

mountUi()

const settings = readSettings()
void boot().catch((error) => {
  setCard('engine offline', 'check phone connection')
  console.error('startup failed', error)
})

async function boot(): Promise<void> {
  const bearerToken = await readRuntimeBearerToken(ENGINE_ORIGIN)
  const client = new EngineClient(ENGINE_ORIGIN, fetch, {
    bearerToken,
    timeoutMs: 5000,
  })
  pushLog(`runtime engine: ${ENGINE_ORIGIN}`)

  // Mock mode drives the identical pipeline from a WAV instead of the glasses.
  const mockUrl = new URLSearchParams(location.search).get('mock')
  if (mockUrl) {
    await runMockMode(mockUrl, client)
  } else {
    await runBridgeMode(client)
  }
}

function readRuntimeBearerToken(engineOrigin: string): Promise<string | undefined> {
  if (isLocalEngineOrigin(engineOrigin)) return Promise.resolve(undefined)

  return new Promise((resolve) => {
    const overlay = document.createElement('div')
    overlay.setAttribute('role', 'dialog')
    overlay.setAttribute('aria-modal', 'true')
    overlay.style.cssText =
      'position:fixed;inset:0;z-index:9999;display:grid;place-items:center;padding:24px;background:#10110feF'

    const form = document.createElement('form')
    form.autocomplete = 'off'
    form.style.cssText =
      'width:min(420px,100%);display:grid;gap:14px;padding:22px;border:1px solid #6f7669;background:#1a1c18;color:#f4f3ed'
    const title = document.createElement('strong')
    title.textContent = 'Connect to secure engine'
    const detail = document.createElement('span')
    detail.textContent = `${engineOrigin} · token stays in memory and is requested again after reload`
    const input = document.createElement('input')
    input.type = 'password'
    input.autocomplete = 'off'
    input.spellcheck = false
    input.setAttribute('autocapitalize', 'none')
    input.setAttribute('autocorrect', 'off')
    input.setAttribute('aria-label', 'Engine bearer token')
    input.placeholder = 'Bearer token'
    input.required = true
    const submit = document.createElement('button')
    submit.type = 'submit'
    submit.textContent = 'Connect'
    form.append(title, detail, input, submit)
    overlay.append(form)
    document.body.append(overlay)
    input.focus()

    form.addEventListener('submit', (event) => {
      event.preventDefault()
      const token = input.value.trim()
      input.value = ''
      overlay.remove()
      resolve(token || undefined)
    })
  })
}

async function runMockMode(wavUrl: string, client: EngineClient): Promise<void> {
  setBadge('MOCK')
  pushLog(`mock mode: fetching ${wavUrl}`)
  try {
    const res = await fetch(wavUrl)
    const buf = await res.arrayBuffer()
    const display = {
      render(card: { line1: string; line2: string }) {
        setCard(card.line1, card.line2)
        const evidence = evidenceFrom(card.line1, card.line2)
        if (evidence) setEvidence(evidence)
      },
    }
    await runMockFromWav(buf, {
      client,
      display,
      mode: 'session',
      tag: settings.tag || 'mock',
      rpm: settings.rpm,
      onStateChange: (s) => {
        setBadge(s)
        pushLog(`state: ${s}`)
      },
    })
    pushLog('mock run complete')
  } catch (err) {
    setCard('engine offline', 'check phone connection')
    console.error('mock mode failed', err)
  }
}

async function runBridgeMode(client: EngineClient): Promise<void> {
  const bridge = await waitForEvenAppBridge()

  // Single 576×288 text container, event capture on so taps reach us.
  const hud = new TextContainerProperty({
    xPosition: 0,
    yPosition: 0,
    width: 576,
    height: 288,
    borderWidth: 0,
    paddingLength: 4,
    containerID: HUD_CONTAINER_ID,
    containerName: 'hud',
    content: 'Applied Resonance starting...',
    isEventCapture: 1,
  })

  const created = await bridge.createStartUpPageContainer(
    new CreateStartUpPageContainer({ containerTotalNum: 1, textObject: [hud] }),
  )
  if (created !== 0) {
    pushLog(`createStartUpPageContainer failed: ${created}`)
    console.error('Failed to create startup page container', created)
  }

  const bridgeDisplay = new BridgeDisplay(bridge)

  // Mirror every card to both the lens and the companion UI.
  const display = {
    render(card: { line1: string; line2: string }) {
      bridgeDisplay.render(card)
      setCard(card.line1, card.line2)
      const evidence = evidenceFrom(card.line1, card.line2)
      if (evidence) setEvidence(evidence)
    },
  }

  // Mode: prefer a saved baseline if the companion settings name a tag that the
  // engine already has; otherwise capture a fresh session baseline.
  const mode = await resolveMode(client, settings.mode, settings.tag)
  pushLog(`engine ${ENGINE_ORIGIN} · mode ${mode} · tag ${settings.tag || '(session)'}`)

  const feed = new BridgeAudioFeed(bridge)
  const pipeline = new Pipeline(feed, {
    client,
    display,
    mode,
    tag: settings.tag || 'glasses-session',
    rpm: settings.rpm,
    onStateChange: (s) => {
      setBadge(s)
      pushLog(`state: ${s}`)
    },
    onCaptureEvent: (event) => {
      switch (event.kind) {
        case 'started':
          setCaptureStatus(true, 0)
          pushLog(`raw capture started: ${event.captureId}`)
          break
        case 'append':
          setCaptureStatus(true, event.secondsTotal)
          break
        case 'stopped':
          setCaptureStatus(false, event.durationS)
          setCaptureResult(event.wavPath, event.durationS)
          pushLog(`raw capture stopped: ${event.wavPath}`)
          break
        case 'warning':
          setCaptureStatus(pipeline.capturing, pipeline.captureSeconds())
          pushLog(event.message)
          break
      }
    },
  })
  bindCaptureControls({
    start: (tag) => pipeline.startCapture(tag),
    stop: () => pipeline.stopCapture(),
    capturing: () => pipeline.capturing,
    seconds: () => pipeline.captureSeconds(),
  })

  let cleanedUp = false
  function cleanup(): void {
    if (cleanedUp) return
    cleanedUp = true
    try {
      pipeline.stop()
      unsubscribe()
    } catch (err) {
      console.error('cleanup failed', err)
    }
  }

  // Event routing (see the audio.ts/template notes):
  //   - audio PCM is handled inside BridgeAudioFeed, not here.
  //   - CLICK_EVENT is 0 and protobuf omits zero values, so a single tap arrives
  //     as sysEvent PRESENT with eventType === undefined. Detect with `?? 0`.
  //   - double tap (3) → confirm-exit dialog; exit events → cleanup.
  const unsubscribe = bridge.onEvenHubEvent((event) => {
    try {
      const sys = event.sysEvent
      if (sys) {
        const type = sys.eventType ?? 0
        if (type === OsEventTypeList.CLICK_EVENT) {
          void pipeline.logTap()
          return
        }
        if (type === OsEventTypeList.DOUBLE_CLICK_EVENT) {
          void bridge.shutDownPageContainer(1)
          return
        }
        if (
          type === OsEventTypeList.SYSTEM_EXIT_EVENT ||
          type === OsEventTypeList.ABNORMAL_EXIT_EVENT
        ) {
          cleanup()
          return
        }
      }
      // Double-tap can also surface via a textEvent on the captured container.
      if (event.textEvent?.eventType === OsEventTypeList.DOUBLE_CLICK_EVENT) {
        void bridge.shutDownPageContainer(1)
      }
    } catch (err) {
      // Never let a handler throw raw text onto the lens.
      const offline = formatHudCard({ state: 'OFFLINE' })
      display.render(offline)
      console.error('event handler error', err)
    }
  })

  window.addEventListener('beforeunload', cleanup)

  try {
    pushLog('pipeline: starting session')
    await pipeline.start()
    pushLog('pipeline: session started')
  } catch (err) {
    const offline = formatHudCard({ state: 'OFFLINE' })
    display.render(offline)
    console.error('pipeline.start failed', err)
  }
}

/**
 * Decide session vs saved. 'auto' → saved if the protected baseline lookup
 * confirms the tag, else session. Explicit 'saved'/'session' are honored
 * but 'saved' falls back to 'session' if the tag is missing or the engine is
 * unreachable (so we never open a session that will 404 on every score).
 */
async function resolveMode(
  client: EngineClient,
  requested: 'auto' | 'saved' | 'session',
  tag: string,
): Promise<'saved' | 'session'> {
  if (requested === 'session') return 'session'
  if (!tag) return 'session'

  const baseline = await client.hasBaseline(tag)
  const hasBaseline =
    baseline.kind === 'ok' && baseline.value.exists

  if (requested === 'saved') return hasBaseline ? 'saved' : 'session'
  // auto
  return hasBaseline ? 'saved' : 'session'
}
