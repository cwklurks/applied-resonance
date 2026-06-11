/**
 * EarSight Even Hub shell: a thin bridge wrapper around the bridge-agnostic
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
import { mountUi, setBadge, setCard, setEvidence, pushLog, readSettings } from './ui'

/** Lift the evidence text (everything before ` · tap to log`) for the panel. */
function evidenceFrom(line1: string, line2: string): string | null {
  if (!(line1.startsWith('!') || line1.startsWith('?'))) return null
  const idx = line2.lastIndexOf(' · tap to log')
  return idx >= 0 ? line2.slice(0, idx) : line2
}

const DEFAULT_ENGINE_URL = 'http://localhost:8000'

mountUi()

const settings = readSettings()
const engineUrl = settings.engineUrl || DEFAULT_ENGINE_URL

// ── Mock mode (browser dogfooding): ?mock=<wavUrl> drives the identical
// pipeline from a WAV instead of the glasses, no bridge required. ──────────
const mockUrl = new URLSearchParams(location.search).get('mock')
if (mockUrl) {
  void runMockMode(mockUrl, engineUrl)
} else {
  void runBridgeMode(engineUrl)
}

async function runMockMode(wavUrl: string, baseUrl: string): Promise<void> {
  setBadge('MOCK')
  pushLog(`mock mode: fetching ${wavUrl}`)
  try {
    const res = await fetch(wavUrl)
    const buf = await res.arrayBuffer()
    const client = new EngineClient(baseUrl)
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

async function runBridgeMode(baseUrl: string): Promise<void> {
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
    content: 'EarSight starting...',
    isEventCapture: 1,
  })

  const created = await bridge.createStartUpPageContainer(
    new CreateStartUpPageContainer({ containerTotalNum: 1, textObject: [hud] }),
  )
  if (created !== 0) {
    pushLog(`createStartUpPageContainer failed: ${created}`)
    console.error('Failed to create startup page container', created)
  }

  const client = new EngineClient(baseUrl)
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
  pushLog(`engine ${baseUrl} · mode ${mode} · tag ${settings.tag || '(session)'}`)

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
    await pipeline.start()
  } catch (err) {
    const offline = formatHudCard({ state: 'OFFLINE' })
    display.render(offline)
    console.error('pipeline.start failed', err)
  }
}

/**
 * Decide session vs saved. 'auto' → saved if the engine's /health lists the tag
 * as an existing baseline, else session. Explicit 'saved'/'session' are honored
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

  const health = await client.health()
  const hasBaseline =
    health.kind === 'ok' && health.value.baselines.includes(tag)

  if (requested === 'saved') return hasBaseline ? 'saved' : 'session'
  // auto
  return hasBaseline ? 'saved' : 'session'
}
