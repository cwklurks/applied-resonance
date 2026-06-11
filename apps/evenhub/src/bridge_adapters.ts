/**
 * Adapters that bind the bridge-agnostic Pipeline to the Even Hub SDK.
 *
 * `BridgeAudioFeed` turns the mic on and routes `audioEvent.audioPcm` chunks
 * into the pipeline. `BridgeDisplay` pushes finished HUD cards to the single
 * 576×288 text container as `line1\nline2`.
 *
 * Defensive PCM handling: the SDK *types* `audioPcm` as `Uint8Array`, but the
 * host may deliver it as a `number[]` or a base64 string after JSON transit.
 * We normalize all three to a `Uint8Array` here so the pipeline only ever sees
 * raw s16le bytes.
 */

import type {
  EvenAppBridge,
  EvenHubEvent,
} from '@evenrealities/even_hub_sdk'
import { TextContainerUpgrade } from '@evenrealities/even_hub_sdk'
import type { HudCard } from '@earsight/display-card'
import type { AudioFeed, DisplayPort } from './pipeline'

const HUD_CONTAINER_ID = 1
const HUD_CONTAINER_NAME = 'hud'

/** Coerce whatever the host sent for audioPcm into a Uint8Array, or null. */
export function coercePcm(raw: unknown): Uint8Array | null {
  if (raw instanceof Uint8Array) return raw.length > 0 ? raw : null
  if (Array.isArray(raw)) {
    return raw.length > 0 ? Uint8Array.from(raw as number[]) : null
  }
  if (typeof raw === 'string' && raw.length > 0) {
    try {
      const binary = atob(raw)
      const out = new Uint8Array(binary.length)
      for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i)
      return out.length > 0 ? out : null
    } catch {
      return null
    }
  }
  return null
}

/**
 * Mic feed backed by the SDK bridge. `start` enables the mic and subscribes to
 * audio events; `stop` unsubscribes and turns the mic off. The owner is
 * expected to also subscribe for sys/text events separately (taps, exits).
 */
export class BridgeAudioFeed implements AudioFeed {
  private readonly bridge: EvenAppBridge
  private unsub: (() => void) | null = null

  constructor(bridge: EvenAppBridge) {
    this.bridge = bridge
  }

  start(onChunk: (bytes: Uint8Array) => void): void {
    this.unsub = this.bridge.onEvenHubEvent((event: EvenHubEvent) => {
      const raw = event.audioEvent?.audioPcm
      if (raw == null) return
      const pcm = coercePcm(raw)
      if (pcm) onChunk(pcm)
    })
    // Fire-and-forget: the mic-on ack is not load-bearing for the pipeline.
    void this.bridge.audioControl(true)
  }

  stop(): void {
    this.unsub?.()
    this.unsub = null
    void this.bridge.audioControl(false)
  }
}

/** Display sink that writes the two HUD lines to the text container. */
export class BridgeDisplay implements DisplayPort {
  private readonly bridge: EvenAppBridge

  constructor(bridge: EvenAppBridge) {
    this.bridge = bridge
  }

  render(card: HudCard): void {
    const content = card.line2 ? `${card.line1}\n${card.line2}` : card.line1
    void this.bridge
      .textContainerUpgrade(
        new TextContainerUpgrade({
          containerID: HUD_CONTAINER_ID,
          containerName: HUD_CONTAINER_NAME,
          content,
        }),
      )
      .catch((err) => {
        // A failed lens write must never bubble up as raw text or a crash.
        console.error('BridgeDisplay: textContainerUpgrade failed', err)
      })
  }
}

export { HUD_CONTAINER_ID, HUD_CONTAINER_NAME }
