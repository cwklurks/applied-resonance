/**
 * Bridge-agnostic anomaly-monitoring pipeline.
 *
 * This is the heart of EarSight on the glasses, deliberately decoupled from the
 * Even Hub SDK so the exact same code path drives:
 *   - the live glasses app (main.ts wires a BridgeAudioFeed + BridgeDisplay),
 *   - headless mock mode (mock.ts wires a WavFileFeed + a recording display),
 *   - the unit + integration tests (a scripted feed + a recording display).
 *
 * Flow: feed → AudioNormalizer → per 1 s frame → EngineClient.score → map the
 * response to a HUD card → HudThrottle → DisplayPort.render. Failures never
 * surface as raw text on the lens: the engine going away becomes an OFFLINE
 * card and the pipeline keeps trying until it answers again.
 */

import {
  EngineClient,
  formatHudCard,
  HudThrottle,
  type HudCard,
  type HudInput,
  type HudState,
} from '@earsight/display-card'
import { AudioNormalizer } from './audio'

/** A source of s16le PCM bytes (glasses mic, a WAV file, a test script). */
export interface AudioFeed {
  start(onChunk: (bytes: Uint8Array) => void): void
  stop(): void
}

/** A sink for finished HUD cards (the glasses lens, a recorder). */
export interface DisplayPort {
  render(card: HudCard): void
}

export interface PipelineOpts {
  client: EngineClient
  display: DisplayPort
  mode: 'session' | 'saved'
  tag: string
  rpm?: number | null
  now?: () => number
  /** Mirror state transitions to a companion UI (e.g. the phone WebView). */
  onStateChange?: (state: string) => void
  /**
   * Fires once per 1 s frame after its scoring settles. Mock/accelerated runs
   * use this to advance a virtual clock so the 2 s HUD throttle behaves as it
   * would in real time (the live app ignores it — frames arrive at 1 Hz).
   */
  onFrameScored?: () => void
  /** Rolling tap-to-log buffer length in seconds (default 10). */
  ringSeconds?: number
}

const FRAME_BYTES = 32000 // 1 s @ 16 kHz s16le
const OFFLINE_RETRY_MS = 5000
const LOGGED_HOLD_MS = 2000
const DEFAULT_RING_SECONDS = 10

/** Engine wire state → HUD state. Only these four come back from /score. */
function engineStateToHud(state: string): HudState {
  switch (state) {
    case 'CAPTURING_BASELINE':
      return 'CAPTURING_BASELINE'
    case 'SUSPECT':
      return 'SUSPECT'
    case 'ALERT':
      return 'ALERT'
    default:
      // LISTENING and any unexpected value default to the safe idle card.
      return 'LISTENING'
  }
}

export class Pipeline {
  private readonly feed: AudioFeed
  private readonly client: EngineClient
  private readonly display: DisplayPort
  private readonly mode: 'session' | 'saved'
  private readonly tag: string
  private readonly rpm: number | null
  private readonly now: () => number
  private readonly onStateChange?: (state: string) => void
  private readonly onFrameScored?: () => void
  private readonly ringFrames: number

  private readonly normalizer: AudioNormalizer
  private readonly throttle: HudThrottle

  private sessionId: string | null = null
  private started = false
  private starting = false

  /**
   * Frames are scored strictly in order, one at a time. The engine session is
   * stateful (capture buffers chunks in arrival order), and the glasses deliver
   * ~1 frame/s anyway, so serializing keeps the wire honest and lets mock/test
   * feeds pump fast without firing dozens of concurrent, out-of-order POSTs.
   */
  private scoreChain: Promise<void> = Promise.resolve()
  private queuedFrames = 0

  /** Rolling ring of the most recent finished 1 s frames (for tap-to-log). */
  private ring: Uint8Array[] = []

  /** Until this wall-clock time, live cards are suppressed (LOGGED hold). */
  private suppressUntil = 0

  private lastEngineState: string | null = null

  constructor(feed: AudioFeed, opts: PipelineOpts) {
    this.feed = feed
    this.client = opts.client
    this.display = opts.display
    this.mode = opts.mode
    this.tag = opts.tag
    this.rpm = opts.rpm ?? null
    this.now = opts.now ?? (() => Date.now())
    this.onStateChange = opts.onStateChange
    this.onFrameScored = opts.onFrameScored
    this.ringFrames = Math.max(1, opts.ringSeconds ?? DEFAULT_RING_SECONDS)

    this.throttle = new HudThrottle(2000, this.now)
    this.normalizer = new AudioNormalizer({
      now: this.now,
      onFrame: (frame) => {
        // The feed delivers frames synchronously; scoring is async. Append each
        // frame to the serial chain so they are scored strictly in order.
        this.queuedFrames += 1
        this.scoreChain = this.scoreChain
          .then(() => this.onFrameAsync(frame))
          .catch((err) => {
            console.error('frame scoring chain error', err)
          })
          .finally(() => {
            this.queuedFrames -= 1
          })
      },
    })
  }

  /**
   * Open a session and start pumping mic audio. If the engine is unreachable we
   * show the OFFLINE card and retry every 5 s; we never throw and never go
   * silent. Once a session opens, the feed is started exactly once.
   */
  async start(): Promise<void> {
    await this.ensureSession()
    if (!this.started) {
      this.started = true
      this.feed.start((bytes) => this.onChunk(bytes))
    }
  }

  stop(): void {
    this.feed.stop()
    this.started = false
  }

  /** Forward end-of-stream to the normalizer (mock/test EOF). */
  flushAudio(): void {
    this.normalizer.flush()
  }

  /**
   * Flush any card the throttle is holding (its latest-wins `pending`). In the
   * live app a new frame every second naturally ticks the throttle, but at
   * end-of-stream (mock/test) the final pending card would otherwise be
   * stranded. Honors the throttle's interval, so the caller's clock must have
   * advanced past it (mock/test drive a virtual clock to make this deterministic).
   */
  flushDisplay(): void {
    if (this.now() >= this.suppressUntil) {
      this.throttle.tick((card) => this.display.render(card))
    }
  }

  /**
   * Resolve once all queued frame scoring has settled. New frames can be queued
   * while we wait (the chain extends), so we loop until the queue is empty.
   * Mock/test only — the live app never needs to wait.
   */
  async drain(): Promise<void> {
    while (this.queuedFrames > 0) {
      await this.scoreChain
    }
  }

  /**
   * Capture the rolling ring buffer to disk via /label, then hold a LOGGED card
   * for 2 s before live cards resume. Never throws.
   */
  async logTap(): Promise<void> {
    const pcm = this.concatRing()
    if (pcm.length > 0) {
      try {
        await this.client.label({
          session_id: this.sessionId,
          machine_type: this.tag,
          note: 'tap-to-log from glasses',
          pcm,
        })
      } catch (err) {
        // Logging is best-effort; a failed write must not crash the lens.
        console.error('logTap: /label failed', err)
      }
    }
    // Suppress live cards and show LOGGED regardless of label success — the tap
    // gesture should always acknowledge on the lens immediately (bypass the
    // throttle: a direct user action is not a rate-limited engine update).
    this.suppressUntil = this.now() + LOGGED_HOLD_MS
    this.notifyState('LOGGED')
    this.display.render(formatHudCard({ state: 'LOGGED' }))
  }

  // ----------------------------------------------------------- internals ----

  private onChunk(bytes: Uint8Array): void {
    this.normalizer.push(bytes)
  }

  private async ensureSession(): Promise<void> {
    if (this.sessionId !== null || this.starting) return
    this.starting = true
    try {
      const res = await this.client.startSession({
        mode: this.mode,
        tag: this.tag,
        rpm: this.rpm,
      })
      if (res.kind === 'ok') {
        this.sessionId = res.value.session_id
        // Surface the engine's opening state immediately (capture countdown or
        // listening) so the lens isn't blank while the first frame buffers.
        if (res.value.state === 'CAPTURING_BASELINE') {
          this.renderEngine({
            state: 'CAPTURING_BASELINE',
            captureRemainingS: res.value.capture_remaining_s ?? null,
          })
        } else {
          this.renderEngine({ state: 'LISTENING' })
        }
      } else {
        // Offline or http error opening the session: show OFFLINE, retry later.
        this.renderState('OFFLINE')
        this.scheduleSessionRetry()
      }
    } finally {
      this.starting = false
    }
  }

  private scheduleSessionRetry(): void {
    setTimeout(() => {
      if (this.sessionId === null) void this.ensureSession()
    }, OFFLINE_RETRY_MS)
  }

  private async onFrameAsync(frame: Uint8Array): Promise<void> {
    try {
      await this.scoreFrame(frame)
    } finally {
      // Per-frame hook (mock/accelerated runs advance their virtual clock here).
      this.onFrameScored?.()
    }
  }

  private async scoreFrame(frame: Uint8Array): Promise<void> {
    this.pushRing(frame)
    // The 1 s frame cadence is our throttle tick. Skip it while a LOGGED card is
    // held so a stale pending card can't flush over the tap acknowledgment.
    if (this.now() >= this.suppressUntil) {
      this.throttle.tick((card) => this.display.render(card))
    }

    if (this.sessionId === null) {
      // No session yet (still retrying). Keep the ring filling; the OFFLINE card
      // is already shown by ensureSession.
      await this.ensureSession()
      return
    }

    let res
    try {
      res = await this.client.score(this.sessionId, frame)
    } catch (err) {
      console.error('score: unexpected throw', err)
      this.renderState('OFFLINE')
      return
    }

    if (res.kind === 'offline') {
      this.renderState('OFFLINE')
      return
    }

    if (res.kind === 'http_error') {
      if (res.status === 404) {
        // Session was dropped (engine restart / eviction). Re-open and retry the
        // frame against the fresh session.
        this.sessionId = null
        await this.ensureSession()
        if (this.sessionId !== null) {
          const retry = await this.client.score(this.sessionId, frame)
          if (retry.kind === 'ok') {
            this.renderEngine(this.toHudInput(retry.value))
            return
          }
        }
      }
      this.renderState('OFFLINE')
      return
    }

    this.renderEngine(this.toHudInput(res.value))
  }

  private toHudInput(value: {
    state: string
    percentile: number | null
    evidence_line: string | null
    capture_remaining_s?: number
  }): HudInput {
    return {
      state: engineStateToHud(value.state),
      percentile: value.percentile,
      evidenceLine: value.evidence_line,
      captureRemainingS: value.capture_remaining_s ?? null,
    }
  }

  /** Push a frame onto the rolling tap-to-log ring, dropping the oldest. */
  private pushRing(frame: Uint8Array): void {
    this.ring = [...this.ring, frame].slice(-this.ringFrames)
  }

  private concatRing(): Uint8Array {
    const total = this.ring.reduce((n, f) => n + f.length, 0)
    const out = new Uint8Array(total)
    let off = 0
    for (const f of this.ring) {
      out.set(f, off)
      off += f.length
    }
    return out
  }

  /** Render from a full engine input, respecting the LOGGED suppression hold. */
  private renderEngine(input: HudInput): void {
    this.notifyState(input.state)
    if (this.now() < this.suppressUntil) return
    this.throttle.push(formatHudCard(input), (card) => this.display.render(card))
  }

  /** Render a bare state card (OFFLINE, LOGGED) bypassing the suppression gate. */
  private renderState(state: HudState): void {
    this.notifyState(state)
    this.throttle.push(formatHudCard({ state }), (card) =>
      this.display.render(card),
    )
  }

  private notifyState(state: string): void {
    if (state !== this.lastEngineState) {
      this.lastEngineState = state
      this.onStateChange?.(state)
    }
  }
}

export { FRAME_BYTES }
