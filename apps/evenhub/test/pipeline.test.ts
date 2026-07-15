import { describe, it, expect } from 'vitest'
import {
  Pipeline,
  type AudioFeed,
  type CaptureEvent,
  type DisplayPort,
} from '../src/pipeline'
import type { HudCard } from '@earsight/display-card'
import type {
  EngineClient,
  EngineResult,
  ScoreResponse,
  StartResponse,
  LabelResponse,
  HealthResponse,
  CaptureStartResponse,
  CaptureAppendResponse,
  CaptureStopResponse,
  StartRequest,
  LabelRequest,
} from '@earsight/display-card'

// ── Test doubles ───────────────────────────────────────────────────────────

const FRAME_BYTES = 32000 // 1 s @ 16 kHz s16le

/** A feed we drive by hand: `emit(bytes)` pushes a chunk into the pipeline. */
class ManualFeed implements AudioFeed {
  private cb: ((bytes: Uint8Array) => void) | null = null
  stopped = false
  start(onChunk: (bytes: Uint8Array) => void): void {
    this.cb = onChunk
  }
  stop(): void {
    this.stopped = true
  }
  emit(bytes: Uint8Array): void {
    this.cb?.(bytes)
  }
}

/** Records every card the pipeline renders, in order. */
class RecordingDisplay implements DisplayPort {
  readonly cards: HudCard[] = []
  render(card: HudCard): void {
    this.cards.push({ ...card })
  }
  get last(): HudCard | undefined {
    return this.cards[this.cards.length - 1]
  }
}

type ScoreResult = EngineResult<ScoreResponse>
type CaptureAppendCall = { captureId: string; pcm: Uint8Array }
type CaptureAppendImpl = (
  captureId: string,
  pcm: Uint8Array,
) => Promise<EngineResult<CaptureAppendResponse>>
type ScoreImpl = (
  sessionId: string,
  pcm: Uint8Array,
) => Promise<ScoreResult>

/**
 * Scripted EngineClient: `startSession` returns a queue of start results (or a
 * default), `score` pulls the next scripted response per call, `label` records
 * the posted PCM. Duck-typed to EngineClient via a cast at the call site.
 */
class FakeEngineClient {
  startQueue: EngineResult<StartResponse>[] = []
  scoreScript: ScoreResult[] = []
  labelCalls: (LabelRequest & { pcm: Uint8Array })[] = []
  startCalls: StartRequest[] = []
  captureStartQueue: EngineResult<CaptureStartResponse>[] = []
  captureAppendResults: EngineResult<CaptureAppendResponse>[] = []
  captureStopQueue: EngineResult<CaptureStopResponse>[] = []
  captureStartCalls: string[] = []
  captureAppendCalls: CaptureAppendCall[] = []
  captureStopCalls: string[] = []
  captureAppendImpl: CaptureAppendImpl | null = null
  scoreImpl: ScoreImpl | null = null
  scoreCalls: { sessionId: string; pcm: Uint8Array }[] = []
  healthResult: EngineResult<HealthResponse> = {
    kind: 'ok',
    value: { status: 'ok' },
  }
  private startCount = 0
  private captureStartCount = 0
  private captureBytes = 0

  async startSession(req: StartRequest): Promise<EngineResult<StartResponse>> {
    this.startCalls.push(req)
    const queued = this.startQueue[this.startCount]
    this.startCount += 1
    return (
      queued ?? {
        kind: 'ok',
        value: { session_id: `s${this.startCount}`, state: 'CAPTURING_BASELINE', capture_remaining_s: 30 },
      }
    )
  }

  async score(sessionId: string, pcm: Uint8Array): Promise<ScoreResult> {
    this.scoreCalls.push({ sessionId, pcm: new Uint8Array(pcm) })
    if (this.scoreImpl) return this.scoreImpl(sessionId, pcm)
    const next = this.scoreScript.shift()
    return next ?? { kind: 'ok', value: listening() }
  }

  async label(req: LabelRequest & { pcm: Uint8Array }): Promise<EngineResult<LabelResponse>> {
    this.labelCalls.push(req)
    return { kind: 'ok', value: { wav_path: 'datakit/data/x.wav', json_path: 'datakit/data/x.json' } }
  }

  async captureStart(tag: string): Promise<EngineResult<CaptureStartResponse>> {
    this.captureStartCalls.push(tag)
    const queued = this.captureStartQueue[this.captureStartCount]
    this.captureStartCount += 1
    return queued ?? { kind: 'ok', value: { capture_id: `cap-${this.captureStartCount}` } }
  }

  async captureAppend(
    captureId: string,
    pcm: Uint8Array,
  ): Promise<EngineResult<CaptureAppendResponse>> {
    if (this.captureAppendImpl) return this.captureAppendImpl(captureId, pcm)

    this.captureAppendCalls.push({ captureId, pcm: new Uint8Array(pcm) })
    this.captureBytes += pcm.length
    const next = this.captureAppendResults.shift()
    return (
      next ?? {
        kind: 'ok',
        value: {
          bytes_total: this.captureBytes,
          seconds_total: this.captureBytes / FRAME_BYTES,
        },
      }
    )
  }

  async captureStop(captureId: string): Promise<EngineResult<CaptureStopResponse>> {
    this.captureStopCalls.push(captureId)
    const next = this.captureStopQueue.shift()
    return (
      next ?? {
        kind: 'ok',
        value: { wav_path: `captures/${captureId}.wav`, duration_s: this.captureBytes / FRAME_BYTES },
      }
    )
  }

  async health(): Promise<EngineResult<HealthResponse>> {
    return this.healthResult
  }
}

function asClient(fake: FakeEngineClient): EngineClient {
  return fake as unknown as EngineClient
}

// ── Response builders ───────────────────────────────────────────────────────

function capturing(remaining: number): ScoreResponse {
  return { state: 'CAPTURING_BASELINE', score: null, percentile: null, evidence_line: null, capture_remaining_s: remaining }
}
function listening(): ScoreResponse {
  return { state: 'LISTENING', score: null, percentile: null, evidence_line: null }
}
function suspect(percentile: number, evidence: string): ScoreResponse {
  return { state: 'SUSPECT', score: 0.5, percentile, evidence_line: evidence }
}
function alert(percentile: number, evidence: string): ScoreResponse {
  return { state: 'ALERT', score: 0.9, percentile, evidence_line: evidence }
}

// ── Clock + frame helpers ────────────────────────────────────────────────────

/** A fake monotonic clock advanced explicitly by the test. */
class FakeClock {
  t = 0
  now = (): number => this.t
  advance(ms: number): void {
    this.t += ms
  }
}

function frame(fill = 1): Uint8Array {
  // Non-zero fill so the bytes survive any silence trimming downstream.
  return new Uint8Array(FRAME_BYTES).fill(fill)
}

const PRIME_BYTES = 16000 * 2 * 10 // 10 s = the normalizer's pre-decision cap

/**
 * Force the AudioNormalizer past its rate-decision window in one shot, so that
 * every subsequent single frame emits immediately. Feeding exactly the 10 s
 * pre-decision cap trips `decideAndDrain` without a clock-based estimate, so the
 * working rate stays 16 kHz (no resample). The 10 frames this drains are scored
 * against the default (LISTENING) — call BEFORE arming the scoreScript so the
 * primer never consumes scripted responses.
 */
async function prime(feed: ManualFeed, pipeline: Pipeline): Promise<void> {
  feed.emit(new Uint8Array(PRIME_BYTES).fill(1))
  await pipeline.drain()
}

/** Feed exactly one 1 s frame and let its scoring settle. */
async function step(feed: ManualFeed, pipeline: Pipeline, fill = 1): Promise<void> {
  feed.emit(frame(fill))
  await pipeline.drain()
}

async function settleAsyncWork(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0))
}

function deferred(): { promise: Promise<void>; resolve: () => void } {
  let resolve!: () => void
  const promise = new Promise<void>((r) => {
    resolve = r
  })
  return { promise, resolve }
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe('Pipeline', () => {
  it('shows capture countdown cards then LISTENING', async () => {
    const clock = new FakeClock()
    const fake = new FakeEngineClient()
    // Engine opens in capture; subsequent scores count down then flip to listening.
    fake.startQueue = [
      { kind: 'ok', value: { session_id: 's1', state: 'CAPTURING_BASELINE', capture_remaining_s: 30 } },
    ]
    const feed = new ManualFeed()
    const display = new RecordingDisplay()
    const pipeline = new Pipeline(feed, { client: asClient(fake), display, mode: 'session', tag: 't', now: clock.now })

    await pipeline.start()
    // Opening card from startSession is the capture countdown.
    expect(display.cards[0].line1).toContain('Capturing baseline')

    await prime(feed, pipeline) // decide rate; drained frames score as default LISTENING

    // Each scored frame must clear the 2 s throttle window to actually render.
    fake.scoreScript = [{ kind: 'ok', value: capturing(20) }]
    clock.advance(2001)
    await step(feed, pipeline)

    fake.scoreScript = [{ kind: 'ok', value: capturing(10) }]
    clock.advance(2001)
    await step(feed, pipeline)

    fake.scoreScript = [{ kind: 'ok', value: listening() }]
    clock.advance(2001)
    await step(feed, pipeline)

    const lines = display.cards.map((c) => c.line1)
    expect(lines.some((l) => l.includes('Capturing baseline'))).toBe(true)
    expect(lines.some((l) => l.startsWith('Listening'))).toBe(true)
    // Listening must come after the first capturing card.
    const firstCapture = lines.findIndex((l) => l.includes('Capturing baseline'))
    const firstListen = lines.findIndex((l) => l.startsWith('Listening'))
    expect(firstListen).toBeGreaterThan(firstCapture)
  })

  it('maps SUSPECT and ALERT with percentile in line1 and evidence + tap-to-log in line2', async () => {
    const clock = new FakeClock()
    const fake = new FakeEngineClient()
    fake.startQueue = [{ kind: 'ok', value: { session_id: 's1', state: 'LISTENING' } }]
    const feed = new ManualFeed()
    const display = new RecordingDisplay()
    const pipeline = new Pipeline(feed, { client: asClient(fake), display, mode: 'saved', tag: 'pump', now: clock.now })

    await pipeline.start()
    await prime(feed, pipeline) // primer scores as default LISTENING; arm script after

    fake.scoreScript = [{ kind: 'ok', value: suspect(88, 'tonal 120Hz') }]
    clock.advance(2001)
    await step(feed, pipeline)
    const suspectCard = display.last!
    expect(suspectCard.line1.startsWith('?')).toBe(true)
    expect(suspectCard.line1).toContain('88%')
    expect(suspectCard.line2).toContain('tonal 120Hz')
    expect(suspectCard.line2).toContain('tap to log')

    fake.scoreScript = [{ kind: 'ok', value: alert(97, 'bearing 3.2x') }]
    clock.advance(2001) // clear throttle interval
    await step(feed, pipeline)
    const alertCard = display.last!
    expect(alertCard.line1.startsWith('!')).toBe(true)
    expect(alertCard.line1).toContain('97%')
    expect(alertCard.line2).toContain('bearing 3.2x')
    expect(alertCard.line2).toContain('tap to log')
  })

  it('shows OFFLINE on {kind:"offline"} and recovers automatically', async () => {
    const clock = new FakeClock()
    const fake = new FakeEngineClient()
    fake.startQueue = [{ kind: 'ok', value: { session_id: 's1', state: 'LISTENING' } }]
    const feed = new ManualFeed()
    const display = new RecordingDisplay()
    const pipeline = new Pipeline(feed, { client: asClient(fake), display, mode: 'saved', tag: 'pump', now: clock.now })

    await pipeline.start()
    await prime(feed, pipeline)

    fake.scoreScript = [{ kind: 'offline' }]
    clock.advance(2001)
    await step(feed, pipeline)
    expect(display.last!.line1).toBe('engine offline')

    fake.scoreScript = [{ kind: 'ok', value: alert(95, 'bearing fault') }]
    clock.advance(2001)
    await step(feed, pipeline) // recovers to ALERT, feed never stopped
    expect(display.last!.line1.startsWith('!')).toBe(true)
  })

  it('re-opens the session on a 404 and re-scores the frame', async () => {
    const clock = new FakeClock()
    const fake = new FakeEngineClient()
    fake.startQueue = [
      { kind: 'ok', value: { session_id: 's1', state: 'LISTENING' } },
      { kind: 'ok', value: { session_id: 's2', state: 'LISTENING' } },
    ]
    const feed = new ManualFeed()
    const display = new RecordingDisplay()
    const pipeline = new Pipeline(feed, { client: asClient(fake), display, mode: 'session', tag: 't', now: clock.now })

    await pipeline.start()
    expect(fake.startCalls.length).toBe(1)
    await prime(feed, pipeline)

    // Next scored frame 404s, forcing a re-open, then the re-score succeeds.
    fake.scoreScript = [
      { kind: 'http_error', status: 404, detail: 'unknown session_id' },
      { kind: 'ok', value: alert(91, 'recovered') },
    ]
    clock.advance(2001)
    await step(feed, pipeline)

    // A second session was opened after the 404, and the re-score landed.
    expect(fake.startCalls.length).toBe(2)
    expect(display.last!.line1.startsWith('!')).toBe(true)
  })

  it('logTap posts exactly the last ringSeconds frames and holds LOGGED for 2 s', async () => {
    const clock = new FakeClock()
    const fake = new FakeEngineClient()
    fake.startQueue = [{ kind: 'ok', value: { session_id: 's1', state: 'LISTENING' } }]
    const feed = new ManualFeed()
    const display = new RecordingDisplay()
    const pipeline = new Pipeline(feed, {
      client: asClient(fake),
      display,
      mode: 'saved',
      tag: 'pump',
      now: clock.now,
      ringSeconds: 10,
    })

    await pipeline.start()
    await prime(feed, pipeline) // 10 frames buffered/drained → ring is full at 10
    // Push 15 more single frames so the ring overflows well past 10 s.
    for (let i = 0; i < 15; i++) {
      clock.advance(1000)
      await step(feed, pipeline)
    }

    clock.advance(1000)
    await pipeline.logTap()
    expect(fake.labelCalls.length).toBe(1)
    // Ring holds exactly the last 10 frames → 10 × 32000 bytes.
    expect(fake.labelCalls[0].pcm.length).toBe(10 * FRAME_BYTES)
    expect(display.last!.line1).toContain('logged')

    // Within the 2 s hold, a live SUSPECT card must NOT overwrite LOGGED.
    fake.scoreScript = [{ kind: 'ok', value: suspect(80, 'x') }]
    clock.advance(500)
    await step(feed, pipeline)
    expect(display.last!.line1).toContain('logged')

    // After the hold elapses, live cards resume.
    fake.scoreScript = [{ kind: 'ok', value: alert(99, 'y') }]
    clock.advance(2001)
    await step(feed, pipeline)
    expect(display.last!.line1.startsWith('!')).toBe(true)
  })

  it('throttles to at most one render per 2 s across many state changes', async () => {
    const clock = new FakeClock()
    const fake = new FakeEngineClient()
    fake.startQueue = [{ kind: 'ok', value: { session_id: 's1', state: 'LISTENING' } }]
    const feed = new ManualFeed()
    const display = new RecordingDisplay()
    const pipeline = new Pipeline(feed, { client: asClient(fake), display, mode: 'saved', tag: 'pump', now: clock.now })

    await pipeline.start()
    await prime(feed, pipeline)
    clock.advance(2001) // open the throttle window

    // Six distinct cards fed at the SAME instant (no clock advance between them).
    fake.scoreScript = [
      { kind: 'ok', value: suspect(70, 'a') },
      { kind: 'ok', value: suspect(72, 'b') },
      { kind: 'ok', value: alert(95, 'c') },
      { kind: 'ok', value: suspect(73, 'd') },
      { kind: 'ok', value: alert(96, 'e') },
      { kind: 'ok', value: alert(97, 'f') },
    ]
    const before = display.cards.length
    for (let i = 0; i < 6; i++) await step(feed, pipeline, i + 1)

    // The throttle permits only the first render within the 2 s window.
    const renderedDuringWindow = display.cards.length - before
    expect(renderedDuringWindow).toBeLessThanOrEqual(1)
  })

  it('bounds a stalled scoring queue to the active frame plus the latest pending frame', async () => {
    const fake = new FakeEngineClient()
    fake.startQueue = [{ kind: 'ok', value: { session_id: 's1', state: 'LISTENING' } }]
    const feed = new ManualFeed()
    const pipeline = new Pipeline(feed, {
      client: asClient(fake),
      display: new RecordingDisplay(),
      mode: 'saved',
      tag: 'pump',
      maxPendingFrames: 1,
    })
    await pipeline.start()
    await prime(feed, pipeline)
    fake.scoreCalls = []

    const held = deferred()
    let first = true
    fake.scoreImpl = async () => {
      if (first) {
        first = false
        await held.promise
      }
      return { kind: 'ok', value: listening() }
    }

    feed.emit(frame(1))
    await settleAsyncWork()
    feed.emit(frame(2))
    feed.emit(frame(3))
    feed.emit(frame(4))
    await settleAsyncWork()
    expect(fake.scoreCalls.map((call) => call.pcm[0])).toEqual([1])

    held.resolve()
    await pipeline.drain()
    expect(fake.scoreCalls.map((call) => call.pcm[0])).toEqual([1, 4])
  })

  it('drops a pending score frame once it is stale', async () => {
    const clock = new FakeClock()
    const fake = new FakeEngineClient()
    fake.startQueue = [{ kind: 'ok', value: { session_id: 's1', state: 'LISTENING' } }]
    const feed = new ManualFeed()
    const pipeline = new Pipeline(feed, {
      client: asClient(fake),
      display: new RecordingDisplay(),
      mode: 'saved',
      tag: 'pump',
      now: clock.now,
      maxPendingFrames: 1,
      maxFrameAgeMs: 1_000,
    })
    await pipeline.start()
    await prime(feed, pipeline)
    fake.scoreCalls = []

    const held = deferred()
    let first = true
    fake.scoreImpl = async () => {
      if (first) {
        first = false
        await held.promise
      }
      return { kind: 'ok', value: listening() }
    }
    feed.emit(frame(1))
    await settleAsyncWork()
    feed.emit(frame(2))
    clock.advance(1_001)
    held.resolve()

    await pipeline.drain()
    expect(fake.scoreCalls.map((call) => call.pcm[0])).toEqual([1])
  })

  it('startCapture stores the engine capture id', async () => {
    const clock = new FakeClock()
    const fake = new FakeEngineClient()
    const feed = new ManualFeed()
    const display = new RecordingDisplay()
    const events: string[] = []
    const pipeline = new Pipeline(feed, {
      client: asClient(fake),
      display,
      mode: 'session',
      tag: 't',
      now: clock.now,
      onCaptureEvent: (event) => events.push(event.kind),
    })

    const id = await pipeline.startCapture('g2-characterization')

    expect(id).toBe('cap-1')
    expect(pipeline.capturing).toBe(true)
    expect(pipeline.captureSeconds()).toBe(0)
    expect(fake.captureStartCalls).toEqual(['g2-characterization'])
    expect(events).toEqual(['started'])
  })

  it('tees exact frames while scoring CAPTURING_BASELINE and LISTENING states', async () => {
    const clock = new FakeClock()
    const fake = new FakeEngineClient()
    fake.startQueue = [{ kind: 'ok', value: { session_id: 's1', state: 'LISTENING' } }]
    const feed = new ManualFeed()
    const display = new RecordingDisplay()
    const pipeline = new Pipeline(feed, { client: asClient(fake), display, mode: 'session', tag: 't', now: clock.now })

    await pipeline.start()
    await prime(feed, pipeline)
    await pipeline.startCapture('g2-characterization')

    fake.scoreScript = [
      { kind: 'ok', value: capturing(9) },
      { kind: 'ok', value: listening() },
    ]
    await step(feed, pipeline, 7)
    await step(feed, pipeline, 8)
    const stopped = await pipeline.stopCapture()

    expect(stopped).toEqual({ wavPath: 'captures/cap-1.wav', durationS: 2 })
    expect(fake.captureAppendCalls.map((c) => c.captureId)).toEqual(['cap-1', 'cap-1'])
    expect(fake.captureAppendCalls[0].pcm).toEqual(frame(7))
    expect(fake.captureAppendCalls[1].pcm).toEqual(frame(8))
  })

  it('serializes capture appends without blocking scoring', async () => {
    const clock = new FakeClock()
    const fake = new FakeEngineClient()
    const first = deferred()
    const second = deferred()
    const gates = [first, second]
    fake.captureAppendImpl = async (captureId, pcm) => {
      const index = fake.captureAppendCalls.length
      fake.captureAppendCalls.push({ captureId, pcm: new Uint8Array(pcm) })
      await gates[index].promise
      return {
        kind: 'ok',
        value: { bytes_total: (index + 1) * FRAME_BYTES, seconds_total: index + 1 },
      }
    }
    fake.startQueue = [{ kind: 'ok', value: { session_id: 's1', state: 'LISTENING' } }]
    const feed = new ManualFeed()
    const display = new RecordingDisplay()
    const pipeline = new Pipeline(feed, { client: asClient(fake), display, mode: 'session', tag: 't', now: clock.now })

    await pipeline.start()
    await prime(feed, pipeline)
    await pipeline.startCapture('g2-characterization')

    fake.scoreScript = [{ kind: 'ok', value: alert(92, 'append held') }]
    clock.advance(2001)
    await step(feed, pipeline, 3)
    expect(display.last!.line1.startsWith('!')).toBe(true)
    expect(fake.captureAppendCalls.length).toBe(1)

    fake.scoreScript = [{ kind: 'ok', value: listening() }]
    await step(feed, pipeline, 4)
    await settleAsyncWork()
    expect(fake.captureAppendCalls.length).toBe(1)

    first.resolve()
    await settleAsyncWork()
    expect(fake.captureAppendCalls.length).toBe(2)
    expect(fake.captureAppendCalls.map((c) => c.pcm[0])).toEqual([3, 4])

    second.resolve()
    const stopped = await pipeline.stopCapture()
    expect(stopped).toEqual({ wavPath: 'captures/cap-1.wav', durationS: 0 })
  })

  it('finalizes a partial capture after an overflow once the active append settles', async () => {
    const fake = new FakeEngineClient()
    const held = deferred()
    fake.captureAppendImpl = async (captureId, pcm) => {
      fake.captureAppendCalls.push({ captureId, pcm: new Uint8Array(pcm) })
      await held.promise
      return { kind: 'offline' }
    }
    fake.startQueue = [{ kind: 'ok', value: { session_id: 's1', state: 'LISTENING' } }]
    const events: CaptureEvent[] = []
    const feed = new ManualFeed()
    const pipeline = new Pipeline(feed, {
      client: asClient(fake),
      display: new RecordingDisplay(),
      mode: 'session',
      tag: 'pump',
      maxCapturePendingFrames: 1,
      onCaptureEvent: (event) => events.push(event),
    })

    await pipeline.start()
    await prime(feed, pipeline)
    await pipeline.startCapture('bounded')
    await step(feed, pipeline, 1)
    await settleAsyncWork()
    await step(feed, pipeline, 2)
    await step(feed, pipeline, 3)

    expect(pipeline.capturing).toBe(false)
    expect(events).toContainEqual({
      kind: 'warning',
      message: 'raw capture stopping early: append queue limit reached',
    })
    expect(fake.captureAppendCalls.map((call) => call.pcm[0])).toEqual([1])
    expect(fake.captureStopCalls).toEqual([])

    held.resolve()
    await settleAsyncWork()

    expect(fake.captureStopCalls).toEqual(['cap-1'])
    expect(events).toContainEqual({
      kind: 'warning',
      message: 'raw capture finalized partial recording: captures/cap-1.wav',
    })
    expect(events).toContainEqual({
      kind: 'stopped',
      wavPath: 'captures/cap-1.wav',
      durationS: 0,
    })
    expect(await pipeline.startCapture('after-overflow')).toBe('cap-2')

    pipeline.stop()
    await settleAsyncWork()
    expect(fake.captureStopCalls).toEqual(['cap-1'])
  })

  it('stopCapture returns the wav path and halts future appends', async () => {
    const clock = new FakeClock()
    const fake = new FakeEngineClient()
    const feed = new ManualFeed()
    const display = new RecordingDisplay()
    const pipeline = new Pipeline(feed, { client: asClient(fake), display, mode: 'session', tag: 't', now: clock.now })

    await pipeline.start()
    await prime(feed, pipeline)
    await pipeline.startCapture('g2-characterization')
    await step(feed, pipeline, 5)

    const stopped = await pipeline.stopCapture()
    expect(stopped).toEqual({ wavPath: 'captures/cap-1.wav', durationS: 1 })
    expect(pipeline.capturing).toBe(false)

    await step(feed, pipeline, 6)
    expect(fake.captureAppendCalls.length).toBe(1)
    expect(fake.captureStopCalls).toEqual(['cap-1'])
  })

  it('falls back to server idle cleanup when append and cleanup are offline', async () => {
    const clock = new FakeClock()
    const fake = new FakeEngineClient()
    fake.captureAppendResults = [{ kind: 'offline' }]
    fake.captureStopQueue = [{ kind: 'offline' }]
    const warnings: string[] = []
    const feed = new ManualFeed()
    const display = new RecordingDisplay()
    const pipeline = new Pipeline(feed, {
      client: asClient(fake),
      display,
      mode: 'session',
      tag: 't',
      now: clock.now,
      onCaptureEvent: (event) => {
        if (event.kind === 'warning') warnings.push(event.message)
      },
    })

    await pipeline.start()
    await prime(feed, pipeline)
    await pipeline.startCapture('g2-characterization')
    await step(feed, pipeline, 9)
    await settleAsyncWork()
    expect(pipeline.capturing).toBe(false)

    await step(feed, pipeline, 10)
    await settleAsyncWork()

    expect(fake.captureAppendCalls.length).toBe(1)
    expect(warnings).toEqual([
      'raw capture append failed: engine offline',
      'raw capture cleanup failed: engine offline; server idle cleanup pending',
    ])
    expect(fake.captureStopCalls).toEqual(['cap-1'])

    pipeline.stop()
    await settleAsyncWork()
    expect(fake.captureStopCalls).toEqual(['cap-1'])
  })

  it('keeps scoring flow unaffected while raw capture is active', async () => {
    const clock = new FakeClock()
    const fake = new FakeEngineClient()
    const heldAppend = deferred()
    fake.captureAppendImpl = async (captureId, pcm) => {
      fake.captureAppendCalls.push({ captureId, pcm: new Uint8Array(pcm) })
      await heldAppend.promise
      return { kind: 'ok', value: { bytes_total: FRAME_BYTES, seconds_total: 1 } }
    }
    fake.startQueue = [{ kind: 'ok', value: { session_id: 's1', state: 'LISTENING' } }]
    const feed = new ManualFeed()
    const display = new RecordingDisplay()
    const pipeline = new Pipeline(feed, { client: asClient(fake), display, mode: 'session', tag: 't', now: clock.now })

    await pipeline.start()
    await prime(feed, pipeline)
    await pipeline.startCapture('g2-characterization')

    fake.scoreScript = [{ kind: 'ok', value: alert(99, 'scoring continues') }]
    clock.advance(2001)
    await step(feed, pipeline, 11)

    expect(fake.captureAppendCalls.length).toBe(1)
    expect(display.last!.line1.startsWith('!')).toBe(true)
    expect(display.last!.line2).toContain('scoring continues')

    heldAppend.resolve()
    const stopped = await pipeline.stopCapture()
    expect(stopped).toEqual({ wavPath: 'captures/cap-1.wav', durationS: 0 })
  })
})
