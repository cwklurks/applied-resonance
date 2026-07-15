/**
 * Headless mock mode: drive the *identical* Pipeline from a WAV file instead of
 * the glasses bridge. Used by the integration test (real engine, real MIMII
 * audio) and by the browser build via `?mock=<url>`.
 *
 * The WAV parser is intentionally minimal: it understands PCM-16 RIFF/WAVE,
 * including WAVE_FORMAT_EXTENSIBLE (format tag 0xFFFE, which the MIMII files
 * use), skips unknown chunks (fact, LIST, etc.), and takes channel 0 of a
 * multichannel file — matching what the engine's loader does.
 */

import { Pipeline, type AudioFeed, type DisplayPort, type PipelineOpts } from './pipeline'

export interface ParsedWav {
  /** Channel-0 s16le bytes (mono), ready to feed the normalizer. */
  bytes: Uint8Array
  sampleRate: number
  channels: number
}

/** RIFF chunk id as a 4-char ASCII string. */
function chunkId(view: DataView, off: number): string {
  return String.fromCharCode(
    view.getUint8(off),
    view.getUint8(off + 1),
    view.getUint8(off + 2),
    view.getUint8(off + 3),
  )
}

/**
 * Parse a 16-bit PCM WAV into channel-0 s16le bytes. Throws on a non-RIFF/WAVE
 * container or a non-16-bit sample width (we only support what the engine eats).
 */
export function parseWav(input: ArrayBuffer | Uint8Array): ParsedWav {
  const u8 = input instanceof Uint8Array ? input : new Uint8Array(input)
  const view = new DataView(u8.buffer, u8.byteOffset, u8.byteLength)

  if (u8.byteLength < 12 || chunkId(view, 0) !== 'RIFF' || chunkId(view, 8) !== 'WAVE') {
    throw new Error('parseWav: not a RIFF/WAVE file')
  }

  let channels = 0
  let sampleRate = 0
  let bitsPerSample = 0
  let dataOffset = -1
  let dataLength = 0

  // Walk the chunk list starting after the 12-byte RIFF/WAVE header.
  let off = 12
  while (off + 8 <= u8.byteLength) {
    const id = chunkId(view, off)
    const size = view.getUint32(off + 4, true)
    const body = off + 8

    if (id === 'fmt ') {
      // PCM (1) and EXTENSIBLE (0xFFFE) share the first 16 bytes of the fmt body.
      channels = view.getUint16(body + 2, true)
      sampleRate = view.getUint32(body + 4, true)
      bitsPerSample = view.getUint16(body + 14, true)
    } else if (id === 'data') {
      dataOffset = body
      // Clamp to the actual buffer in case the header over-reports.
      dataLength = Math.min(size, u8.byteLength - body)
    }

    // Chunks are word-aligned: an odd size is padded with one byte.
    off = body + size + (size & 1)
  }

  if (bitsPerSample !== 16) {
    throw new Error(`parseWav: only 16-bit PCM supported, got ${bitsPerSample}-bit`)
  }
  if (channels < 1) throw new Error('parseWav: invalid channel count')
  if (dataOffset < 0) throw new Error('parseWav: no data chunk')

  const bytesPerSample = 2
  const frameBytes = bytesPerSample * channels
  const frameCount = Math.floor(dataLength / frameBytes)

  // Extract channel 0 only (the engine's loader does the same for multichannel).
  const mono = new Uint8Array(frameCount * bytesPerSample)
  for (let f = 0; f < frameCount; f++) {
    const src = dataOffset + f * frameBytes // channel 0 is the first sample
    mono[f * 2] = u8[src]
    mono[f * 2 + 1] = u8[src + 1]
  }

  return { bytes: mono, sampleRate, channels }
}

/**
 * Emits one or more blocks of s16le bytes in randomized 0.25–1.5 s chunks at
 * accelerated (non-real-time) pace, then completes. Enqueue all clips before
 * `start`; `whenDone()` resolves once the whole queue has drained.
 */
export class WavFileFeed implements AudioFeed {
  private readonly queue: Uint8Array[] = []
  private readonly rng: () => number
  private readonly sampleRate: number
  private onChunk: ((bytes: Uint8Array) => void) | null = null
  private stopped = false
  private donePromise: Promise<void>
  private resolveDone!: () => void

  constructor(opts: { sampleRate?: number; seed?: number } = {}) {
    this.sampleRate = opts.sampleRate ?? 16000
    this.rng = mulberry32(opts.seed ?? 0x1234abcd)
    this.donePromise = new Promise<void>((r) => {
      this.resolveDone = r
    })
  }

  /** Append a clip's channel-0 bytes to the playback queue. */
  enqueue(bytes: Uint8Array): void {
    this.queue.push(bytes)
  }

  /** Resolves once every enqueued clip has been emitted (or stop() was called). */
  whenDone(): Promise<void> {
    return this.donePromise
  }

  start(onChunk: (bytes: Uint8Array) => void): void {
    this.onChunk = onChunk
    void this.pump()
  }

  stop(): void {
    this.stopped = true
    this.resolveDone()
  }

  private async pump(): Promise<void> {
    const bytesPerSample = 2
    for (const clip of this.queue) {
      let pos = 0
      while (pos < clip.length && !this.stopped) {
        // 0.25–1.5 s of audio, snapped to whole samples.
        const seconds = 0.25 + this.rng() * 1.25
        const chunkSamples = Math.max(1, Math.round(seconds * this.sampleRate))
        const chunkBytes = chunkSamples * bytesPerSample
        const end = Math.min(pos + chunkBytes, clip.length)
        this.onChunk?.(clip.subarray(pos, end))
        pos = end
        // Yield to the event loop without real-time pacing.
        await new Promise<void>((r) => setTimeout(r, 0))
      }
      if (this.stopped) break
    }
    this.resolveDone()
  }
}

/** A DisplayPort that keeps every rendered card for later assertions. */
export class RecordingDisplay implements DisplayPort {
  readonly cards: { line1: string; line2: string }[] = []
  render(card: { line1: string; line2: string }): void {
    this.cards.push({ line1: card.line1, line2: card.line2 })
  }
}

export interface MockRunResult {
  display: RecordingDisplay
  pipeline: Pipeline
  feed: WavFileFeed
}

/**
 * Tests/browser convenience: parse a WAV buffer, run the whole clip through the
 * pipeline, then flush at EOF. Returns the recording display and pipeline (so a
 * caller can still call `logTap()` afterwards).
 */
export async function runMockFromWav(
  wav: ArrayBuffer | Uint8Array,
  opts: Omit<PipelineOpts, 'display'> & { display?: DisplayPort; seed?: number },
): Promise<MockRunResult> {
  const parsed = parseWav(wav)
  const display = opts.display ?? new RecordingDisplay()
  const feed = new WavFileFeed({ sampleRate: parsed.sampleRate, seed: opts.seed })
  const pipeline = new Pipeline(feed, {
    maxPendingFrames: 64,
    maxFrameAgeMs: 120_000,
    ...opts,
    display,
  })
  feed.enqueue(parsed.bytes)

  await pipeline.start()
  await feed.whenDone()
  await pipeline.drain()
  pipeline.flushAudio()
  await pipeline.drain()
  return { display: display as RecordingDisplay, pipeline, feed }
}

// ── Deterministic seeded PRNG (mulberry32) ─────────────────────────────────
function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
