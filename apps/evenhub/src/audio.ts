// Defensive audio layer for the glasses microphone stream.
//
// The SDK delivers mic PCM as Uint8Array chunks of s16le bytes via
// `event.audioEvent.audioPcm`. The *documented* format is 16 kHz mono, but
// that claim comes from the template, not the SDK, and chunk sizes are
// variable and undocumented. The engine downstream wants exactly 16 kHz mono
// int16 LE in 1.0 s frames (16000 samples / 32000 bytes). Everything here
// assumes nothing about the inbound stream and normalizes it to that target.
//
// Browser-safe: no Node imports. Pure functions + two small state machines.

const TARGET_RATE = 16000
const BYTES_PER_SAMPLE = 2
const FRAME_SAMPLES = TARGET_RATE // 1.0 s at 16 kHz (= 32000 bytes s16le)
const INT16_MIN = -32768
const INT16_MAX = 32767

const SUPPORTED_RATES = [8000, 16000, 22050, 24000, 32000, 44100, 48000] as const

// Safety cap on pre-decision buffering. The rate window is normally 3 s; this
// only bites when the supplied clock never advances (frozen/unusable), in which
// case we fall back to the declared rate rather than buffer unboundedly.
const MAX_BUFFER_SEC = 10

let warnedOddByte = false

/**
 * s16le bytes -> Float32Array in [-1, 1).
 * A trailing odd byte (incomplete sample) is dropped, warning once via console.
 */
export function pcmBytesToFloat32(bytes: Uint8Array): Float32Array {
  let sampleCount = bytes.length >> 1
  if (bytes.length & 1) {
    if (!warnedOddByte) {
      warnedOddByte = true
      console.warn('pcmBytesToFloat32: dropped trailing odd byte (incomplete s16le sample).')
    }
  }
  // Read little-endian int16 via DataView; do not assume buffer alignment, and
  // honor the byteOffset/length of the incoming view (it may be a subarray).
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const out = new Float32Array(sampleCount)
  for (let i = 0; i < sampleCount; i++) {
    const s = view.getInt16(i * 2, true)
    // Map symmetrically so the round-trip back to int16 is lossless: divide by
    // 32768 keeps the value in [-1, 1) for negatives and < 1 for max positive.
    out[i] = s / 32768
  }
  return out
}

/** Float32 [-1,1) -> s16le bytes (clamp + round). */
export function float32ToPcmBytes(samples: Float32Array): Uint8Array {
  const out = new Uint8Array(samples.length * BYTES_PER_SAMPLE)
  const view = new DataView(out.buffer)
  for (let i = 0; i < samples.length; i++) {
    // Inverse of /32768. Round to nearest, then clamp into int16 range.
    let v = Math.round(samples[i] * 32768)
    if (v > INT16_MAX) v = INT16_MAX
    else if (v < INT16_MIN) v = INT16_MIN
    view.setInt16(i * 2, v, true)
  }
  return out
}

/** Interleaved multichannel -> mono by channel averaging. channels >= 1. */
export function mixToMono(samples: Float32Array, channels: number): Float32Array {
  if (channels < 1 || !Number.isInteger(channels)) {
    throw new Error(`mixToMono: channels must be an integer >= 1, got ${channels}`)
  }
  if (channels === 1) return samples
  const frames = Math.floor(samples.length / channels)
  const out = new Float32Array(frames)
  for (let f = 0; f < frames; f++) {
    let sum = 0
    const base = f * channels
    for (let c = 0; c < channels; c++) sum += samples[base + c]
    out[f] = sum / channels
  }
  return out
}

/** Linear-interpolation resampler. Returns a new Float32Array at targetRate. */
export function resampleLinear(
  samples: Float32Array,
  sourceRate: number,
  targetRate: number,
): Float32Array {
  if (sourceRate === targetRate) return samples
  if (samples.length === 0) return new Float32Array(0)
  const ratio = targetRate / sourceRate
  const outLen = Math.round(samples.length * ratio)
  const out = new Float32Array(outLen)
  const step = sourceRate / targetRate // float64 source position increment
  let pos = 0
  for (let i = 0; i < outLen; i++) {
    const i0 = Math.floor(pos)
    const frac = pos - i0
    const a = samples[i0] ?? samples[samples.length - 1] ?? 0
    const b = samples[i0 + 1] ?? a
    out[i] = a + (b - a) * frac
    pos += step
  }
  return out
}

/**
 * Estimates the effective inbound sample rate from wall-clock byte throughput.
 * Call feed(byteCount, nowMs) on every chunk. After >= minWindowMs of observed
 * span, estimate() returns bytes/sec ÷ 2 ÷ channels (samples/sec); else null.
 */
export class RateEstimator {
  private readonly channels: number
  private readonly minWindowMs: number
  private firstMs: number | null = null
  private lastMs: number | null = null
  // Bytes that arrived *after* the first chunk; the first chunk only fixes t0.
  private bytesSinceFirst = 0

  constructor(channels = 1, minWindowMs = 3000) {
    if (channels < 1) throw new Error('RateEstimator: channels must be >= 1')
    this.channels = channels
    this.minWindowMs = minWindowMs
  }

  feed(byteCount: number, nowMs: number): void {
    if (this.firstMs === null) {
      this.firstMs = nowMs
      this.lastMs = nowMs
      return
    }
    this.bytesSinceFirst += byteCount
    this.lastMs = nowMs
  }

  estimate(): number | null {
    if (this.firstMs === null || this.lastMs === null) return null
    const spanMs = this.lastMs - this.firstMs
    if (spanMs < this.minWindowMs || spanMs <= 0) return null
    const bytesPerSec = (this.bytesSinceFirst * 1000) / spanMs
    return bytesPerSec / BYTES_PER_SAMPLE / this.channels
  }

  reset(): void {
    this.firstMs = null
    this.lastMs = null
    this.bytesSinceFirst = 0
  }
}

function nearestSupportedRate(rate: number): number {
  let best: number = SUPPORTED_RATES[0]
  let bestDist = Math.abs(rate - best)
  for (const r of SUPPORTED_RATES) {
    const d = Math.abs(rate - r)
    if (d < bestDist) {
      bestDist = d
      best = r
    }
  }
  return best
}

interface AudioNormalizerOpts {
  declaredRate?: number
  channels?: number
  onFrame: (frame: Uint8Array) => void
  onWarning?: (msg: string) => void
  now?: () => number
}

/**
 * Full defensive pipeline. Accepts raw Uint8Array s16le chunks of unknown
 * pacing with a declared format, detects a mismatched effective rate via
 * RateEstimator, switches the working rate to the nearest supported estimate
 * when off by >20%, mixes to mono, resamples to 16 kHz, and emits exact 1.0 s
 * frames (32000 bytes). Carry-over between chunks is sample-exact.
 */
export class AudioNormalizer {
  private readonly declaredRate: number
  private readonly channels: number
  private readonly onFrame: (frame: Uint8Array) => void
  private readonly onWarning?: (msg: string) => void
  private readonly now: () => number

  private workingRate: number
  private readonly estimator: RateEstimator

  // Resampler carry: fractional source position relative to the last sample of
  // the previous chunk's mono buffer, plus the final mono sample for boundary
  // interpolation. Tracked in float64 to avoid drift across chunk boundaries.
  private prevTail = 0 // last mono sample of previous chunk (for interpolation)
  private hasPrevTail = false
  private fracPos = 0 // fractional source position within the next chunk

  // Output accumulator: target-rate mono samples awaiting frame emission.
  private pending: number[] = []

  // A single trailing byte from the previous chunk (s16le samples can straddle
  // chunk boundaries since chunk sizes are arbitrary). Carried so no inbound
  // byte is ever lost mid-stream. null = no pending byte.
  private leftoverByte: number | null = null

  // Until the working rate is decided we cannot know how to resample, so raw
  // (s16le-aligned) bytes are buffered. Once decided, the backlog is drained
  // through the pipeline at the final rate and future chunks process directly.
  private decided = false
  private predecision: Uint8Array[] = []
  private predecisionBytes = 0

  constructor(opts: AudioNormalizerOpts) {
    this.declaredRate = opts.declaredRate ?? TARGET_RATE
    this.channels = opts.channels ?? 1
    if (this.channels < 1 || !Number.isInteger(this.channels)) {
      throw new Error('AudioNormalizer: channels must be an integer >= 1')
    }
    this.onFrame = opts.onFrame
    this.onWarning = opts.onWarning
    this.now = opts.now ?? (() => Date.now())
    this.workingRate = this.declaredRate
    this.estimator = new RateEstimator(this.channels, 3000)
  }

  push(bytes: Uint8Array): void {
    if (bytes.length === 0) return

    // Observe inbound throughput on the RAW byte count for an accurate rate.
    this.estimator.feed(bytes.length, this.now())

    const aligned = this.alignBytes(bytes)

    if (this.decided) {
      this.process(aligned)
      return
    }

    // Pre-decision: buffer the audio while we characterize the stream rate.
    if (aligned.length > 0) {
      this.predecision.push(aligned)
      this.predecisionBytes += aligned.length
    }
    this.tryDecide()
  }

  /**
   * Reassemble s16le-aligned bytes: prepend any leftover byte from the prior
   * chunk, then hold back a new trailing byte if this leaves an odd count.
   */
  private alignBytes(bytes: Uint8Array): Uint8Array {
    let aligned: Uint8Array
    if (this.leftoverByte !== null) {
      aligned = new Uint8Array(bytes.length + 1)
      aligned[0] = this.leftoverByte
      aligned.set(bytes, 1)
      this.leftoverByte = null
    } else {
      aligned = bytes
    }
    if (aligned.length & 1) {
      this.leftoverByte = aligned[aligned.length - 1]
      aligned = aligned.subarray(0, aligned.length - 1)
    }
    return aligned
  }

  /**
   * Once the estimator yields a verdict, lock the working rate and drain the
   * pre-decision backlog through the pipeline at that rate. Without a verdict
   * yet, do nothing (keep buffering).
   */
  private tryDecide(): void {
    const est = this.estimator.estimate()
    if (est === null) {
      // No rate verdict yet. Keep buffering unless we've held more than the
      // safety cap (e.g. a frozen/unusable clock): then commit to the declared
      // rate so audio still flows and memory stays bounded.
      const capBytes = this.declaredRate * this.channels * BYTES_PER_SAMPLE * MAX_BUFFER_SEC
      if (this.predecisionBytes >= capBytes) this.decideAndDrain()
      return
    }

    const drift = Math.abs(est - this.declaredRate) / this.declaredRate
    if (drift > 0.2) {
      const corrected = nearestSupportedRate(est)
      if (corrected !== this.declaredRate) {
        this.workingRate = corrected
        this.onWarning?.(
          `Inbound rate mismatch: declared ${this.declaredRate} Hz but effective ` +
            `~${Math.round(est)} Hz; switching working rate to ${corrected} Hz.`,
        )
      }
    }
    this.decideAndDrain()
  }

  private decideAndDrain(): void {
    this.decided = true
    const backlog = this.predecision
    this.predecision = []
    this.predecisionBytes = 0
    for (const chunk of backlog) this.process(chunk)
  }

  private process(aligned: Uint8Array): void {
    if (aligned.length === 0) return
    const floats = pcmBytesToFloat32(aligned)
    const mono = mixToMono(floats, this.channels)
    this.resampleChunkInto(mono)
    this.emitFrames()
  }

  /**
   * Resample one mono chunk at `workingRate` into TARGET_RATE, appending to the
   * pending buffer. Fractional source position carries across chunk boundaries
   * so concatenated output equals whole-signal resampling within 1e-6.
   */
  private resampleChunkInto(mono: Float32Array): void {
    if (mono.length === 0) return

    if (this.workingRate === TARGET_RATE) {
      for (let i = 0; i < mono.length; i++) this.pending.push(mono[i])
      this.prevTail = mono[mono.length - 1]
      this.hasPrevTail = true
      return
    }

    const step = this.workingRate / TARGET_RATE // source samples per output sample

    // Build a virtual source indexed so that index -1 is the previous chunk's
    // tail sample, index 0..n-1 are this chunk. fracPos is the source position
    // (in this index space) of the next output sample to produce.
    const sampleAt = (idx: number): number => {
      if (idx < 0) return this.hasPrevTail ? this.prevTail : mono[0]
      if (idx >= mono.length) return mono[mono.length - 1]
      return mono[idx]
    }

    let pos = this.fracPos
    // Highest source position we can interpolate using samples available in
    // this chunk is mono.length - 1 (need idx and idx+1; idx+1 <= length-1).
    const maxPos = mono.length - 1
    while (pos <= maxPos) {
      const i0 = Math.floor(pos)
      const frac = pos - i0
      const a = sampleAt(i0)
      const b = sampleAt(i0 + 1)
      this.pending.push(a + (b - a) * frac)
      pos += step
    }

    // Re-base fracPos relative to the next chunk: shift the index origin by
    // mono.length so index -1 of the next chunk maps to this chunk's last
    // sample. Keep float64 precision to avoid cumulative drift.
    this.fracPos = pos - mono.length
    this.prevTail = mono[mono.length - 1]
    this.hasPrevTail = true
  }

  private emitFrames(): void {
    while (this.pending.length >= FRAME_SAMPLES) {
      const frameSamples = this.pending.splice(0, FRAME_SAMPLES)
      this.onFrame(float32ToPcmBytes(Float32Array.from(frameSamples)))
    }
  }

  /** Emit any >= 0.5 s remainder zero-padded to a full 1 s frame; drop less. */
  flush(): void {
    // Stream ended before the rate window elapsed: commit to the declared rate
    // and drain whatever was buffered so short clips still produce frames.
    if (!this.decided) this.decideAndDrain()
    this.emitFrames()
    const remainder = this.pending.length
    if (remainder >= FRAME_SAMPLES / 2) {
      const frame = new Float32Array(FRAME_SAMPLES)
      for (let i = 0; i < remainder; i++) frame[i] = this.pending[i]
      this.pending = []
      this.onFrame(float32ToPcmBytes(frame))
    } else {
      this.pending = []
    }
  }

  reset(): void {
    this.workingRate = this.declaredRate
    this.estimator.reset()
    this.prevTail = 0
    this.hasPrevTail = false
    this.fracPos = 0
    this.pending = []
    this.leftoverByte = null
    this.decided = false
    this.predecision = []
    this.predecisionBytes = 0
  }
}
