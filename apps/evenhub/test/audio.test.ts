import { describe, it, expect } from 'vitest'
import {
  pcmBytesToFloat32,
  float32ToPcmBytes,
  mixToMono,
  resampleLinear,
  RateEstimator,
  AudioNormalizer,
} from '../src/audio'

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

function randInt16(rng: () => number): number {
  // Uniform over the full int16 domain [-32768, 32767].
  return Math.floor(rng() * 65536) - 32768
}

function rms(a: Float32Array): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * a[i]
  return Math.sqrt(s / a.length)
}

function int16ToBytes(samples: number[]): Uint8Array {
  const out = new Uint8Array(samples.length * 2)
  const view = new DataView(out.buffer)
  for (let i = 0; i < samples.length; i++) view.setInt16(i * 2, samples[i], true)
  return out
}

function concatBytes(parts: Uint8Array[]): Uint8Array {
  const total = parts.reduce((n, p) => n + p.length, 0)
  const out = new Uint8Array(total)
  let off = 0
  for (const p of parts) {
    out.set(p, off)
    off += p.length
  }
  return out
}

// Synthesize a sine as int16 samples at a given rate.
function sineInt16(freq: number, rate: number, durationSec: number, amp = 0.6): number[] {
  const n = Math.round(rate * durationSec)
  const out: number[] = new Array(n)
  for (let i = 0; i < n; i++) {
    out[i] = Math.round(Math.sin((2 * Math.PI * freq * i) / rate) * amp * 32767)
  }
  return out
}

// ── Test 1: PCM round-trip ─────────────────────────────────────────────────
describe('pcm round-trip', () => {
  it('int16 -> bytes -> float -> bytes is lossless within int16 domain', () => {
    const rng = mulberry32(12345)
    const samples: number[] = []
    for (let i = 0; i < 2048; i++) samples.push(randInt16(rng))
    const bytes = int16ToBytes(samples)

    const floats = pcmBytesToFloat32(bytes)
    const back = float32ToPcmBytes(floats)

    expect(back.length).toBe(bytes.length)
    expect(Array.from(back)).toEqual(Array.from(bytes))
  })

  it('drops a trailing odd byte without throwing', () => {
    const samples = [100, -200, 30000, -30000]
    const bytes = int16ToBytes(samples)
    const odd = new Uint8Array(bytes.length + 1)
    odd.set(bytes, 0)
    odd[bytes.length] = 0x7f // stray byte

    let floats!: Float32Array
    expect(() => {
      floats = pcmBytesToFloat32(odd)
    }).not.toThrow()
    expect(floats.length).toBe(samples.length) // tail byte dropped
  })
})

// ── Test 2: mixToMono ──────────────────────────────────────────────────────
describe('mixToMono', () => {
  it('stereo L=0.5 R=-0.5 averages to exact zeros', () => {
    const n = 256
    const stereo = new Float32Array(n * 2)
    for (let i = 0; i < n; i++) {
      stereo[i * 2] = 0.5
      stereo[i * 2 + 1] = -0.5
    }
    const mono = mixToMono(stereo, 2)
    expect(mono.length).toBe(n)
    for (let i = 0; i < n; i++) expect(mono[i]).toBe(0)
  })

  it('1-channel passthrough returns the same values', () => {
    const data = new Float32Array([0.1, -0.2, 0.3, -0.4])
    const mono = mixToMono(data, 1)
    expect(Array.from(mono)).toEqual([0.1, -0.2, 0.3, -0.4].map((v) => Math.fround(v)))
  })
})

// ── Test 3: resampleLinear ─────────────────────────────────────────────────
describe('resampleLinear', () => {
  it('440 Hz sine 48k -> 16k matches a directly synthesized 16k sine (RMS err < 1%)', () => {
    const freq = 440
    const dur = 0.5
    const src = Float32Array.from(sineInt16(freq, 48000, dur).map((s) => s / 32768))

    const out = resampleLinear(src, 48000, 16000)
    const ref = Float32Array.from(sineInt16(freq, 16000, dur).map((s) => s / 32768))

    // Length within ±1 of the ideal.
    expect(Math.abs(out.length - Math.round(src.length * (16000 / 48000)))).toBeLessThanOrEqual(1)

    const n = Math.min(out.length, ref.length)
    const diff = new Float32Array(n)
    for (let i = 0; i < n; i++) diff[i] = out[i] - ref[i]
    expect(rms(diff)).toBeLessThan(0.01 * rms(ref))
  })
})

// ── Test 4: chunker exactness (no resampling) ──────────────────────────────
describe('chunker exactness', () => {
  it('5 s @ 16k mono in randomized odd byte chunks -> 5 byte-identical frames', () => {
    const rng = mulberry32(99)
    const totalSamples = 80000 // 5 s @ 16k
    const samples: number[] = []
    for (let i = 0; i < totalSamples; i++) samples.push(randInt16(rng))
    const fullBytes = int16ToBytes(samples)

    const frames: Uint8Array[] = []
    const norm = new AudioNormalizer({
      declaredRate: 16000,
      channels: 1,
      onFrame: (f) => frames.push(f),
      now: () => 0, // freeze clock so no rate switch is ever triggered
    })

    // Feed in odd-sized pieces.
    const sizes = [313, 1009, 4096]
    let off = 0
    let si = 0
    while (off < fullBytes.length) {
      const size = Math.min(sizes[si % sizes.length], fullBytes.length - off)
      norm.push(fullBytes.subarray(off, off + size))
      off += size
      si++
    }
    norm.flush() // end of stream; 5 s is an exact multiple so no padding

    expect(frames.length).toBe(5)
    for (const f of frames) expect(f.length).toBe(32000)
    expect(Array.from(concatBytes(frames))).toEqual(Array.from(fullBytes))
  })
})

// ── Test 5: rate-mismatch detection (fake clock) ───────────────────────────
describe('rate-mismatch detection', () => {
  it('declared 16k but fed at 32k pace -> warns once, resamples to wall-clock seconds', () => {
    // Real 32 kHz tone; we declare 16 kHz and pace bytes at 64000 bytes/s.
    const freq = 440
    const durSec = 4
    const realRate = 32000
    const samples = sineInt16(freq, realRate, durSec)
    const fullBytes = int16ToBytes(samples)

    const warnings: string[] = []
    const frames: Uint8Array[] = []

    // Fake clock advances by real-time: 64000 bytes/s => ms = bytes/64 .
    let clockMs = 0
    const norm = new AudioNormalizer({
      declaredRate: 16000,
      channels: 1,
      onFrame: (f) => frames.push(f),
      onWarning: (m) => warnings.push(m),
      now: () => clockMs,
    })

    // Feed ~0.25 s worth of real audio per chunk (16000 bytes), advancing the
    // wall clock by the real elapsed time those bytes represent at 32 kHz.
    const chunkBytes = 16000 // 8000 samples @ 32k = 0.25 s real time
    let off = 0
    while (off < fullBytes.length) {
      const size = Math.min(chunkBytes, fullBytes.length - off)
      const piece = fullBytes.subarray(off, off + size)
      // Advance clock for THIS chunk before pushing so throughput reflects 32k.
      clockMs += (size / 64000) * 1000
      norm.push(piece)
      off += size
    }
    norm.flush()

    // Warned exactly once, mentioning the detected 32000 Hz.
    expect(warnings.length).toBe(1)
    expect(warnings[0]).toContain('32000')

    // Frame count tracks wall-clock seconds (~4 s), not naive byte math (~8).
    expect(frames.length).toBeGreaterThanOrEqual(3)
    expect(frames.length).toBeLessThanOrEqual(5)

    // The tone should still be ~440 Hz after correction. Count zero crossings
    // across all emitted frames and derive frequency from total duration.
    const all = concatBytes(frames)
    const view = new DataView(all.buffer, all.byteOffset, all.byteLength)
    const outSamples = all.length / 2
    let crossings = 0
    let prev = view.getInt16(0, true)
    for (let i = 1; i < outSamples; i++) {
      const cur = view.getInt16(i * 2, true)
      if ((prev < 0 && cur >= 0) || (prev >= 0 && cur < 0)) crossings++
      prev = cur
    }
    const outDurSec = outSamples / 16000
    const measuredFreq = crossings / 2 / outDurSec // 2 crossings per cycle
    expect(measuredFreq).toBeGreaterThan(440 * 0.95)
    expect(measuredFreq).toBeLessThan(440 * 1.05)
  })
})

// ── Test 6: AudioNormalizer multichannel ───────────────────────────────────
describe('AudioNormalizer multichannel', () => {
  it('declared stereo @ 16k -> mono frames with correct count', () => {
    // 3 s of interleaved stereo @ 16k. Interleaved => 2x samples => 2x bytes.
    const durSec = 3
    const frameSamples = 16000 * durSec
    const interleaved: number[] = []
    for (let i = 0; i < frameSamples; i++) {
      interleaved.push(1000) // L
      interleaved.push(3000) // R
    }
    const bytes = int16ToBytes(interleaved)

    const frames: Uint8Array[] = []
    const norm = new AudioNormalizer({
      declaredRate: 16000,
      channels: 2,
      onFrame: (f) => frames.push(f),
      now: () => 0,
    })
    // Feed in two pieces to exercise carry-over; keep piece boundaries on whole
    // stereo frames so mixing stays aligned.
    const mid = Math.floor(bytes.length / 4) * 4
    norm.push(bytes.subarray(0, mid))
    norm.push(bytes.subarray(mid))
    norm.flush() // end of stream; 3 s is an exact multiple so no padding

    expect(frames.length).toBe(3) // 3 s of mono @ 16k
    for (const f of frames) expect(f.length).toBe(32000)

    // Each mono sample should be the average (1000+3000)/2 = 2000.
    const first = new DataView(frames[0].buffer, frames[0].byteOffset, frames[0].byteLength)
    expect(first.getInt16(0, true)).toBe(2000)
  })
})

// ── Test 7: flush remainder handling ───────────────────────────────────────
describe('flush()', () => {
  it('0.7 s remainder -> one zero-padded frame', () => {
    const samples = sineInt16(200, 16000, 0.7)
    const bytes = int16ToBytes(samples)
    const frames: Uint8Array[] = []
    const norm = new AudioNormalizer({
      declaredRate: 16000,
      channels: 1,
      onFrame: (f) => frames.push(f),
      now: () => 0,
    })
    norm.push(bytes)
    expect(frames.length).toBe(0) // not yet a full second
    norm.flush()
    expect(frames.length).toBe(1)
    expect(frames[0].length).toBe(32000)

    // The tail beyond 0.7 s must be zero padding.
    const v = new DataView(frames[0].buffer, frames[0].byteOffset, frames[0].byteLength)
    const padStart = Math.round(16000 * 0.7)
    expect(v.getInt16((16000 - 1) * 2, true)).toBe(0)
    expect(v.getInt16(padStart * 2 + 2 * 100, true)).toBe(0)
  })

  it('0.3 s remainder -> no frame', () => {
    const samples = sineInt16(200, 16000, 0.3)
    const bytes = int16ToBytes(samples)
    const frames: Uint8Array[] = []
    const norm = new AudioNormalizer({
      declaredRate: 16000,
      channels: 1,
      onFrame: (f) => frames.push(f),
      now: () => 0,
    })
    norm.push(bytes)
    norm.flush()
    expect(frames.length).toBe(0)
  })
})

// ── Extra: RateEstimator unit behavior + whole-signal resample equivalence ──
describe('RateEstimator', () => {
  it('returns null before the window and bytes/sec ÷ 2 ÷ channels after', () => {
    const est = new RateEstimator(1, 3000)
    est.feed(32000, 0)
    expect(est.estimate()).toBeNull()
    est.feed(32000, 1000)
    expect(est.estimate()).toBeNull() // span 1000ms < 3000
    est.feed(32000, 2000)
    est.feed(32000, 3000)
    // 3 chunks of 32000 bytes after the first, over 3000 ms span => 32000 B/s.
    // /2 /1 channel => 16000 samples/s.
    expect(est.estimate()).toBeCloseTo(16000, 5)
  })

  it('reset clears state', () => {
    const est = new RateEstimator(1, 1000)
    est.feed(1000, 0)
    est.feed(1000, 2000)
    expect(est.estimate()).not.toBeNull()
    est.reset()
    expect(est.estimate()).toBeNull()
  })
})

describe('resampler carry-over equivalence', () => {
  it('chunked resample equals whole-signal resample within 1e-6', () => {
    // Drive AudioNormalizer with a real 48k tone declared as 48k (so it
    // resamples 48k->16k) and compare against resampleLinear on the whole
    // signal. Feeding 48k declared keeps the estimator from switching since
    // the clock is frozen.
    const freq = 330
    const src48 = Float32Array.from(sineInt16(freq, 48000, 2).map((s) => s / 32768))
    const wholeOut = resampleLinear(src48, 48000, 16000)

    const bytes = float32ToPcmBytes(src48)
    const frames: Uint8Array[] = []
    const norm = new AudioNormalizer({
      declaredRate: 48000,
      channels: 1,
      onFrame: (f) => frames.push(f),
      now: () => 0,
    })
    // Odd-sized chunks to stress the fractional carry.
    const sizes = [1001, 4096, 777, 20000]
    let off = 0
    let si = 0
    while (off < bytes.length) {
      const size = Math.min(sizes[si % sizes.length], bytes.length - off)
      norm.push(bytes.subarray(off, off + size))
      off += size
      si++
    }
    norm.flush()

    const all = concatBytes(frames)
    const v = new DataView(all.buffer, all.byteOffset, all.byteLength)
    const chunkedFloats: number[] = []
    for (let i = 0; i < all.length / 2; i++) chunkedFloats.push(v.getInt16(i * 2, true) / 32768)

    // Compare on the overlap (the padded tail of the last frame is ignored).
    const n = Math.min(chunkedFloats.length, wholeOut.length)
    let maxErr = 0
    for (let i = 0; i < n; i++) {
      maxErr = Math.max(maxErr, Math.abs(chunkedFloats[i] - wholeOut[i]))
    }
    // float32ToPcmBytes quantizes to int16, so allow one int16 quantum on top
    // of the 1e-6 algorithmic tolerance.
    expect(maxErr).toBeLessThan(1 / 32768 + 1e-6)
  })
})
