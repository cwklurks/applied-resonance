import { describe, expect, it } from 'vitest'

import { computeLevel, encodePcm16Base64, resampleLinear } from '../src/audio'

describe('audio helpers', () => {
  it('detects clipping on hot input', () => {
    const level = computeLevel(new Float32Array([0, 0.5, -0.99]))
    expect(level.clipped).toBe(true)
    expect(level.peak).toBeCloseTo(0.99)
  })

  it('resamples to the target length', () => {
    const source = new Float32Array(48000)
    const out = resampleLinear(source, 48000, 16000)
    expect(out).toHaveLength(16000)
  })

  it('encodes int16 pcm as base64', () => {
    const encoded = encodePcm16Base64(new Float32Array([0, 0.5, -0.5]))
    expect(atob(encoded)).toHaveLength(6)
  })
})

