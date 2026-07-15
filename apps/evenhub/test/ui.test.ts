import { describe, expect, it } from 'vitest'

import { isLoopbackEngineOrigin } from '../src/config'

describe('engine origin classification', () => {
  it('recognizes only HTTP loopback origins as token-free development', () => {
    expect(isLoopbackEngineOrigin('http://localhost:8000')).toBe(true)
    expect(isLoopbackEngineOrigin('http://127.0.0.1:8000')).toBe(true)
  })

  it('requires runtime credentials for HTTPS and rejects LAN HTTP classification', () => {
    expect(isLoopbackEngineOrigin('https://engine.example.test')).toBe(false)
    expect(isLoopbackEngineOrigin('http://192.0.2.10:8000')).toBe(false)
    expect(isLoopbackEngineOrigin('not a URL')).toBe(false)
  })
})
