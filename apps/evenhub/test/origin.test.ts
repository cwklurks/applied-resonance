import { describe, expect, it } from 'vitest'

import { buildManifest, validateEngineOrigin } from '../scripts/engine-origin.mjs'

const base = {
  permissions: [
    {
      name: 'network',
      whitelist: [
        'http://localhost:8000',
        'http://127.0.0.1:8000',
        'http://192.0.2.10:8000',
      ],
    },
  ],
}

describe('packaged engine origin', () => {
  it('accepts and canonicalizes one stable HTTPS hostname', () => {
    expect(validateEngineOrigin('https://engine.example.test:443')).toBe(
      'https://engine.example.test',
    )
  })

  it.each([
    '',
    'http://engine.example.test',
    'https://localhost',
    'https://engine.localhost',
    'https://192.0.2.10',
    'https://user:pass@engine.example.test',
    'https://engine.example.test/path',
    'https://engine.example.test?token=nope',
  ])('rejects unsafe or ephemeral origin %j', (origin) => {
    expect(() => validateEngineOrigin(origin)).toThrow()
  })

  it('generates a manifest with loopback dev origins and exactly one remote origin', () => {
    const manifest = buildManifest(base, 'https://engine.example.test')
    const network = manifest.permissions.find((permission) => permission.name === 'network')!
    expect(network.whitelist).toEqual([
      'http://localhost:8000',
      'http://127.0.0.1:8000',
      'https://engine.example.test',
    ])
  })
})
