import { readFile } from 'node:fs/promises'

import { validateEngineOrigin } from './engine-origin.mjs'

const selected = validateEngineOrigin(process.env.EARSIGHT_ENGINE_ORIGIN)
const manifest = JSON.parse(
  await readFile(new URL('../app.generated.json', import.meta.url), 'utf8'),
)
const runtime = JSON.parse(
  await readFile(new URL('../dist/engine-origin.json', import.meta.url), 'utf8'),
)
const network = manifest.permissions?.find((permission) => permission.name === 'network')
const remoteOrigins = (network?.whitelist ?? []).filter((value) => value.startsWith('https://'))

if (runtime.engineOrigin !== selected) {
  throw new Error('built runtime engine origin does not match EARSIGHT_ENGINE_ORIGIN')
}
if (remoteOrigins.length !== 1 || remoteOrigins[0] !== selected) {
  throw new Error('generated manifest engine origin does not match EARSIGHT_ENGINE_ORIGIN')
}

console.log(`validated runtime + manifest engine origin: ${selected}`)
