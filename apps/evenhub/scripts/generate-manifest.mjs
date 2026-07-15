import { readFile, writeFile } from 'node:fs/promises'

import { buildManifest } from './engine-origin.mjs'

const base = JSON.parse(await readFile(new URL('../app.json', import.meta.url), 'utf8'))
const manifest = buildManifest(base, process.env.EARSIGHT_ENGINE_ORIGIN)
await writeFile(
  new URL('../app.generated.json', import.meta.url),
  `${JSON.stringify(manifest, null, 2)}\n`,
  { mode: 0o600 },
)
