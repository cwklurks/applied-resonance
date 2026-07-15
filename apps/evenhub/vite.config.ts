import { defineConfig, loadEnv, type Plugin } from 'vite'

import { validateEngineOrigin } from './scripts/engine-origin.mjs'

const LOCAL_ENGINE_ORIGIN = 'http://localhost:8000'

function engineOriginAsset(engineOrigin: string): Plugin {
  return {
    name: 'engine-origin-asset',
    generateBundle() {
      this.emitFile({
        type: 'asset',
        fileName: 'engine-origin.json',
        source: `${JSON.stringify({ engineOrigin })}\n`,
      })
    },
  }
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const engineOrigin = env.EARSIGHT_ENGINE_ORIGIN
    ? validateEngineOrigin(env.EARSIGHT_ENGINE_ORIGIN)
    : LOCAL_ENGINE_ORIGIN

  return {
    define: {
      __EARSIGHT_ENGINE_ORIGIN__: JSON.stringify(engineOrigin),
    },
    plugins: [engineOriginAsset(engineOrigin)],
    server: { host: '127.0.0.1', port: 5173 },
    build: { target: 'esnext' },
  }
})
