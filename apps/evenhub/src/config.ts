/** Injected by Vite from the validated packaging configuration. */
export const ENGINE_ORIGIN = __EARSIGHT_ENGINE_ORIGIN__

export function isLoopbackEngineOrigin(origin: string): boolean {
  try {
    const url = new URL(origin)
    return (
      url.protocol === 'http:' &&
      (url.hostname === 'localhost' || url.hostname === '127.0.0.1')
    )
  } catch {
    return false
  }
}

/** Token-free origins used only by local development and the opt-in G2 proxy. */
export function isLocalEngineOrigin(origin: string): boolean {
  return origin.startsWith('/') || isLoopbackEngineOrigin(origin)
}
