import { isIP } from 'node:net'

/** Validate the single stable remote engine origin used for packaged builds. */
export function validateEngineOrigin(value) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error('EARSIGHT_ENGINE_ORIGIN is required')
  }
  let url
  try {
    url = new URL(value.trim())
  } catch {
    throw new Error('EARSIGHT_ENGINE_ORIGIN must be a valid URL')
  }
  if (url.protocol !== 'https:') {
    throw new Error('EARSIGHT_ENGINE_ORIGIN must use https')
  }
  if (url.username || url.password || url.search || url.hash || url.pathname !== '/') {
    throw new Error('EARSIGHT_ENGINE_ORIGIN must be an origin without credentials, path, query, or fragment')
  }
  if (isIP(url.hostname)) {
    throw new Error('EARSIGHT_ENGINE_ORIGIN must use a stable hostname, not a machine IP')
  }
  if (url.hostname === 'localhost' || url.hostname.endsWith('.localhost')) {
    throw new Error('EARSIGHT_ENGINE_ORIGIN must not use a loopback hostname')
  }
  return url.origin
}

function isLoopbackOrigin(value) {
  try {
    const url = new URL(value)
    return (
      url.protocol === 'http:' &&
      (url.hostname === 'localhost' || url.hostname === '127.0.0.1') &&
      !url.username &&
      !url.password &&
      url.pathname === '/' &&
      !url.search &&
      !url.hash
    )
  } catch {
    return false
  }
}

/** Clone the development manifest and add exactly one validated remote origin. */
export function buildManifest(baseManifest, engineOrigin) {
  const origin = validateEngineOrigin(engineOrigin)
  const manifest = JSON.parse(JSON.stringify(baseManifest))
  const network = manifest.permissions?.find((permission) => permission.name === 'network')
  if (!network) throw new Error('app.json is missing the network permission')
  const loopback = (network.whitelist ?? []).filter(isLoopbackOrigin)
  network.whitelist = [...new Set([...loopback, origin])]
  return manifest
}
