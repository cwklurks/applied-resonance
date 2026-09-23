import os from 'node:os'

const port = process.env.PORT || process.env.VITE_PORT || '5174'
const host = process.env.HOST_IP || findLanAddress() || '127.0.0.1'
const url = `http://${host}:${port}`

console.log(`Field recorder LAN URL: ${url}`)
console.log('')

try {
  const qrcode = await import('qrcode-terminal')
  qrcode.default.generate(url, { small: true })
} catch {
  console.log('Install datakit npm dependencies to print the QR code.')
}

console.log('')
console.log('Start the service:')
console.log('  uv run uvicorn engine.serve:create_app --factory --host 0.0.0.0 --port 8000')
console.log('  npm --prefix datakit run dev')

function findLanAddress() {
  const interfaces = os.networkInterfaces()
  for (const entries of Object.values(interfaces)) {
    for (const entry of entries || []) {
      if (entry.family === 'IPv4' && !entry.internal) {
        return entry.address
      }
    }
  }
  return null
}

