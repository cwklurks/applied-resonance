import fs from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { chromium } from 'playwright'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(__dirname, '../..')
const dataDir = path.join(repoRoot, 'datakit', 'data')
const appUrl = process.env.FIELD_RECORDER_URL || 'http://127.0.0.1:5174/'

const before = await listStems()
const browser = await chromium.launch()
try {
  const page = await browser.newPage({ viewport: { width: 375, height: 812 } })
  await installFakeMicrophone(page)
  await page.goto(appUrl)

  await page.getByRole('button', { name: 'Record' }).click()
  await page.locator('#clipFlag').waitFor({ state: 'visible', timeout: 5000 })
  await page.getByRole('button', { name: 'Stop' }).click()

  await page.locator('#machineType').selectOption('fan')
  await page.locator('#condition').selectOption('sounds normal')
  await page.locator('#siteTag').fill('real-save-check-fan-1')
  await page.locator('#note').fill('browser-to-engine real-save verification')
  await page.getByRole('button', { name: 'Save clip' }).click()
  await page.getByText('Saved').waitFor({ state: 'visible', timeout: 10000 })
} finally {
  await browser.close()
}

const after = await listStems()
const added = [...after].filter((stem) => !before.has(stem)).sort()
if (added.length !== 1) {
  throw new Error(`expected one new datakit clip, found ${added.length}: ${added.join(', ')}`)
}

const stem = added[0]
const jsonPath = path.join(dataDir, `${stem}.json`)
const wavPath = path.join(dataDir, `${stem}.wav`)
const meta = JSON.parse(await fs.readFile(jsonPath, 'utf8'))
const wavStat = await fs.stat(wavPath)

if (meta.site_tag !== 'real-save-check-fan-1') {
  throw new Error(`unexpected site_tag ${meta.site_tag}`)
}
if (meta.machine_type !== 'fan' || meta.condition !== 'sounds normal') {
  throw new Error(`unexpected labels ${meta.machine_type}/${meta.condition}`)
}
if (meta.sr !== 16000 || meta.n_samples <= 0 || wavStat.size <= 44) {
  throw new Error('saved clip did not produce a valid 16 kHz WAV sidecar pair')
}

console.log(`saved ${path.relative(repoRoot, wavPath)}`)
console.log(`sidecar ${path.relative(repoRoot, jsonPath)}`)
console.log(`samples ${meta.n_samples} sr ${meta.sr} clipped ${meta.clipped}`)

async function listStems() {
  const names = await fs.readdir(dataDir)
  const wavs = new Set(names.filter((name) => name.endsWith('.wav')).map(stripExt))
  const jsons = new Set(names.filter((name) => name.endsWith('.json')).map(stripExt))
  return new Set([...wavs].filter((stem) => jsons.has(stem)))
}

function stripExt(name) {
  return name.replace(/\.(wav|json)$/, '')
}

async function installFakeMicrophone(page) {
  await page.addInitScript(() => {
    class FakeMediaRecorder extends EventTarget {
      state = 'inactive'

      start() {
        this.state = 'recording'
      }

      stop() {
        this.state = 'inactive'
        this.dispatchEvent(new Event('stop'))
      }
    }

    class FakeTrack {
      stop() {}
    }

    class FakeStream {
      getTracks() {
        return [new FakeTrack()]
      }
    }

    class FakeAudioContext {
      sampleRate = 48000
      destination = {}

      createMediaStreamSource() {
        return {
          connect() {},
          disconnect() {},
        }
      }

      createScriptProcessor() {
        const processor = {
          onaudioprocess: null,
          connect: () => {
            const samples = new Float32Array(48000)
            for (let i = 0; i < samples.length; i += 1) {
              samples[i] = Math.sin(i / 14) * 0.99
            }
            window.setTimeout(() => {
              processor.onaudioprocess?.({
                inputBuffer: {
                  getChannelData: () => samples,
                },
              })
            }, 0)
          },
          disconnect: () => {},
        }
        return processor
      }

      close() {
        return Promise.resolve()
      }
    }

    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: {
        getUserMedia: () => Promise.resolve(new FakeStream()),
      },
    })
    Object.defineProperty(window, 'MediaRecorder', {
      configurable: true,
      value: FakeMediaRecorder,
    })
    Object.defineProperty(window, 'AudioContext', {
      configurable: true,
      value: FakeAudioContext,
    })
  })
}
