import { expect, test, type Page } from '@playwright/test'

test('375 px layout has required controls and glove-friendly targets', async ({ page }) => {
  await page.goto('/')

  await expect(page.getByRole('heading', { name: 'Field recorder' })).toBeVisible()
  await expect(page.getByText('Queue 0')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Record' })).toBeVisible()
  await expect(page.getByLabel('Live input level')).toBeVisible()

  const touchTargets = [
    page.locator('#machineType'),
    page.locator('#condition'),
    page.locator('#siteTag'),
    page.locator('.check-row'),
    page.getByRole('button', { name: 'Save clip' }),
  ]

  for (const locator of touchTargets) {
    const box = await locator.boundingBox()
    expect(box).not.toBeNull()
    expect(box!.height).toBeGreaterThanOrEqual(56)
  }

  const recordBox = await page.getByRole('button', { name: 'Record' }).boundingBox()
  expect(recordBox).not.toBeNull()
  expect(recordBox!.width).toBeGreaterThanOrEqual(240)
  expect(recordBox!.height).toBeGreaterThanOrEqual(240)

  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)
  expect(overflow).toBe(false)
})

test('records a clipped clip and posts the field label payload', async ({ page }) => {
  await installFakeMicrophone(page)
  const uploads: unknown[] = []

  await page.route('**/label', async (route) => {
    uploads.push(route.request().postDataJSON())
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        wav_path: 'datakit/data/test.wav',
        json_path: 'datakit/data/test.json',
      }),
    })
  })

  await page.goto('/')
  await page.getByRole('button', { name: 'Record' }).click()
  await expect(page.getByRole('button', { name: 'Stop' })).toBeVisible()
  await expect(page.locator('#clipFlag')).toBeVisible()
  await page.getByRole('button', { name: 'Stop' }).click()

  await page.locator('#machineType').selectOption('pump')
  await page.locator('#condition').selectOption('suspected issue')
  await page.locator('#siteTag').fill('shop-7-pump-2')
  await page.locator('#containsSpeech').check()
  await page.locator('#note').fill('rattle at startup')
  await page.getByRole('button', { name: 'Save clip' }).click()

  await expect(page.getByText('Saved')).toBeVisible()
  expect(uploads).toHaveLength(1)

  const payload = uploads[0] as Record<string, unknown>
  expect(payload.machine_type).toBe('pump')
  expect(payload.condition).toBe('suspected issue')
  expect(payload.site_tag).toBe('shop-7-pump-2')
  expect(payload.contains_speech).toBe(true)
  expect(payload.note).toBe('rattle at startup')
  expect(payload.client_clipped).toBe(true)
  expect(payload.client_duration_s).toBeGreaterThan(0)
  expect(typeof payload.pcm_b64).toBe('string')
  expect((payload.pcm_b64 as string).length).toBeGreaterThan(100)
})

test('queues failed uploads in IndexedDB and retries visibly', async ({ page }) => {
  await installFakeMicrophone(page)
  let requestCount = 0

  await page.route('**/label', async (route) => {
    requestCount += 1
    if (requestCount === 1) {
      await route.fulfill({ status: 503, body: 'offline' })
      return
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        wav_path: 'datakit/data/retry.wav',
        json_path: 'datakit/data/retry.json',
      }),
    })
  })

  await page.goto('/')
  await page.getByRole('button', { name: 'Record' }).click()
  await expect(page.locator('#clipFlag')).toBeVisible()
  await page.getByRole('button', { name: 'Stop' }).click()
  await page.locator('#siteTag').fill('offline-fan-1')
  await page.getByRole('button', { name: 'Save clip' }).click()

  await expect(page.getByText('Offline: clip queued')).toBeVisible()
  await expect(page.getByText('Queue 1')).toBeVisible()

  await page.reload()
  await expect(page.getByText('Queue 0')).toBeVisible()
  await expect(page.getByText('Uploaded 1 queued clip')).toBeVisible()
  expect(requestCount).toBeGreaterThanOrEqual(2)
})

async function installFakeMicrophone(page: Page) {
  await page.addInitScript(() => {
    class FakeMediaRecorder extends EventTarget {
      state = 'inactive'

      constructor(_stream: MediaStream) {
        super()
      }

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

      createMediaStreamSource(_stream: MediaStream) {
        return {
          connect() {},
          disconnect() {},
        }
      }

      createScriptProcessor() {
        const processor: {
          onaudioprocess: ((event: { inputBuffer: { getChannelData: () => Float32Array } }) => void) | null
          connect: () => void
          disconnect: () => void
        } = {
          onaudioprocess: null,
          connect: () => {
            const samples = new Float32Array(4096)
            for (let i = 0; i < samples.length; i += 1) {
              samples[i] = i % 2 === 0 ? 0.99 : -0.99
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
    Object.defineProperty(window, 'webkitAudioContext', {
      configurable: true,
      value: FakeAudioContext,
    })
  })
}
