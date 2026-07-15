/**
 * The criterion test: the IDENTICAL Pipeline that drives the glasses, fed real
 * MIMII pump audio through the real FastAPI engine over HTTP. No bridge, no
 * mocks of the engine — only the audio source and display port differ from the
 * live app.
 *
 * Shape:
 *   1. spawn `uv run uvicorn engine.serve:create_app --factory --port 8731`
 *      (cwd = repo root, EARSIGHT_DEVICE=cpu), wait for /health,
 *   2. feed THREE normal clips (30 s → captures the baseline) then TWO abnormal
 *      clips through one Pipeline in session mode,
 *   3. assert the recorded card sequence: a CAPTURING_BASELINE countdown card,
 *      then LISTENING, then a "!"/"?" anomaly card during the abnormal clips
 *      whose line2 carries a non-empty evidence prefix + "tap to log",
 *   4. logTap() → a LOGGED card and a /label write on disk.
 *
 * Skips gracefully if data/mimii is absent.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import { spawn, type ChildProcess } from 'node:child_process'
import { existsSync, readFileSync, readdirSync } from 'node:fs'
import { resolve } from 'node:path'
import { Pipeline } from '../src/pipeline'
import { WavFileFeed, RecordingDisplay, parseWav } from '../src/mock'
import { EngineClient } from '@earsight/display-card'

const REPO_ROOT = resolve(__dirname, '../../..')
const MIMII_DIR = resolve(REPO_ROOT, 'data/mimii/0_dB/pump/id_00')
const NORMAL_DIR = resolve(MIMII_DIR, 'normal')
const ABNORMAL_DIR = resolve(MIMII_DIR, 'abnormal')
const DATAKIT_DIR = resolve(REPO_ROOT, 'datakit/data')
const PORT = 8731
const BASE_URL = `http://127.0.0.1:${PORT}`

const haveData =
  existsSync(NORMAL_DIR) && existsSync(ABNORMAL_DIR) && firstWavs(NORMAL_DIR, 3).length === 3

function firstWavs(dir: string, n: number): string[] {
  if (!existsSync(dir)) return []
  return readdirSync(dir)
    .filter((f) => f.endsWith('.wav'))
    .sort()
    .slice(0, n)
    .map((f) => resolve(dir, f))
}

function loadMonoBytes(path: string): Uint8Array {
  const buf = readFileSync(path)
  const u8 = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength)
  return parseWav(u8).bytes
}

async function waitForHealth(client: EngineClient, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const res = await client.health()
    if (res.kind === 'ok') return
    await new Promise((r) => setTimeout(r, 500))
  }
  throw new Error('engine /health did not come up in time')
}

describe.skipIf(!haveData)('integration: real engine + real MIMII audio', () => {
  let engine: ChildProcess
  const client = new EngineClient(BASE_URL)

  beforeAll(async () => {
    engine = spawn(
      'uv',
      ['run', 'uvicorn', 'engine.serve:create_app', '--factory', '--port', String(PORT)],
      {
        cwd: REPO_ROOT,
        env: { ...process.env, EARSIGHT_DEVICE: 'cpu' },
        stdio: ['ignore', 'pipe', 'pipe'],
      },
    )
    engine.stdout?.on('data', (d) => process.stdout.write(`[engine] ${d}`))
    engine.stderr?.on('data', (d) => process.stdout.write(`[engine] ${d}`))
    await waitForHealth(client, 120_000)
  }, 180_000)

  afterAll(() => {
    engine?.kill('SIGKILL')
  })

  it(
    'captures a baseline, listens, flags an anomaly, and logs a clip',
    async () => {
      const display = new RecordingDisplay()
      const feed = new WavFileFeed({ sampleRate: 16000, seed: 0xc0ffee })

      // Three normal clips (30 s) to capture the baseline, then two abnormal.
      for (const p of firstWavs(NORMAL_DIR, 3)) feed.enqueue(loadMonoBytes(p))
      for (const p of firstWavs(ABNORMAL_DIR, 2)) feed.enqueue(loadMonoBytes(p))

      const states: string[] = []
      // Virtual clock: the feed pumps ~50 s of audio in ~1 s of real time, so we
      // drive the 2 s HUD throttle with simulated time that advances 1 s per
      // scored frame. The pipeline logic is byte-identical to the live app; only
      // the clock source differs (exactly as in production where `now=Date.now`).
      let vclock = 0
      const pipeline = new Pipeline(feed, {
        client,
        display,
        mode: 'session',
        tag: 'pump-int-test',
        now: () => vclock,
        onStateChange: (s) => states.push(s),
        onFrameScored: () => {
          vclock += 1000
        },
        // Accelerated criterion input is finite (~50 frames) but much faster
        // than live cadence, so retain it in a still-bounded integration queue.
        maxPendingFrames: 64,
        maxFrameAgeMs: 120_000,
      })

      await pipeline.start()
      await feed.whenDone()
      await pipeline.drain()
      pipeline.flushAudio()
      await pipeline.drain()
      // Flush the throttle's final pending card (the anomaly) now the stream ended.
      vclock += 2001
      pipeline.flushDisplay()

      // Card sequence log (printed for the report).
      // eslint-disable-next-line no-console
      console.log(
        '\n--- CARD SEQUENCE ---\n' +
          display.cards.map((c, i) => `${i}: ${c.line1} | ${c.line2}`).join('\n') +
          '\n--- STATES ---\n' +
          states.join(' -> ') +
          '\n',
      )

      // 1) A capturing-baseline countdown card appeared.
      const captureCard = display.cards.find(
        (c) => c.line1.includes('Capturing baseline') && /\d+s/.test(c.line1),
      )
      expect(captureCard, 'expected a CAPTURING_BASELINE countdown card').toBeTruthy()

      // 2) LISTENING was reached after capture, before any anomaly. The state
      //    stream is the authoritative signal: the transient "Listening..." card
      //    can legitimately be coalesced by the 2 s HUD throttle when an anomaly
      //    follows quickly, so we assert ordering on states, not on the card list.
      const capStateIdx = states.indexOf('CAPTURING_BASELINE')
      const listenStateIdx = states.indexOf('LISTENING')
      expect(listenStateIdx, 'expected a LISTENING state').toBeGreaterThanOrEqual(0)
      expect(listenStateIdx).toBeGreaterThan(capStateIdx)

      // 3) An anomaly card (SUSPECT "?" or ALERT "!") with evidence + tap-to-log.
      const anomaly = display.cards.find(
        (c) =>
          (c.line1.startsWith('!') || c.line1.startsWith('?')) &&
          c.line2.includes('tap to log'),
      )
      expect(anomaly, 'expected a SUSPECT/ALERT card with tap-to-log').toBeTruthy()
      // Non-empty evidence prefix before " · tap to log".
      const evidencePrefix = anomaly!.line2.split(' · tap to log')[0]
      expect(evidencePrefix.length, 'expected a non-empty evidence prefix').toBeGreaterThan(0)

      // 4) logTap writes a labeled clip and shows LOGGED.
      const before = listFiles(DATAKIT_DIR)
      await pipeline.logTap()
      const loggedCard = display.cards[display.cards.length - 1]
      expect(loggedCard.line1.toLowerCase()).toContain('logged')

      const after = listFiles(DATAKIT_DIR)
      const newFiles = after.filter((f) => !before.includes(f))
      // /label writes a .wav + .json pair.
      expect(newFiles.some((f) => f.endsWith('.wav'))).toBe(true)
      expect(newFiles.some((f) => f.endsWith('.json'))).toBe(true)

      pipeline.stop()
    },
    300_000,
  )
})

function listFiles(dir: string): string[] {
  if (!existsSync(dir)) return []
  return readdirSync(dir)
}
