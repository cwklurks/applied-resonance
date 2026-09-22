import { TARGET_SAMPLE_RATE, type RecordedClip } from './types'

export interface InputLevel {
  peak: number
  rms: number
  clipped: boolean
}

type LevelHandler = (level: InputLevel) => void

export class FieldRecorder {
  private stream: MediaStream | null = null
  private audioContext: AudioContext | null = null
  private processor: ScriptProcessorNode | null = null
  private source: MediaStreamAudioSourceNode | null = null
  private mediaRecorder: MediaRecorder | null = null
  private sampleRate = TARGET_SAMPLE_RATE
  private buffers: Float32Array[] = []
  private peakAbs = 0
  private clipped = false
  private startedAt = 0

  async start(onLevel: LevelHandler): Promise<void> {
    if (this.stream) {
      throw new Error('recording already active')
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('microphone capture is not available in this browser')
    }
    if (typeof MediaRecorder === 'undefined') {
      throw new Error('MediaRecorder is not available in this browser')
    }

    this.buffers = []
    this.peakAbs = 0
    this.clipped = false
    this.startedAt = Date.now()

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
      video: false,
    })

    this.audioContext = new AudioContext()
    this.sampleRate = this.audioContext.sampleRate
    this.source = this.audioContext.createMediaStreamSource(this.stream)
    this.processor = this.audioContext.createScriptProcessor(4096, 1, 1)

    this.processor.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0)
      const copy = new Float32Array(input.length)
      copy.set(input)
      this.buffers.push(copy)

      const level = computeLevel(input)
      this.peakAbs = Math.max(this.peakAbs, level.peak)
      this.clipped = this.clipped || level.clipped
      onLevel(level)
    }

    this.source.connect(this.processor)
    this.processor.connect(this.audioContext.destination)

    this.mediaRecorder = new MediaRecorder(this.stream)
    this.mediaRecorder.start()
  }

  async stop(): Promise<RecordedClip> {
    if (!this.stream || !this.audioContext || !this.processor || !this.source) {
      throw new Error('recording is not active')
    }

    const recorderStopped = new Promise<void>((resolve) => {
      if (!this.mediaRecorder || this.mediaRecorder.state === 'inactive') {
        resolve()
        return
      }
      this.mediaRecorder.addEventListener('stop', () => resolve(), { once: true })
      this.mediaRecorder.stop()
    })

    this.processor.disconnect()
    this.source.disconnect()
    this.stream.getTracks().forEach((track) => track.stop())
    await recorderStopped
    await this.audioContext.close()

    const sourcePcm = concatFloat32(this.buffers)
    const resampled = resampleLinear(sourcePcm, this.sampleRate, TARGET_SAMPLE_RATE)
    const pcm_b64 = encodePcm16Base64(resampled)

    const duration_s =
      resampled.length > 0
        ? resampled.length / TARGET_SAMPLE_RATE
        : Math.max(0, (Date.now() - this.startedAt) / 1000)

    const clip: RecordedClip = {
      pcm_b64,
      duration_s,
      peak_abs: this.peakAbs,
      clipped: this.clipped || this.peakAbs >= 0.999,
      recorded_at: new Date(this.startedAt).toISOString(),
    }

    this.stream = null
    this.audioContext = null
    this.processor = null
    this.source = null
    this.mediaRecorder = null
    this.buffers = []

    return clip
  }
}

export function computeLevel(input: Float32Array): InputLevel {
  let sumSquares = 0
  let peak = 0
  for (const sample of input) {
    const abs = Math.abs(sample)
    peak = Math.max(peak, abs)
    sumSquares += sample * sample
  }
  return {
    peak,
    rms: input.length > 0 ? Math.sqrt(sumSquares / input.length) : 0,
    clipped: peak >= 0.98,
  }
}

export function concatFloat32(chunks: Float32Array[]): Float32Array {
  const length = chunks.reduce((total, chunk) => total + chunk.length, 0)
  const out = new Float32Array(length)
  let offset = 0
  for (const chunk of chunks) {
    out.set(chunk, offset)
    offset += chunk.length
  }
  return out
}

export function resampleLinear(
  input: Float32Array,
  sourceSampleRate: number,
  targetSampleRate: number,
): Float32Array {
  if (sourceSampleRate <= 0 || targetSampleRate <= 0) {
    throw new Error('sample rates must be positive')
  }
  if (input.length === 0 || sourceSampleRate === targetSampleRate) {
    return new Float32Array(input)
  }

  const ratio = sourceSampleRate / targetSampleRate
  const outputLength = Math.max(1, Math.round(input.length / ratio))
  const output = new Float32Array(outputLength)

  for (let i = 0; i < outputLength; i += 1) {
    const sourceIndex = i * ratio
    const left = Math.floor(sourceIndex)
    const right = Math.min(input.length - 1, left + 1)
    const frac = sourceIndex - left
    output[i] = input[left] * (1 - frac) + input[right] * frac
  }

  return output
}

export function encodePcm16Base64(input: Float32Array): string {
  const bytes = new Uint8Array(input.length * 2)
  const view = new DataView(bytes.buffer)
  input.forEach((sample, index) => {
    const clipped = Math.max(-1, Math.min(0.999969, sample))
    const int16 = clipped < 0 ? clipped * 32768 : clipped * 32767
    view.setInt16(index * 2, Math.round(int16), true)
  })
  return bytesToBase64(bytes)
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = ''
  const chunkSize = 0x8000
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize)
    binary += String.fromCharCode(...chunk)
  }
  return btoa(binary)
}

