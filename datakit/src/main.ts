import './style.css'
import { FieldRecorder, type InputLevel } from './audio'
import { uploadLabel } from './api'
import {
  addQueuedUpload,
  countQueuedUploads,
  retryQueuedUploads,
} from './queue'
import type { LabelPayload, RecordedClip } from './types'

const recorder = new FieldRecorder()
let currentClip: RecordedClip | null = null
let recording = false
let timerId: number | null = null
let startedAt = 0

const recordButton = requireElement<HTMLButtonElement>('recordButton')
const recordButtonText = requireElement<HTMLSpanElement>('recordButtonText')
const recordTime = requireElement<HTMLSpanElement>('recordTime')
const meterBar = requireElement<HTMLDivElement>('meterBar')
const clipFlag = requireElement<HTMLDivElement>('clipFlag')
const statusText = requireElement<HTMLParagraphElement>('statusText')
const queueCount = requireElement<HTMLDivElement>('queueCount')
const labelForm = requireElement<HTMLFormElement>('labelForm')
const formFields = requireElement<HTMLFieldSetElement>('formFields')
const machineType = requireElement<HTMLSelectElement>('machineType')
const condition = requireElement<HTMLSelectElement>('condition')
const siteTag = requireElement<HTMLInputElement>('siteTag')
const containsSpeech = requireElement<HTMLInputElement>('containsSpeech')
const note = requireElement<HTMLTextAreaElement>('note')

recordButton.addEventListener('click', () => {
  if (recording) {
    void stopRecording()
  } else {
    void startRecording()
  }
})

labelForm.addEventListener('submit', (event) => {
  event.preventDefault()
  void saveCurrentClip()
})

window.addEventListener('online', () => {
  void flushQueue()
})

window.setInterval(() => {
  void flushQueue()
}, 15000)

void refreshQueueCount()
void flushQueue()

async function startRecording(): Promise<void> {
  setStatus('Requesting microphone...')
  currentClip = null
  formFields.disabled = true
  renderLevel({ peak: 0, rms: 0, clipped: false })

  try {
    await recorder.start(renderLevel)
    recording = true
    startedAt = Date.now()
    recordButton.classList.add('is-recording')
    recordButton.setAttribute('aria-pressed', 'true')
    recordButtonText.textContent = 'Stop'
    setStatus('Recording')
    timerId = window.setInterval(renderTimer, 250)
    renderTimer()
  } catch (error) {
    setStatus(error instanceof Error ? error.message : 'Could not start microphone')
  }
}

async function stopRecording(): Promise<void> {
  recordButton.disabled = true
  setStatus('Stopping...')
  try {
    currentClip = await recorder.stop()
    formFields.disabled = false
    setStatus(`Clip ready: ${formatDuration(currentClip.duration_s)}`)
  } catch (error) {
    setStatus(error instanceof Error ? error.message : 'Could not stop recording')
  } finally {
    recording = false
    recordButton.disabled = false
    recordButton.classList.remove('is-recording')
    recordButton.setAttribute('aria-pressed', 'false')
    recordButtonText.textContent = 'Record'
    if (timerId !== null) {
      window.clearInterval(timerId)
      timerId = null
    }
  }
}

async function saveCurrentClip(): Promise<void> {
  if (!currentClip) {
    setStatus('Record a clip first')
    return
  }
  if (!siteTag.value.trim()) {
    siteTag.focus()
    setStatus('SITE/MACHINE TAG is required')
    return
  }

  const payload: LabelPayload = {
    session_id: null,
    pcm_b64: currentClip.pcm_b64,
    machine_type: machineType.value,
    condition: condition.value,
    suspected_fault: '',
    contains_speech: containsSpeech.checked,
    note: note.value.trim(),
    site_tag: siteTag.value.trim(),
    client_recorded_at: currentClip.recorded_at,
    client_duration_s: currentClip.duration_s,
    client_peak_abs: currentClip.peak_abs,
    client_clipped: currentClip.clipped,
  }

  formFields.disabled = true
  setStatus('Saving...')
  try {
    await uploadLabel(payload)
    currentClip = null
    labelForm.reset()
    renderLevel({ peak: 0, rms: 0, clipped: false })
    setStatus('Saved')
    await flushQueue()
  } catch (error) {
    const message = error instanceof Error ? error.message : 'upload failed'
    await addQueuedUpload(payload, message)
    currentClip = null
    labelForm.reset()
    renderLevel({ peak: 0, rms: 0, clipped: false })
    setStatus('Offline: clip queued')
  } finally {
    await refreshQueueCount()
  }
}

async function flushQueue(): Promise<void> {
  const count = await countQueuedUploads()
  if (count === 0) {
    await refreshQueueCount()
    return
  }

  const result = await retryQueuedUploads(uploadLabel)
  await refreshQueueCount()
  if (result.saved > 0 && !recording && !currentClip) {
    setStatus(`Uploaded ${result.saved} queued clip${result.saved === 1 ? '' : 's'}`)
  }
}

async function refreshQueueCount(): Promise<void> {
  const count = await countQueuedUploads()
  queueCount.textContent = `Queue ${count}`
  queueCount.classList.toggle('has-queue', count > 0)
}

function renderLevel(level: InputLevel): void {
  const pct = Math.min(100, Math.max(2, Math.round(level.peak * 100)))
  meterBar.style.width = `${pct}%`
  meterBar.classList.toggle('is-hot', level.peak >= 0.75)
  meterBar.classList.toggle('is-clipped', level.clipped)
  clipFlag.classList.toggle('is-visible', level.clipped)
}

function renderTimer(): void {
  recordTime.textContent = formatDuration((Date.now() - startedAt) / 1000)
}

function formatDuration(seconds: number): string {
  const safe = Math.max(0, Math.floor(seconds))
  const mins = Math.floor(safe / 60)
  const secs = safe % 60
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
}

function setStatus(message: string): void {
  statusText.textContent = message
}

function requireElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id)
  if (!element) {
    throw new Error(`missing #${id}`)
  }
  return element as T
}

