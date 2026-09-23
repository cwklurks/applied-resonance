export const TARGET_SAMPLE_RATE = 16000

export interface RecordedClip {
  pcm_b64: string
  duration_s: number
  peak_abs: number
  clipped: boolean
  recorded_at: string
}

export interface LabelPayload {
  session_id: string | null
  pcm_b64: string
  machine_type: string
  condition: string
  suspected_fault: string
  contains_speech: boolean
  note: string
  site_tag: string
  client_recorded_at: string
  client_duration_s: number
  client_peak_abs: number
  client_clipped: boolean
}

export interface LabelResponse {
  wav_path: string
  json_path: string
}

