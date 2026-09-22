import type { LabelPayload, LabelResponse } from './types'

export async function uploadLabel(payload: LabelPayload): Promise<LabelResponse> {
  const response = await fetch('/label', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  if (!response.ok) {
    const body = await response.text()
    throw new Error(`upload failed (${response.status}): ${body || response.statusText}`)
  }

  return (await response.json()) as LabelResponse
}

