import type { LabelPayload } from './types'

const DB_NAME = 'earsight-field-recorder'
const STORE_NAME = 'pending-labels'
const DB_VERSION = 1

export interface QueuedUpload {
  id?: number
  payload: LabelPayload
  createdAt: string
  lastError?: string
}

export async function addQueuedUpload(payload: LabelPayload, lastError?: string): Promise<number> {
  const db = await openDb()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    const store = tx.objectStore(STORE_NAME)
    const request = store.add({
      payload,
      createdAt: new Date().toISOString(),
      lastError,
    } satisfies QueuedUpload)
    request.onsuccess = () => resolve(Number(request.result))
    request.onerror = () => reject(request.error)
  })
}

export async function countQueuedUploads(): Promise<number> {
  const db = await openDb()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly')
    const request = tx.objectStore(STORE_NAME).count()
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
  })
}

export async function listQueuedUploads(): Promise<QueuedUpload[]> {
  const db = await openDb()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly')
    const request = tx.objectStore(STORE_NAME).getAll()
    request.onsuccess = () => resolve(request.result as QueuedUpload[])
    request.onerror = () => reject(request.error)
  })
}

export async function removeQueuedUpload(id: number): Promise<void> {
  const db = await openDb()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    const request = tx.objectStore(STORE_NAME).delete(id)
    request.onsuccess = () => resolve()
    request.onerror = () => reject(request.error)
  })
}

export async function retryQueuedUploads(
  upload: (payload: LabelPayload) => Promise<unknown>,
): Promise<{ tried: number; saved: number }> {
  const queued = await listQueuedUploads()
  let saved = 0
  for (const item of queued) {
    if (typeof item.id !== 'number') {
      continue
    }
    try {
      await upload(item.payload)
      await removeQueuedUpload(item.id)
      saved += 1
    } catch {
      break
    }
  }
  return { tried: queued.length, saved }
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION)
    request.onupgradeneeded = () => {
      const db = request.result
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id', autoIncrement: true })
      }
    }
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
  })
}

