/**
 * Browser- and Node-safe base64 encoding of raw bytes.
 *
 * We deliberately avoid Node's `Buffer` (absent in the WebView) and avoid
 * passing a huge array to `String.fromCharCode(...bytes)` (stack overflow on
 * large inputs). Bytes are converted to a binary string in fixed-size chunks,
 * then `btoa` produces standard base64. `btoa` exists in modern browsers and in
 * Node 16+, so the same path runs in both environments.
 */

const CHUNK_SIZE = 0x8000; // 32768 bytes per chunk keeps the apply() arg list small

/** Encode raw bytes to a standard base64 string (browser + Node safe). */
export function pcmToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.length; i += CHUNK_SIZE) {
    const chunk = bytes.subarray(i, i + CHUNK_SIZE);
    binary += String.fromCharCode.apply(null, chunk as unknown as number[]);
  }
  return btoa(binary);
}
