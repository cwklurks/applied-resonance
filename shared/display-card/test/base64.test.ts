import { describe, it, expect } from "vitest";
import { Buffer } from "node:buffer";
import { pcmToBase64 } from "../src/base64.js";

/** Decode our base64 back to bytes via Node's Buffer for round-trip checks. */
function decode(b64: string): Uint8Array {
  return new Uint8Array(Buffer.from(b64, "base64"));
}

function bytesEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

describe("pcmToBase64", () => {
  it("matches Node Buffer base64 for a known 4-byte PCM sample", () => {
    // int16 LE [1, -1] = bytes 01 00 ff ff
    const bytes = new Uint8Array([0x01, 0x00, 0xff, 0xff]);
    expect(pcmToBase64(bytes)).toBe(Buffer.from(bytes).toString("base64"));
  });

  it.each([0, 1, 3, 100_000])(
    "round-trips and matches Node Buffer for size %i (chunking edge)",
    (size) => {
      const bytes = new Uint8Array(size);
      for (let i = 0; i < size; i++) {
        bytes[i] = (i * 31 + 7) & 0xff; // deterministic, spans the full byte range
      }

      const encoded = pcmToBase64(bytes);

      // 1. Exactly matches Node's reference encoder.
      expect(encoded).toBe(Buffer.from(bytes).toString("base64"));

      // 2. Decodes back to the original bytes.
      expect(bytesEqual(decode(encoded), bytes)).toBe(true);
    },
  );

  it("encodes an empty array to an empty string", () => {
    expect(pcmToBase64(new Uint8Array([]))).toBe("");
  });
});
