import { describe, it, expect, vi } from "vitest";
import { Buffer } from "node:buffer";
import { EngineClient } from "../src/client.js";

/** Build a stub `fetch` returning a JSON 2xx response, capturing the call. */
function jsonOk(body: unknown, status = 200): typeof fetch {
  return vi.fn(async () =>
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    }),
  ) as unknown as typeof fetch;
}

/** Build a stub `fetch` returning a FastAPI-style error with `{detail}`. */
function jsonError(status: number, detail: string): typeof fetch {
  return vi.fn(async () =>
    new Response(JSON.stringify({ detail }), {
      status,
      statusText: "Error",
      headers: { "Content-Type": "application/json" },
    }),
  ) as unknown as typeof fetch;
}

const BASE = "http://localhost:8000";

describe("EngineClient — happy paths", () => {
  it("startSession POSTs the right URL/method/body and returns ok", async () => {
    const fetchFn = jsonOk({
      session_id: "abc",
      state: "CAPTURING_BASELINE",
      capture_remaining_s: 30,
    });
    const client = new EngineClient(BASE, fetchFn);

    const result = await client.startSession({
      mode: "session",
      tag: "pump-7",
      rpm: 1750,
    });

    expect(result).toEqual({
      kind: "ok",
      value: {
        session_id: "abc",
        state: "CAPTURING_BASELINE",
        capture_remaining_s: 30,
      },
    });

    const [url, init] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(url).toBe("http://localhost:8000/session/start");
    expect(init.method).toBe("POST");
    expect(init.headers).toEqual({ "Content-Type": "application/json" });
    expect(JSON.parse(init.body)).toEqual({
      mode: "session",
      tag: "pump-7",
      rpm: 1750,
    });
  });

  it("startSession defaults missing rpm to null in the body", async () => {
    const fetchFn = jsonOk({ session_id: "x", state: "LISTENING" });
    const client = new EngineClient(BASE, fetchFn);

    await client.startSession({ mode: "saved", tag: "saved-1" });

    const [, init] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(JSON.parse(init.body)).toEqual({
      mode: "saved",
      tag: "saved-1",
      rpm: null,
    });
  });

  it("score base64-encodes a known 4-byte PCM correctly and returns ok", async () => {
    const fetchFn = jsonOk({
      state: "ALERT",
      score: 0.9,
      percentile: 81,
      evidence_line: "impulse train ~88 Hz",
    });
    const client = new EngineClient(BASE, fetchFn);

    // int16 LE [1, -1] -> bytes 01 00 ff ff
    const pcm = new Uint8Array([0x01, 0x00, 0xff, 0xff]);
    const result = await client.score("sess-1", pcm);

    expect(result).toEqual({
      kind: "ok",
      value: {
        state: "ALERT",
        score: 0.9,
        percentile: 81,
        evidence_line: "impulse train ~88 Hz",
      },
    });

    const [url, init] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(url).toBe("http://localhost:8000/score");
    expect(init.method).toBe("POST");
    const body = JSON.parse(init.body);
    expect(body.session_id).toBe("sess-1");
    expect(body.pcm_b64).toBe(Buffer.from(pcm).toString("base64"));
  });

  it("label sends full request shape with defaults filled in", async () => {
    const fetchFn = jsonOk({
      wav_path: "labeled/clip.wav",
      json_path: "labeled/clip.json",
    });
    const client = new EngineClient(BASE, fetchFn);

    const pcm = new Uint8Array([0x10, 0x20]);
    const result = await client.label({
      session_id: "sess-2",
      machine_type: "pump",
      pcm,
    });

    expect(result).toEqual({
      kind: "ok",
      value: { wav_path: "labeled/clip.wav", json_path: "labeled/clip.json" },
    });

    const [url, init] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(url).toBe("http://localhost:8000/label");
    expect(JSON.parse(init.body)).toEqual({
      session_id: "sess-2",
      pcm_b64: Buffer.from(pcm).toString("base64"),
      machine_type: "pump",
      suspected_fault: "",
      contains_speech: false,
      note: "",
      site_tag: "",
    });
  });

  it("label defaults session_id to null when omitted", async () => {
    const fetchFn = jsonOk({ wav_path: "a", json_path: "b" });
    const client = new EngineClient(BASE, fetchFn);

    await client.label({ machine_type: "fan", pcm: new Uint8Array([1, 2]) });

    const [, init] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(JSON.parse(init.body).session_id).toBeNull();
  });

  it("captureStart POSTs tag and returns the capture id", async () => {
    const fetchFn = jsonOk({ capture_id: "cap-1" });
    const client = new EngineClient(BASE, fetchFn);

    const result = await client.captureStart("g2-characterization");

    expect(result).toEqual({
      kind: "ok",
      value: { capture_id: "cap-1" },
    });

    const [url, init] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(url).toBe("http://localhost:8000/capture/start");
    expect(init.method).toBe("POST");
    expect(init.headers).toEqual({ "Content-Type": "application/json" });
    expect(JSON.parse(init.body)).toEqual({ tag: "g2-characterization" });
  });

  it("captureAppend base64-encodes a known 4-byte PCM frame", async () => {
    const fetchFn = jsonOk({ bytes_total: 4, seconds_total: 0.000125 });
    const client = new EngineClient(BASE, fetchFn);

    const pcm = new Uint8Array([0x01, 0x00, 0xff, 0xff]);
    const result = await client.captureAppend("cap-1", pcm);

    expect(result).toEqual({
      kind: "ok",
      value: { bytes_total: 4, seconds_total: 0.000125 },
    });

    const [url, init] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(url).toBe("http://localhost:8000/capture/append");
    expect(init.method).toBe("POST");
    expect(init.headers).toEqual({ "Content-Type": "application/json" });
    expect(JSON.parse(init.body)).toEqual({
      capture_id: "cap-1",
      pcm_b64: Buffer.from(pcm).toString("base64"),
    });
  });

  it("captureStop POSTs capture id and returns the wav path", async () => {
    const fetchFn = jsonOk({
      wav_path: "captures/cap-1.wav",
      duration_s: 12.5,
    });
    const client = new EngineClient(BASE, fetchFn);

    const result = await client.captureStop("cap-1");

    expect(result).toEqual({
      kind: "ok",
      value: { wav_path: "captures/cap-1.wav", duration_s: 12.5 },
    });

    const [url, init] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(url).toBe("http://localhost:8000/capture/stop");
    expect(init.method).toBe("POST");
    expect(init.headers).toEqual({ "Content-Type": "application/json" });
    expect(JSON.parse(init.body)).toEqual({ capture_id: "cap-1" });
  });

  it("health GETs /health and returns ok", async () => {
    const fetchFn = jsonOk({
      status: "ok",
      backend: "torch",
      baselines: ["pump-7"],
      sessions: 2,
    });
    const client = new EngineClient(BASE, fetchFn);

    const result = await client.health();

    expect(result).toEqual({
      kind: "ok",
      value: { status: "ok", backend: "torch", baselines: ["pump-7"], sessions: 2 },
    });

    const [url, init] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(url).toBe("http://localhost:8000/health");
    expect(init.method).toBe("GET");
  });
});

describe("EngineClient — failure handling", () => {
  it("network rejection becomes {kind:'offline'}", async () => {
    const fetchFn = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    }) as unknown as typeof fetch;
    const client = new EngineClient(BASE, fetchFn);

    const result = await client.health();
    expect(result).toEqual({ kind: "offline" });
  });

  it("captureStart network rejection becomes {kind:'offline'}", async () => {
    const fetchFn = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    }) as unknown as typeof fetch;
    const client = new EngineClient(BASE, fetchFn);

    const result = await client.captureStart("g2-characterization");
    expect(result).toEqual({ kind: "offline" });
  });

  it("404 becomes http_error with FastAPI detail", async () => {
    const fetchFn = jsonError(404, "no saved baseline for tag 'pump-7'");
    const client = new EngineClient(BASE, fetchFn);

    const result = await client.startSession({ mode: "saved", tag: "pump-7" });
    expect(result).toEqual({
      kind: "http_error",
      status: 404,
      detail: "no saved baseline for tag 'pump-7'",
    });
  });

  it("404 from /capture/append becomes http_error with detail", async () => {
    const fetchFn = jsonError(404, "unknown capture_id");
    const client = new EngineClient(BASE, fetchFn);

    const result = await client.captureAppend("missing", new Uint8Array([1, 2]));
    expect(result).toEqual({
      kind: "http_error",
      status: 404,
      detail: "unknown capture_id",
    });
  });

  it("400 from /score becomes http_error with detail", async () => {
    const fetchFn = jsonError(400, "invalid base64 pcm");
    const client = new EngineClient(BASE, fetchFn);

    const result = await client.score("s", new Uint8Array([1, 2]));
    expect(result).toEqual({
      kind: "http_error",
      status: 400,
      detail: "invalid base64 pcm",
    });
  });

  it("non-2xx with non-JSON body falls back to statusText for detail", async () => {
    const fetchFn = vi.fn(async () =>
      new Response("Internal Server Error", {
        status: 500,
        statusText: "Internal Server Error",
      }),
    ) as unknown as typeof fetch;
    const client = new EngineClient(BASE, fetchFn);

    const result = await client.health();
    expect(result).toEqual({
      kind: "http_error",
      status: 500,
      detail: "Internal Server Error",
    });
  });

  it("trims a trailing slash from the base URL", async () => {
    const fetchFn = jsonOk({ status: "ok", backend: "x", baselines: [], sessions: 0 });
    const client = new EngineClient("http://localhost:8000/", fetchFn);

    await client.health();

    const [url] = (fetchFn as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(url).toBe("http://localhost:8000/health");
  });
});
