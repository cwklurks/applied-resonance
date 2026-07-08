/**
 * HTTP client for the Earsight scoring engine (FastAPI). Mirrors the API in
 * `engine/serve.py`. The client NEVER throws: transport failures (network/DNS/
 * abort) become `{kind: "offline"}`, and non-2xx responses become
 * `{kind: "http_error", status, detail}` with `detail` lifted from FastAPI's
 * `{detail: ...}` body when present.
 */

import { pcmToBase64 } from "./base64.js";

// ----------------------------------------------------------- API types ----

export type EngineState =
  | "CAPTURING_BASELINE"
  | "LISTENING"
  | "SUSPECT"
  | "ALERT";

export interface StartRequest {
  mode: "session" | "saved" | "library";
  tag: string;
  rpm?: number | null;
}

export interface StartResponse {
  session_id: string;
  state: "CAPTURING_BASELINE" | "LISTENING";
  capture_remaining_s?: number;
}

export interface ScoreResponse {
  state: EngineState;
  score: number | null;
  percentile: number | null;
  evidence_line: string | null;
  capture_remaining_s?: number;
}

export interface LabelRequest {
  session_id?: string | null;
  machine_type: string;
  suspected_fault?: string;
  contains_speech?: boolean;
  note?: string;
  site_tag?: string;
}

export interface LabelResponse {
  wav_path: string;
  json_path: string;
}

export interface CaptureStartResponse {
  capture_id: string;
}

export interface CaptureAppendResponse {
  bytes_total: number;
  seconds_total: number;
}

export interface CaptureStopResponse {
  wav_path: string;
  duration_s: number;
}

export interface HealthResponse {
  status: string;
  backend: string;
  baselines: string[];
  sessions: number;
}

// -------------------------------------------------------------- result ----

export type EngineResult<T> =
  | { kind: "ok"; value: T }
  | { kind: "offline" }
  | { kind: "http_error"; status: number; detail: string };

// -------------------------------------------------------------- client ----

const JSON_HEADERS = { "Content-Type": "application/json" } as const;

export class EngineClient {
  private readonly baseUrl: string;
  private readonly fetchFn: typeof fetch;

  constructor(baseUrl: string, fetchFn: typeof fetch = fetch) {
    // Trim a single trailing slash so `${baseUrl}/health` never doubles up.
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.fetchFn = fetchFn;
  }

  startSession(req: StartRequest): Promise<EngineResult<StartResponse>> {
    return this.postJson<StartResponse>("/session/start", {
      mode: req.mode,
      tag: req.tag,
      rpm: req.rpm ?? null,
    });
  }

  score(sessionId: string, pcm: Uint8Array): Promise<EngineResult<ScoreResponse>> {
    return this.postJson<ScoreResponse>("/score", {
      session_id: sessionId,
      pcm_b64: pcmToBase64(pcm),
    });
  }

  label(
    req: LabelRequest & { pcm: Uint8Array },
  ): Promise<EngineResult<LabelResponse>> {
    return this.postJson<LabelResponse>("/label", {
      session_id: req.session_id ?? null,
      pcm_b64: pcmToBase64(req.pcm),
      machine_type: req.machine_type,
      suspected_fault: req.suspected_fault ?? "",
      contains_speech: req.contains_speech ?? false,
      note: req.note ?? "",
      site_tag: req.site_tag ?? "",
    });
  }

  captureStart(tag: string): Promise<EngineResult<CaptureStartResponse>> {
    return this.postJson<CaptureStartResponse>("/capture/start", { tag });
  }

  captureAppend(
    captureId: string,
    pcm: Uint8Array,
  ): Promise<EngineResult<CaptureAppendResponse>> {
    return this.postJson<CaptureAppendResponse>("/capture/append", {
      capture_id: captureId,
      pcm_b64: pcmToBase64(pcm),
    });
  }

  captureStop(captureId: string): Promise<EngineResult<CaptureStopResponse>> {
    return this.postJson<CaptureStopResponse>("/capture/stop", {
      capture_id: captureId,
    });
  }

  health(): Promise<EngineResult<HealthResponse>> {
    return this.request<HealthResponse>("/health", { method: "GET" });
  }

  // ------------------------------------------------------------ internals ----

  private postJson<T>(
    path: string,
    body: unknown,
  ): Promise<EngineResult<T>> {
    return this.request<T>(path, {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify(body),
    });
  }

  private async request<T>(
    path: string,
    init: RequestInit,
  ): Promise<EngineResult<T>> {
    let response: Response;
    try {
      response = await this.fetchFn(`${this.baseUrl}${path}`, init);
    } catch {
      // Network down, DNS failure, CORS rejection, aborted request, etc.
      return { kind: "offline" };
    }

    if (!response.ok) {
      const detail = await readDetail(response);
      return { kind: "http_error", status: response.status, detail };
    }

    try {
      const value = (await response.json()) as T;
      return { kind: "ok", value };
    } catch {
      // 2xx with an unparseable body is treated as a transport-level failure.
      return { kind: "offline" };
    }
  }
}

/** Lift FastAPI's `{detail: ...}` message from an error body when parseable. */
async function readDetail(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as unknown;
    if (
      body !== null &&
      typeof body === "object" &&
      "detail" in body &&
      typeof (body as { detail: unknown }).detail === "string"
    ) {
      return (body as { detail: string }).detail;
    }
    return response.statusText || `HTTP ${response.status}`;
  } catch {
    return response.statusText || `HTTP ${response.status}`;
  }
}
