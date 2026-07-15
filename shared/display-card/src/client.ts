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
}

export interface BaselineResponse {
  exists: boolean;
}

// -------------------------------------------------------------- result ----

export type EngineResult<T> =
  | { kind: "ok"; value: T }
  | { kind: "offline" }
  | { kind: "http_error"; status: number; detail: string };

// -------------------------------------------------------------- client ----

const JSON_HEADERS = { "Content-Type": "application/json" } as const;
const DEFAULT_TIMEOUT_MS = 5_000;

export interface EngineClientOptions {
  /** Runtime-only secret. Callers must not persist or bundle this value. */
  bearerToken?: string;
  /** One deadline covering fetch plus response-body parsing. */
  timeoutMs?: number;
}

export class EngineClient {
  private readonly baseUrl: string;
  private readonly fetchFn: typeof fetch;
  private readonly bearerToken?: string;
  private readonly timeoutMs: number;

  constructor(
    baseUrl: string,
    fetchFn: typeof fetch = fetch,
    options: EngineClientOptions = {},
  ) {
    // Trim a single trailing slash so `${baseUrl}/health` never doubles up.
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.fetchFn = fetchFn;
    this.bearerToken = options.bearerToken || undefined;
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    if (!Number.isFinite(this.timeoutMs) || this.timeoutMs <= 0) {
      throw new TypeError("timeoutMs must be a positive finite number");
    }
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

  hasBaseline(tag: string): Promise<EngineResult<BaselineResponse>> {
    return this.request<BaselineResponse>(`/baselines/${encodeURIComponent(tag)}`, {
      method: "GET",
    });
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
    const controller = new AbortController();
    let timeout: ReturnType<typeof setTimeout>;
    const deadline = new Promise<never>((_, reject) => {
      timeout = setTimeout(() => {
        controller.abort();
        reject(new Error("engine request deadline exceeded"));
      }, this.timeoutMs);
    });

    let headers = init.headers;
    if (path !== "/health" && this.bearerToken) {
      headers = {
        ...(headers as Record<string, string> | undefined),
        Authorization: `Bearer ${this.bearerToken}`,
      };
    }
    const requestInit: RequestInit = {
      ...init,
      ...(headers === undefined ? {} : { headers }),
      signal: controller.signal,
    };

    try {
      // Some WebViews require native fetch to be invoked with the global object
      // as its receiver. Calling a stored fetch as `this.fetchFn(...)` binds the
      // EngineClient instance instead and can throw before any request is sent.
      const response = await Promise.race([
        this.fetchFn.call(globalThis, `${this.baseUrl}${path}`, requestInit),
        deadline,
      ]);

      if (!response.ok) {
        const detail = await Promise.race([readDetail(response), deadline]);
        return { kind: "http_error", status: response.status, detail };
      }

      const value = (await Promise.race([response.json(), deadline])) as T;
      return { kind: "ok", value };
    } catch {
      // Network/DNS/CORS failures, aborts, and invalid/late bodies are offline.
      return { kind: "offline" };
    } finally {
      clearTimeout(timeout!);
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
