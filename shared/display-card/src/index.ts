/**
 * @earsight/display-card — the single source of truth for the glasses HUD.
 *
 * Consumed by both the Even Hub app and the future MentraOS app so the two
 * shells cannot drift in wording, throttling, or engine wire format.
 */

export {
  formatHudCard,
  HUD_MAX_CHARS,
  type HudState,
  type HudInput,
  type HudCard,
} from "./card.js";

export { HudThrottle } from "./throttle.js";

export { pcmToBase64 } from "./base64.js";

export {
  EngineClient,
  type EngineResult,
  type EngineState,
  type StartRequest,
  type StartResponse,
  type ScoreResponse,
  type LabelRequest,
  type LabelResponse,
  type HealthResponse,
} from "./client.js";
