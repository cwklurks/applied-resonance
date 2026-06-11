/**
 * HUD card formatting: the single source of truth for what two lines appear on
 * the glasses lens. Both the Even Hub app and the future MentraOS app render
 * the output of `formatHudCard`, so the wording cannot drift between shells.
 *
 * Product rule: at most two glanceable lines, each capped at `HUD_MAX_CHARS`.
 */

export type HudState =
  | "CAPTURING_BASELINE"
  | "LISTENING"
  | "SUSPECT"
  | "ALERT"
  | "LOGGED"
  | "OFFLINE";

export interface HudInput {
  state: HudState;
  percentile?: number | null; // 0-100
  evidenceLine?: string | null;
  captureRemainingS?: number | null;
}

export interface HudCard {
  line1: string;
  line2: string;
}

/** Per-line hard character cap. */
export const HUD_MAX_CHARS = 60;

const TAP_TO_LOG = "tap to log";
const TAP_SUFFIX = ` · ${TAP_TO_LOG}`;

/** Truncate a single line to at most `HUD_MAX_CHARS` characters. */
function clampLine(line: string): string {
  return line.length > HUD_MAX_CHARS ? line.slice(0, HUD_MAX_CHARS) : line;
}

/** A finite number is required to render the `· N%` suffix; NaN/null omit it. */
function isFiniteNumber(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/** Render the `· N%` suffix (rounded) when percentile is a finite number. */
function percentSuffix(percentile: number | null | undefined): string {
  return isFiniteNumber(percentile) ? ` · ${Math.round(percentile)}%` : "";
}

/**
 * Build the anomaly second line. The evidence text is truncated first so that
 * the trailing ` · tap to log` always survives intact within `HUD_MAX_CHARS`.
 */
function anomalyLine2(evidenceLine: string | null | undefined): string {
  const evidence = typeof evidenceLine === "string" ? evidenceLine : "";
  if (evidence.length === 0) {
    return TAP_TO_LOG;
  }
  const budget = HUD_MAX_CHARS - TAP_SUFFIX.length;
  const trimmedEvidence =
    evidence.length > budget ? evidence.slice(0, Math.max(0, budget)) : evidence;
  return `${trimmedEvidence}${TAP_SUFFIX}`;
}

/** Format a HUD input into the two glanceable lens lines. Never throws. */
export function formatHudCard(input: HudInput): HudCard {
  const { state, percentile, evidenceLine, captureRemainingS } = input;

  let line1: string;
  let line2: string;

  switch (state) {
    case "LISTENING": {
      line1 = "Listening...";
      line2 = "";
      break;
    }
    case "CAPTURING_BASELINE": {
      line1 = isFiniteNumber(captureRemainingS)
        ? `Capturing baseline · ${Math.ceil(captureRemainingS)}s`
        : "Capturing baseline...";
      line2 = "keep the machine sounding normal";
      break;
    }
    case "SUSPECT": {
      line1 = `? Possible anomaly${percentSuffix(percentile)}`;
      line2 = anomalyLine2(evidenceLine);
      break;
    }
    case "ALERT": {
      line1 = `! Bearing-like anomaly${percentSuffix(percentile)}`;
      line2 = anomalyLine2(evidenceLine);
      break;
    }
    case "LOGGED": {
      line1 = "logged ✓";
      line2 = "";
      break;
    }
    case "OFFLINE": {
      line1 = "engine offline";
      line2 = "check phone connection";
      break;
    }
    default: {
      // Defensive: unknown state should never crash the lens.
      line1 = "";
      line2 = "";
      break;
    }
  }

  return { line1: clampLine(line1), line2: clampLine(line2) };
}
