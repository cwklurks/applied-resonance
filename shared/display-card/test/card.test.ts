import { describe, it, expect } from "vitest";
import { formatHudCard, HUD_MAX_CHARS } from "../src/card.js";

describe("formatHudCard — exact strings per state", () => {
  it("LISTENING", () => {
    expect(formatHudCard({ state: "LISTENING" })).toEqual({
      line1: "Listening...",
      line2: "",
    });
  });

  it("CAPTURING_BASELINE with remaining seconds (ceil)", () => {
    expect(
      formatHudCard({ state: "CAPTURING_BASELINE", captureRemainingS: 12.1 }),
    ).toEqual({
      line1: "Capturing baseline · 13s",
      line2: "keep the machine sounding normal",
    });
  });

  it("CAPTURING_BASELINE ceils a whole number unchanged", () => {
    expect(
      formatHudCard({ state: "CAPTURING_BASELINE", captureRemainingS: 30 }).line1,
    ).toBe("Capturing baseline · 30s");
  });

  it("CAPTURING_BASELINE without remaining", () => {
    expect(formatHudCard({ state: "CAPTURING_BASELINE" })).toEqual({
      line1: "Capturing baseline...",
      line2: "keep the machine sounding normal",
    });
  });

  it("CAPTURING_BASELINE with null remaining falls back to ellipsis", () => {
    expect(
      formatHudCard({ state: "CAPTURING_BASELINE", captureRemainingS: null }).line1,
    ).toBe("Capturing baseline...");
  });

  it("SUSPECT with percentile and evidence", () => {
    expect(
      formatHudCard({
        state: "SUSPECT",
        percentile: 74.6,
        evidenceLine: "impulse train ~88 Hz",
      }),
    ).toEqual({
      line1: "? Possible anomaly · 75%",
      line2: "impulse train ~88 Hz · tap to log",
    });
  });

  it("ALERT matches the product brief example exactly", () => {
    expect(
      formatHudCard({
        state: "ALERT",
        percentile: 81,
        evidenceLine: "impulse train ~88 Hz",
      }),
    ).toEqual({
      line1: "! Bearing-like anomaly · 81%",
      line2: "impulse train ~88 Hz · tap to log",
    });
  });

  it("ALERT without evidence falls back to bare tap-to-log", () => {
    expect(
      formatHudCard({ state: "ALERT", percentile: 81 }),
    ).toEqual({
      line1: "! Bearing-like anomaly · 81%",
      line2: "tap to log",
    });
  });

  it("SUSPECT with null evidence falls back to bare tap-to-log", () => {
    expect(
      formatHudCard({ state: "SUSPECT", percentile: 50, evidenceLine: null }).line2,
    ).toBe("tap to log");
  });

  it("LOGGED", () => {
    expect(formatHudCard({ state: "LOGGED" })).toEqual({
      line1: "logged ✓",
      line2: "",
    });
  });

  it("OFFLINE", () => {
    expect(formatHudCard({ state: "OFFLINE" })).toEqual({
      line1: "engine offline",
      line2: "check phone connection",
    });
  });
});

describe("formatHudCard — percentile edge cases", () => {
  it("rounds percentile", () => {
    expect(formatHudCard({ state: "ALERT", percentile: 80.5 }).line1).toBe(
      "! Bearing-like anomaly · 81%",
    );
  });

  it("omits the · N% suffix when percentile is NaN", () => {
    expect(formatHudCard({ state: "ALERT", percentile: NaN }).line1).toBe(
      "! Bearing-like anomaly",
    );
  });

  it("omits the · N% suffix when percentile is undefined", () => {
    expect(formatHudCard({ state: "SUSPECT" }).line1).toBe("? Possible anomaly");
  });

  it("omits the · N% suffix when percentile is null", () => {
    expect(formatHudCard({ state: "ALERT", percentile: null }).line1).toBe(
      "! Bearing-like anomaly",
    );
  });

  it("omits the · N% suffix when percentile is Infinity", () => {
    expect(
      formatHudCard({ state: "ALERT", percentile: Infinity }).line1,
    ).toBe("! Bearing-like anomaly");
  });
});

describe("formatHudCard — truncation to HUD_MAX_CHARS", () => {
  it("a 70-char evidence line still ends with the tap-to-log suffix and fits", () => {
    const longEvidence = "x".repeat(70);
    const card = formatHudCard({
      state: "ALERT",
      percentile: 81,
      evidenceLine: longEvidence,
    });
    expect(card.line2.length).toBeLessThanOrEqual(HUD_MAX_CHARS);
    expect(card.line2.endsWith(" · tap to log")).toBe(true);
  });

  it("never lets line1 exceed HUD_MAX_CHARS", () => {
    const card = formatHudCard({ state: "ALERT", percentile: 81 });
    expect(card.line1.length).toBeLessThanOrEqual(HUD_MAX_CHARS);
  });

  it("evidence that exactly fits is preserved verbatim", () => {
    const suffixLen = " · tap to log".length;
    const evidence = "y".repeat(HUD_MAX_CHARS - suffixLen);
    const card = formatHudCard({
      state: "SUSPECT",
      evidenceLine: evidence,
    });
    expect(card.line2).toBe(`${evidence} · tap to log`);
    expect(card.line2.length).toBe(HUD_MAX_CHARS);
  });
});

describe("formatHudCard — robustness", () => {
  it("does not throw on an unknown state", () => {
    // @ts-expect-error intentionally passing an invalid state
    expect(() => formatHudCard({ state: "WAT" })).not.toThrow();
  });
});
