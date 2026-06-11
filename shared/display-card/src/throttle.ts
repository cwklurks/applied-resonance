/**
 * Rate-limits HUD updates to the product cap of at most one lens update every
 * `minIntervalMs` (2 s by default). Latest-wins: a card pushed inside the
 * interval is stored and flushed by the next `push`/`tick` once the interval
 * has elapsed. Identical consecutive cards are deduped so the lens never
 * re-renders the same thing.
 *
 * Deterministic by design: time comes from an injected `now`, and there is no
 * `setTimeout` inside — callers own the cadence (e.g. by calling `tick`).
 */

import type { HudCard } from "./card.js";

function sameCard(a: HudCard | null, b: HudCard): boolean {
  return a !== null && a.line1 === b.line1 && a.line2 === b.line2;
}

export class HudThrottle {
  private readonly minIntervalMs: number;
  private readonly now: () => number;

  /** Timestamp of the last actually-rendered card; null until first render. */
  private lastRenderedAt: number | null = null;
  /** The last card we rendered, for dedupe. */
  private lastRenderedCard: HudCard | null = null;
  /** A card that arrived too soon and is waiting to be flushed. */
  private pending: HudCard | null = null;

  constructor(minIntervalMs = 2000, now: () => number = () => Date.now()) {
    this.minIntervalMs = minIntervalMs;
    this.now = now;
  }

  /**
   * Render immediately if the interval has elapsed (or this is the first card),
   * otherwise store it as the pending latest-wins card.
   */
  push(card: HudCard, render: (card: HudCard) => void): void {
    if (sameCard(this.lastRenderedCard, card)) {
      // Identical to what's on the lens; drop any stale pending duplicate too.
      this.pending = null;
      return;
    }

    if (this.canRenderNow()) {
      this.doRender(card, render);
      return;
    }

    // Too soon: keep only the latest card.
    this.pending = card;
  }

  /** Flush a pending card if the interval has elapsed. Call periodically. */
  tick(render: (card: HudCard) => void): void {
    if (this.pending === null || !this.canRenderNow()) {
      return;
    }
    const card = this.pending;
    this.pending = null;

    if (sameCard(this.lastRenderedCard, card)) {
      return;
    }
    this.doRender(card, render);
  }

  private canRenderNow(): boolean {
    return (
      this.lastRenderedAt === null ||
      this.now() - this.lastRenderedAt >= this.minIntervalMs
    );
  }

  private doRender(card: HudCard, render: (card: HudCard) => void): void {
    this.lastRenderedAt = this.now();
    this.lastRenderedCard = card;
    this.pending = null;
    render(card);
  }
}
