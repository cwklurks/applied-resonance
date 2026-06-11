import { describe, it, expect, vi } from "vitest";
import { HudThrottle } from "../src/throttle.js";
import type { HudCard } from "../src/card.js";

function card(line1: string, line2 = ""): HudCard {
  return { line1, line2 };
}

/** A controllable clock for deterministic throttle tests. */
function clock(start = 0) {
  let t = start;
  return {
    now: () => t,
    advance: (ms: number) => {
      t += ms;
    },
    set: (ms: number) => {
      t = ms;
    },
  };
}

describe("HudThrottle", () => {
  it("renders the first push immediately", () => {
    const c = clock();
    const throttle = new HudThrottle(2000, c.now);
    const render = vi.fn();

    throttle.push(card("a"), render);

    expect(render).toHaveBeenCalledTimes(1);
    expect(render).toHaveBeenCalledWith(card("a"));
  });

  it("a second push at +500ms does NOT render but is stored", () => {
    const c = clock();
    const throttle = new HudThrottle(2000, c.now);
    const render = vi.fn();

    throttle.push(card("a"), render);
    render.mockClear();

    c.advance(500);
    throttle.push(card("b"), render);

    expect(render).not.toHaveBeenCalled();
  });

  it("tick at +2100ms flushes the stored (latest) card", () => {
    const c = clock();
    const throttle = new HudThrottle(2000, c.now);
    const render = vi.fn();

    throttle.push(card("a"), render); // renders at t=0
    c.advance(500);
    throttle.push(card("b"), render); // stored
    render.mockClear();

    c.set(2100);
    throttle.tick(render);

    expect(render).toHaveBeenCalledTimes(1);
    expect(render).toHaveBeenCalledWith(card("b"));
  });

  it("tick before the interval elapses does nothing", () => {
    const c = clock();
    const throttle = new HudThrottle(2000, c.now);
    const render = vi.fn();

    throttle.push(card("a"), render); // t=0
    c.advance(500);
    throttle.push(card("b"), render); // stored
    render.mockClear();

    c.set(1500);
    throttle.tick(render);

    expect(render).not.toHaveBeenCalled();
  });

  it("latest-wins when 3 pushes queue inside the interval", () => {
    const c = clock();
    const throttle = new HudThrottle(2000, c.now);
    const render = vi.fn();

    throttle.push(card("a"), render); // renders at t=0
    c.advance(200);
    throttle.push(card("b"), render); // stored
    c.advance(200);
    throttle.push(card("c"), render); // replaces stored
    c.advance(200);
    throttle.push(card("d"), render); // replaces stored
    render.mockClear();

    c.set(2100);
    throttle.tick(render);

    expect(render).toHaveBeenCalledTimes(1);
    expect(render).toHaveBeenCalledWith(card("d"));
  });

  it("dedupes identical consecutive cards (no render)", () => {
    const c = clock();
    const throttle = new HudThrottle(2000, c.now);
    const render = vi.fn();

    throttle.push(card("a", "x"), render); // renders
    render.mockClear();

    c.set(5000); // well past the interval
    throttle.push(card("a", "x"), render); // identical -> no render

    expect(render).not.toHaveBeenCalled();
  });

  it("renders a changed card immediately once the interval has elapsed", () => {
    const c = clock();
    const throttle = new HudThrottle(2000, c.now);
    const render = vi.fn();

    throttle.push(card("a"), render); // t=0
    render.mockClear();

    c.set(2500);
    throttle.push(card("b"), render);

    expect(render).toHaveBeenCalledTimes(1);
    expect(render).toHaveBeenCalledWith(card("b"));
  });

  it("a pending card identical to what was last rendered is dropped, not re-rendered", () => {
    const c = clock();
    const throttle = new HudThrottle(2000, c.now);
    const render = vi.fn();

    throttle.push(card("a"), render); // renders "a" at t=0
    c.advance(300);
    throttle.push(card("b"), render); // stored "b"
    c.advance(300);
    throttle.push(card("a"), render); // identical to last rendered -> clears pending
    render.mockClear();

    c.set(3000);
    throttle.tick(render);

    expect(render).not.toHaveBeenCalled();
  });

  it("tick is a no-op when nothing is pending", () => {
    const c = clock();
    const throttle = new HudThrottle(2000, c.now);
    const render = vi.fn();

    throttle.push(card("a"), render);
    render.mockClear();

    c.set(9999);
    throttle.tick(render);

    expect(render).not.toHaveBeenCalled();
  });
});
