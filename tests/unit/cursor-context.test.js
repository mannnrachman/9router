import { describe, it, expect } from "vitest";
import { tailOnlyCursorMessages } from "../../open-sse/executors/cursor.js";
import {
  inferContextWindow,
  scaleCursorUsageTokens,
  buildScaledCursorUsage,
} from "../../open-sse/services/cursorContext.js";

describe("cursorContext infer + scale", () => {
  it("infers family windows", () => {
    expect(inferContextWindow("claude-4.6-sonnet")).toBe(200_000);
    expect(inferContextWindow("gemini-2.5-pro")).toBe(1_048_576);
    expect(inferContextWindow("gpt-5.2")).toBe(400_000);
    expect(inferContextWindow("gpt-5.2-mini")).toBe(128_000);
    expect(inferContextWindow("grok-4")).toBe(256_000);
    expect(inferContextWindow("composer-1")).toBe(200_000);
  });

  it("scales used tokens when Cursor cap is tighter", () => {
    expect(scaleCursorUsageTokens(197_000, {
      cursorMaxTokens: 200_000,
      inferredWindow: 1_048_576,
      scaling: true,
    })).toBe(Math.round(197_000 * 1_048_576 / 200_000));
  });

  it("does not scale when Cursor window is equal or larger", () => {
    expect(scaleCursorUsageTokens(50_000, {
      cursorMaxTokens: 400_000,
      inferredWindow: 200_000,
      scaling: true,
    })).toBe(50_000);
  });

  it("builds OpenAI usage with scaled totals", () => {
    const usage = buildScaledCursorUsage({
      usedTokens: 197_000,
      outputTokens: 100,
      cursorMaxTokens: 200_000,
      modelId: "gemini-2.5-pro",
      scaling: true,
    });
    expect(usage.total_tokens).toBe(Math.round(197_000 * 1_048_576 / 200_000));
    expect(usage.completion_tokens).toBe(100);
    expect(usage.prompt_tokens).toBe(usage.total_tokens - 100);
  });
});

describe("tailOnlyCursorMessages", () => {
  it("keeps only the latest turn from a long history", () => {
    const long = Array.from({ length: 100 }, (_, i) => ([
      { role: "user", content: `u${i}` },
      { role: "assistant", content: `a${i}` },
    ])).flat();
    long.push({ role: "user", content: "latest" });
    const trimmed = tailOnlyCursorMessages(long);
    expect(trimmed.length).toBeLessThanOrEqual(4);
    expect(trimmed.some((m) => m.content === "latest")).toBe(true);
    expect(trimmed.some((m) => String(m.content).includes("[9router]"))).toBe(true);
  });
});
