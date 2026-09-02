/**
 * Cursor context-window inference + usage scaling (pi-cursor parity).
 *
 * Cursor's GetUsableModels catalog does not return window sizes. Clients
 * (Claude Code, Pi, etc.) need accurate total_tokens so auto-compact fires
 * before Cursor's actual enforced cap overflows.
 *
 * Scaling rule (pi-cursor README):
 *   when Cursor maxTokens < inferredWindow:
 *     total_tokens = round(usedTokens × inferredWindow / cursorMaxTokens)
 */

const CURSOR_DEFAULT_CONTEXT_WINDOW = (() => {
  const configured = Number(process.env.CURSOR_CONTEXT_WINDOW);
  return Number.isFinite(configured) && configured > 0
    ? Math.floor(configured)
    : 200_000;
})();

const CURSOR_USAGE_SCALING = process.env.CURSOR_USAGE_SCALING !== "0";

/**
 * Infer the client-facing context window for a Cursor model id.
 * Matches pi-cursor `inferContextWindow` family table.
 */
export function inferContextWindow(modelId) {
  const configured = Number(process.env.CURSOR_INFERRED_CONTEXT_WINDOW);
  if (Number.isFinite(configured) && configured > 0) return Math.floor(configured);

  const lower = String(modelId || "").toLowerCase();
  if (!lower) return CURSOR_DEFAULT_CONTEXT_WINDOW;

  if (lower.includes("-1m")) return 1_048_576;

  // Claude: Cursor typically enforces ~200k even when catalog implies 1M.
  if (lower.startsWith("claude-") || lower.includes("claude-fable-")) return 200_000;

  if (lower.startsWith("gemini-")) return 1_048_576;

  if (/^gpt-[0-9.]*-(nano|mini)/.test(lower)) return 128_000;
  if (lower.startsWith("gpt-5.5")) return 1_048_576;
  if (lower.startsWith("gpt-")) return 400_000;

  if (lower.startsWith("grok-")) return 256_000;
  if (lower.startsWith("kimi-")) return 262_144;

  return CURSOR_DEFAULT_CONTEXT_WINDOW;
}

export function getDefaultCursorContextWindow() {
  return CURSOR_DEFAULT_CONTEXT_WINDOW;
}

export function isCursorUsageScalingEnabled() {
  return CURSOR_USAGE_SCALING;
}

/**
 * Scale Cursor usedTokens so client compact thresholds fire relative to the
 * window the client believes the model has (inferred), not only Cursor's cap.
 */
export function scaleCursorUsageTokens(usedTokens, {
  cursorMaxTokens = 0,
  inferredWindow = 0,
  scaling = CURSOR_USAGE_SCALING,
} = {}) {
  const used = Number(usedTokens) || 0;
  if (!scaling || used <= 0) return used;
  const cursorWindow = Number(cursorMaxTokens) || 0;
  const piWindow = Number(inferredWindow) || 0;
  if (cursorWindow > 0 && piWindow > cursorWindow) {
    return Math.round(used * piWindow / cursorWindow);
  }
  return used;
}

/**
 * Build OpenAI-style usage from Cursor conversation token state.
 */
export function buildScaledCursorUsage({
  usedTokens = 0,
  outputTokens = 0,
  cursorMaxTokens = 0,
  modelId = "",
  inferredWindow = inferContextWindow(modelId),
  scaling = CURSOR_USAGE_SCALING,
} = {}) {
  const output = Math.max(0, Number(outputTokens) || 0);
  const rawUsed = Math.max(output, Number(usedTokens) || 0);
  if (!rawUsed) return null;
  const total = scaleCursorUsageTokens(rawUsed, {
    cursorMaxTokens,
    inferredWindow,
    scaling,
  });
  return {
    prompt_tokens: Math.max(0, total - output),
    completion_tokens: output,
    total_tokens: total,
  };
}
