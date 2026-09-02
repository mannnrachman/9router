export function sseChunk(data) {
  return `data: ${JSON.stringify(data)}\n\n`;
}

// Build OpenAI chat.completion.chunk SSE frame. Key order: id, object, created, model, choices[, usage].
export function chatChunkSse({ id, created, model, delta, finishReason = null, usage = null }) {
  const chunk = {
    id,
    object: "chat.completion.chunk",
    created,
    model,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  };
  if (usage && typeof usage === "object") chunk.usage = usage;
  return sseChunk(chunk);
}
