import { BaseExecutor } from "./base.js";
import { PROVIDERS, PROVIDER_OAUTH } from "../config/providers.js";
import { HTTP_STATUS } from "../config/runtimeConfig.js";
import {
  generateCursorBody,
  encodeField,
  wrapConnectRPCFrame,
  decodeMessage,
  parseConnectRPCFrame,
  extractTextFromResponse,
  encodeMcpTools,
  decodeMcpArgs,
  encodeMcpResultSuccess,
  decodeAgentKvServerEvent,
  encodeAgentKvGetResult,
  encodeAgentKvSetResult,
  encodeAgentHeartbeat,
} from "../utils/cursorProtobuf.js";
import {
  createAgentConversationRecord,
  validateConversationLineage,
  commitConversationTurn,
  resyncConversationLineageAfterCompact,
  archiveCheckpointTurns,
  shouldArchiveByTokenUsage,
  splitMessagesIntoTurns,
  buildTurnsTranscript,
  encodeTurnBlobFromParsedTurn,
  encodeConversationSummaryArchive,
  encodeConversationState,
  storeCursorBlob,
  pruneAgentBlobs,
  CURSOR_AGENT_CONVERSATION_DEFAULTS,
} from "../utils/cursorConversationState.js";
import {
  inferContextWindow,
  getDefaultCursorContextWindow,
  buildScaledCursorUsage,
  isCursorUsageScalingEnabled,
} from "../services/cursorContext.js";
import { resolveSessionId } from "../utils/sessionManager.js";
import { buildCursorHeaders } from "../utils/cursorChecksum.js";
import { estimateUsage } from "../utils/usageTracking.js";
import { SSE_DONE, SSE_HEADERS } from "../utils/sseConstants.js";
import { chatChunkSse, sseChunk } from "../utils/sse.js";
import { FORMATS } from "../translator/formats.js";
import { ROLE, OPENAI_BLOCK } from "../translator/schema/index.js";
import { proxyAwareFetch } from "../utils/proxyFetch.js";
import zlib from "zlib";
import crypto from "crypto";

// Detect cloud environment
const isCloudEnv = () => {
  if (typeof caches !== "undefined" && typeof caches === "object") return true;
  if (typeof EdgeRuntime !== "undefined") return true;
  return false;
};

// Lazy import http2 (only in Node.js environment)
let http2 = null;
if (!isCloudEnv()) {
  try {
    http2 = await import("http2");
  } catch {
    // http2 not available
  }
}

const COMPRESS_FLAG = {
  NONE: 0x00,
  GZIP: 0x01,
  TRAILER: 0x02,
  GZIP_TRAILER: 0x03
};

const AGENT_RUN_PATH = "/agent.v1.AgentService/Run";
const PROTOBUF_LEN = 2;
const PROTOBUF_VARINT = 0;

// ─── Retained-session + checkpoint subsystem ───────────────────────────
// Retains the AgentService h2 session across a tool-call turn so the tool
// result can be written to the SAME server-side conversation, and persists
// per-owner conversation state (checkpoint + KV blobs) so later turns resume
// instead of replaying the whole client history every request.

const CURSOR_AGENT_SESSION_TTL_MS = (() => {
  const value = Number(process.env.CURSOR_AGENT_SESSION_TTL_MS);
  return Number.isFinite(value) && value > 0 ? Math.floor(value) : 5 * 60 * 1000;
})();
const CURSOR_AGENT_STREAM_TIMEOUT_MS = Number(process.env.CURSOR_STREAM_TIMEOUT_MS || 300000);
const CURSOR_AGENT_HEARTBEAT_MS = Number(process.env.CURSOR_HEARTBEAT_MS || 5000);
const CURSOR_SSE_KEEPALIVE_MS = Number(process.env.CURSOR_SSE_KEEPALIVE_MS || 15000);
const CURSOR_AGENT_MAX_RETRIES = (() => {
  const value = Number(process.env.CURSOR_AGENT_MAX_RETRIES);
  return Number.isFinite(value) && value >= 0 ? Math.floor(value) : 2;
})();
const CURSOR_ARCHIVE_TOKEN_SOFT_RATIO = (() => {
  const value = Number(process.env.CURSOR_ARCHIVE_TOKEN_SOFT_RATIO);
  return Number.isFinite(value) && value > 0 && value < 1 ? value : 0.55;
})();
const CURSOR_CONTEXT_RESUME_RATIO = 0.70; // drop checkpoint when estimated tokens exceed this share of the window

const retainedAgentSessions = new Map(); // h2 session → { session, owner, pending, buffered, closed, timer }
const retainedAgentToolCalls = new Map(); // `${owner}\0${toolCallId}` → retained state
const agentConversations = new Map(); // owner → conversation record (see cursorConversationState.js)

// Owner key: model + account + client session. A different model, account, or
// client chat must never share a Cursor conversation or retained h2 session.
function agentSessionOwner(credentials, model, body = null) {
  const account = credentials?.connectionId
    || credentials?.id
    || credentials?.accessToken
    || credentials?.apiKey
    || "anonymous";
  const sessionId = resolveSessionId({
    headers: credentials?.rawHeaders,
    body,
    connectionId: account,
    scope: "cursor",
  });
  return crypto.createHash("sha256").update(`${model}\0${account}\0${sessionId}`).digest("hex");
}

function getAgentConversation(owner) {
  let conv = agentConversations.get(owner);
  if (!conv) {
    conv = createAgentConversationRecord();
    conv.tokenUsage = null;
    conv.effectiveContextWindow = null;
    conv.outputTokens = 0;
    agentConversations.set(owner, conv);
  }
  conv.lastAccessMs = Date.now();
  clearTimeout(conv.timer);
  conv.timer = setTimeout(() => {
    if (agentConversations.get(owner) === conv) agentConversations.delete(owner);
  }, CURSOR_AGENT_CONVERSATION_DEFAULTS.conversationTtlMs);
  conv.timer.unref?.();
  return conv;
}

// Discard checkpoint, blobs, and lineage; assign a new conversation ID.
// Used on lineage mismatch (edit/fork) and blob_not_found retry.
function resetAgentConversation(conv) {
  conv.conversationId = crypto.randomUUID();
  conv.checkpoint = null;
  conv.blobStore.clear();
  conv.turnCount = 0;
  conv.fingerprint = null;
  conv.userTexts = [];
  conv.tokenUsage = null;
  conv.effectiveContextWindow = null;
  conv.outputTokens = 0;
}

function extractUserTexts(body) {
  return (body?.messages || [])
    .filter((message) => message?.role === ROLE.USER)
    .map((message) => textFromContent(message?.content));
}

function getCursorContextWindow(modelId = "") {
  if (modelId) return inferContextWindow(modelId);
  return getDefaultCursorContextWindow();
}

export function estimateCursorContextTokens(messages, tools = []) {
  try {
    const body = { messages: Array.isArray(messages) ? messages : [] };
    if (Array.isArray(tools) && tools.length > 0) body.tools = tools;
    return Math.ceil(Buffer.byteLength(JSON.stringify(body), "utf8") / 4);
  } catch {
    return 0;
  }
}

// ─── Retained h2 session plumbing ──────────────────────────────────────

function normalizeAgentToolCallId(value) {
  return String(value || "").split("\n")[0].trim();
}

function retainedToolCallKey(owner, toolCallId) {
  return `${owner}\0${normalizeAgentToolCallId(toolCallId)}`;
}

function decodeXmlEntities(value) {
  return String(value || "")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&amp;/g, "&");
}

// Client tool results arrive inline as <tool_result> XML inside a user turn.
function extractCursorToolResults(body) {
  const results = [];
  for (const message of body?.messages || []) {
    const content = textFromContent(message?.content);
    const blocks = content.matchAll(/<tool_result>([\s\S]*?)<\/tool_result>/g);
    for (const block of blocks) {
      const value = block[1];
      const toolName = value.match(/<tool_name>([\s\S]*?)<\/tool_name>/)?.[1];
      const toolCallId = value.match(/<tool_call_id>([\s\S]*?)<\/tool_call_id>/)?.[1];
      const result = value.match(/<result>([\s\S]*?)<\/result>/)?.[1];
      if (toolName == null || toolCallId == null || result == null) continue;
      const isError = value.match(/<is_error>([\s\S]*?)<\/is_error>/)?.[1];
      results.push({
        toolName: decodeXmlEntities(toolName).trim(),
        toolCallId: normalizeAgentToolCallId(decodeXmlEntities(toolCallId)),
        content: decodeXmlEntities(result),
        isError: decodeXmlEntities(isError).trim() === "true",
      });
    }
  }
  return results;
}

function closeRetainedAgentSession(state) {
  if (!state || state.closed) return;
  state.closed = true;
  clearTimeout(state.timer);
  for (const toolCallId of state.pending.keys()) {
    const key = retainedToolCallKey(state.owner, toolCallId);
    if (retainedAgentToolCalls.get(key) === state) {
      retainedAgentToolCalls.delete(key);
    }
  }
  retainedAgentSessions.delete(state.session);
  try { state.session.close(); } catch {}
}

function retainAgentToolCall(session, owner, toolCallId, execRequest) {
  toolCallId = normalizeAgentToolCallId(toolCallId);
  let state = retainedAgentSessions.get(session);
  if (!state) {
    state = { session, owner, pending: new Map(), buffered: Buffer.alloc(0), closed: false, timer: null };
    retainedAgentSessions.set(session, state);
  }
  const key = retainedToolCallKey(owner, toolCallId);
  const previousState = retainedAgentToolCalls.get(key);
  if (previousState && previousState !== state) {
    closeRetainedAgentSession(previousState);
  }
  state.pending.set(toolCallId, {
    id: extractAgentVarint(execRequest, 1),
    execId: extractAgentString(execRequest, 15),
  });
  retainedAgentToolCalls.set(key, state);
  clearTimeout(state.timer);
  state.timer = setTimeout(() => closeRetainedAgentSession(state), CURSOR_AGENT_SESSION_TTL_MS);
  state.timer.unref?.();
  return state;
}

// A follow-up request resumes a retained session only when at least one tool
// result correlates to a pending call on that same session.
function acquireRetainedAgentSession(owner, toolResults) {
  if (!toolResults.length) return null;
  const matchedResults = toolResults.filter((result) =>
    retainedAgentToolCalls.has(retainedToolCallKey(owner, result.toolCallId))
  );
  if (!matchedResults.length) return null;
  const state = retainedAgentToolCalls.get(retainedToolCallKey(owner, matchedResults[0].toolCallId));
  if (!state || state.closed) return null;
  if (matchedResults.some((result) =>
    retainedAgentToolCalls.get(retainedToolCallKey(owner, result.toolCallId)) !== state
  )) return null;
  clearTimeout(state.timer);
  return { state, matchedResults };
}

function consumeRetainedToolResults(state, toolResults) {
  for (const result of toolResults) {
    const toolCallId = normalizeAgentToolCallId(result.toolCallId);
    const pending = state.pending.get(toolCallId);
    if (!pending) continue;
    state.session.write(createMcpResultResponse(pending, result.content, result.isError));
    state.pending.delete(toolCallId);
    const key = retainedToolCallKey(state.owner, toolCallId);
    if (retainedAgentToolCalls.get(key) === state) {
      retainedAgentToolCalls.delete(key);
    }
  }
}

function releaseRetainedAgentSession(state, keepOpen) {
  if (!state || state.closed) return;
  if (keepOpen && state.pending.size > 0) {
    state.timer = setTimeout(() => closeRetainedAgentSession(state), CURSOR_AGENT_SESSION_TTL_MS);
    state.timer.unref?.();
    return;
  }
  closeRetainedAgentSession(state);
}

// ─── Retry classification ──────────────────────────────────────────────

function classifyCursorError(message) {
  const text = String(message || "");
  if (/blob not found/i.test(text)) return "blob_not_found";
  if (text === "empty_turn" || /empty[_ ](?:turn|response)/i.test(text)) return "empty_turn";
  if (text === "context_exhausted" || /context[_ ]exhausted/i.test(text)) return "context_exhausted";
  if (/resource_exhausted/i.test(text)) return "resource_exhausted";
  if (text === "timeout" || /stream timeout/i.test(text)) return "timeout";
  return null;
}

function mapCursorAgentErrorResponse(error) {
  const hint = error?.retryHint || classifyCursorError(error?.message || "");
  if (hint === "resource_exhausted") {
    return {
      status: HTTP_STATUS.RATE_LIMITED,
      type: "rate_limit_error",
      code: "rate_limit_exceeded",
      message: error?.message || "resource_exhausted",
    };
  }
  if (hint === "timeout") {
    return {
      status: HTTP_STATUS.GATEWAY_TIMEOUT,
      type: "server_error",
      code: "gateway_timeout",
      message: error?.message || "timeout",
    };
  }
  if (hint === "empty_turn") {
    return {
      status: HTTP_STATUS.BAD_REQUEST,
      type: "api_error",
      code: "empty_response",
      message: error?.message || "Cursor AgentService returned an empty turn",
    };
  }
  if (hint === "context_exhausted") {
    return {
      status: HTTP_STATUS.SERVICE_UNAVAILABLE,
      type: "api_error",
      code: "context_exhausted",
      message: error?.message || "Cursor session context too large; start a new chat or run /compact in the IDE",
    };
  }
  return {
    status: HTTP_STATUS.SERVER_ERROR,
    type: "connection_error",
    code: "",
    message: error?.message || "Cursor AgentService request failed",
  };
}

function sleepMs(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function retryDelayMs(hint) {
  let base;
  switch (hint) {
    case "blob_not_found":
    case "empty_turn": base = 200; break;
    case "resource_exhausted": base = 2000; break;
    case "timeout": base = 3000; break;
    default: base = 1000; break;
  }
  return Math.round(base * (1 + Math.random() * 0.5));
}

function concatBuffers(...parts) {
  const length = parts.reduce((total, part) => total + part.length, 0);
  const result = new Uint8Array(length);
  let offset = 0;
  for (const part of parts) {
    result.set(part, offset);
    offset += part.length;
  }
  return result;
}

const agentString = (field, value) => encodeField(field, PROTOBUF_LEN, value);
const agentMessage = (field, value) => encodeField(field, PROTOBUF_LEN, value);
const agentBool = (field, value) => encodeField(field, PROTOBUF_VARINT, value ? 1 : 0);

function textFromContent(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((part) => part?.type === OPENAI_BLOCK.TEXT && typeof part.text === "string")
    .map((part) => part.text)
    .join("\n");
}

function isTextPart(part) {
  return !part || part.type === OPENAI_BLOCK.TEXT || typeof part === "string";
}

export function isAgentCapableRequest(body) {
  // ChatService rejects auto/composer and most thinking variants. AgentService
  // can answer text turns (including declared tool schemas) and tool-call
  // history. Image parts still need the legacy protobuf path.
  if (!Array.isArray(body?.messages) || body.messages.length === 0) return false;
  return body.messages.every((message) => {
    if (Array.isArray(message?.content)) return message.content.every(isTextPart);
    return message?.content == null || typeof message.content === "string";
  });
}

function encodeHistoryMessage(message) {
  const content = textFromContent(message?.content);
  const extras = [];
  if (message?.role === ROLE.ASSISTANT && message.tool_calls?.length) {
    for (const tc of message.tool_calls) {
      extras.push(`[tool_call id=${tc.id || ""} name=${tc.function?.name || "tool"} args=${tc.function?.arguments || "{}"}]`);
    }
  }
  if (message?.role === ROLE.TOOL) {
    extras.push(`[tool_result id=${message.tool_call_id || ""}]`);
  }
  const textBody = [content, ...extras].filter(Boolean).join("\n");
  if (!textBody) return null;

  // ConversationHistoryMessage.user / .assistant -> repeated content -> text.
  const text = agentString(1, textBody);
  if (message.role === ROLE.ASSISTANT) {
    return agentMessage(2, agentMessage(1, agentMessage(1, text)));
  }
  return agentMessage(1, agentMessage(1, agentMessage(1, text)));
}

export function buildAgentRunFrame(messages, model, tools = [], reasoningEffort = null, resume = null, { omitTools = false } = {}) {
  // resume = { conversationId, checkpoint, blobStore, tokenUsage } — continue
  // stored server-side state instead of replaying the whole history inline.
  const blobStore = resume?.blobStore || null;
  let conversationState = resume?.checkpoint || null;
  const conversationId = resume?.conversationId || null;
  const contextWindow = getCursorContextWindow(model);
  if (conversationState && estimateCursorContextTokens(messages, tools) > Math.floor(contextWindow * CURSOR_CONTEXT_RESUME_RATIO)) {
    conversationState = null;
  }

  // custom_system_prompt (RunRequest field 8) makes AgentService return an
  // empty turn. Fold system text into the current user message instead.
  const system = messages
    .filter((message) => message?.role === ROLE.SYSTEM)
    .map((message) => textFromContent(message.content))
    .filter(Boolean)
    .join("\n\n");
  const chatMessages = messages.filter((message) => message?.role !== ROLE.SYSTEM);
  const currentIndex = [...chatMessages].map((message) => message?.role).lastIndexOf(ROLE.USER);
  const current = currentIndex >= 0 ? chatMessages[currentIndex] : chatMessages.at(-1);
  const priorMessages = chatMessages.slice(0, currentIndex >= 0 ? currentIndex : -1);

  // Token-pressure pre-archive: shrink the checkpoint before resuming it.
  if (conversationState && blobStore) {
    const tokenForce = shouldArchiveByTokenUsage(resume?.tokenUsage, CURSOR_ARCHIVE_TOKEN_SOFT_RATIO)
      || shouldArchiveByTokenUsage(resume?.tokenUsage);
    conversationState = archiveCheckpointTurns(conversationState, blobStore, {
      force: tokenForce,
      turnThreshold: tokenForce
        ? Math.max(5, Math.floor(CURSOR_AGENT_CONVERSATION_DEFAULTS.turnArchiveThreshold / 2))
        : CURSOR_AGENT_CONVERSATION_DEFAULTS.turnArchiveThreshold,
    });
    if (tokenForce && conversationState === resume?.checkpoint) {
      conversationState = null;
    }
  } else if (!conversationState && blobStore && priorMessages.length > 0) {
    // No checkpoint yet but the client already has a long history: bootstrap a
    // ConversationState from real turn blobs + one summary archive so the
    // server resumes rather than re-reading the whole inline history.
    const turns = splitMessagesIntoTurns(priorMessages, { textFromContent });
    if (turns.length > CURSOR_AGENT_CONVERSATION_DEFAULTS.turnArchiveThreshold) {
      const older = turns.slice(0, turns.length - CURSOR_AGENT_CONVERSATION_DEFAULTS.turnArchiveThreshold);
      const recent = turns.slice(-CURSOR_AGENT_CONVERSATION_DEFAULTS.turnArchiveThreshold);
      const archiveBlobId = storeCursorBlob(
        encodeConversationSummaryArchive({
          summarizedMessages: [],
          summary: buildTurnsTranscript(older),
          windowTail: older.length,
        }),
        blobStore,
      );
      const turnBlobIds = recent.map((turn) => encodeTurnBlobFromParsedTurn(turn, blobStore));
      pruneAgentBlobs(blobStore);
      conversationState = encodeConversationState({
        turns: turnBlobIds,
        summaryArchives: [archiveBlobId],
      });
    }
  }

  // With conversation_state the server resumes from blobs; only replay inline
  // history when we have no serialized state yet.
  const history = !conversationState
    ? priorMessages.map(encodeHistoryMessage).filter(Boolean)
    : [];
  const rawUser = textFromContent(current?.content) || "Continue.";
  const userText = system ? `${system}\n\n${rawUser}` : rawUser;

  // agent.v1.UserMessageAction.user_message and its optional history.
  // selected_context (3) + mode=1 (4) match cursor-agent's wire format; without
  // them the server may accept the RPC and stream an empty turn.
  const userMessage = concatBuffers(
    agentString(1, userText),
    agentString(2, crypto.randomUUID()),
    agentMessage(3, new Uint8Array()),
    encodeField(4, PROTOBUF_VARINT, 1),
  );
  const conversationHistory = history.length
    ? concatBuffers(...history.map((entry) => agentMessage(1, entry)))
    : null;
  const userAction = concatBuffers(
    agentMessage(1, userMessage),
    ...(conversationHistory ? [agentMessage(7, conversationHistory)] : []),
  );
  const conversationAction = agentMessage(1, userAction);
  const effort = String(reasoningEffort || "").toLowerCase();
  const effortValue = effort === "ultra" ? "max" : effort === "minimal" ? "low" : effort;
  const parameter = effortValue && effortValue !== "none"
    ? agentMessage(3, concatBuffers(
      agentString(1, /gpt-/i.test(model) ? "reasoning" : "effort"),
      agentString(2, effortValue),
    ))
    : null;
  const requestedModel = concatBuffers(
    agentString(1, model), agentBool(7, true),
    ...(parameter ? [parameter] : []),
  );
  // ModelDetails (field 3): thinking variants (Composer, Grok, *-thinking)
  // return an empty turn when only RequestedModel (field 9) is set.
  const modelDetails = concatBuffers(
    agentString(1, model),
    agentString(3, model),
    agentString(4, model),
  );
  const mcpTools = omitTools ? new Uint8Array() : encodeMcpTools(tools);
  const runRequest = concatBuffers(
    // An empty ConversationStateStructure starts a fresh local agent session;
    // a stored checkpoint resumes the server-side conversation.
    agentMessage(1, conversationState || new Uint8Array()),
    agentMessage(2, conversationAction),
    agentMessage(3, modelDetails),
    ...(mcpTools.length ? [agentMessage(4, mcpTools)] : []),
    ...(conversationId ? [agentString(5, conversationId)] : []),
    agentMessage(9, requestedModel),
  );

  // agent.v1.AgentClientMessage.run_request.
  return wrapConnectRPCFrame(agentMessage(1, runRequest));
}

function extractAgentString(message, field) {
  const value = message?.get(field)?.[0]?.value;
  return value ? Buffer.from(value).toString("utf8") : "";
}

function extractAgentVarint(message, field) {
  const value = message?.get(field)?.[0]?.value;
  return typeof value === "number" ? value : null;
}

// MCP tool result written back onto the retained AgentService session
// (ExecClientMessage.mcp_result), correlated by exec id / exec message id.
function createMcpResultResponse(pending, content, isError = false) {
  const mcpResult = encodeMcpResultSuccess({ textItems: [String(content ?? "")], isError });
  const execClientMessage = concatBuffers(
    ...(pending.id != null ? [encodeField(1, PROTOBUF_VARINT, pending.id)] : []),
    agentMessage(11, mcpResult),
    ...(pending.execId ? [agentMessage(15, pending.execId)] : []),
  );
  return wrapConnectRPCFrame(agentMessage(2, execClientMessage));
}

function decodeAgentFrames(buffer, onFrame) {
  let pending = Buffer.from(buffer || []);
  while (pending.length >= 5) {
    const flags = pending[0];
    const length = pending.readUInt32BE(1);
    if (pending.length < 5 + length) break;
    let payload = pending.subarray(5, 5 + length);
    pending = pending.subarray(5 + length);
    if (flags & COMPRESS_FLAG.GZIP) {
      payload = zlib.gunzipSync(payload);
    }
    if (!(flags & COMPRESS_FLAG.TRAILER)) onFrame(payload);
  }
  return pending;
}

function execIds(execRequest) {
  const id = Number(execRequest?.get(1)?.[0]?.value || 0);
  const execId = extractAgentString(execRequest, 15);
  return { id, execId };
}

function wrapExecClientMessage(execMsgId, execId, resultField, resultPayload) {
  const parts = [];
  if (execMsgId) parts.push(encodeField(1, PROTOBUF_VARINT, execMsgId));
  parts.push(agentString(15, execId || ""));
  parts.push(encodeField(resultField, PROTOBUF_LEN, resultPayload || new Uint8Array()));
  return wrapConnectRPCFrame(agentMessage(2, concatBuffers(...parts)));
}

function createRequestContextResponse(execRequest) {
  // Tools already go out on AgentRunRequest.mcp_tools. Echoing them again on
  // this ack makes AgentService stall silently (0 SSE bytes until abort).
  const { id, execId } = execIds(execRequest);
  const requestContextSuccess = agentMessage(1, new Uint8Array());
  const requestContextResult = agentMessage(1, requestContextSuccess);
  return wrapExecClientMessage(id, execId, 10, requestContextResult);
}

// ExecServerMessage variant → ExecClientMessage result field (same numbers).
const EXEC_RESULT_FIELD = {
  2: 2, 3: 3, 4: 4, 5: 5, 7: 7, 8: 8, 9: 9, 16: 16, 20: 20, 23: 23,
};

function rejectExecRequest(execRequest) {
  const { id, execId } = execIds(execRequest);
  const variant = [...(execRequest?.keys?.() || [])].find((field) => field !== 1 && field !== 15);
  const resultField = EXEC_RESULT_FIELD[variant];
  if (!resultField) return null;
  // Diagnostics has no rejected variant — empty success unblocks the stream.
  if (variant === 9) return wrapExecClientMessage(id, execId, 9, new Uint8Array());
  const rejected = agentMessage(2, agentString(2, "Tool not available in this environment. Use the MCP tools provided instead."));
  return wrapExecClientMessage(id, execId, resultField, rejected);
}

const CURSOR_STREAM_DEBUG = process.env.CURSOR_STREAM_DEBUG === "1";
const debugLog = (...args) => {
  if (CURSOR_STREAM_DEBUG) console.log(...args);
};

function isComposerModel(model) {
  const modelId = String(model || "").split("/").pop();
  return /^composer(?:-|$)/i.test(modelId);
}

function visibleComposerContentFromThinking(thinking) {
  if (!thinking) return "";
  const endTag = "</think>";
  const endIdx = thinking.lastIndexOf(endTag);
  if (endIdx < 0) return "";
  return thinking.slice(endIdx + endTag.length).trimStart();
}

function decompressPayload(payload, flags) {
  // Check if payload is JSON error (starts with {"error")
  if (payload.length > 10 && payload[0] === 0x7b && payload[1] === 0x22) {
    try {
      const text = payload.toString("utf-8");
      if (text.startsWith('{"error"')) {
        debugLog(`[DECOMPRESS] Detected JSON error, skipping decompression`);
        return payload;
      }
    } catch {}
  }

  if (
    flags === COMPRESS_FLAG.GZIP ||
    flags === COMPRESS_FLAG.TRAILER ||
    flags === COMPRESS_FLAG.GZIP_TRAILER
  ) {
    // Primary: try gzip decompression (standard gzip header 0x1f 0x8b)
    try {
      return zlib.gunzipSync(payload);
    } catch (gzipErr) {
      // Fallback: TRAILER and GZIP_TRAILER frames sometimes use raw zlib deflate format
      try {
        return zlib.inflateSync(payload);
      } catch (deflateErr) {
        // Last resort: try raw deflate (no zlib header)
        try {
          return zlib.inflateRawSync(payload);
        } catch (rawErr) {
          debugLog(
            `[DECOMPRESS ERROR] flags=${flags}, payloadSize=${payload.length}, gzip=${gzipErr.message}, deflate=${deflateErr.message}, raw=${rawErr.message}`
          );
          debugLog(
            `[DECOMPRESS ERROR] First 50 bytes (hex):`,
            payload.slice(0, 50).toString("hex")
          );
          return payload;
        }
      }
    }
  }
  return payload;
}

// Read one cursor protobuf frame: header + bounds + decompress. Returns status + payload + new offset.
function readCursorFrame(buffer, offset, frameNum, tag) {
  if (offset + 5 > buffer.length) {
    debugLog(`[CURSOR BUFFER${tag}] Reached end, offset=${offset}, remaining=${buffer.length - offset}`);
    return { status: "done" };
  }

  const flags = buffer[offset];
  const length = buffer.readUInt32BE(offset + 1);
  debugLog(`[CURSOR BUFFER${tag}] Frame ${frameNum + 1}: flags=0x${flags.toString(16).padStart(2, "0")}, length=${length}`);

  if (offset + 5 + length > buffer.length) {
    debugLog(`[CURSOR BUFFER${tag}] Incomplete frame, offset=${offset}, length=${length}, buffer.length=${buffer.length}`);
    return { status: "done" };
  }

  let payload = buffer.slice(offset + 5, offset + 5 + length);
  const newOffset = offset + 5 + length;
  payload = decompressPayload(payload, flags);
  if (!payload) {
    debugLog(`[CURSOR BUFFER${tag}] Frame ${frameNum + 1}: decompression failed, skipping`);
    return { status: "skip", offset: newOffset };
  }
  return { status: "ok", payload, offset: newOffset };
}

function createErrorResponse(jsonError) {
  const errorMsg = jsonError?.error?.details?.[0]?.debug?.details?.title
    || jsonError?.error?.details?.[0]?.debug?.details?.detail
    || jsonError?.error?.message
    || "API Error";
  
  const isRateLimit = jsonError?.error?.code === "resource_exhausted";
  
  return new Response(JSON.stringify({
    error: {
      message: errorMsg,
      type: isRateLimit ? "rate_limit_error" : "api_error",
      code: jsonError?.error?.details?.[0]?.debug?.error || "unknown"
    }
  }), {
    status: isRateLimit ? HTTP_STATUS.RATE_LIMITED : HTTP_STATUS.BAD_REQUEST,
    headers: { "Content-Type": "application/json" }
  });
}

export class CursorExecutor extends BaseExecutor {
  constructor() {
    super("cursor", PROVIDERS.cursor);
  }

  buildUrl() {
    return `${this.config.baseUrl}${this.config.chatPath}`;
  }

  buildHeaders(credentials) {
    const accessToken = credentials.accessToken;
    const machineId = credentials.providerSpecificData?.machineId;
    const ghostMode = credentials.providerSpecificData?.ghostMode !== false;

    if (!machineId) {
      throw new Error("Machine ID is required for Cursor API");
    }

    return buildCursorHeaders(accessToken, machineId, ghostMode);
  }

  transformRequest(model, body, stream, credentials) {
    // Messages are already translated by chatCore (claude→openai→cursor)
    // Do NOT call openaiToCursorRequest again — double-translation drops tool_results
    const messages = body.messages || [];
    const tools = body.tools || [];
    const reasoningEffort = body.reasoning_effort || null;
    // Detect Claude Code UA to force Agent mode (issue #643)
    const ua = credentials?.rawHeaders?.["user-agent"] || "";
    const forceAgentMode = ua.includes("claude-cli") || ua.includes("claude-code") || ua.includes("Claude Code");
    return generateCursorBody(messages, model, tools, reasoningEffort, forceAgentMode);
  }

  async makeFetchRequest(url, headers, body, signal, proxyOptions = null) {
    const response = await proxyAwareFetch(url, {
      method: "POST",
      headers,
      body,
      signal
    }, proxyOptions);

    return {
      status: response.status,
      headers: Object.fromEntries(response.headers.entries()),
      body: Buffer.from(await response.arrayBuffer())
    };
  }

  makeHttp2Request(url, headers, body, signal) {
    if (!http2) {
      throw new Error("http2 module not available");
    }

    const HTTP2_TIMEOUT_MS = 60000; // 60s max — prevent hung sessions

    return new Promise((resolve, reject) => {
      const urlObj = new URL(url);
      const client = http2.connect(`https://${urlObj.host}`);
      const chunks = [];
      let responseHeaders = {};
      let settled = false;

      // Ensure client is always closed on settle
      const finish = (fn) => (...args) => {
        if (settled) return;
        settled = true;
        clearTimeout(hangTimeout);
        client.close();
        fn(...args);
      };

      // Hard timeout: close session if server never responds
      const hangTimeout = setTimeout(finish(() => {
        reject(new Error("HTTP/2 request timed out"));
      }), HTTP2_TIMEOUT_MS);

      client.on("error", finish(reject));

      const req = client.request({
        ":method": "POST",
        ":path": urlObj.pathname,
        ":authority": urlObj.host,
        ":scheme": "https",
        ...headers
      });

      req.on("response", (hdrs) => { responseHeaders = hdrs; });
      req.on("data", (chunk) => { chunks.push(chunk); });
      req.on("end", finish(() => {
        resolve({
          status: responseHeaders[":status"],
          headers: responseHeaders,
          body: Buffer.concat(chunks)
        });
      }));
      req.on("error", finish(reject));

      if (signal) {
        const onAbort = finish(() => reject(new Error("Request aborted")));
        signal.addEventListener("abort", onAbort, { once: true });
      }

      req.write(body);
      req.end();
    });
  }

  /**
   * AgentService (agent.api5.cursor.sh) is HTTP/2-only. Node's fetch/undici speaks
   * HTTP/1.1 and fails with HTTPParserError on the h2 preface — use http2 duplex.
   */
  openAgentHttp2Stream(url, headers, signal) {
    if (!http2) {
      throw new Error("HTTP/2 is required for Cursor AgentService (endpoint is h2-only)");
    }

    const urlObj = new URL(url);
    const client = http2.connect(`https://${urlObj.host}`);
    const chunkQueue = [];
    let waiting = null;
    let ended = false;
    let streamError = null;
    let req = null;

    const wake = (result) => {
      if (!waiting) return;
      const resolve = waiting;
      waiting = null;
      resolve(result);
    };

    const fail = (error) => {
      if (streamError) return;
      streamError = error;
      ended = true;
      wake(null);
    };

    const close = () => {
      try { req?.destroy(); } catch {}
      try { client.close(); } catch {}
    };

    client.on("error", fail);

    req = client.request({
      ":method": "POST",
      ":path": urlObj.pathname,
      ":authority": urlObj.host,
      ":scheme": "https",
      ...headers,
    });

    req.on("error", fail);
    req.on("data", (chunk) => {
      if (waiting) wake({ value: chunk, done: false });
      else chunkQueue.push(chunk);
    });
    req.on("end", () => {
      ended = true;
      wake({ value: undefined, done: true });
    });

    if (signal) {
      const onAbort = () => {
        fail(new Error("Request aborted"));
        close();
      };
      if (signal.aborted) onAbort();
      else signal.addEventListener("abort", onAbort, { once: true });
    }

    const responseHeaders = new Promise((resolve, reject) => {
      const onEarlyError = (error) => reject(error);
      client.once("error", onEarlyError);
      req.once("error", onEarlyError);
      req.once("response", (hdrs) => {
        client.off("error", onEarlyError);
        req.off("error", onEarlyError);
        resolve(hdrs);
      });
    });

    return {
      responseHeaders,
      write(frame) {
        if (req && !req.destroyed) req.write(Buffer.from(frame));
      },
      end() {
        try { if (req && !req.destroyed) req.end(); } catch {}
      },
      close,
      async read() {
        if (chunkQueue.length) return { value: chunkQueue.shift(), done: false };
        if (ended) {
          if (streamError) throw streamError;
          return { value: undefined, done: true };
        }
        const result = await new Promise((resolve) => { waiting = resolve; });
        if (streamError) throw streamError;
        return result || { value: undefined, done: true };
      },
    };
  }

  async executeAgent({ model, body, stream, credentials, signal, log }) {
    const agentEndpoint = PROVIDER_OAUTH.cursor?.agentEndpoint;
    if (!agentEndpoint) throw new Error("Cursor AgentService endpoint is not configured");

    const url = `${agentEndpoint}${AGENT_RUN_PATH}`;
    const headers = this.buildHeaders(credentials);
    const requestController = new AbortController();
    if (signal?.addEventListener) {
      if (signal.aborted) requestController.abort(signal.reason);
      else signal.addEventListener("abort", () => requestController.abort(signal.reason), { once: true });
    }

    // ─── Retained-session + conversation state resolution ─────────────
    const toolResults = extractCursorToolResults(body);
    const sessionOwner = agentSessionOwner(credentials, model, body);
    const userTexts = extractUserTexts(body);
    const conv = getAgentConversation(sessionOwner);
    const lineage = validateConversationLineage(conv, userTexts, body, textFromContent);
    if (lineage === false) {
      resetAgentConversation(conv);
    } else if (lineage === "compact") {
      resyncConversationLineageAfterCompact(conv, userTexts);
    }

    const tools = body.tools || [];

    // The Claude SSE translator derives Anthropic's message ID by stripping
    // `chatcmpl-`. Keep the remaining ID in Anthropic's required `msg_` form
    // so strict clients such as Claude Code accept the completed stream.
    const responseId = `chatcmpl-msg_${Date.now()}`;
    const created = Math.floor(Date.now() / 1000);

    // Build the run frame against the current conversation state. Rebuilt per
    // attempt so a blob_not_found reset (new conversation ID) is picked up.
    const buildRunFrame = () => buildAgentRunFrame(
      body.messages || [],
      model,
      tools,
      body.reasoning_effort || body.reasoning?.effort,
      conv.conversationId
        ? {
          conversationId: conv.conversationId,
          checkpoint: conv.checkpoint,
          blobStore: conv.blobStore,
          tokenUsage: conv.tokenUsage,
        }
        : null,
    );

    // One full attempt: (re)open the session, write the run frame, consume.
    // Emits events via onEvent. Retryable failures throw an Error whose
    // message classifies via classifyCursorError.
    const runAttempt = async (onEvent) => {
      const retained = acquireRetainedAgentSession(sessionOwner, toolResults);
      let retainedState = retained?.state || null;
      let keepSessionOpen = false;
      let sawContent = false;
      let session;
      let responseHeaders;
      const closeRetainedOnAbort = () => {
        if (retainedState) closeRetainedAgentSession(retainedState);
      };
      if (requestController.signal.aborted) closeRetainedOnAbort();
      else requestController.signal.addEventListener("abort", closeRetainedOnAbort, { once: true });
      try {
        if (requestController.signal.aborted) throw new Error("Request aborted");
        if (retainedState) {
          // Tool-result resume: write the correlated MCP results onto the SAME
          // server-side session so the agent continues its own conversation.
          session = retainedState.session;
          consumeRetainedToolResults(retainedState, retained.matchedResults);
          keepSessionOpen = true;
          responseHeaders = { ":status": 200 };
        } else {
          session = this.openAgentHttp2Stream(url, headers, requestController.signal);
          session.write(buildRunFrame());
        }
      } catch (error) {
        if (retainedState) closeRetainedAgentSession(retainedState);
        else try { session?.close(); } catch {}
        throw new Error(`Cursor AgentService request failed: ${error.message}`);
      }

      if (!responseHeaders) {
        try {
          responseHeaders = await session.responseHeaders;
        } catch (error) {
          session.close();
          throw new Error(`Cursor AgentService request failed: ${error.message}`);
        }
      }

      const status = Number(responseHeaders[":status"] || 0);
      if (status !== 200) {
        let errorText = "";
        try {
          while (true) {
            const { done, value } = await session.read();
            if (done) break;
            errorText += Buffer.from(value).toString("utf8");
          }
        } catch {}
        session.close();
        const hint = classifyCursorError(errorText);
        if (hint) {
          const err = new Error(hint);
          err.retryHint = hint;
          throw err;
        }
        return {
          response: new Response(JSON.stringify({
            error: { message: `Cursor AgentService ${status}: ${errorText || "request failed"}`, type: "api_error" },
          }), { status: status || HTTP_STATUS.SERVER_ERROR, headers: { "Content-Type": "application/json" } }),
          url,
          headers,
          transformedBody: body,
          responseFormat: FORMATS.OPENAI,
        };
      }

      // The Claude SSE translator derives Anthropic's message ID by stripping
      // `chatcmpl-`. Keep the remaining ID in Anthropic's required `msg_` form
      // so strict clients such as Claude Code accept the completed stream.
      const composerModel = isComposerModel(model);
      let pending = retainedState?.buffered || Buffer.alloc(0);
      if (retainedState) retainedState.buffered = Buffer.alloc(0);
      let finished = false;
      let timedOut = false;
      let thinkingAcc = "";
      let emittedVisible = 0;
      let emittedText = false;

      const flushThinkingFallback = (onEvent) => {
        if (emittedText || !thinkingAcc) return;
        const fallback = composerModel
          ? visibleComposerContentFromThinking(thinkingAcc)
          : thinkingAcc.trim();
        if (fallback) {
          emittedText = true;
          sawContent = true;
          onEvent({ type: "text", value: fallback });
        }
      };

      const consume = async (onEvent) => {
        const finishTurn = () => {
          flushThinkingFallback(onEvent);
          finished = true;
          if (emittedText) onEvent({ type: "done" });
          // Empty turns emit NO error event here — the retry loop owns the
          // decision (retry vs terminal) so premature errors don't wedge it.
        };
        let heartbeatTimer;
        let safetyTimer;
        try {
          heartbeatTimer = setInterval(() => {
            if (!finished && !retainedState) {
              try { session.write(encodeAgentHeartbeat()); } catch {}
            }
          }, CURSOR_AGENT_HEARTBEAT_MS);
          heartbeatTimer.unref?.();
          safetyTimer = setTimeout(() => {
            if (finished) return;
            finished = true;
            timedOut = true;
            try { session.close(); } catch {}
          }, CURSOR_AGENT_STREAM_TIMEOUT_MS);
          safetyTimer.unref?.();
          while (!finished) {
            const { done, value } = await session.read();
            if (done) break;
            pending = Buffer.concat([pending, Buffer.from(value)]);
            pending = decodeAgentFrames(pending, (payload) => {
              // A single read can carry several frames; once the turn is over the
              // remaining complete frames stay buffered for the retained resume.
              if (finished) return false;
              const serverMessage = decodeMessage(payload);

              // agent.v1.AgentServerMessage.interaction_update
              if (serverMessage.has(1)) {
                const update = decodeMessage(serverMessage.get(1)[0].value);
                if (update.has(1)) {
                  const textDelta = extractAgentString(decodeMessage(update.get(1)[0].value), 1);
                  if (textDelta) {
                    emittedText = true;
                    sawContent = true;
                    onEvent({ type: "text", value: textDelta });
                  }
                }
                // thinking_delta (field 4). Composer (and some Grok variants) put
                // the visible answer after </think> here and never send text_delta.
                if (update.has(4)) {
                  const thinkingDelta = extractAgentString(decodeMessage(update.get(4)[0].value), 1);
                  if (thinkingDelta) {
                    thinkingAcc += thinkingDelta;
                    sawContent = true; // thinking-only turns are real content
                    if (composerModel) {
                      const visible = visibleComposerContentFromThinking(thinkingAcc);
                      if (visible.length > emittedVisible) {
                        const deltaContent = visible.slice(emittedVisible);
                        emittedVisible = visible.length;
                        emittedText = true;
                        onEvent({ type: "text", value: deltaContent });
                      }
                    }
                  }
                }
                if (update.has(8)) {
                  try {
                    const tokenDelta = decodeMessage(update.get(8)[0].value);
                    const tokens = extractAgentVarint(tokenDelta, 1);
                    if (typeof tokens === "number" && tokens > 0) conv.outputTokens += tokens;
                  } catch {}
                }
                if (update.has(14)) {
                  finishTurn();
                  return false;
                }
              }

              // KvServerMessage (field 4): KV is a bidirectional side channel.
              // GetBlob answers from the per-conversation blob store; SetBlob
              // stores for later turns — an empty GetBlob breaks long conversations.
              const kvEvent = decodeAgentKvServerEvent(payload);
              if (kvEvent) {
                if (kvEvent.kind === "get") {
                  const key = Buffer.from(kvEvent.blobId).toString("hex");
                  const blob = conv.blobStore.get(key);
                  session.write(encodeAgentKvGetResult(kvEvent.id, blob || new Uint8Array(), kvEvent.metadata));
                } else {
                  conv.blobStore.set(Buffer.from(kvEvent.blobId).toString("hex"), Buffer.from(kvEvent.blobData));
                  pruneAgentBlobs(conv.blobStore);
                  session.write(encodeAgentKvSetResult(kvEvent.id, kvEvent.metadata));
                }
                return true;
              }

              // agent.v1.AgentServerMessage.conversation_checkpoint_update (field 3).
              // Persist the serialized conversation state so the next turn resumes
              // this conversation instead of replaying history.
              if (serverMessage.has(3)) {
                const bytes = Buffer.from(serverMessage.get(3)[0].value);
                if (bytes.length) {
                  conv.checkpoint = bytes;
                  try {
                    // ConversationStateStructure.token_details (field 5) →
                    // { used_tokens: 1, max_tokens: 2 }
                    const state = decodeMessage(bytes);
                    const td = state.get(5)?.[0];
                    if (td && td.wireType === PROTOBUF_LEN) {
                      const details = decodeMessage(td.value);
                      const used = extractAgentVarint(details, 1);
                      const max = extractAgentVarint(details, 2);
                      if (typeof used === "number") {
                        conv.tokenUsage = { used, max: typeof max === "number" ? max : null };
                      }
                      if (typeof max === "number" && max > 0) conv.effectiveContextWindow = max;
                    }
                  } catch {}
                }
                return true;
              }

              // AgentService requests IDE context before producing a response.
              if (serverMessage.has(2)) {
                const execRequest = decodeMessage(serverMessage.get(2)[0].value);
                if (execRequest.has(10)) {
                  log?.info?.("CURSOR", "AgentService request_context ack");
                  session.write(createRequestContextResponse(execRequest));
                } else if (execRequest.has(11)) {
                  const mcp = decodeMcpArgs(execRequest.get(11)[0].value);
                  const name = mcp.toolName || mcp.name;
                  if (name) {
                    log?.info?.("CURSOR", `AgentService MCP tool_call ${name}`);
                    // Retain the session so the tool result resumes this exact
                    // conversation (owner = model+account+client-session).
                    const toolCallId = normalizeAgentToolCallId(
                      mcp.toolCallId || `call_${crypto.randomUUID()}`
                    );
                    retainedState = retainAgentToolCall(session, sessionOwner, toolCallId, execRequest);
                    keepSessionOpen = true;
                    sawContent = true; // a tool call is a real terminal outcome
                    finished = true;
                    onEvent({
                      type: "tool_call",
                      value: {
                        id: toolCallId,
                        name,
                        arguments: JSON.stringify(mcp.args || {}),
                      },
                    });
                    onEvent({ type: "done", finishReason: "tool_calls" });
                    return false;
                  } else {
                    debugLog(`[CURSOR AGENT] Unsupported exec request fields: ${[...execRequest.keys()].join(",")}`);
                    finished = true;
                    sawContent = true; // protocol error is a real terminal outcome
                    onEvent({ type: "error", value: "Cursor AgentService requested an unsupported IDE tool" });
                  }
                } else {
                  // Auto/Composer often probe IDE builtins (shell/read/…). Reject
                  // them so the model can continue with MCP tools or a text answer
                  // instead of stalling the h2 stream.
                  const rejection = rejectExecRequest(execRequest);
                  if (rejection) {
                    log?.info?.("CURSOR", `AgentService rejected IDE exec fields=${[...execRequest.keys()].join(",")}`);
                    session.write(rejection);
                  } else {
                    debugLog(`[CURSOR AGENT] Unsupported exec request fields: ${[...execRequest.keys()].join(",")}`);
                    finished = true;
                    sawContent = true;
                    onEvent({ type: "error", value: "Cursor AgentService requested an unsupported IDE tool" });
                  }
                }
              }
              return true;
            });
            if (finished) break;
          }
          if (!finished) finishTurn();
        } catch (error) {
          if (retainedState) closeRetainedAgentSession(retainedState);
          throw error;
        } finally {
          if (heartbeatTimer) clearInterval(heartbeatTimer);
          if (safetyTimer) clearTimeout(safetyTimer);
          // Retained sessions stay open (TTL'd) so the tool result can resume
          // them; fresh sessions are closed as before.
          if (keepSessionOpen && retainedState) {
            if (!sawContent) {
              closeRetainedAgentSession(retainedState);
              keepSessionOpen = false;
            } else {
              retainedState.buffered = pending;
              releaseRetainedAgentSession(retainedState, true);
            }
          } else {
            try { session.end(); } catch {}
            try { session.close(); } catch {}
          }
          if (timedOut) {
            const err = new Error("timeout");
            err.retryHint = "timeout";
            throw err;
          }
        }
      };

      await consume(onEvent);
      return { keepSessionOpen, sawContent };
    };

    // Wrap events so we can refuse to retry once real output has started.
    let outputStarted = false;
    const makeWrappedEvent = (onEvent) => (event) => {
      if (event.type === "text" || event.type === "thinking" || event.type === "tool_call") outputStarted = true;
      onEvent(event);
    };

    // Retry empty upstream turns only when a long/stale session makes an empty
    // turn likely transient (stored checkpoint or an already-long history);
    // a fresh chat that streams nothing fails fast after one retry.

    // Retry loop: fresh session per attempt; blob_not_found / empty_turn resets
    // the conversation (new ID) so Cursor replays from client history.
    const runWithRetry = async (onEvent) => {
      outputStarted = false;
      let finishReason = "stop";
      const wrapped = (event) => {
        if (event.type === "tool_call") finishReason = "tool_calls";
        makeWrappedEvent(onEvent)(event);
      };
      let attempt = 0;
      for (;;) {
        try {
          const result = await runAttempt(wrapped);
          // Stale checkpoint or oversized replay can yield HTTP 200 with no
          // deltas — treat as retryable instead of a fake successful empty stop.
          if (!result.sawContent) {
            const err = new Error("empty_turn");
            err.retryHint = "empty_turn";
            throw err;
          }
          commitConversationTurn(conv, userTexts);
          return { ...result, finishReason };
        } catch (error) {
          if (error?.retryHint === "empty_turn" && outputStarted) {
            // Real output already streamed; surface what we have.
            return { keepSessionOpen: false, sawContent: true, finishReason };
          }
          const hint = error.retryHint || classifyCursorError(error.message);
          const maxRetries = hint === "empty_turn" ? CURSOR_AGENT_MAX_RETRIES + 1 : CURSOR_AGENT_MAX_RETRIES;
          if (!hint || attempt >= maxRetries || outputStarted) {
            if (hint === "empty_turn") {
              // Retries exhausted (or nothing retryable): terminal empty-turn
              // error — one event, matching the pre-subsystem kernel contract.
              onEvent({ type: "error", value: "Cursor AgentService returned an empty turn" });
              return { keepSessionOpen: false, sawContent: false, finishReason };
            }
            throw error;
          }
          attempt++;
          if (hint === "blob_not_found" || hint === "empty_turn") {
            resetAgentConversation(conv);
          }
          await sleepMs(retryDelayMs(hint));
        }
      }
    };

    const buildAgentUsage = (contentLength) => {
      const tokenUsage = conv.tokenUsage;
      const output = conv.outputTokens || 0;
      if (tokenUsage?.used) {
        const scaled = buildScaledCursorUsage({
          usedTokens: tokenUsage.used,
          outputTokens: output,
          cursorMaxTokens: tokenUsage.max || conv.effectiveContextWindow || 0,
          modelId: model,
          inferredWindow: inferContextWindow(model),
          scaling: isCursorUsageScalingEnabled(),
        });
        if (scaled) return scaled;
      }
      return estimateUsage(body, contentLength, FORMATS.OPENAI);
    };

    if (stream === false) {
      let content = "";
      let reasoning = "";
      let agentError = null;
      const toolCalls = [];
      let finishReason = "stop";
      await runWithRetry((event) => {
        if (event.type === "text") content += event.value;
        else if (event.type === "thinking") reasoning += event.value;
        else if (event.type === "tool_call") {
          toolCalls.push({
            id: event.value.id,
            type: "function",
            function: { name: event.value.name, arguments: event.value.arguments },
          });
          finishReason = "tool_calls";
        }
        else if (event.type === "error") agentError = event.value;
        else if (event.type === "done" && event.finishReason) finishReason = event.finishReason;
      });
      if (agentError) {
        return {
          response: new Response(JSON.stringify({ error: { message: agentError, type: "api_error" } }), {
            status: HTTP_STATUS.BAD_REQUEST,
            headers: { "Content-Type": "application/json" },
          }),
          url,
          headers,
          transformedBody: body,
          responseFormat: FORMATS.OPENAI,
        };
      }
      const message = {
        role: "assistant",
        content: content || null,
        ...(reasoning ? { reasoning_content: reasoning } : {}),
        ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
      };
      return {
        response: new Response(JSON.stringify({
          id: responseId,
          object: "chat.completion",
          created,
          model,
          choices: [{ index: 0, message, finish_reason: finishReason }],
          usage: buildAgentUsage(content.length),
        }), { headers: { "Content-Type": "application/json" } }),
        url,
        headers,
        transformedBody: body,
        responseFormat: FORMATS.OPENAI,
      };
    }

    const encoder = new TextEncoder();
    const streamState = { clearKeepalive: null };
    const responseStream = new ReadableStream({
      start(controller) {
        let streamClosed = false;
        const clearKeepalive = () => {
          if (streamState.keepaliveTimer) {
            clearInterval(streamState.keepaliveTimer);
            streamState.keepaliveTimer = null;
          }
        };
        streamState.clearKeepalive = clearKeepalive;
        const closeWithAgentError = (error) => {
          if (streamClosed) return;
          streamClosed = true;
          clearKeepalive();
          const mapped = mapCursorAgentErrorResponse(error);
          try {
            controller.enqueue(encoder.encode(sseChunk({
              error: {
                message: mapped.message,
                type: mapped.type,
                ...(mapped.code ? { code: mapped.code } : {}),
              },
            })));
            controller.enqueue(encoder.encode(SSE_DONE));
            controller.close();
          } catch {}
        };

        // SSE keepalive comments before first output keep proxies from
        // buffering/closing an idle stream during long thinking turns.
        streamState.keepaliveTimer = setInterval(() => {
          if (!streamClosed && !requestController.signal.aborted) {
            try {
              controller.enqueue(encoder.encode(": keepalive\n\n"));
            } catch {}
          }
        }, CURSOR_SSE_KEEPALIVE_MS);
        streamState.keepaliveTimer.unref?.();

        // Wrap per-attempt events: during retry rounds the stream must stay
        // open (an attempt's premature done/error would wedge the retry), so
        // only content deltas flow through; terminal frames are emitted once
        // runWithRetry settles.
        const streamHadContent = () => outputStarted;
        runWithRetry((event) => {
          if (streamClosed) return;
          if (event.type === "text") {
            clearKeepalive();
            controller.enqueue(encoder.encode(chatChunkSse({ id: responseId, created, model, delta: { content: event.value } })));
          } else if (event.type === "thinking") {
            clearKeepalive();
            controller.enqueue(encoder.encode(chatChunkSse({ id: responseId, created, model, delta: { reasoning_content: event.value } })));
          } else if (event.type === "tool_call") {
            clearKeepalive();
            controller.enqueue(encoder.encode(chatChunkSse({
              id: responseId, created, model,
              delta: {
                tool_calls: [{
                  index: 0,
                  id: event.value.id,
                  type: "function",
                  function: { name: event.value.name, arguments: event.value.arguments },
                }],
              },
            })));
          } else if (event.type === "error" && outputStarted) {
            // Terminal protocol error after real content (e.g. unsupported IDE
            // tool late in the turn): close with the error frame immediately.
            streamClosed = true;
            clearKeepalive();
            controller.enqueue(encoder.encode(sseChunk({ error: { message: event.value, type: "api_error" } })));
            controller.enqueue(encoder.encode(SSE_DONE));
            controller.close();
          }
          // done/empty-error during an attempt are deferred: the retry loop
          // decides whether they are terminal (see .then/.catch below).
        }).then((result) => {
          if (streamClosed) return;
          streamClosed = true;
          clearKeepalive();
          if (!streamHadContent()) {
            // Turn produced no real output — terminal error, not a fake stop.
            controller.enqueue(encoder.encode(sseChunk({
              error: { message: "Cursor AgentService returned an empty turn", type: "api_error" },
            })));
            controller.enqueue(encoder.encode(SSE_DONE));
            controller.close();
            return;
          }
          const finishReason = result?.finishReason === "tool_calls" ? "tool_calls" : "stop";
          controller.enqueue(encoder.encode(chatChunkSse({
            id: responseId, created, model, delta: {},
            finishReason,
            usage: buildAgentUsage(0),
          })));
          controller.enqueue(encoder.encode(SSE_DONE));
          controller.close();
        }).catch((error) => {
          // Classified agent failures (rate limit / timeout / empty / context)
          // close with a mapped terminal SSE error; unclassified transport
          // errors must reject the body — a broken stream must not look like a
          // fabricated server error frame to the client.
          if (error?.retryHint || classifyCursorError(error?.message)) {
            closeWithAgentError(error);
          } else {
            clearKeepalive();
            controller.error(error);
          }
        });
      },
      cancel() {
        streamState.clearKeepalive?.();
        requestController.abort();
      },
    });

    return {
      response: new Response(responseStream, { headers: SSE_HEADERS }),
      url,
      headers,
      transformedBody: body,
      responseFormat: FORMATS.OPENAI,
    };
  }

  async execute({ model, body, stream, credentials, signal, log, proxyOptions = null }) {
    if (isAgentCapableRequest(body)) {
      try {
        return await this.executeAgent({ model, body, stream, credentials, signal, log });
      } catch (error) {
        // Classify retryable agent failures (rate limit / timeout / empty /
        // context) into client-meaningful HTTP codes instead of a bare 500.
        const mapped = mapCursorAgentErrorResponse(error);
        return {
          response: new Response(JSON.stringify({
            error: {
              message: mapped.message,
              type: mapped.type,
              ...(mapped.code ? { code: mapped.code } : {}),
            },
          }), { status: mapped.status, headers: { "Content-Type": "application/json" } }),
          url: `${PROVIDER_OAUTH.cursor?.agentEndpoint || ""}${AGENT_RUN_PATH}`,
          headers: {},
          transformedBody: body,
        };
      }
    }

    const url = this.buildUrl();
    const headers = this.buildHeaders(credentials);
    const transformedBody = this.transformRequest(model, body, stream, credentials);

    try {
      const shouldForceFetch = proxyOptions?.enabled === true || proxyOptions?.connectionProxyEnabled === true || !!proxyOptions?.vercelRelayUrl;
      const response = (http2 && !shouldForceFetch)
        ? await this.makeHttp2Request(url, headers, transformedBody, signal)
        : await this.makeFetchRequest(url, headers, transformedBody, signal, proxyOptions);

      if (response.status !== 200) {
        const errorText = response.body?.toString() || "Unknown error";
        const errorResponse = new Response(JSON.stringify({
          error: {
            message: `[${response.status}]: ${errorText}`,
            type: "invalid_request_error",
            code: ""
          }
        }), {
          status: response.status,
          headers: { "Content-Type": "application/json" }
        });
        return { response: errorResponse, url, headers, transformedBody: body };
      }

      const transformedResponse = stream !== false
        ? this.transformProtobufToSSE(response.body, model, body)
        : this.transformProtobufToJSON(response.body, model, body);

      return { response: transformedResponse, url, headers, transformedBody: body };
    } catch (error) {
      const errorResponse = new Response(JSON.stringify({
        error: {
          message: error.message,
          type: "connection_error",
          code: ""
        }
      }), {
        status: HTTP_STATUS.SERVER_ERROR,
        headers: { "Content-Type": "application/json" }
      });
      return { response: errorResponse, url, headers, transformedBody: body };
    }
  }

  transformProtobufToJSON(buffer, model, body) {
    const responseId = `chatcmpl-cursor-${Date.now()}`;
    const created = Math.floor(Date.now() / 1000);

    let offset = 0;
    let totalContent = "";
    let totalThinking = "";
    const toolCalls = [];
    const toolCallsMap = new Map(); // Track streaming tool calls by ID
    const finalizedIds = new Set();
    let frameCount = 0;

    debugLog(`[CURSOR BUFFER] Total length: ${buffer.length} bytes`);

    while (offset < buffer.length) {
      const frame = readCursorFrame(buffer, offset, frameCount, "");
      if (frame.status === "done") break;
      offset = frame.offset;
      frameCount++;
      if (frame.status === "skip") continue;
      const payload = frame.payload;

      // Check for JSON error frames (byte guard: skip toString on non-JSON frames)
      if (payload.length > 0 && payload[0] === 0x7b) {
        try {
          const text = payload.toString("utf-8");
          if (text.includes('"error"')) {
            const hasContent = totalContent || toolCallsMap.size > 0;
            debugLog(
              `[CURSOR BUFFER] Error frame (hasContent=${hasContent}): ${text.slice(0, 500)}`
            );
            if (hasContent) {
              break;
            }
            return createErrorResponse(JSON.parse(text));
          }
        } catch {}
      }

      const result = extractTextFromResponse(new Uint8Array(payload));
      debugLog(`[CURSOR DECODED] Frame ${frameCount}:`, result);

      if (result.error) {
        const hasContent = totalContent || toolCallsMap.size > 0;
        debugLog(`[CURSOR BUFFER] Decoded error (hasContent=${hasContent}): ${result.error}`);
        if (hasContent) {
          break;
        }
        return new Response(
          JSON.stringify({
            error: {
              message: result.error,
              type: "rate_limit_error",
              code: "rate_limited"
            }
          }),
          {
            status: HTTP_STATUS.RATE_LIMITED,
            headers: { "Content-Type": "application/json" }
          }
        );
      }

      if (result.toolCall) {
        const tc = result.toolCall;

        if (toolCallsMap.has(tc.id)) {
          // Accumulate arguments for existing tool call
          const existing = toolCallsMap.get(tc.id);
          existing.function.arguments += tc.function.arguments;
          existing.isLast = tc.isLast;
        } else {
          // New tool call
          toolCallsMap.set(tc.id, { ...tc });
        }

        // Push to final array when isLast is true
        if (tc.isLast) {
          const finalToolCall = toolCallsMap.get(tc.id);
          finalizedIds.add(tc.id);
          toolCalls.push({
            id: finalToolCall.id,
            type: finalToolCall.type,
            function: {
              name: finalToolCall.function.name,
              arguments: finalToolCall.function.arguments
            }
          });
        }
      }

      if (result.text) totalContent += result.text;
      if (result.thinking) totalThinking += result.thinking;
    }

    const visibleComposerContent = isComposerModel(model)
      ? visibleComposerContentFromThinking(totalThinking)
      : "";
    const finalContent = totalContent || visibleComposerContent;

    debugLog(
      `[CURSOR BUFFER] Parsed ${frameCount} frames, toolCallsMap size: ${toolCallsMap.size}, finalized toolCalls: ${toolCalls.length}`
    );

    // Finalize all remaining tool calls in map (in case stream ended without isLast=true)
    for (const [id, tc] of toolCallsMap.entries()) {
      // Check if already in final array
      if (!finalizedIds.has(id)) {
        debugLog(`[CURSOR BUFFER] Finalizing incomplete tool call: ${id}, isLast=${tc.isLast}`);
        toolCalls.push({
          id: tc.id,
          type: tc.type,
          function: {
            name: tc.function.name,
            arguments: tc.function.arguments
          }
        });
      }
    }

    debugLog(`[CURSOR BUFFER] Final toolCalls count: ${toolCalls.length}`);


    const message = {
      role: "assistant",
      content: finalContent || null
    };

    if (toolCalls.length > 0) {
      message.tool_calls = toolCalls;
    }

    const usage = estimateUsage(body, finalContent.length, FORMATS.OPENAI);

    const completion = {
      id: responseId,
      object: "chat.completion",
      created,
      model,
      choices: [{
        index: 0,
        message,
        finish_reason: toolCalls.length > 0 ? "tool_calls" : "stop"
      }],
      usage
    };

    return new Response(JSON.stringify(completion), {
      status: 200,
      headers: { "Content-Type": "application/json" }
    });
  }

  transformProtobufToSSE(buffer, model, body) {
    const responseId = `chatcmpl-cursor-${Date.now()}`;
    const created = Math.floor(Date.now() / 1000);

    const chunks = [];
    let offset = 0;
    let totalContent = "";
    let totalThinking = "";
    let emittedComposerThinkingContentLength = 0;
    const toolCalls = [];
    const toolCallsMap = new Map(); // Track streaming tool calls by ID
    const finalizedIds = new Set();
    const emittedToolCallIds = new Set();
    let frameCount = 0;

    debugLog(`[CURSOR BUFFER SSE] Total length: ${buffer.length} bytes`);

    while (offset < buffer.length) {
      const frame = readCursorFrame(buffer, offset, frameCount, " SSE");
      if (frame.status === "done") break;
      offset = frame.offset;
      frameCount++;
      if (frame.status === "skip") continue;
      const payload = frame.payload;

      // Check for JSON error frames (byte-guard: only decode if starts with '{')
      if (payload[0] === 0x7b) {
        try {
          const text = payload.toString("utf-8");
          if (text.includes('"error"')) {
            const hasContent = chunks.length > 0 || totalContent || toolCallsMap.size > 0;
            debugLog(
              `[CURSOR BUFFER SSE] Error frame (hasContent=${hasContent}): ${text.slice(0, 500)}`
            );
            if (hasContent) {
              break;
            }
            return createErrorResponse(JSON.parse(text));
          }
        } catch {}
      }

      const result = extractTextFromResponse(new Uint8Array(payload));
      debugLog(`[CURSOR DECODED SSE] Frame ${frameCount}:`, result);

      if (result.error) {
        const hasContent = chunks.length > 0 || totalContent || toolCallsMap.size > 0;
        debugLog(`[CURSOR BUFFER SSE] Decoded error (hasContent=${hasContent}): ${result.error}`);
        if (hasContent) {
          break;
        }
        return new Response(
          JSON.stringify({
            error: {
              message: result.error,
              type: "rate_limit_error",
              code: "rate_limited"
            }
          }),
          {
            status: HTTP_STATUS.RATE_LIMITED,
            headers: { "Content-Type": "application/json" }
          }
        );
      }

      if (result.toolCall) {
        const tc = result.toolCall;

        if (chunks.length === 0) {
          chunks.push(chatChunkSse({ id: responseId, created, model, delta: { role: "assistant", content: "" } }));
        }

        if (toolCallsMap.has(tc.id)) {
          // Accumulate arguments for existing tool call
          const existing = toolCallsMap.get(tc.id);
          const oldArgsLen = existing.function.arguments.length;
          existing.function.arguments += tc.function.arguments;
          existing.isLast = tc.isLast;

          // Stream the delta arguments
          if (tc.function.arguments) {
            emittedToolCallIds.add(tc.id);
            chunks.push(chatChunkSse({
              id: responseId, created, model,
              delta: {
                tool_calls: [
                  {
                    index: existing.index,
                    id: tc.id,
                    type: "function",
                    function: {
                      name: tc.function.name,
                      arguments: tc.function.arguments
                    }
                  }
                ]
              }
            }));
          }
        } else {
          // New tool call - assign index and add to map
          const toolCallIndex = toolCalls.length;
          finalizedIds.add(tc.id);
          toolCalls.push({ ...tc, index: toolCallIndex });
          toolCallsMap.set(tc.id, { ...tc, index: toolCallIndex });

          // Stream initial tool call with name
          emittedToolCallIds.add(tc.id);
          chunks.push(chatChunkSse({
            id: responseId, created, model,
            delta: {
              tool_calls: [
                {
                  index: toolCallIndex,
                  id: tc.id,
                  type: "function",
                  function: {
                    name: tc.function.name,
                    arguments: tc.function.arguments
                  }
                }
              ]
            }
          }));
        }
      }

      if (result.text) {
        totalContent += result.text;
        chunks.push(chatChunkSse({
          id: responseId, created, model,
          delta:
            chunks.length === 0 && toolCalls.length === 0
              ? { role: "assistant", content: result.text }
              : { content: result.text }
        }));
      }

      if (isComposerModel(model) && result.thinking) {
        totalThinking += result.thinking;
        const visibleContent = visibleComposerContentFromThinking(totalThinking);
        if (visibleContent.length > emittedComposerThinkingContentLength) {
          const deltaContent = visibleContent.slice(emittedComposerThinkingContentLength);
          emittedComposerThinkingContentLength = visibleContent.length;
          totalContent += deltaContent;
          chunks.push(chatChunkSse({
            id: responseId, created, model,
            delta:
              chunks.length === 0 && toolCalls.length === 0
                ? { role: "assistant", content: deltaContent }
                : { content: deltaContent }
          }));
        }
      }
    }

    debugLog(
      `[CURSOR BUFFER SSE] Parsed ${frameCount} frames, toolCallsMap size: ${toolCallsMap.size}, toolCalls array: ${toolCalls.length}`
    );

    // Finalize all remaining tool calls in map (stream may have ended without isLast=true)
    for (const [id, tc] of toolCallsMap.entries()) {
      if (!finalizedIds.has(id)) {
        debugLog(`[CURSOR BUFFER SSE] Finalizing incomplete tool call: ${id}, isLast=${tc.isLast}`);
        const toolCallIndex = toolCalls.length;
        toolCalls.push({
          id: tc.id,
          type: tc.type,
          index: toolCallIndex,
          function: {
            name: tc.function.name,
            arguments: tc.function.arguments
          }
        });

        // Emit SSE chunk for the finalized tool call if not already emitted
        if (!emittedToolCallIds.has(tc.id)) {
          chunks.push(chatChunkSse({
            id: responseId, created, model,
            delta: {
              tool_calls: [
                {
                  index: toolCallIndex,
                  id: tc.id,
                  type: "function",
                  function: {
                    name: tc.function.name,
                    arguments: tc.function.arguments
                  }
                }
              ]
            }
          }));
        }
      }
    }

    if (chunks.length === 0 && toolCalls.length === 0) {
      chunks.push(chatChunkSse({ id: responseId, created, model, delta: { role: "assistant", content: "" } }));
    }

    const usage = estimateUsage(body, totalContent.length, FORMATS.OPENAI);

    chunks.push(
      `data: ${JSON.stringify({
        id: responseId,
        object: "chat.completion.chunk",
        created,
        model,
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: toolCalls.length > 0 ? "tool_calls" : "stop"
          }
        ],
        usage
      })}\n\n`
    );
    chunks.push(SSE_DONE);

    return new Response(chunks.join(""), {
      status: 200,
      headers: { ...SSE_HEADERS }
    });
  }

  async refreshCredentials() {
    return null;
  }
}

export default CursorExecutor;
