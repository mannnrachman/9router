import { BaseExecutor } from "./base.js";
import { PROVIDERS, PROVIDER_OAUTH } from "../config/providers.js";
import { HTTP_STATUS } from "../config/runtimeConfig.js";
import {
  generateCursorBody,
  encodeField,
  wrapConnectRPCFrame,
  decodeMessage,
  extractTextFromResponse,
  encodeMcpToolDefinition,
  encodeMcpTools,
  encodeMcpResultSuccess,
  decodeExecServerEvent,
  encodeAgentNativeRejection,
  encodeAgentEmptyListMcpResources,
  encodeAgentHeartbeat,
  decodeAgentKvServerEvent,
  encodeAgentKvGetResult,
  encodeAgentKvSetResult,
  decodeAgentInteractionQuery,
  encodeAgentInteractionResponse,
  encodeAgentReadSuccess,
  encodeAgentGrepSuccess,
  encodeAgentLsSuccess,
  encodeAgentDiagnosticsSuccess,
  encodeAgentFetchSuccess,
  encodeAgentWriteSuccess,
  encodeAgentDeleteSuccess,
  encodeAgentShellSuccess,
  encodeAgentShellFailure,
  encodeAgentShellTimeout,
} from "../utils/cursorProtobuf.js";
import { buildCursorHeaders } from "../utils/cursorChecksum.js";
import { estimateUsage } from "../utils/usageTracking.js";
import { SSE_DONE, SSE_HEADERS } from "../utils/sseConstants.js";
import { chatChunkSse, sseChunk } from "../utils/sse.js";
import { FORMATS } from "../translator/formats.js";
import { proxyAwareFetch } from "../utils/proxyFetch.js";
import { resolveCursorModel, resolveCursorModelSelection } from "../services/cursorModels.js";
import zlib from "node:zlib";
import { promises as fs } from "node:fs";
import { spawn } from "node:child_process";
import path from "node:path";

// Native IDE tool execution. Read-only tools (read/grep/ls/diagnostics/fetch)
// always run inside CURSOR_WORKSPACE (default: server cwd). Mutating tools
// (write/delete/shell) require CURSOR_NATIVE_EXEC=1.
const NATIVE_WORKSPACE = path.resolve(process.env.CURSOR_WORKSPACE || process.cwd());
const NATIVE_EXEC_MUTATE = process.env.CURSOR_NATIVE_EXEC === "1";
const NATIVE_READ_CAP = Number(process.env.CURSOR_READ_CAP) || 1024 * 1024;
const NATIVE_SHELL_TIMEOUT_MS = Number(process.env.CURSOR_SHELL_TIMEOUT_MS) || 30000;
const NATIVE_GREP_MAX = 500;
const NATIVE_LS_MAX = 500;
const CURSOR_MCP_RESULT_CAP = (() => {
  const value = Number(process.env.CURSOR_MCP_RESULT_CAP);
  return Number.isFinite(value) && value > 0 ? Math.floor(value) : 64 * 1024;
})();
const CURSOR_CONTEXT_COMPACT_TRIGGER = 0.8;
const CURSOR_CONTEXT_COMPACT_TARGET = 0.6;
const CURSOR_DEFAULT_CONTEXT_WINDOW = 200_000;

function getCursorContextWindow() {
  const configured = Number(process.env.CURSOR_CONTEXT_WINDOW);
  return Number.isFinite(configured) && configured > 0
    ? Math.floor(configured)
    : CURSOR_DEFAULT_CONTEXT_WINDOW;
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

export function capCursorToolResult(content, cap = CURSOR_MCP_RESULT_CAP) {
  const text = typeof content === "string" ? content : String(content ?? "");
  const limit = Number.isFinite(cap) && cap > 0 ? Math.floor(cap) : CURSOR_MCP_RESULT_CAP;
  const bytes = Buffer.byteLength(text, "utf8");
  if (bytes <= limit) return text;
  const marker = "\n[9router: MCP result truncated; use a narrower query or pagination]";
  if (limit <= marker.length) return marker.slice(0, limit);
  const available = limit - Buffer.byteLength(marker, "utf8");
  const prefix = Buffer.from(text, "utf8").subarray(0, available).toString("utf8");
  return `${prefix}${marker}`;
}

function capCursorToolXml(content) {
  const text = typeof content === "string" ? content : String(content ?? "");
  if (!text.includes("<tool_result>")) return text;
  return text.replace(/(<result>)([\s\S]*?)(<\/result>)/g, (_match, open, result, close) => (
    `${open}${capCursorToolResult(result)}${close}`
  ));
}

export function compactCursorMessages(messages, _model, tools = [], contextWindow = getCursorContextWindow()) {
  if (!Array.isArray(messages) || messages.length < 3) return messages || [];
  const contextTokens = estimateCursorContextTokens(messages, tools);
  const trigger = Math.floor(contextWindow * CURSOR_CONTEXT_COMPACT_TRIGGER);
  if (!contextTokens || contextTokens <= trigger) return messages;

  const systemMessages = messages.filter((message) => message?.role === "system");
  const chatMessages = messages.filter((message) => message?.role !== "system");
  const currentIndex = [...chatMessages]
    .map((message) => message?.role)
    .findLastIndex((role) => role === "user" || role === "tool");
  let suffixStart = currentIndex >= 0 ? currentIndex : chatMessages.length - 1;
  if (chatMessages[suffixStart]?.role === "tool" && chatMessages[suffixStart - 1]?.role === "assistant") suffixStart--;
  const suffix = chatMessages.slice(suffixStart);
  const history = chatMessages.slice(0, suffixStart);
  if (!suffix.length || !history.length) return messages;

  const target = Math.floor(contextWindow * CURSOR_CONTEXT_COMPACT_TARGET);
  const kept = [];
  let keptTokens = estimateCursorContextTokens([...systemMessages, ...suffix], tools);
  for (let i = history.length - 1; i >= 0; i--) {
    const messageTokens = estimateCursorContextTokens([history[i]]);
    if (keptTokens + messageTokens > target) break;
    kept.unshift(history[i]);
    keptTokens += messageTokens;
  }

  const dropped = history.length - kept.length;
  if (dropped <= 0) return messages;

  // ponytail: deterministic suffix drop; may lose old details, but agent can reread files. No nested LLM call.
  const note = {
    role: "system",
    content: `[9router] Earlier context compacted: ${dropped} messages omitted. Re-read files and current tool state before continuing.`,
  };
  return [...systemMessages, note, ...kept, ...suffix];
}

function inWorkspace(p) {
  const resolved = path.resolve(p || "");
  return resolved === NATIVE_WORKSPACE || resolved.startsWith(NATIVE_WORKSPACE + path.sep);
}

async function readFileSafe(p) {
  const buf = await fs.readFile(p);
  const truncated = buf.length > NATIVE_READ_CAP;
  const slice = truncated ? buf.subarray(0, NATIVE_READ_CAP) : buf;
  return { text: slice.toString("utf8"), truncated, size: buf.length };
}

async function* walkFiles(dir, { depth = 0, maxDepth = 4, cap = NATIVE_GREP_MAX } = {}) {
  if (depth > maxDepth) return;
  let entries;
  try { entries = await fs.readdir(dir, { withFileTypes: true }); } catch { return; }
  for (const e of entries) {
    if (e.name === ".git" || e.name === "node_modules" || e.name.startsWith(".")) continue;
    const full = path.join(dir, e.name);
    if (e.isDirectory()) yield* walkFiles(full, { depth: depth + 1, maxDepth, cap });
    else if (e.isFile()) { yield full; if (--cap <= 0) return; }
  }
}

async function execNativeTool(execEvent, session) {
  const { kind } = execEvent;
  const reply = (buf) => buf && session.write(buf);
  try {
    switch (kind) {
      case "exec_read": {
        if (!inWorkspace(execEvent.path)) return reply(encodeAgentNativeRejection(execEvent, "Path is outside the allowed workspace"));
        const { text, truncated, size } = await readFileSafe(execEvent.path);
        return reply(encodeAgentReadSuccess(execEvent.execMsgId, execEvent.execId, { path: execEvent.path, content: text, truncated, fileSize: size }));
      }
      case "exec_grep": {
        if (execEvent.path && !inWorkspace(execEvent.path)) return reply(encodeAgentNativeRejection(execEvent, "Path is outside the allowed workspace"));
        const root = execEvent.path || NATIVE_WORKSPACE;
        const flags = execEvent.caseInsensitive ? "i" : "";
        let re;
        try { re = new RegExp(execEvent.pattern, flags); } catch { return reply(encodeAgentNativeRejection(execEvent, `Invalid regex: ${execEvent.pattern}`)); }
        const matches = [];
        for await (const file of walkFiles(root)) {
          if (matches.length >= NATIVE_GREP_MAX) break;
          let text;
          try { text = (await fs.readFile(file, "utf8")).slice(0, NATIVE_READ_CAP); } catch { continue; }
          const lines = [];
          text.split("\n").forEach((line, i) => { if (re.test(line)) lines.push({ lineNumber: i + 1, content: line }); });
          if (lines.length) matches.push({ file, lines });
        }
        return reply(encodeAgentGrepSuccess(execEvent.execMsgId, execEvent.execId, { pattern: execEvent.pattern, path: path.relative(NATIVE_WORKSPACE, root) || ".", matches, truncated: matches.length >= NATIVE_GREP_MAX }));
      }
      case "exec_ls": {
        if (execEvent.path && !inWorkspace(execEvent.path)) return reply(encodeAgentNativeRejection(execEvent, "Path is outside the allowed workspace"));
        const root = execEvent.path || NATIVE_WORKSPACE;
        const files = [], dirs = [];
        let entries;
        try { entries = await fs.readdir(root, { withFileTypes: true }); } catch (e) { return reply(encodeAgentNativeRejection(execEvent, `Cannot read dir: ${e.message}`)); }
        for (const e of entries) {
          if (files.length + dirs.length >= NATIVE_LS_MAX) break;
          if (e.name.startsWith(".")) continue;
          if (e.isDirectory()) dirs.push(e.name); else if (e.isFile()) files.push(e.name);
        }
        return reply(encodeAgentLsSuccess(execEvent.execMsgId, execEvent.execId, { path: root, files, dirs, numFiles: files.length, truncated: files.length + dirs.length >= NATIVE_LS_MAX }));
      }
      case "exec_diagnostics":
        // No linter on the server; empty success means "no diagnostics".
        return reply(encodeAgentDiagnosticsSuccess(execEvent.execMsgId, execEvent.execId, execEvent.path || ""));
      case "exec_fetch": {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 30000);
        try {
          const res = await fetch(execEvent.url, { signal: controller.signal, redirect: "follow" });
          const text = (await res.text()).slice(0, NATIVE_READ_CAP);
          return reply(encodeAgentFetchSuccess(execEvent.execMsgId, execEvent.execId, { url: execEvent.url, content: text, statusCode: res.status, contentType: res.headers.get("content-type") || "" }));
        } finally { clearTimeout(timer); }
      }
      case "exec_write":
      case "exec_delete":
        if (!NATIVE_EXEC_MUTATE) return reply(encodeAgentNativeRejection(execEvent, "File mutations disabled. Set CURSOR_NATIVE_EXEC=1 to enable write/delete."));
        if (!inWorkspace(execEvent.path)) return reply(encodeAgentNativeRejection(execEvent, "Path is outside the allowed workspace"));
        if (kind === "exec_delete") {
          const prev = (await fs.readFile(execEvent.path, "utf8")).slice(0, NATIVE_READ_CAP);
          const stat = await fs.stat(execEvent.path);
          await fs.unlink(execEvent.path);
          return reply(encodeAgentDeleteSuccess(execEvent.execMsgId, execEvent.execId, { path: execEvent.path, deletedFile: execEvent.path, fileSize: stat.size, prevContent: prev }));
        }
        {
          await fs.mkdir(path.dirname(execEvent.path), { recursive: true });
          await fs.writeFile(execEvent.path, execEvent.fileText ?? "");
          const stat = await fs.stat(execEvent.path);
          return reply(encodeAgentWriteSuccess(execEvent.execMsgId, execEvent.execId, { path: execEvent.path, linesCreated: (execEvent.fileText || "").split("\n").length, fileSize: stat.size }));
        }
      case "exec_shell":
        if (!NATIVE_EXEC_MUTATE) return reply(encodeAgentNativeRejection(execEvent, "Shell disabled. Set CURSOR_NATIVE_EXEC=1 to enable shell."));
        return reply(await runNativeShell(execEvent));
      default:
        return reply(encodeAgentNativeRejection(execEvent));
    }
  } catch (e) {
    return reply(encodeAgentNativeRejection(execEvent, `Tool failed: ${e.message}`));
  }
}

async function runNativeShell({ execMsgId, execId, command = "", workingDir = "" }) {
  const cwd = workingDir && inWorkspace(workingDir) ? workingDir : NATIVE_WORKSPACE;
  const started = Date.now();
  const child = spawn("/bin/sh", ["-c", command], { cwd, maxBuffer: NATIVE_READ_CAP });
  let stdout = "", stderr = "";
  const out = [], err = [];
  child.stdout.on("data", (c) => { out.push(c); if (Buffer.concat(out).length > NATIVE_READ_CAP) child.stdout.pause(); });
  child.stderr.on("data", (c) => { err.push(c); if (Buffer.concat(err).length > NATIVE_READ_CAP) child.stderr.pause(); });
  const [code, error, timedOut] = await new Promise((resolve) => {
    let settled = false;
    const finish = (v) => { if (!settled) { settled = true; resolve(v); } };
    const timer = setTimeout(() => {
      try { child.kill("SIGKILL"); } catch {}
      finish([null, null, true]);
    }, NATIVE_SHELL_TIMEOUT_MS);
    child.on("error", (e) => { clearTimeout(timer); finish([null, e.message, false]); });
    child.on("close", (c) => { clearTimeout(timer); finish([c, null, false]); });
  });
  stdout = Buffer.concat(out).toString("utf8").slice(0, NATIVE_READ_CAP);
  stderr = Buffer.concat(err).toString("utf8").slice(0, NATIVE_READ_CAP);
  const executionTime = Date.now() - started;
  if (timedOut) return encodeAgentShellTimeout(execMsgId, execId, { command, cwd, timeoutMs: NATIVE_SHELL_TIMEOUT_MS });
  if (error) return encodeAgentShellFailure(execMsgId, execId, { command, cwd, exitCode: 1, stderr: error, executionTime });
  return code === 0
    ? encodeAgentShellSuccess(execMsgId, execId, { command, cwd, exitCode: code, stdout, stderr, executionTime })
    : encodeAgentShellFailure(execMsgId, execId, { command, cwd, exitCode: code ?? 1, stdout, stderr, executionTime });
}

// Reject (or execute) a native tool request. Returns false when the message
// was handled without a wire reply (request_context/mcp paths handled by caller).
function handleNativeExec(execEvent, session) {
  if (!execEvent) return false;
  if (execEvent.kind === "exec_request_context" || execEvent.kind === "exec_mcp") return false;
  execNativeTool(execEvent, session);
  return true;
}
import crypto from "node:crypto";

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
    http2 = await import("node:http2");
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

const CURSOR_AGENT_SESSION_TTL_MS = 5 * 60 * 1000;
const CURSOR_AGENT_STREAM_TIMEOUT_MS = Number(process.env.CURSOR_STREAM_TIMEOUT_MS || 300000);
const CURSOR_AGENT_HEARTBEAT_MS = Number(process.env.CURSOR_HEARTBEAT_MS || 5000);
const CURSOR_SSE_KEEPALIVE_MS = Number(process.env.CURSOR_SSE_KEEPALIVE_MS || 15000);
const CURSOR_AGENT_MAX_RETRIES = (() => {
  const value = Number(process.env.CURSOR_AGENT_MAX_RETRIES);
  return Number.isFinite(value) && value >= 0 ? Math.floor(value) : 2;
})();
// Conversation persistence: blobs + checkpoints survive past the 5-min tool-call
// session TTL, so a later turn can resume the same Cursor conversation (pi-cursor parity).
const CURSOR_CONVERSATION_TTL_MS = 30 * 60 * 1000;
const MAX_BLOBS_PER_OWNER = 128;
const retainedAgentSessions = new Map();
const retainedAgentToolCalls = new Map();
const agentConversations = new Map(); // owner -> { conversationId, checkpoint, blobStore, turnCount, fingerprint }

function getAgentConversation(owner) {
  let conv = agentConversations.get(owner);
  if (!conv) {
    conv = {
      conversationId: crypto.randomUUID(),
      checkpoint: null,
      blobStore: new Map(),
      turnCount: 0,
      fingerprint: null,
      tokenUsage: null,
      outputTokens: 0,
      lastAccessMs: Date.now(),
      timer: null,
    };
    agentConversations.set(owner, conv);
  }
  conv.lastAccessMs = Date.now();
  clearTimeout(conv.timer);
  conv.timer = setTimeout(() => {
    if (agentConversations.get(owner) === conv) agentConversations.delete(owner);
  }, CURSOR_CONVERSATION_TTL_MS);
  conv.timer.unref?.();
  return conv;
}

// Discard checkpoint, blobs, and lineage; assign a new conversation ID.
// Used on lineage mismatch (edit/fork/compaction) and blob_not_found retry.
function resetAgentConversation(conv) {
  conv.conversationId = crypto.randomUUID();
  conv.checkpoint = null;
  conv.blobStore.clear();
  conv.turnCount = 0;
  conv.fingerprint = null;
  conv.tokenUsage = null;
  conv.outputTokens = 0;
}

// SHA256 over user-text turns in order; mismatch means the client history no
// longer matches the checkpoint (edited/forked conversation) -> start fresh.
function computeLineageFingerprint(userTexts) {
  const hash = crypto.createHash("sha256");
  for (const text of userTexts) {
    hash.update(text);
    hash.update("\0");
  }
  return hash.digest("hex");
}

function extractUserTexts(body) {
  return (body?.messages || [])
    .filter((message) => message?.role === "user")
    .map((message) => textFromContent(message?.content));
}

function validateConversationLineage(conv, userTexts) {
  if (conv.checkpoint === null) return true; // nothing stored yet -> fresh
  const turns = userTexts.slice(0, -1);
  if (conv.turnCount !== turns.length) return false;
  if (conv.fingerprint === null) return true;
  return conv.fingerprint === computeLineageFingerprint(turns);
}

function commitConversationTurn(conv, userTexts) {
  conv.turnCount = userTexts.length;
  conv.fingerprint = computeLineageFingerprint(userTexts);
}

function pruneAgentBlobs(blobStore) {
  const excess = blobStore.size - MAX_BLOBS_PER_OWNER;
  if (excess <= 0) return;
  for (const key of blobStore.keys()) {
    if (excess <= 0) break;
    blobStore.delete(key);
    excess--;
  }
}

function retryDelayMs(hint) {
  let base;
  switch (hint) {
    case "blob_not_found": base = 200; break;
    case "resource_exhausted": base = 2000; break;
    case "timeout": base = 3000; break;
    default: base = 1000; break;
  }
  return Math.round(base * (1 + Math.random() * 0.5));
}

function classifyCursorError(message) {
  const text = String(message || "");
  if (/blob not found/i.test(text)) return "blob_not_found";
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

function encodeAgentModelParameter(parameter) {
  if (!parameter?.id) return null;
  return agentMessage(3, concatBuffers(
    agentString(1, parameter.id),
    agentString(2, parameter.value ?? ""),
  ));
}

function encodeRequestedAgentModel(model, selection = null, reasoningEffort = null) {
  const modelId = selection?.modelId || model;
  const rawParameters = [...(selection?.parameters || [])];
  const effort = String(reasoningEffort || "").toLowerCase();
  if (effort && effort !== "none") {
    const value = effort === "ultra" ? "max" : effort === "minimal" ? "low" : effort;
    const existingIndex = rawParameters.findIndex((parameter) => ["effort", "reasoning"].includes(parameter?.id));
    if (existingIndex >= 0) rawParameters[existingIndex] = { ...rawParameters[existingIndex], value };
    else rawParameters.push({ id: /gpt-/i.test(modelId) ? "reasoning" : "effort", value });
  }
  const parameters = rawParameters
    .map(encodeAgentModelParameter)
    .filter(Boolean);
  return concatBuffers(
    agentString(1, modelId),
    ...(selection?.maxMode === true ? [agentBool(2, true)] : []),
    ...parameters,
    agentBool(7, selection?.builtInModel !== false),
    ...(selection?.isVariantStringRepresentation === true ? [agentBool(8, true)] : []),
  );
}

function shouldResolveCursorModel(model) {
  return typeof model === "string" && (
    model === "auto"
    || model === "default"
    || model.startsWith("cursor-")
    || /(?:^|-)fast(?:-|$)/i.test(model)
    || /(?:^|-)x?(?:high|medium|low|max|ultra)(?:-|$)/i.test(model)
    || /(?:^|-)thinking(?:-|$)/i.test(model)
    || /\[[^\]]+=/.test(model)
  );
}

function decodeXmlEntities(value) {
  return String(value || "")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&amp;/g, "&");
}

function normalizeAgentToolCallId(value) {
  return String(value || "").split("\n")[0].trim();
}

function agentSessionOwner(credentials, model) {
  const account = credentials?.connectionId
    || credentials?.id
    || credentials?.accessToken
    || credentials?.apiKey
    || "anonymous";
  return crypto.createHash("sha256").update(`${model}\0${account}`).digest("hex");
}

function retainedToolCallKey(owner, toolCallId) {
  return `${owner}\0${normalizeAgentToolCallId(toolCallId)}`;
}

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
        content: capCursorToolResult(decodeXmlEntities(result)),
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

function textFromContent(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((part) => part?.type === "text" && typeof part.text === "string")
    .map((part) => part.text)
    .join("\n");
}

export function isAgentCapableRequest(body) {
  // Many compatible clients always attach their built-in tool schemas, even
  // for a normal text turn. Cursor's retired ChatService rejects those
  // requests. AgentService accepts text history, declared MCP tools, and prior
  // tool calls/results reconstructed as conversation text.
  return Array.isArray(body?.messages) && body.messages.length > 0 && body.messages.every((message) => {
    if (!["system", "user", "assistant", "tool"].includes(message?.role)) return false;
    const textContent = typeof message?.content === "string"
      || message?.content == null
      || Array.isArray(message?.content) && message.content.every((part) => part?.type === "text" && typeof part.text === "string");
    if (!textContent) return false;
    if (message.role === "assistant" && message.tool_calls != null && !Array.isArray(message.tool_calls)) return false;
    return true;
  });
}

const isAgentTextRequest = isAgentCapableRequest;

function toolCallsFromMessage(message) {
  if (!Array.isArray(message?.tool_calls) || message.tool_calls.length === 0) return "";
  return message.tool_calls.map((toolCall) => {
    const fn = toolCall?.function || {};
    return `[Tool call ${toolCall?.id || ""}: ${fn.name || "tool"}(${fn.arguments || "{}"})]`;
  }).join("\n");
}

function toolResultsFromMessage(message) {
  if (!Array.isArray(message?.tool_results) || message.tool_results.length === 0) return "";
  return message.tool_results.map((result) =>
    `[Tool result ${result?.tool_call_id || ""}${result?.tool_name ? ` (${result.tool_name})` : ""}]\n${capCursorToolResult(result?.result_content || result?.result || "")}`
  ).join("\n");
}

function encodeHistoryMessage(message) {
  const rawContent = textFromContent(message?.content);
  const content = message?.role === "tool"
    ? capCursorToolResult(rawContent)
    : capCursorToolXml(rawContent);
  const contentParts = [content];
  if (message?.role === "assistant") contentParts.push(toolCallsFromMessage(message));
  if (message?.role === "assistant") contentParts.push(toolResultsFromMessage(message));
  if (message?.role === "tool") {
    contentParts.unshift(`[Tool result ${message.tool_call_id || ""}${message.name ? ` (${message.name})` : ""}]`);
  }
  const encodedContent = contentParts.filter(Boolean).join("\n");
  if (!encodedContent) return null;

  // ConversationHistoryMessage.user / .assistant -> repeated content -> text.
  const text = agentString(1, encodedContent);
  if (message.role === "assistant") {
    return agentMessage(2, agentMessage(1, agentMessage(1, text)));
  }
  return agentMessage(1, agentMessage(1, agentMessage(1, text)));
}

export function buildAgentRunFrame(messages, model, tools = [], modelSelection = null, reasoningEffort = null, resume = null) {
  // resume = { conversationId, checkpoint } — continue a stored Cursor conversation
  // instead of replaying the whole history as fresh context.
  const checkpoint = resume?.checkpoint || null;
  const conversationId = resume?.conversationId || null;
  if (!checkpoint) messages = compactCursorMessages(messages, model, tools);
  const system = messages
    .filter((message) => message?.role === "system")
    .map((message) => textFromContent(message.content))
    .filter(Boolean)
    .join("\n\n");
  const chatMessages = messages.filter((message) => message?.role !== "system");
  const currentIndex = [...chatMessages].map((message) => message?.role).findLastIndex((role) => role === "user" || role === "tool");
  const current = currentIndex >= 0 ? chatMessages[currentIndex] : chatMessages.at(-1);
  // With a checkpoint the conversation state already holds prior turns; send only
  // the current user message so Cursor resumes instead of replaying history.
  const history = !checkpoint
    ? chatMessages
      .slice(0, currentIndex >= 0 ? currentIndex : -1)
      .map(encodeHistoryMessage)
      .filter(Boolean)
    : [];
  const rawCurrentText = textFromContent(current?.content);
  const currentText = current?.role === "tool"
    ? [`[Tool result ${current.tool_call_id || ""}${current.name ? ` (${current.name})` : ""}]`, capCursorToolResult(rawCurrentText)].filter(Boolean).join("\n")
    : capCursorToolXml(rawCurrentText);
  const userText = currentText || "Continue.";

  // agent.v1.UserMessageAction.user_message and its optional history.
  const userMessage = concatBuffers(
    agentString(1, userText),
    agentString(2, crypto.randomUUID()),
  );
  const conversationHistory = history.length
    ? concatBuffers(...history.map((entry) => agentMessage(1, entry)))
    : null;
  const userAction = concatBuffers(
    agentMessage(1, userMessage),
    ...(conversationHistory ? [agentMessage(7, conversationHistory)] : []),
  );
  const conversationAction = agentMessage(1, userAction);
  const requestedModel = encodeRequestedAgentModel(model, modelSelection, reasoningEffort);
  const runRequest = concatBuffers(
    // An empty ConversationStateStructure starts a fresh local agent session.
    agentMessage(1, checkpoint || new Uint8Array()),
    agentMessage(2, conversationAction),
    ...(tools.length ? [agentMessage(4, encodeMcpTools(tools))] : []),
    ...(conversationId ? [agentString(5, conversationId)] : []),
    ...(system ? [agentString(8, system)] : []),
    agentMessage(9, requestedModel),
  );

  // agent.v1.AgentClientMessage.run_request.
  return wrapConnectRPCFrame(agentMessage(1, runRequest));
}

function extractAgentString(message, field) {
  const value = message?.get(field)?.[0]?.value;
  return value instanceof Uint8Array || Buffer.isBuffer(value)
    ? Buffer.from(value).toString("utf8")
    : "";
}

function extractAgentVarint(message, field) {
  const value = message?.get(field)?.[0]?.value;
  return typeof value === "number" ? value : null;
}

function describeAgentFields(message) {
  return [...(message?.entries?.() || [])].map(([field, entries]) => {
    const values = entries.map(({ wireType, value }) => {
      if (wireType === PROTOBUF_VARINT) return `v${value}`;
      if (value instanceof Uint8Array || Buffer.isBuffer(value)) return `l${value.length}`;
      return `w${wireType}`;
    }).join(",");
    return `${field}:${values}`;
  }).join(" ");
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
    if (!(flags & COMPRESS_FLAG.TRAILER) && onFrame(payload) === false) break;
  }
  return pending;
}

function createRequestContextResponse(execRequest, tools = []) {
  // AgentService asks every run for client context. 9router has no IDE file
  // context, so acknowledge with an empty RequestContext.
  const requestContext = tools.length
    ? concatBuffers(...tools.map((tool) => agentMessage(7, encodeMcpToolDefinition(tool))))
    : new Uint8Array();
  const requestContextSuccess = agentMessage(1, requestContext);
  const requestContextResult = agentMessage(1, requestContextSuccess);
  const execClientMessage = concatBuffers(
    ...(extractAgentVarint(execRequest, 1) != null ? [encodeField(1, PROTOBUF_VARINT, extractAgentVarint(execRequest, 1))] : []),
    agentMessage(10, requestContextResult),
    ...(execRequest?.has(15) ? [agentMessage(15, execRequest.get(15)[0].value)] : []),
  );
  return wrapConnectRPCFrame(agentMessage(2, execClientMessage));
}

function createListMcpResourcesResponse(execRequest) {
  // 9router exposes callable MCP tools, but it has no Cursor MCP resource
  // catalogue. Return a successful empty list so the agent can continue.
  const result = agentMessage(1, new Uint8Array());
  const execClientMessage = concatBuffers(
    ...(extractAgentVarint(execRequest, 1) != null ? [encodeField(1, PROTOBUF_VARINT, extractAgentVarint(execRequest, 1))] : []),
    agentMessage(17, result),
    ...(execRequest?.has(15) ? [agentMessage(15, execRequest.get(15)[0].value)] : []),
  );
  return wrapConnectRPCFrame(agentMessage(2, execClientMessage));
}

function createMcpResultResponse(pending, content, isError = false) {
  const mcpResult = encodeMcpResultSuccess({ textItems: [capCursorToolResult(content)], isError });
  const execClientMessage = concatBuffers(
    ...(pending.id != null ? [encodeField(1, PROTOBUF_VARINT, pending.id)] : []),
    agentMessage(11, mcpResult),
    ...(pending.execId ? [agentMessage(15, pending.execId)] : []),
  );
  return wrapConnectRPCFrame(agentMessage(2, execClientMessage));
}

const CURSOR_STREAM_DEBUG = process.env.CURSOR_STREAM_DEBUG === "1";
const CURSOR_TRACE_CONTENT = process.env.CURSOR_TRACE_CONTENT === "1";
const debugLog = (...args) => {
  if (CURSOR_STREAM_DEBUG) console.log(...args);
};
const traceLog = (runId, ...args) => {
  if (CURSOR_TRACE_CONTENT) console.log(`[CURSOR TRACE ${runId}]`, ...args);
};

function traceInteractionUpdate(runId, update) {
  for (const [field, entries] of update.entries()) {
    for (const { wireType, value } of entries) {
      if (wireType === PROTOBUF_VARINT) {
        traceLog(runId, `interaction_update field=${field} varint=${value}`);
        continue;
      }

      const bytes = Buffer.from(value || []);
      if (field === 1 || field === 4) {
        const text = extractAgentString(decodeMessage(bytes), 1);
        traceLog(runId, `${field === 1 ? "text_delta" : "thinking_delta"}=${JSON.stringify(text)}`);
      } else if (field === 5 || field === 8) {
        const nested = decodeMessage(bytes);
        traceLog(runId, `${field === 5 ? "thinking_completed" : "token_delta"} fields=${describeAgentFields(nested)}`);
      } else {
        traceLog(
          runId,
          `interaction_update field=${field} length=${bytes.length} base64=${bytes.toString("base64")}`
        );
      }
    }
  }
}

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
    const reasoningEffort = body.reasoning_effort || body.reasoning?.effort || null;
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

  async executeAgent({ model, body, stream, credentials, signal, log, modelCatalog, proxyOptions = null }) {
    const agentEndpoint = PROVIDER_OAUTH.cursor?.agentEndpoint;
    if (!agentEndpoint) throw new Error("Cursor AgentService endpoint is not configured");

    const url = `${agentEndpoint}${AGENT_RUN_PATH}`;
    const headers = this.buildHeaders(credentials);
    const runId = crypto.randomUUID().slice(0, 8);
    const requestController = new AbortController();
    if (signal?.addEventListener) {
      if (signal.aborted) requestController.abort(signal.reason);
      else signal.addEventListener("abort", () => requestController.abort(signal.reason), { once: true });
    }

    if (proxyOptions?.enabled || proxyOptions?.connectionProxyEnabled || proxyOptions?.vercelRelayUrl) {
      traceLog(runId, `proxyOptions pending http2 proxy wiring enabled=${Boolean(proxyOptions?.enabled)} conn=${Boolean(proxyOptions?.connectionProxyEnabled)}`);
    }

    const toolResults = extractCursorToolResults(body);
    const sessionOwner = agentSessionOwner(credentials, model);
    const userTexts = extractUserTexts(body);
    const conv = getAgentConversation(sessionOwner);
    if (!validateConversationLineage(conv, userTexts)) {
      traceLog(runId, `lineage_mismatch reset conv=${conv.conversationId.slice(0, 8)}`);
      resetAgentConversation(conv);
    }

    const reasoningEffort = body.reasoning_effort || body.reasoning?.effort || null;
    const modelSelection = modelCatalog !== undefined
      ? resolveCursorModelSelection(modelCatalog, model)
      : shouldResolveCursorModel(model)
        ? await resolveCursorModel(credentials, model, { signal: requestController.signal, log })
        : null;
    if (modelSelection) {
      debugLog(
        `[CURSOR AGENT ${runId}] Catalog model=${modelSelection.modelId}, `
        + `matchedBy=${modelSelection.matchedBy}, params=${modelSelection.parameters?.length || 0}`,
      );
    }

    // Cursor's AgentService streams thinking deltas for OpenAI-format clients.
    // Claude Code (Anthropic) requires cryptographically signed thinking blocks,
    // so reasoning stays upstream-only for that UA (issue #643).
    const ua = credentials?.rawHeaders?.["user-agent"] || "";
    const forwardThinking = !(ua.includes("claude-cli") || ua.includes("claude-code") || ua.includes("Claude Code"));

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
      body.tools || [],
      modelSelection,
      reasoningEffort,
      conv.conversationId ? { conversationId: conv.conversationId, checkpoint: conv.checkpoint } : null,
    );

    // One full attempt: (re)open the session, write the run frame, consume.
    // Emits events via onEvent. Retryable failures throw an Error whose
    // message classifies via classifyCursorError (blob_not_found / resource_exhausted).
    const runAttempt = async (onEvent) => {
      const retained = acquireRetainedAgentSession(sessionOwner, toolResults);
      let retainedState = retained?.state || null;
      let keepSessionOpen = false;
      let session;
      let responseHeaders;
      const closeRetainedOnAbort = () => {
        if (retainedState) closeRetainedAgentSession(retainedState);
      };
      if (requestController.signal.aborted) closeRetainedOnAbort();
      else requestController.signal.addEventListener("abort", closeRetainedOnAbort, { once: true });
      traceLog(runId, `tool_results=${JSON.stringify(toolResults)} retained=${Boolean(retainedState)}`);
      try {
        if (requestController.signal.aborted) throw new Error("Request aborted");
        if (retainedState) {
          session = retainedState.session;
          consumeRetainedToolResults(retainedState, retained.matchedResults);
          keepSessionOpen = true;
          responseHeaders = { ":status": 200 };
          debugLog(
            `[CURSOR AGENT ${runId}] Resume messages=${body.messages?.length || 0}, `
            + `toolResults=${retained.matchedResults.map((result) => result.toolCallId).join(",")}`
          );
          traceLog(runId, `resume=${JSON.stringify(retained.matchedResults)}`);
        } else {
          session = this.openAgentHttp2Stream(url, headers, requestController.signal);
          debugLog(
            `[CURSOR AGENT ${runId}] Run model=${model}, messages=${body.messages?.length || 0}, `
            + `roles=${(body.messages || []).map((message) => message?.role || "?").join(",")}, `
            + `tools=${body.tools?.length || 0}`
          );
          traceLog(runId, `request=${JSON.stringify({ model, stream, body })}`);
          const runFrame = buildRunFrame();
          traceLog(runId, `client_frame length=${runFrame.length} base64=${Buffer.from(runFrame).toString("base64")}`);
          session.write(runFrame);
          responseHeaders = await session.responseHeaders;
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
            if (value) errorText += Buffer.from(value).toString("utf8");
            if (done) break;
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

      let pending = retainedState?.buffered || Buffer.alloc(0);
      if (retainedState) retainedState.buffered = Buffer.alloc(0);
      let finished = false;
      let timedOut = false;

      const consume = async () => {
        let heartbeatTimer;
        let safetyTimer;
        try {
          heartbeatTimer = setInterval(() => {
            if (!finished) {
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
            pending = decodeAgentFrames(pending, (payload) => {
              // A single read can carry several frames; once the turn is over the
              // remaining complete frames stay buffered for the retained resume.
              if (finished) return false;
              traceLog(runId, `server_frame length=${payload.length} base64=${Buffer.from(payload).toString("base64")}`);
              const serverMessage = decodeMessage(payload);
              debugLog(`[CURSOR AGENT ${runId}] Server fields: ${describeAgentFields(serverMessage)}`);

              // KV is a bidirectional side channel. GetBlob answers from the
              // per-conversation blob store; SetBlob stores for later turns
              // (pi-cursor parity — an empty GetBlob breaks long conversations).
              const kvEvent = decodeAgentKvServerEvent(payload);
              if (kvEvent) {
                if (kvEvent.kind === "get") {
                  const key = Buffer.from(kvEvent.blobId).toString("hex");
                  const blob = conv.blobStore.get(key);
                  if (!blob) traceLog(runId, `blob_get_miss key=${key.slice(0, 16)} store=${conv.blobStore.size}`);
                  session.write(encodeAgentKvGetResult(kvEvent.id, blob || new Uint8Array(), kvEvent.metadata));
                } else {
                  conv.blobStore.set(Buffer.from(kvEvent.blobId).toString("hex"), kvEvent.blobData);
                  pruneAgentBlobs(conv.blobStore);
                  session.write(encodeAgentKvSetResult(kvEvent.id, kvEvent.metadata));
                }
                return true;
              }

              // agent.v1.AgentServerMessage.conversation_checkpoint_update (field 3).
              // Persist the serialized conversation state so the next turn resumes
              // this conversation instead of replaying history (pi-cursor parity).
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
                      if (typeof used === "number") conv.tokenUsage = { used, max: typeof max === "number" ? max : null };
                    }
                  } catch {}
                  traceLog(runId, `checkpoint bytes=${bytes.length}`);
                }
                return true;
              }

              // agent.v1.AgentServerMessage.interaction_update
              if (serverMessage.has(1)) {
                const update = decodeMessage(serverMessage.get(1)[0].value);
                traceInteractionUpdate(runId, update);
                if (update.has(1)) {
                  const textDelta = extractAgentString(decodeMessage(update.get(1)[0].value), 1);
                  if (textDelta) onEvent({ type: "text", value: textDelta });
                }
                if (update.has(4) && forwardThinking) {
                  const thinkingDelta = extractAgentString(decodeMessage(update.get(4)[0].value), 1);
                  if (thinkingDelta) onEvent({ type: "thinking", value: thinkingDelta });
                }
                if (update.has(8)) {
                  try {
                    const tokenDelta = decodeMessage(update.get(8)[0].value);
                    const tokens = extractAgentVarint(tokenDelta, 1);
                    if (typeof tokens === "number" && tokens > 0) conv.outputTokens += tokens;
                  } catch {}
                }
                if (update.has(14)) {
                  finished = true;
                  onEvent({ type: "done" });
                  return false;
                }
              }

              // AgentService requests IDE context before producing a response.
              // Return an empty context; 9router is not coupled to an editor.
              if (serverMessage.has(2)) {
                const execRequest = decodeMessage(serverMessage.get(2)[0].value);
                traceLog(runId, `exec_request fields=${describeAgentFields(execRequest)}`);
                const execEvent = decodeExecServerEvent(payload);
                if (execEvent?.kind === "exec_request_context") {
                  // Include declared MCP tools in the context envelope. Cursor
                  // versions that require context-local tool discovery otherwise
                  // accept the handshake but never expose the tools to the model.
                  session.write(createRequestContextResponse(execRequest, body.tools || []));
                } else if (execEvent?.kind === "exec_list_mcp_resources") {
                  session.write(encodeAgentEmptyListMcpResources(execEvent.execMsgId, execEvent.execId));
                } else if (execEvent?.kind === "exec_mcp") {
                  const toolCallId = normalizeAgentToolCallId(
                    execEvent.toolCallId || execEvent.execId || `call_${crypto.randomUUID()}`
                  );
                  const toolName = execEvent.toolName || "tool";
                  retainedState = retainAgentToolCall(session, sessionOwner, toolCallId, execRequest);
                  keepSessionOpen = true;
                  traceLog(runId, `mcp_exec tool=${toolName} call_id=${toolCallId} args=${JSON.stringify(execEvent.args || {})}`);
                  finished = true;
                  onEvent({
                    type: "tool_call",
                    value: {
                      id: toolCallId,
                      type: "function",
                      function: {
                        name: toolName,
                        arguments: JSON.stringify(execEvent.args || {}),
                      },
                    },
                  });
                  return false;
                } else if (execEvent) {
                  // Typed rejection lets Cursor continue/fallback instead of
                  // wedging the bidirectional stream on an IDE-only operation.
                  handleNativeExec(execEvent, session);
                } else {
                  debugLog(`[CURSOR AGENT ${runId}] Unsupported exec request fields: ${describeAgentFields(execRequest)}`);
                  finished = true;
                  onEvent({ type: "error", value: "Cursor AgentService requested an unsupported IDE tool" });
                  return false;
                }
              }

              if (serverMessage.has(7)) {
                const query = Buffer.from(serverMessage.get(7)[0].value);
                const interaction = decodeAgentInteractionQuery(query);
                if (interaction?.kind) {
                  // Auto-approve harmless searches/fetches so the agent loop
                  // never stalls waiting on an IDE dialog; reject questions,
                  // mode switches, and plan writes (no UI here to answer).
                  const approved = [2, 5, 6, 8, 9].includes(interaction.kind);
                  const reason = "Interaction unavailable in headless proxy mode";
                  const response = encodeAgentInteractionResponse(interaction.id, interaction.kind, approved, reason);
                  if (response) session.write(response);
                  traceLog(runId, `interaction_query kind=${interaction.kind} id=${interaction.id} approved=${approved}`);
                } else {
                  traceLog(
                    runId,
                    `interaction_query fields=${describeAgentFields(decodeMessage(query))} base64=${query.toString("base64")}`
                  );
                }
              }
              return true;
            });
            if (finished) break;
            const { done, value } = await session.read();
            if (done) break;
            pending = Buffer.concat([pending, Buffer.from(value)]);
          }
        } catch (error) {
          if (retainedState) closeRetainedAgentSession(retainedState);
          const hint = classifyCursorError(error.message);
          if (hint && !outputStarted) {
            const err = new Error(hint);
            err.retryHint = hint;
            throw err;
          }
          throw error;
        } finally {
          if (heartbeatTimer) clearInterval(heartbeatTimer);
          if (safetyTimer) clearTimeout(safetyTimer);
          traceLog(runId, `consume_finally finished=${finished} pending=${pending.length}`);
          if (keepSessionOpen && retainedState) {
            retainedState.buffered = pending;
            releaseRetainedAgentSession(retainedState, true);
          } else {
            try { session.end(); } catch {}
            try { session.close(); } catch {}
          }
          if (timedOut && !outputStarted) {
            const err = new Error("timeout");
            err.retryHint = "timeout";
            throw err;
          }
          if (!finished && !timedOut) onEvent({ type: "done" });
        }
      };

      await consume();
      return { keepSessionOpen, retainedState };
    };

    // Wrap events so we can refuse to retry once real output has started.
    let outputStarted = false;
    const makeWrappedEvent = (onEvent) => (event) => {
      if (event.type === "text" || event.type === "thinking" || event.type === "tool_call") outputStarted = true;
      onEvent(event);
    };

    // Retry loop: fresh session per attempt; blob_not_found resets the
    // conversation (new ID) so Cursor replays from client history.
    const runWithRetry = async (onEvent) => {
      outputStarted = false;
      const wrapped = makeWrappedEvent(onEvent);
      let attempt = 0;
      for (;;) {
        try {
          const result = await runAttempt(wrapped);
          // Commit lineage only after a successful attempt (done or tool_call).
          commitConversationTurn(conv, userTexts);
          return result;
        } catch (error) {
          const hint = error.retryHint || classifyCursorError(error.message);
          if (!hint || attempt >= CURSOR_AGENT_MAX_RETRIES || outputStarted) throw error;
          attempt++;
          if (hint === "blob_not_found") {
            traceLog(runId, `retry blob_not_found: reset conversation`);
            resetAgentConversation(conv);
          }
          traceLog(runId, `retry ${attempt}/${CURSOR_AGENT_MAX_RETRIES} hint=${hint}`);
          await sleepMs(retryDelayMs(hint));
        }
      }
    };

    const buildAgentUsage = (contentLength) => {
      const tokenUsage = conv.tokenUsage;
      const output = conv.outputTokens || 0;
      if (tokenUsage?.used) {
        const total = Math.max(output, tokenUsage.used);
        return {
          prompt_tokens: Math.max(0, total - output),
          completion_tokens: output,
          total_tokens: total,
        };
      }
      return estimateUsage(body, contentLength, FORMATS.OPENAI);
    };

    if (stream === false) {
      let content = "";
      let reasoning = "";
      let toolCall = null;
      let agentError = null;
      // Connection/stream failures propagate (reject) so callers treat them as
      // transport errors; only agent-level protocol errors become BAD_REQUEST.
      await runWithRetry((event) => {
        if (event.type === "text") content += event.value;
        else if (event.type === "thinking") reasoning += event.value;
        else if (event.type === "tool_call") toolCall = event.value;
        else if (event.type === "error") agentError = event.value;
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
      return {
        response: new Response(JSON.stringify({
          id: responseId,
          object: "chat.completion",
          created,
          model,
          choices: [{
            index: 0,
            message: {
              role: "assistant",
              content: content || null,
              ...(reasoning ? { reasoning_content: reasoning } : {}),
              ...(toolCall ? { tool_calls: [toolCall] } : {}),
            },
            finish_reason: toolCall ? "tool_calls" : "stop",
          }],
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
        let keepaliveTimer;
        let streamOutputStarted = false;
        const clearKeepalive = () => {
          if (keepaliveTimer) {
            clearInterval(keepaliveTimer);
            keepaliveTimer = null;
          }
        };
        streamState.clearKeepalive = clearKeepalive;
        const markStreamOutput = () => {
          if (!streamOutputStarted) {
            streamOutputStarted = true;
            clearKeepalive();
          }
        };
        const closeWithAgentError = (error) => {
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

        keepaliveTimer = setInterval(() => {
          if (!streamOutputStarted && !requestController.signal.aborted) {
            try {
              controller.enqueue(encoder.encode(": keepalive\n\n"));
            } catch {}
          }
        }, CURSOR_SSE_KEEPALIVE_MS);
        keepaliveTimer.unref?.();

        runWithRetry((event) => {
          if (event.type === "text") {
            markStreamOutput();
            controller.enqueue(encoder.encode(chatChunkSse({ id: responseId, created, model, delta: { content: event.value } })));
          } else if (event.type === "thinking") {
            markStreamOutput();
            controller.enqueue(encoder.encode(chatChunkSse({ id: responseId, created, model, delta: { reasoning_content: event.value } })));
          } else if (event.type === "tool_call") {
            markStreamOutput();
            controller.enqueue(encoder.encode(chatChunkSse({
              id: responseId,
              created,
              model,
              delta: { tool_calls: [{ index: 0, ...event.value }] },
            })));
            controller.enqueue(encoder.encode(chatChunkSse({ id: responseId, created, model, delta: {}, finishReason: "tool_calls" })));
            clearKeepalive();
            controller.enqueue(encoder.encode(SSE_DONE));
            controller.close();
          } else if (event.type === "error") {
            // An SSE error frame, not a content delta: a protocol failure must not
            // be rendered to the user as the assistant's reply, and downstream
            // usage tracking must not record the turn as a success.
            clearKeepalive();
            controller.enqueue(encoder.encode(sseChunk({ error: { message: event.value, type: "api_error" } })));
            controller.enqueue(encoder.encode(SSE_DONE));
            controller.close();
          } else if (event.type === "done") {
            clearKeepalive();
            controller.enqueue(encoder.encode(chatChunkSse({ id: responseId, created, model, delta: {}, finishReason: "stop" })));
            controller.enqueue(encoder.encode(SSE_DONE));
            controller.close();
          }
        }).catch((error) => closeWithAgentError(error));
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
    if (isAgentTextRequest(body)) {
      try {
        return await this.executeAgent({ model, body, stream, credentials, signal, log, proxyOptions });
      } catch (error) {
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
