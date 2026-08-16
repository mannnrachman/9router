/**
 * Cursor Protobuf Encoder/Decoder
 * Implements ConnectRPC protobuf wire format for Cursor API
 */

import { v4 as uuidv4 } from "uuid";
import zlib from "zlib";

const DEBUG = process.env.CURSOR_PROTOBUF_DEBUG === "1";
const log = (tag, ...args) => DEBUG && console.log(`[PROTOBUF:${tag}]`, ...args);
const textDecoder = new TextDecoder();

const PROTOBUF_SCHEMA_VERSION = "1.1.3";

// ==================== SCHEMAS ====================

const WIRE_TYPE = { VARINT: 0, FIXED64: 1, LEN: 2, FIXED32: 5 };

const ROLE = { USER: 1, ASSISTANT: 2 };

const UNIFIED_MODE = { CHAT: 1, AGENT: 2 };

const THINKING_LEVEL = { UNSPECIFIED: 0, MEDIUM: 1, HIGH: 2 };

function normalizeCursorThinkingLevel(reasoningEffort) {
  const effort = String(reasoningEffort || "").toLowerCase();
  if (["high", "xhigh", "max", "ultra"].includes(effort)) return THINKING_LEVEL.HIGH;
  if (["low", "medium"].includes(effort)) return THINKING_LEVEL.MEDIUM;
  return THINKING_LEVEL.UNSPECIFIED;
}
const CLIENT_SIDE_TOOL_V2 = { MCP: 19 };
const CLIENT_SIDE_TOOL_V2_MCP = 19;

const FIELD = {
  // StreamUnifiedChatRequestWithTools (top level)
  REQUEST: 1,

  // StreamUnifiedChatRequest
  MESSAGES: 1,
  UNKNOWN_2: 2,
  INSTRUCTION: 3,
  UNKNOWN_4: 4,
  MODEL: 5,
  WEB_TOOL: 8,
  UNKNOWN_13: 13,
  CURSOR_SETTING: 15,
  UNKNOWN_19: 19,
  CONVERSATION_ID: 23,
  METADATA: 26,
  IS_AGENTIC: 27,
  SUPPORTED_TOOLS: 29,
  MESSAGE_IDS: 30,
  MCP_TOOLS: 34,
  LARGE_CONTEXT: 35,
  UNKNOWN_38: 38,
  UNIFIED_MODE: 46,
  UNKNOWN_47: 47,
  SHOULD_DISABLE_TOOLS: 48,
  THINKING_LEVEL: 49,
  UNKNOWN_51: 51,
  UNKNOWN_53: 53,
  UNIFIED_MODE_NAME: 54,

  // ConversationMessage
  MSG_CONTENT: 1,
  MSG_ROLE: 2,
  MSG_ID: 13,
  MSG_TOOL_RESULTS: 18,
  MSG_IS_AGENTIC: 29,
  MSG_SERVER_BUBBLE_ID: 32,
  MSG_UNIFIED_MODE: 47,
  MSG_SUPPORTED_TOOLS: 51,

  // ConversationMessage.ToolResult
  TOOL_RESULT_CALL_ID: 1,
  TOOL_RESULT_NAME: 2,
  TOOL_RESULT_INDEX: 3,
  TOOL_RESULT_RAW_ARGS: 5,
  TOOL_RESULT_RESULT: 8,
  TOOL_RESULT_TOOL_CALL: 11,
  TOOL_RESULT_MODEL_CALL_ID: 12,

  // ClientSideToolV2Result (nested inside ToolResult.result)
  CLIENT_RESULT_TOOL: 1,
  CLIENT_RESULT_MCP_RESULT: 28,
  CLIENT_RESULT_TOOL_CALL_ID: 35,
  CLIENT_RESULT_MODEL_CALL_ID: 48,
  CLIENT_RESULT_TOOL_INDEX: 49,
  // Aliases used by encodeClientSideToolV2Result
  CV2R_TOOL: 1,
  CV2R_MCP_RESULT: 28,
  CV2R_CALL_ID: 35,
  CV2R_MODEL_CALL_ID: 48,
  CV2R_TOOL_INDEX: 49,

  // MCPResult (nested inside ClientSideToolV2Result.mcp_result)
  MCP_RESULT_SELECTED_TOOL: 1,
  MCP_RESULT_RESULT: 2,
  // Aliases used by encodeMcpResult
  MCPR_SELECTED_TOOL: 1,
  MCPR_RESULT: 2,

  // ClientSideToolV2Call (nested inside ToolResult.tool_call)
  CLIENT_CALL_TOOL: 1,
  CLIENT_CALL_MCP_PARAMS: 27,
  CLIENT_CALL_TOOL_CALL_ID: 3,
  CLIENT_CALL_NAME: 9,
  CLIENT_CALL_RAW_ARGS: 10,
  CLIENT_CALL_TOOL_INDEX: 48,
  CLIENT_CALL_MODEL_CALL_ID: 49,
  // Aliases used by encodeClientSideToolV2Call
  CV2C_TOOL: 1,
  CV2C_MCP_PARAMS: 27,
  CV2C_CALL_ID: 3,
  CV2C_NAME: 9,
  CV2C_RAW_ARGS: 10,
  CV2C_TOOL_INDEX: 48,
  CV2C_MODEL_CALL_ID: 49,

  // Model
  MODEL_NAME: 1,
  MODEL_EMPTY: 4,

  // Instruction
  INSTRUCTION_TEXT: 1,

  // CursorSetting
  SETTING_PATH: 1,
  SETTING_UNKNOWN_3: 3,
  SETTING_UNKNOWN_6: 6,
  SETTING_UNKNOWN_8: 8,
  SETTING_UNKNOWN_9: 9,

  // CursorSetting.Unknown6
  SETTING6_FIELD_1: 1,
  SETTING6_FIELD_2: 2,

  // Metadata
  META_PLATFORM: 1,
  META_ARCH: 2,
  META_VERSION: 3,
  META_CWD: 4,
  META_TIMESTAMP: 5,

  // MessageId
  MSGID_ID: 1,
  MSGID_SUMMARY: 2,
  MSGID_ROLE: 3,

  // MCPTool
  MCP_TOOL_NAME: 1,
  MCP_TOOL_DESC: 2,
  MCP_TOOL_PARAMS: 3,
  MCP_TOOL_SERVER: 4,

  // StreamUnifiedChatResponseWithTools (response)
  TOOL_CALL: 1,
  RESPONSE: 2,

  // ClientSideToolV2Call
  TOOL_ID: 3,
  TOOL_NAME: 9,
  TOOL_RAW_ARGS: 10,
  TOOL_IS_LAST: 11,
  TOOL_IS_LAST_ALT: 15,
  TOOL_MCP_PARAMS: 27,

  // MCPParams
  MCP_TOOLS_LIST: 1,

  // MCPParams.Tool (nested)
  MCP_NESTED_NAME: 1,
  MCP_NESTED_PARAMS: 3,

  // StreamUnifiedChatResponse
  RESPONSE_TEXT: 1,
  THINKING: 25,

  // Thinking
  THINKING_TEXT: 1
};

// Known response field numbers — used to detect unknown fields from protocol updates
const KNOWN_RESPONSE_FIELDS = new Set([
  FIELD.TOOL_CALL,
  FIELD.RESPONSE,
  FIELD.TOOL_ID,
  FIELD.TOOL_NAME,
  FIELD.TOOL_RAW_ARGS,
  FIELD.TOOL_IS_LAST,
  FIELD.TOOL_MCP_PARAMS,
  FIELD.RESPONSE_TEXT,
  FIELD.THINKING
]);

// ==================== PRIMITIVE ENCODING ====================

export function encodeVarint(value) {
  const bytes = [];
  while (value >= 0x80) {
    bytes.push((value & 0x7F) | 0x80);
    value >>>= 7;
  }
  bytes.push(value & 0x7F);
  return new Uint8Array(bytes);
}

export function encodeField(fieldNum, wireType, value) {
  const tag = (fieldNum << 3) | wireType;
  const tagBytes = encodeVarint(tag);

  if (wireType === WIRE_TYPE.VARINT) {
    const valueBytes = encodeVarint(value);
    return concatArrays(tagBytes, valueBytes);
  }

  if (wireType === WIRE_TYPE.LEN) {
    const dataBytes = typeof value === "string" 
      ? new TextEncoder().encode(value)
      : value instanceof Uint8Array ? value
      : Buffer.isBuffer(value) ? new Uint8Array(value)
      : new Uint8Array(0);
    
    const lengthBytes = encodeVarint(dataBytes.length);
    return concatArrays(tagBytes, lengthBytes, dataBytes);
  }

  if (wireType === WIRE_TYPE.FIXED64 || wireType === WIRE_TYPE.FIXED32) {
    const length = wireType === WIRE_TYPE.FIXED64 ? 8 : 4;
    const valueBytes = value instanceof Uint8Array
      ? value
      : Buffer.isBuffer(value) ? new Uint8Array(value) : new Uint8Array(0);
    if (valueBytes.length !== length) return new Uint8Array(0);
    return concatArrays(tagBytes, valueBytes);
  }

  return new Uint8Array(0);
}

function concatArrays(...arrays) {
  const totalLength = arrays.reduce((sum, arr) => sum + arr.length, 0);
  const result = new Uint8Array(totalLength);
  let offset = 0;
  for (const arr of arrays) {
    result.set(arr, offset);
    offset += arr.length;
  }
  return result;
}

// google.protobuf.Value and the AgentService MCP messages use the standard
// protobuf JSON representation rather than JSON strings.
export function encodeAgentValue(value) {
  if (value === null || value === undefined) return encodeField(1, WIRE_TYPE.VARINT, 0);
  if (typeof value === "number") {
    const bytes = new Uint8Array(8);
    new DataView(bytes.buffer).setFloat64(0, value, true);
    return encodeField(2, WIRE_TYPE.FIXED64, bytes);
  }
  if (typeof value === "string") return encodeField(3, WIRE_TYPE.LEN, value);
  if (typeof value === "boolean") return encodeField(4, WIRE_TYPE.VARINT, value ? 1 : 0);
  if (Array.isArray(value)) {
    const list = concatArrays(...value.map((item) => encodeField(1, WIRE_TYPE.LEN, encodeAgentValue(item))));
    return encodeField(6, WIRE_TYPE.LEN, list);
  }
  if (typeof value === "object") {
    const struct = concatArrays(...Object.entries(value).map(([key, item]) => {
      const entry = concatArrays(
        encodeField(1, WIRE_TYPE.LEN, key),
        encodeField(2, WIRE_TYPE.LEN, encodeAgentValue(item))
      );
      return encodeField(1, WIRE_TYPE.LEN, entry);
    }));
    return encodeField(5, WIRE_TYPE.LEN, struct);
  }
  return encodeField(1, WIRE_TYPE.VARINT, 0);
}

export function decodeAgentValue(data) {
  const value = decodeMessage(data);
  if (value.has(1)) return null;
  if (value.has(2)) {
    const bytes = value.get(2)[0].value;
    return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getFloat64(0, true);
  }
  if (value.has(3)) return textDecoder.decode(value.get(3)[0].value);
  if (value.has(4)) return value.get(4)[0].value !== 0;
  if (value.has(5)) {
    const result = {};
    const struct = decodeMessage(value.get(5)[0].value);
    for (const field of struct.get(1) || []) {
      const entry = decodeMessage(field.value);
      const key = entry.has(1) ? textDecoder.decode(entry.get(1)[0].value) : "";
      if (key && entry.has(2)) result[key] = decodeAgentValue(entry.get(2)[0].value);
    }
    return result;
  }
  if (value.has(6)) {
    const list = decodeMessage(value.get(6)[0].value);
    return (list.get(1) || []).map((field) => decodeAgentValue(field.value));
  }
  return null;
}

export function encodeMcpToolDefinition(tool) {
  const definition = tool?.function || tool || {};
  const name = definition.name || "tool";
  const description = definition.description || "";
  const inputSchema = definition.parameters || definition.inputSchema || definition.input_schema || {};
  return concatArrays(
    encodeField(1, WIRE_TYPE.LEN, name),
    ...(description ? [encodeField(2, WIRE_TYPE.LEN, description)] : []),
    encodeField(3, WIRE_TYPE.LEN, encodeAgentValue(inputSchema)),
    encodeField(4, WIRE_TYPE.LEN, "9router"),
    encodeField(5, WIRE_TYPE.LEN, name)
  );
}

export function encodeMcpTools(tools = []) {
  return concatArrays(...(tools || []).map((tool) =>
    encodeField(1, WIRE_TYPE.LEN, encodeMcpToolDefinition(tool))
  ));
}

export function decodeMcpArgs(data) {
  const message = decodeMessage(data);
  const stringField = (field) => message.has(field)
    ? textDecoder.decode(message.get(field)[0].value)
    : "";
  const args = {};
  for (const field of message.get(2) || []) {
    const entry = decodeMessage(field.value);
    if (!entry.has(1) || !entry.has(2)) continue;
    args[textDecoder.decode(entry.get(1)[0].value)] = decodeAgentValue(entry.get(2)[0].value);
  }
  return { name: stringField(1), args, toolCallId: stringField(3), toolName: stringField(5) };
}

export function encodeMcpResultSuccess({ textItems = [], imageItems = [], isError = false } = {}) {
  const content = [
    ...textItems.map((text) => encodeField(1, WIRE_TYPE.LEN,
      encodeField(1, WIRE_TYPE.LEN, encodeField(1, WIRE_TYPE.LEN, String(text)))
    )),
    ...imageItems.map((image) => encodeField(1, WIRE_TYPE.LEN,
      encodeField(2, WIRE_TYPE.LEN, concatArrays(
        encodeField(1, WIRE_TYPE.LEN, image?.data || new Uint8Array()),
        encodeField(2, WIRE_TYPE.LEN, image?.mimeType || "application/octet-stream")
      ))
    )),
  ];
  return encodeField(1, WIRE_TYPE.LEN, concatArrays(
    ...content,
    encodeField(2, WIRE_TYPE.VARINT, isError ? 1 : 0)
  ));
}

export function encodeMcpResultError(message) {
  return encodeField(2, WIRE_TYPE.LEN, encodeField(1, WIRE_TYPE.LEN, String(message || "")));
}

export function encodeMcpResultToolNotFound(toolName) {
  return encodeField(5, WIRE_TYPE.LEN, encodeField(1, WIRE_TYPE.LEN, String(toolName || "")));
}

// ==================== MESSAGE ENCODING ====================

/**
 * Format tool name: "toolName" → "mcp_custom_toolName"
 * Also handles: "mcp__server__tool" → "mcp_server_tool"
 */
function formatToolName(name) {
  const base = typeof name === "string" && name.length > 0 ? name : "tool";

  if (base.startsWith("mcp__")) {
    const rest = base.slice("mcp__".length);
    const splitIdx = rest.indexOf("__");
    if (splitIdx >= 0) {
      const server = rest.slice(0, splitIdx) || "custom";
      const toolName = rest.slice(splitIdx + 2) || "tool";
      return `mcp_${server}_${toolName}`;
    }
    return `mcp_custom_${rest || "tool"}`;
  }

  if (base.startsWith("mcp_")) return base;
  return `mcp_custom_${base}`;
}

/**
 * Parse formatted tool name: "mcp_server_tool" → { serverName, selectedTool }
 */
function parseToolName(formattedName) {
  if (typeof formattedName !== "string" || !formattedName.startsWith("mcp_")) {
    return { serverName: "custom", selectedTool: formattedName || "tool" };
  }

  const tail = formattedName.slice("mcp_".length);
  const splitIdx = tail.indexOf("_");
  if (splitIdx < 0) {
    return { serverName: "custom", selectedTool: tail || "tool" };
  }

  return {
    serverName: tail.slice(0, splitIdx) || "custom",
    selectedTool: tail.slice(splitIdx + 1) || "tool"
  };
}

/**
 * Parse tool_call_id into { toolCallId, modelCallId }
 * Cursor uses "\nmc_" delimiter for model_call_id
 */
function parseToolId(id) {
  const delimiter = "\nmc_";
  const idx = id.indexOf(delimiter);
  if (idx >= 0) {
    return { toolCallId: id.slice(0, idx), modelCallId: id.slice(idx + delimiter.length) };
  }
  return { toolCallId: id, modelCallId: null };
}

/**
 * Encode MCPResult proto: { selected_tool, result }
 */
function encodeMcpResult(selectedTool, resultContent) {
  return concatArrays(
    encodeField(FIELD.MCPR_SELECTED_TOOL, WIRE_TYPE.LEN, selectedTool),
    encodeField(FIELD.MCPR_RESULT, WIRE_TYPE.LEN, resultContent)
  );
}

/**
 * Encode ClientSideToolV2Result proto: { tool, mcp_result, call_id, model_call_id, tool_index }
 * Represents the result of executing a tool
 */
function encodeClientSideToolV2Result(toolCallId, modelCallId, selectedTool, resultContent, toolIndex = 1) {
  return concatArrays(
    encodeField(FIELD.CV2R_TOOL, WIRE_TYPE.VARINT, CLIENT_SIDE_TOOL_V2_MCP),
    encodeField(FIELD.CV2R_MCP_RESULT, WIRE_TYPE.LEN, encodeMcpResult(selectedTool, resultContent)),
    encodeField(FIELD.CV2R_CALL_ID, WIRE_TYPE.LEN, toolCallId),
    ...(modelCallId ? [encodeField(FIELD.CV2R_MODEL_CALL_ID, WIRE_TYPE.LEN, modelCallId)] : []),
    encodeField(FIELD.CV2R_TOOL_INDEX, WIRE_TYPE.VARINT, toolIndex > 0 ? toolIndex : 1)
  );
}

/**
 * Encode MCPParams.Tool nested inside ClientSideToolV2Call
 */
function encodeMcpParamsForCall(toolName, rawArgs, serverName) {
  const tool = concatArrays(
    encodeField(FIELD.MCP_TOOL_NAME, WIRE_TYPE.LEN, toolName),
    encodeField(FIELD.MCP_TOOL_PARAMS, WIRE_TYPE.LEN, rawArgs),
    encodeField(FIELD.MCP_TOOL_SERVER, WIRE_TYPE.LEN, serverName)
  );
  return encodeField(FIELD.MCP_TOOLS_LIST, WIRE_TYPE.LEN, tool);
}

/**
 * Encode ClientSideToolV2Call proto: { tool, mcp_params, call_id, name, raw_args, tool_index, model_call_id }
 * Represents a tool call definition
 */
function encodeClientSideToolV2Call(toolCallId, toolName, selectedTool, serverName, rawArgs, modelCallId, toolIndex = 1) {
  return concatArrays(
    encodeField(FIELD.CV2C_TOOL, WIRE_TYPE.VARINT, CLIENT_SIDE_TOOL_V2_MCP),
    encodeField(FIELD.CV2C_MCP_PARAMS, WIRE_TYPE.LEN, encodeMcpParamsForCall(selectedTool, rawArgs, serverName)),
    encodeField(FIELD.CV2C_CALL_ID, WIRE_TYPE.LEN, toolCallId),
    encodeField(FIELD.CV2C_NAME, WIRE_TYPE.LEN, toolName),
    encodeField(FIELD.CV2C_RAW_ARGS, WIRE_TYPE.LEN, rawArgs),
    encodeField(FIELD.CV2C_TOOL_INDEX, WIRE_TYPE.VARINT, toolIndex > 0 ? toolIndex : 1),
    ...(modelCallId ? [encodeField(FIELD.CV2C_MODEL_CALL_ID, WIRE_TYPE.LEN, modelCallId)] : [])
  );
}

/**
 * Encode ConversationMessage.ToolResult with full structure
 * Matches Cursor proto: tool_call_id, tool_name, tool_index, raw_args, result, tool_call
 */
export function encodeToolResult(toolResult) {
  const originalName = toolResult.tool_name || toolResult.name || "";
  const toolName = formatToolName(originalName);
  const rawArgs = toolResult.raw_args || "{}";
  const resultContent = toolResult.result_content || toolResult.result || "";
  const { toolCallId, modelCallId } = parseToolId(toolResult.tool_call_id || "");
  const toolIndex = toolResult.tool_index || toolResult.index || 1;

  // Parse tool name to extract server and selected tool
  const { serverName, selectedTool } = parseToolName(toolName);

  return concatArrays(
    encodeField(FIELD.TOOL_RESULT_CALL_ID, WIRE_TYPE.LEN, toolCallId),
    encodeField(FIELD.TOOL_RESULT_NAME, WIRE_TYPE.LEN, toolName),
    encodeField(FIELD.TOOL_RESULT_INDEX, WIRE_TYPE.VARINT, toolIndex > 0 ? toolIndex : 1),
    ...(modelCallId ? [encodeField(FIELD.TOOL_RESULT_MODEL_CALL_ID, WIRE_TYPE.LEN, modelCallId)] : []),
    encodeField(FIELD.TOOL_RESULT_RAW_ARGS, WIRE_TYPE.LEN, rawArgs),
    encodeField(FIELD.TOOL_RESULT_RESULT, WIRE_TYPE.LEN,
      encodeClientSideToolV2Result(toolCallId, modelCallId, selectedTool, resultContent, toolIndex)
    ),
    encodeField(FIELD.TOOL_RESULT_TOOL_CALL, WIRE_TYPE.LEN,
      encodeClientSideToolV2Call(toolCallId, toolName, selectedTool, serverName, rawArgs, modelCallId, toolIndex)
    )
  );
}

export function encodeMessage(content, role, messageId, chatModeEnum = null, isLast = false, hasTools = false, toolResults = [], serverBubbleId = null) {
  const hasToolResults = toolResults.length > 0;
  return concatArrays(
    encodeField(FIELD.MSG_CONTENT, WIRE_TYPE.LEN, content),
    encodeField(FIELD.MSG_ROLE, WIRE_TYPE.VARINT, role),
    encodeField(FIELD.MSG_ID, WIRE_TYPE.LEN, messageId),
    // Only include server_bubble_id if explicitly provided (last assistant message only)
    ...(serverBubbleId ? [encodeField(FIELD.MSG_SERVER_BUBBLE_ID, WIRE_TYPE.LEN, serverBubbleId)] : []),
    ...(hasToolResults ? toolResults.map(tr =>
      encodeField(FIELD.MSG_TOOL_RESULTS, WIRE_TYPE.LEN, encodeToolResult(tr))
    ) : []),
    encodeField(FIELD.MSG_IS_AGENTIC, WIRE_TYPE.VARINT, hasTools ? 1 : 0),
    encodeField(FIELD.MSG_UNIFIED_MODE, WIRE_TYPE.VARINT, hasTools ? UNIFIED_MODE.AGENT : UNIFIED_MODE.CHAT),
    ...(isLast && hasTools ? [encodeField(FIELD.MSG_SUPPORTED_TOOLS, WIRE_TYPE.LEN, encodeVarint(1))] : [])
  );
}

export function encodeInstruction(text) {
  return text ? encodeField(FIELD.INSTRUCTION_TEXT, WIRE_TYPE.LEN, text) : new Uint8Array(0);
}

export function encodeModel(modelName) {
  return concatArrays(
    encodeField(FIELD.MODEL_NAME, WIRE_TYPE.LEN, modelName),
    encodeField(FIELD.MODEL_EMPTY, WIRE_TYPE.LEN, new Uint8Array(0))
  );
}

export function encodeCursorSetting() {
  const unknown6 = concatArrays(
    encodeField(FIELD.SETTING6_FIELD_1, WIRE_TYPE.LEN, new Uint8Array(0)),
    encodeField(FIELD.SETTING6_FIELD_2, WIRE_TYPE.LEN, new Uint8Array(0))
  );

  return concatArrays(
    encodeField(FIELD.SETTING_PATH, WIRE_TYPE.LEN, "cursor\\aisettings"),
    encodeField(FIELD.SETTING_UNKNOWN_3, WIRE_TYPE.LEN, new Uint8Array(0)),
    encodeField(FIELD.SETTING_UNKNOWN_6, WIRE_TYPE.LEN, unknown6),
    encodeField(FIELD.SETTING_UNKNOWN_8, WIRE_TYPE.VARINT, 1),
    encodeField(FIELD.SETTING_UNKNOWN_9, WIRE_TYPE.VARINT, 1)
  );
}

export function encodeMetadata() {
  return concatArrays(
    encodeField(FIELD.META_PLATFORM, WIRE_TYPE.LEN, process.platform || "linux"),
    encodeField(FIELD.META_ARCH, WIRE_TYPE.LEN, process.arch || "x64"),
    encodeField(FIELD.META_VERSION, WIRE_TYPE.LEN, process.version || "v20.0.0"),
    encodeField(FIELD.META_CWD, WIRE_TYPE.LEN, process.cwd?.() || "/"),
    encodeField(FIELD.META_TIMESTAMP, WIRE_TYPE.LEN, new Date().toISOString())
  );
}

export function encodeMessageId(messageId, role, summaryId = null) {
  return concatArrays(
    encodeField(FIELD.MSGID_ID, WIRE_TYPE.LEN, messageId),
    ...(summaryId ? [encodeField(FIELD.MSGID_SUMMARY, WIRE_TYPE.LEN, summaryId)] : []),
    encodeField(FIELD.MSGID_ROLE, WIRE_TYPE.VARINT, role)
  );
}

export function encodeMcpTool(tool) {
  const toolName = tool.function?.name || tool.name || "";
  const toolDesc = tool.function?.description || tool.description || "";
  const inputSchema = tool.function?.parameters || tool.input_schema || {};

  return concatArrays(
    ...(toolName ? [encodeField(FIELD.MCP_TOOL_NAME, WIRE_TYPE.LEN, toolName)] : []),
    ...(toolDesc ? [encodeField(FIELD.MCP_TOOL_DESC, WIRE_TYPE.LEN, toolDesc)] : []),
    ...(Object.keys(inputSchema).length > 0 ? [encodeField(FIELD.MCP_TOOL_PARAMS, WIRE_TYPE.LEN, JSON.stringify(inputSchema))] : []),
    encodeField(FIELD.MCP_TOOL_SERVER, WIRE_TYPE.LEN, "custom")
  );
}

// ==================== REQUEST BUILDING ====================

export function encodeRequest(messages, modelName, tools = [], reasoningEffort = null, forceAgentMode = false) {
  const hasTools = tools?.length > 0;
  const isAgentic = hasTools || forceAgentMode;
  const formattedMessages = [];
  const messageIds = [];
  const normalizedMessages = [];

  // Guardrail: split mixed assistant payload into separate assistant messages
  // This prevents protobuf encoding errors when tool calls and results are in same message
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    const hasToolCalls = Array.isArray(msg?.tool_calls) && msg.tool_calls.length > 0;
    const hasToolResults = Array.isArray(msg?.tool_results) && msg.tool_results.length > 0;

    if (msg?.role === "assistant" && hasToolCalls && hasToolResults) {
      log(
        "ENCODE",
        `normalizing mixed assistant tool payload at msg[${i}] (calls=${msg.tool_calls.length}, results=${msg.tool_results.length})`
      );

      // Keep assistant tool call message without embedded results
      normalizedMessages.push({
        ...msg,
        tool_results: []
      });

      // Avoid inserting duplicate assistant tool-result message if next one already matches
      const nextMsg = messages[i + 1];
      const nextHasToolResults =
        nextMsg?.role === "assistant" &&
        Array.isArray(nextMsg?.tool_results) &&
        nextMsg.tool_results.length > 0;
      const currentIds = new Set(
        msg.tool_results.map(tr => tr?.tool_call_id).filter(id => typeof id === "string")
      );
      const nextIds = new Set(
        (nextMsg?.tool_results || [])
          .map(tr => tr?.tool_call_id)
          .filter(id => typeof id === "string")
      );
      let sameIds = currentIds.size > 0 && currentIds.size === nextIds.size;
      if (sameIds) {
        for (const id of currentIds) {
          if (!nextIds.has(id)) {
            sameIds = false;
            break;
          }
        }
      }

      if (!(nextHasToolResults && sameIds)) {
        normalizedMessages.push({
          role: "assistant",
          content: "",
          tool_results: msg.tool_results
        });
      }

      continue;
    }

    normalizedMessages.push(msg);
  }

  // Prepare messages
  for (let i = 0; i < normalizedMessages.length; i++) {
    const msg = normalizedMessages[i];
    const role = msg.role === "user" ? ROLE.USER : ROLE.ASSISTANT;
    const msgId = uuidv4();
    const isLast = i === normalizedMessages.length - 1;

    formattedMessages.push({
      content: msg.content,
      role,
      messageId: msgId,
      isLast,
      hasTools,
      toolResults: msg.tool_results || []
    });

    messageIds.push({ messageId: msgId, role });
  }

  // Cursor exposes medium/high on this legacy wire path. Clamp newer effort
  // names to the nearest supported level instead of silently disabling thinking.
  const thinkingLevel = normalizeCursorThinkingLevel(reasoningEffort);

  // Build request
  return concatArrays(
    // Messages
    ...formattedMessages.map(fm => 
      encodeField(FIELD.MESSAGES, WIRE_TYPE.LEN, 
        encodeMessage(fm.content, fm.role, fm.messageId, null, fm.isLast, fm.hasTools, fm.toolResults)
      )
    ),
    
    // Static fields
    encodeField(FIELD.UNKNOWN_2, WIRE_TYPE.VARINT, 1),
    encodeField(FIELD.INSTRUCTION, WIRE_TYPE.LEN, encodeInstruction("")),
    encodeField(FIELD.UNKNOWN_4, WIRE_TYPE.VARINT, 1),
    encodeField(FIELD.MODEL, WIRE_TYPE.LEN, encodeModel(modelName)),
    encodeField(FIELD.WEB_TOOL, WIRE_TYPE.LEN, ""),
    encodeField(FIELD.UNKNOWN_13, WIRE_TYPE.VARINT, 1),
    encodeField(FIELD.CURSOR_SETTING, WIRE_TYPE.LEN, encodeCursorSetting()),
    encodeField(FIELD.UNKNOWN_19, WIRE_TYPE.VARINT, 1),
    encodeField(FIELD.CONVERSATION_ID, WIRE_TYPE.LEN, uuidv4()),
    encodeField(FIELD.METADATA, WIRE_TYPE.LEN, encodeMetadata()),

    // Tool-related fields
    encodeField(FIELD.IS_AGENTIC, WIRE_TYPE.VARINT, isAgentic ? 1 : 0),
    ...(isAgentic ? [encodeField(FIELD.SUPPORTED_TOOLS, WIRE_TYPE.LEN, encodeVarint(1))] : []),
    
    // Message IDs
    ...messageIds.map(mid => 
      encodeField(FIELD.MESSAGE_IDS, WIRE_TYPE.LEN, encodeMessageId(mid.messageId, mid.role))
    ),

    // MCP Tools
    ...(tools?.length > 0 ? tools.map(tool => 
      encodeField(FIELD.MCP_TOOLS, WIRE_TYPE.LEN, encodeMcpTool(tool))
    ) : []),

    // Mode fields
    encodeField(FIELD.LARGE_CONTEXT, WIRE_TYPE.VARINT, 0),
    encodeField(FIELD.UNKNOWN_38, WIRE_TYPE.VARINT, 0),
    encodeField(FIELD.UNIFIED_MODE, WIRE_TYPE.VARINT, isAgentic ? UNIFIED_MODE.AGENT : UNIFIED_MODE.CHAT),
    encodeField(FIELD.UNKNOWN_47, WIRE_TYPE.LEN, ""),
    encodeField(FIELD.SHOULD_DISABLE_TOOLS, WIRE_TYPE.VARINT, isAgentic ? 0 : 1),
    encodeField(FIELD.THINKING_LEVEL, WIRE_TYPE.VARINT, thinkingLevel),
    encodeField(FIELD.UNKNOWN_51, WIRE_TYPE.VARINT, 0),
    encodeField(FIELD.UNKNOWN_53, WIRE_TYPE.VARINT, 1),
    encodeField(FIELD.UNIFIED_MODE_NAME, WIRE_TYPE.LEN, isAgentic ? "Agent" : "Ask")
  );
}

export function buildChatRequest(messages, modelName, tools = [], reasoningEffort = null, forceAgentMode = false) {
  return encodeField(FIELD.REQUEST, WIRE_TYPE.LEN, encodeRequest(messages, modelName, tools, reasoningEffort, forceAgentMode));
}

/**
 * Encode a tool result as ClientSideToolV2Result (field 2 of StreamUnifiedChatRequestWithTools)
 * This is sent as a SEPARATE request frame, not inside conversation messages.
 * Proto: StreamUnifiedChatRequestWithTools.client_side_tool_v2_result = 2
 */
export function buildToolResultRequest(toolResult) {
  const { toolCallId, modelCallId } = parseToolId(toolResult.tool_call_id || "");
  const rawName = toolResult.tool_name || "";
  const resultContent = toolResult.result_content || "";

  // selected_tool = raw tool name (e.g. "Write", "Read") per cursor-api Rust source:
  // McpResult { selected_tool: tool_name, result } where tool_name is the mcpParams.tools[0].name
  // which is the name AFTER server prefix stripping (e.g. "custom_Write" -> name = "Write")
  // Actually cursor-api uses: name = tool_name.slice_unchecked(d+1..) → raw name without "custom_"
  // So selected_tool = raw tool name without any prefix
  const selectedTool = rawName.startsWith("mcp_custom_")
    ? rawName.slice("mcp_custom_".length)
    : rawName.startsWith("mcp_")
    ? rawName.slice(4)
    : rawName;

  // ClientSideToolV2Result per proto:
  //   field 1 (tool): varint = 19 (MCP)
  //   field 28 (mcp_result): LEN { field 1: selected_tool, field 2: result }
  //   field 35 (tool_call_id): string
  //   field 48 (model_call_id): string (optional)
  //   NO tool_index (None in Rust source: encode_tool_result sets tool_index: None)
  const cv2Result = concatArrays(
    encodeField(FIELD.CV2R_TOOL, WIRE_TYPE.VARINT, CLIENT_SIDE_TOOL_V2_MCP),
    encodeField(FIELD.CV2R_MCP_RESULT, WIRE_TYPE.LEN, encodeMcpResult(selectedTool, resultContent)),
    encodeField(FIELD.CV2R_CALL_ID, WIRE_TYPE.LEN, toolCallId),
    ...(modelCallId ? [encodeField(FIELD.CV2R_MODEL_CALL_ID, WIRE_TYPE.LEN, modelCallId)] : [])
    // tool_index intentionally omitted (None per Rust source)
  );

  // StreamUnifiedChatRequestWithTools: field 2 = client_side_tool_v2_result
  return encodeField(2, WIRE_TYPE.LEN, cv2Result);
}

export function wrapConnectRPCFrame(payload, compress = false) {
  let finalPayload = payload;
  let flags = 0x00;

  if (compress) {
    finalPayload = new Uint8Array(zlib.gzipSync(Buffer.from(payload)));
    flags = 0x01;
  }

  const frame = new Uint8Array(5 + finalPayload.length);
  frame[0] = flags;
  frame[1] = (finalPayload.length >> 24) & 0xFF;
  frame[2] = (finalPayload.length >> 16) & 0xFF;
  frame[3] = (finalPayload.length >> 8) & 0xFF;
  frame[4] = finalPayload.length & 0xFF;
  frame.set(finalPayload, 5);

  return frame;
}

export function generateCursorBody(messages, modelName, tools = [], reasoningEffort = null, forceAgentMode = false) {
  log("BODY", `Generating: ${messages.length} msgs, model=${modelName}, tools=${tools.length}, reasoning=${reasoningEffort || "none"}, forceAgentMode=${forceAgentMode}`);
  
  const protobuf = buildChatRequest(messages, modelName, tools, reasoningEffort, forceAgentMode);
  const framed = wrapConnectRPCFrame(protobuf, false); // Cursor doesn't support compressed requests
  
  log("BODY", `Protobuf=${protobuf.length}B, Framed=${framed.length}B`);
  return framed;
}

/**
 * Generate a framed tool result body to send as a separate request frame.
 * Uses field 2 (client_side_tool_v2_result) of StreamUnifiedChatRequestWithTools.
 */
export function generateToolResultBody(toolResult) {
  const protobuf = buildToolResultRequest(toolResult);
  return wrapConnectRPCFrame(protobuf, false);
}

// ==================== PRIMITIVE DECODING ====================

export function decodeVarint(buffer, offset) {
  let result = 0;
  let shift = 0;
  let pos = offset;

  while (pos < buffer.length) {
    const b = buffer[pos];
    result |= (b & 0x7F) << shift;
    pos++;
    if (!(b & 0x80)) break;
    shift += 7;
  }

  return [result, pos];
}

export function decodeField(buffer, offset) {
  if (offset >= buffer.length) return [null, null, null, offset];

  const [tag, pos1] = decodeVarint(buffer, offset);
  const fieldNum = tag >> 3;
  const wireType = tag & 0x07;

  let value;
  let pos = pos1;

  if (wireType === WIRE_TYPE.VARINT) {
    [value, pos] = decodeVarint(buffer, pos);
  } else if (wireType === WIRE_TYPE.LEN) {
    const [length, pos2] = decodeVarint(buffer, pos);
    value = buffer.slice(pos2, pos2 + length);
    pos = pos2 + length;
  } else if (wireType === WIRE_TYPE.FIXED64) {
    value = buffer.slice(pos, pos + 8);
    pos += 8;
  } else if (wireType === WIRE_TYPE.FIXED32) {
    value = buffer.slice(pos, pos + 4);
    pos += 4;
  } else {
    value = null;
  }

  return [fieldNum, wireType, value, pos];
}

export function decodeMessage(data) {
  const fields = new Map();
  let pos = 0;

  while (pos < data.length) {
    const [fieldNum, wireType, value, newPos] = decodeField(data, pos);
    if (fieldNum === null) break;

    if (!fields.has(fieldNum)) fields.set(fieldNum, []);
    fields.get(fieldNum).push({ wireType, value });
    pos = newPos;
  }

  return fields;
}

// ==================== RESPONSE PARSING ====================

export function parseConnectRPCFrame(buffer) {
  if (buffer.length < 5) return null;

  const flags = buffer[0];
  const length = (buffer[1] << 24) | (buffer[2] << 16) | (buffer[3] << 8) | buffer[4];

  if (buffer.length < 5 + length) return null;

  let payload = buffer.slice(5, 5 + length);

  // Decompress if gzip
  if (flags === 0x01) {
    try {
      payload = new Uint8Array(zlib.gunzipSync(Buffer.from(payload)));
    } catch (err) {
      log("PARSE", `Decompression failed: ${err.message}`);
    }
  }

  return { flags, length, payload, consumed: 5 + length };
}

function extractToolCall(toolCallData) {
  const toolCall = decodeMessage(toolCallData);
  let toolCallId = "";
  let toolName = "";
  let rawArgs = "";
  let isLast = false;

  // Extract tool call ID
  if (toolCall.has(FIELD.TOOL_ID)) {
    const fullId = new TextDecoder().decode(toolCall.get(FIELD.TOOL_ID)[0].value);
    toolCallId = fullId.split("\n")[0]; // Cursor returns multi-line ID, take first line
  }

  // Extract tool name
  if (toolCall.has(FIELD.TOOL_NAME)) {
    toolName = new TextDecoder().decode(toolCall.get(FIELD.TOOL_NAME)[0].value);
  }

  // Extract is_last flag
  if (toolCall.has(FIELD.TOOL_IS_LAST)) {
    isLast = toolCall.get(FIELD.TOOL_IS_LAST)[0].value !== 0;
  }

  // Extract MCP params - nested real tool info
  if (toolCall.has(FIELD.TOOL_MCP_PARAMS)) {
    try {
      const mcpParams = decodeMessage(toolCall.get(FIELD.TOOL_MCP_PARAMS)[0].value);
      
      if (mcpParams.has(FIELD.MCP_TOOLS_LIST)) {
        const tool = decodeMessage(mcpParams.get(FIELD.MCP_TOOLS_LIST)[0].value);
        
        if (tool.has(FIELD.MCP_NESTED_NAME)) {
          toolName = new TextDecoder().decode(tool.get(FIELD.MCP_NESTED_NAME)[0].value);
        }
        
        if (tool.has(FIELD.MCP_NESTED_PARAMS)) {
          rawArgs = new TextDecoder().decode(tool.get(FIELD.MCP_NESTED_PARAMS)[0].value);
        }
      }
    } catch (err) {
      log("EXTRACT", `MCP parse error: ${err.message}`);
    }
  }

  // Fallback to raw_args
  if (!rawArgs && toolCall.has(FIELD.TOOL_RAW_ARGS)) {
    rawArgs = new TextDecoder().decode(toolCall.get(FIELD.TOOL_RAW_ARGS)[0].value);
  }

  if (toolCallId && toolName) {
    return {
      id: toolCallId,
      type: "function",
      function: {
        name: toolName,
        arguments: rawArgs || "{}"
      },
      isLast
    };
  }

  return null;
}

function extractTextAndThinking(responseData) {
  const nested = decodeMessage(responseData);
  let text = null;
  let thinking = null;

  // Extract text
  if (nested.has(FIELD.RESPONSE_TEXT)) {
    text = new TextDecoder().decode(nested.get(FIELD.RESPONSE_TEXT)[0].value);
  }

  // Extract thinking
  if (nested.has(FIELD.THINKING)) {
    try {
      const thinkingMsg = decodeMessage(nested.get(FIELD.THINKING)[0].value);
      if (thinkingMsg.has(FIELD.THINKING_TEXT)) {
        thinking = new TextDecoder().decode(thinkingMsg.get(FIELD.THINKING_TEXT)[0].value);
      }
    } catch (err) {
      log("EXTRACT", `Thinking parse error: ${err.message}`);
    }
  }

  return { text, thinking };
}

export function extractTextFromResponse(payload) {
  try {
    const fields = decodeMessage(payload);

    // Warn about unknown field numbers — may indicate a Cursor protocol update
    for (const fieldNum of fields.keys()) {
      if (!KNOWN_RESPONSE_FIELDS.has(fieldNum)) {
        log(
          "SCHEMA",
          `Unknown response field #${fieldNum} detected. Schema v${PROTOBUF_SCHEMA_VERSION} may be outdated.`
        );
      }
    }

    // Field 1: ClientSideToolV2Call
    if (fields.has(FIELD.TOOL_CALL)) {
      const toolCall = extractToolCall(fields.get(FIELD.TOOL_CALL)[0].value);
      if (toolCall) {
        log("EXTRACT", `Tool call: ${toolCall.function.name}`);
        return { text: null, error: null, toolCall, thinking: null };
      }
    }

    // Field 2: StreamUnifiedChatResponse
    if (fields.has(FIELD.RESPONSE)) {
      const { text, thinking } = extractTextAndThinking(fields.get(FIELD.RESPONSE)[0].value);

      if (text || thinking) {
        return { text, error: null, toolCall: null, thinking };
      }
    }

    return { text: null, error: null, toolCall: null, thinking: null };
  } catch (err) {
    log("EXTRACT", `Decode failed (schema v${PROTOBUF_SCHEMA_VERSION}): ${err.message}`);
    return {
      text: null,
      error: null,
      toolCall: null,
      thinking: null,
      raw: Buffer.from(payload).toString("base64"),
      decodeError: err.message
    };
  }
}

// ==================== EXPORTS ====================

// ==================== AgentService native tool bridge ====================
// AgentService multiplexes IDE execution through ExecServerMessage. Keep this
// codec separate from the legacy ChatService fields above.
const AGENT_EXEC_SERVER_MESSAGE = 2;
const AGENT_EXEC_ID = 15;
const AGENT_EXEC_VARIANTS = {
  shell: 2, write: 3, delete: 4, grep: 5, read: 7, ls: 8,
  diagnostics: 9, requestContext: 10, mcp: 11, shellStream: 14,
  backgroundShell: 16, listMcpResources: 17, readMcpResource: 18,
  fetch: 20, recordScreen: 21, computerUse: 22, writeShellStdin: 23,
  executeHook: 27,
};

function agentStringField(fields, field) {
  const item = fields.get(field)?.[0];
  return item?.value == null || typeof item.value === "number"
    ? "" : textDecoder.decode(item.value);
}

function agentExecEnvelope(payload) {
  const outer = decodeMessage(payload);
  const item = outer.get(AGENT_EXEC_SERVER_MESSAGE)?.[0];
  if (!item || item.wireType !== WIRE_TYPE.LEN) return null;
  const fields = decodeMessage(item.value);
  const id = fields.get(1)?.[0]?.value ?? 0;
  const execId = agentStringField(fields, AGENT_EXEC_ID);
  for (const [field, values] of fields) {
    if (field === 1 || field === AGENT_EXEC_ID || field === 19) continue;
    const value = values[0];
    if (value?.wireType === WIRE_TYPE.LEN) return { id, execId, field, value: value.value };
  }
  return null;
}

export function decodeExecServerEvent(payload) {
  const envelope = agentExecEnvelope(payload);
  if (!envelope) return null;
  const { id: execMsgId, execId, field, value } = envelope;
  if (!value?.length && ![AGENT_EXEC_VARIANTS.requestContext, AGENT_EXEC_VARIANTS.mcp, AGENT_EXEC_VARIANTS.listMcpResources].includes(field)) return null;
  const args = decodeMessage(value || new Uint8Array());
  const text = (n) => agentStringField(args, n);
  switch (field) {
    case AGENT_EXEC_VARIANTS.requestContext: return { kind: "exec_request_context", execMsgId, execId };
    case AGENT_EXEC_VARIANTS.read: return { kind: "exec_read", execMsgId, execId, path: text(1) };
    case AGENT_EXEC_VARIANTS.write: return { kind: "exec_write", execMsgId, execId, path: text(1), fileText: text(2) };
    case AGENT_EXEC_VARIANTS.delete: return { kind: "exec_delete", execMsgId, execId, path: text(1) };
    case AGENT_EXEC_VARIANTS.ls: return { kind: "exec_ls", execMsgId, execId, path: text(1) };
    case AGENT_EXEC_VARIANTS.grep: return { kind: "exec_grep", execMsgId, execId, pattern: text(1), path: text(2), glob: text(3) };
    case AGENT_EXEC_VARIANTS.diagnostics: return { kind: "exec_diagnostics", execMsgId, execId, path: text(1) };
    case AGENT_EXEC_VARIANTS.shell:
      return { kind: "exec_shell", execMsgId, execId, command: text(1), workingDir: text(2) };
    case AGENT_EXEC_VARIANTS.shellStream:
      return { kind: "exec_shell_stream", execMsgId, execId, command: text(1), workingDir: text(2) };
    case AGENT_EXEC_VARIANTS.backgroundShell:
      return { kind: "exec_bg_shell", execMsgId, execId, command: text(1), workingDir: text(2) };
    case AGENT_EXEC_VARIANTS.fetch: return { kind: "exec_fetch", execMsgId, execId, url: text(1) };
    case AGENT_EXEC_VARIANTS.writeShellStdin:
      return { kind: "exec_write_shell_stdin", execMsgId, execId, shellId: args.get(1)?.[0]?.value ?? 0, chars: text(2) };
    case AGENT_EXEC_VARIANTS.listMcpResources: return { kind: "exec_list_mcp_resources", execMsgId, execId, server: text(1) };
    case AGENT_EXEC_VARIANTS.readMcpResource: return { kind: "exec_read_mcp_resource", execMsgId, execId, server: text(1), uri: text(2), downloadPath: text(3) };
    case AGENT_EXEC_VARIANTS.mcp: {
      const toolName = text(5) || text(1);
      const toolCallId = text(3);
      const decodedArgs = {};
      for (const entry of args.get(2) || []) {
        const map = decodeMessage(entry.value);
        const key = agentStringField(map, 1);
        const encoded = map.get(2)?.[0]?.value;
        if (key && encoded) decodedArgs[key] = decodeAgentValue(encoded);
      }
      return { kind: "exec_mcp", execMsgId, execId, toolName, toolCallId, args: decodedArgs };
    }
    case AGENT_EXEC_VARIANTS.recordScreen: return { kind: "exec_record_screen", execMsgId, execId, mode: args.get(1)?.[0]?.value ?? 0, saveAsFilename: text(3) };
    case AGENT_EXEC_VARIANTS.computerUse: return { kind: "exec_computer_use", execMsgId, execId, rawArgs: value };
    case AGENT_EXEC_VARIANTS.executeHook: {
      const request = args.get(1)?.[0]?.value;
      const requestFields = request ? decodeMessage(request) : new Map();
      const hookType = requestFields.has(1)
        ? "pre_compact"
        : requestFields.has(2)
          ? "subagent_start"
          : requestFields.has(3)
            ? "subagent_stop"
            : "pre_compact";
      return { kind: "exec_execute_hook", execMsgId, execId, hookType, rawArgs: value };
    }
    default: return { kind: "exec_unknown", execMsgId, execId, field };
  }
}

function wrapAgentExecResult(id, execId, resultField, resultVariant, variantField = null) {
  const result = variantField == null ? resultVariant : encodeField(variantField, WIRE_TYPE.LEN, resultVariant);
  const exec = concatArrays(
    encodeField(1, WIRE_TYPE.VARINT, id || 0),
    ...(execId ? [encodeField(AGENT_EXEC_ID, WIRE_TYPE.LEN, execId)] : []),
    encodeField(resultField, WIRE_TYPE.LEN, result)
  );
  return wrapConnectRPCFrame(encodeField(2, WIRE_TYPE.LEN, exec));
}
function rejected(path, reason) { return concatArrays(encodeField(1, WIRE_TYPE.LEN, path || ""), encodeField(2, WIRE_TYPE.LEN, reason)); }
function shellRejected(command, cwd, reason) { return concatArrays(encodeField(1, WIRE_TYPE.LEN, command || ""), encodeField(2, WIRE_TYPE.LEN, cwd || ""), encodeField(3, WIRE_TYPE.LEN, reason)); }
function errorMessage(reason) { return encodeField(1, WIRE_TYPE.LEN, reason); }
const BUILTIN_REJECT = "Tool not available in this environment. Use the MCP tools provided instead.";

export function encodeAgentNativeRejection(event, reason = BUILTIN_REJECT) {
  if (!event || event.kind === "exec_request_context" || event.kind === "exec_mcp") return null;
  const map = {
    exec_read: [7, rejected(event.path, reason), 3],
    exec_write: [3, rejected(event.path, reason), 6],
    exec_delete: [4, rejected(event.path, reason), 6],
    exec_ls: [8, rejected(event.path, reason), 3],
    exec_shell: [2, shellRejected(event.command, event.workingDir, reason), 4],
    exec_shell_stream: [14, shellRejected(event.command, event.workingDir, reason), 5],
    exec_bg_shell: [16, shellRejected(event.command, event.workingDir, reason), 3],
    exec_grep: [5, errorMessage(reason), 2],    exec_fetch: [20, concatArrays(encodeField(1, WIRE_TYPE.LEN, event.url || ""), encodeField(2, WIRE_TYPE.LEN, reason)), 2],
    exec_write_shell_stdin: [23, errorMessage(reason), 2],
    exec_diagnostics: [9, rejected(event.path, reason), 3],
    exec_list_mcp_resources: [17, errorMessage(reason), 3],
    exec_read_mcp_resource: [18, concatArrays(encodeField(1, WIRE_TYPE.LEN, event.uri || ""), encodeField(2, WIRE_TYPE.LEN, reason)), 3],
    exec_record_screen: [21, errorMessage(reason), 4],
    exec_computer_use: [22, errorMessage(reason), 2],
  };
  if (event.kind === "exec_execute_hook") {
    const responseField = event.hookType === "subagent_start" ? 2 : event.hookType === "subagent_stop" ? 3 : 1;
    const response = encodeField(1, WIRE_TYPE.LEN, encodeField(responseField, WIRE_TYPE.LEN, new Uint8Array()));
    return wrapAgentExecResult(event.execMsgId, event.execId, 27, response);
  }
  const [resultField, payload, variantField] = map[event.kind] || [null, null, null];
  return resultField ? wrapAgentExecResult(event.execMsgId, event.execId, resultField, payload, variantField) : null;
}

export function encodeAgentRequestContextResponse(execMsgId, execId) {
  const success = encodeField(1, WIRE_TYPE.LEN, encodeField(1, WIRE_TYPE.LEN, new Uint8Array()));
  const exec = concatArrays(encodeField(1, WIRE_TYPE.VARINT, execMsgId || 0), ...(execId ? [encodeField(AGENT_EXEC_ID, WIRE_TYPE.LEN, execId)] : []), encodeField(10, WIRE_TYPE.LEN, success));
  return wrapConnectRPCFrame(encodeField(2, WIRE_TYPE.LEN, exec));
}

export function encodeAgentEmptyListMcpResources(execMsgId, execId) {
  const success = encodeField(1, WIRE_TYPE.LEN, new Uint8Array());
  const exec = concatArrays(encodeField(1, WIRE_TYPE.VARINT, execMsgId || 0), ...(execId ? [encodeField(AGENT_EXEC_ID, WIRE_TYPE.LEN, execId)] : []), encodeField(17, WIRE_TYPE.LEN, success));
  return wrapConnectRPCFrame(encodeField(2, WIRE_TYPE.LEN, exec));
}

export function encodeAgentMcpResult(execMsgId, execId, content, isError = false) {
  const text = encodeField(1, WIRE_TYPE.LEN, encodeField(1, WIRE_TYPE.LEN, String(content ?? "")));
  const success = concatArrays(encodeField(1, WIRE_TYPE.LEN, text), ...(isError ? [encodeField(2, WIRE_TYPE.VARINT, 1)] : []));
  return wrapAgentExecResult(execMsgId, execId, 11, success, 1);
}

export function decodeAgentKvServerEvent(payload) {
  const outer = decodeMessage(payload);
  const item = outer.get(4)?.[0];
  if (!item || item.wireType !== WIRE_TYPE.LEN) return null;
  const fields = decodeMessage(item.value);
  const id = fields.get(1)?.[0]?.value ?? 0;
  const metadata = fields.get(4)?.[0]?.value || null;
  const getArgs = fields.get(2)?.[0]?.value;
  const setArgs = fields.get(3)?.[0]?.value;
  const readBytes = (data, field) => decodeMessage(data).get(field)?.[0]?.value || new Uint8Array();
  if (getArgs) return { kind: "get", id, blobId: readBytes(getArgs, 1), metadata };
  if (setArgs) return { kind: "set", id, blobId: readBytes(setArgs, 1), blobData: readBytes(setArgs, 2), metadata };
  return null;
}

function encodeAgentKvClientMessage(id, variantField, variant, metadata) {
  return wrapConnectRPCFrame(encodeField(3, WIRE_TYPE.LEN, concatArrays(
    ...(id ? [encodeField(1, WIRE_TYPE.VARINT, id)] : []),
    encodeField(variantField, WIRE_TYPE.LEN, variant),
    ...(metadata?.length ? [encodeField(4, WIRE_TYPE.LEN, metadata)] : []),
  )));
}

export function encodeAgentKvGetResult(id, blob, metadata) {
  return encodeAgentKvClientMessage(id, 2, encodeField(1, WIRE_TYPE.LEN, blob || new Uint8Array()), metadata);
}
export function encodeAgentKvSetResult(id, metadata) {
  return encodeAgentKvClientMessage(id, 3, new Uint8Array(), metadata);
}

// ─── Native tool SUCCESS encoders (official agent/v1/*_exec.proto) ───────

export function encodeAgentReadSuccess(execMsgId, execId, { path = "", content = "", truncated = false, fileSize = 0 } = {}) {
  const lines = content ? content.split("\n").length : 0;
  const success = concatArrays(
    encodeField(1, WIRE_TYPE.LEN, path),
    encodeField(2, WIRE_TYPE.LEN, content),
    encodeField(3, WIRE_TYPE.VARINT, lines),
    encodeField(4, WIRE_TYPE.VARINT, fileSize),
    ...(truncated ? [encodeField(6, WIRE_TYPE.VARINT, 1)] : [])
  );
  return wrapAgentExecResult(execMsgId, execId, 7, success, 1);
}

export function encodeAgentGrepSuccess(execMsgId, execId, { pattern = "", path = ".", matches = [], truncated = false } = {}) {
  // GrepFileMatch -> repeated inside GrepContentResult.matches (field 1)
  const fileMatches = matches.map(({ file = "", lines = [] }) =>
    encodeField(1, WIRE_TYPE.LEN, concatArrays(
      encodeField(1, WIRE_TYPE.LEN, file),
      ...lines.map(({ lineNumber = 0, content = "" }) =>
        encodeField(2, WIRE_TYPE.LEN, concatArrays(encodeField(1, WIRE_TYPE.VARINT, lineNumber), encodeField(2, WIRE_TYPE.LEN, content)))
      )
    ))
  );
  const totalMatched = matches.reduce((n, m) => n + (m.lines?.length || 0), 0);
  // GrepContentResult
  const contentResult = concatArrays(
    ...fileMatches,
    encodeField(2, WIRE_TYPE.VARINT, totalMatched),
    encodeField(3, WIRE_TYPE.VARINT, totalMatched),
    ...(truncated ? [encodeField(4, WIRE_TYPE.VARINT, 1)] : [])
  );
  // GrepUnionResult.content = 3 ; map entry key=1 value=2 ; workspace_results=4
  const entry = concatArrays(encodeField(1, WIRE_TYPE.LEN, path || "."), encodeField(2, WIRE_TYPE.LEN, encodeField(3, WIRE_TYPE.LEN, contentResult)));
  const success = concatArrays(
    encodeField(1, WIRE_TYPE.LEN, pattern),
    encodeField(2, WIRE_TYPE.LEN, path || "."),
    encodeField(3, WIRE_TYPE.LEN, "content"),
    encodeField(4, WIRE_TYPE.LEN, entry)
  );
  return wrapAgentExecResult(execMsgId, execId, 5, success, 1);
}

export function encodeAgentLsSuccess(execMsgId, execId, { path = "", files = [], dirs = [], truncated = false, numFiles = 0 } = {}) {
  // LsDirectoryTreeNode: abs_path=1, children_dirs=2, children_files=3,
  // children_were_processed=4, full_subtree_extension_counts=5, num_files=6
  const node = concatArrays(
    encodeField(1, WIRE_TYPE.LEN, path),
    ...dirs.map((d) => encodeField(2, WIRE_TYPE.LEN, encodeField(1, WIRE_TYPE.LEN, d))),
    ...files.map((f) => encodeField(3, WIRE_TYPE.LEN, encodeField(1, WIRE_TYPE.LEN, f))),
    encodeField(4, WIRE_TYPE.VARINT, 1),
    ...(truncated ? [encodeField(5, WIRE_TYPE.VARINT, 0)] : []),
    encodeField(6, WIRE_TYPE.VARINT, numFiles)
  );
  const success = encodeField(1, WIRE_TYPE.LEN, node);
  return wrapAgentExecResult(execMsgId, execId, 8, success, 1);
}

export function encodeAgentDiagnosticsSuccess(execMsgId, execId, path = "") {
  const success = concatArrays(encodeField(1, WIRE_TYPE.LEN, path), encodeField(3, WIRE_TYPE.VARINT, 0));
  return wrapAgentExecResult(execMsgId, execId, 9, success, 1);
}

export function encodeAgentFetchSuccess(execMsgId, execId, { url = "", content = "", statusCode = 200, contentType = "" } = {}) {
  const success = concatArrays(
    encodeField(1, WIRE_TYPE.LEN, url),
    encodeField(2, WIRE_TYPE.LEN, content),
    encodeField(3, WIRE_TYPE.VARINT, statusCode),
    ...(contentType ? [encodeField(4, WIRE_TYPE.LEN, contentType)] : [])
  );
  return wrapAgentExecResult(execMsgId, execId, 20, success, 1);
}

export function encodeAgentWriteSuccess(execMsgId, execId, { path = "", linesCreated = 0, fileSize = 0, contentAfter = null } = {}) {
  const success = concatArrays(
    encodeField(1, WIRE_TYPE.LEN, path),
    encodeField(2, WIRE_TYPE.VARINT, linesCreated),
    encodeField(3, WIRE_TYPE.VARINT, fileSize),
    ...(contentAfter != null ? [encodeField(4, WIRE_TYPE.LEN, contentAfter)] : [])
  );
  return wrapAgentExecResult(execMsgId, execId, 3, success, 1);
}

export function encodeAgentDeleteSuccess(execMsgId, execId, { path = "", deletedFile = "", fileSize = 0, prevContent = "" } = {}) {
  const success = concatArrays(
    encodeField(1, WIRE_TYPE.LEN, path),
    encodeField(2, WIRE_TYPE.LEN, deletedFile),
    encodeField(3, WIRE_TYPE.VARINT, fileSize),
    encodeField(4, WIRE_TYPE.LEN, prevContent)
  );
  return wrapAgentExecResult(execMsgId, execId, 4, success, 1);
}

export function encodeAgentShellSuccess(execMsgId, execId, { command = "", cwd = "", exitCode = 0, stdout = "", stderr = "", executionTime = 0 } = {}) {
  const success = concatArrays(
    encodeField(1, WIRE_TYPE.LEN, command),
    encodeField(2, WIRE_TYPE.LEN, cwd),
    encodeField(3, WIRE_TYPE.VARINT, exitCode),
    ...(stdout ? [encodeField(5, WIRE_TYPE.LEN, stdout)] : []),
    ...(stderr ? [encodeField(6, WIRE_TYPE.LEN, stderr)] : []),
    encodeField(7, WIRE_TYPE.VARINT, executionTime)
  );
  return wrapAgentExecResult(execMsgId, execId, 2, success, 1);
}

export function encodeAgentShellFailure(execMsgId, execId, { command = "", cwd = "", exitCode = 1, stdout = "", stderr = "", executionTime = 0, error = "" } = {}) {
  const failure = concatArrays(
    encodeField(1, WIRE_TYPE.LEN, command),
    encodeField(2, WIRE_TYPE.LEN, cwd),
    encodeField(3, WIRE_TYPE.VARINT, exitCode),
    ...(stdout ? [encodeField(5, WIRE_TYPE.LEN, stdout)] : []),
    ...(stderr ? [encodeField(6, WIRE_TYPE.LEN, stderr)] : []),
    encodeField(7, WIRE_TYPE.VARINT, executionTime),
    ...(error ? [encodeField(9, WIRE_TYPE.LEN, error)] : [])
  );
  return wrapAgentExecResult(execMsgId, execId, 2, failure, 2);
}

export function encodeAgentShellTimeout(execMsgId, execId, { command = "", cwd = "", timeoutMs = 30000 } = {}) {
  const timeout = concatArrays(
    encodeField(1, WIRE_TYPE.LEN, command),
    encodeField(2, WIRE_TYPE.LEN, cwd),
    encodeField(3, WIRE_TYPE.VARINT, timeoutMs)
  );
  return wrapAgentExecResult(execMsgId, execId, 2, timeout, 3);
}

// ─── Interaction query (permission) auto-response ─────────────────────────
// AgentServerMessage.interaction_query=7 -> AgentClientMessage.interaction_response=6.
// InteractionResponse { id=1, oneof result: web_search=2, ask_question=3,
// switch_mode=4, exa_search=5, exa_fetch=6, create_plan=7, setup_vm=8, web_fetch=9 }

export function decodeAgentInteractionQuery(payload) {
  const outer = decodeMessage(payload);
  const item = outer.get(7)?.[0];
  if (!item || item.wireType !== WIRE_TYPE.LEN) return null;
  const fields = decodeMessage(item.value);
  const id = fields.get(1)?.[0]?.value ?? 0;
  const kind = [2, 3, 4, 5, 6, 7, 8, 9].find((f) => fields.has(f)) || 0;
  return { id, kind };
}

export function encodeAgentInteractionResponse(id, queryKind, approved = true, reason = "") {
  // Body per kind (official interaction tool protos). Field number in
  // InteractionResponse mirrors the query kind.
  let body;
  switch (queryKind) {
    case 2: // web_search
    case 5: // exa_search
    case 6: // exa_fetch
    case 9: // web_fetch
      body = approved
        ? encodeField(1, WIRE_TYPE.LEN, new Uint8Array())
        : encodeField(2, WIRE_TYPE.LEN, encodeField(1, WIRE_TYPE.LEN, reason));
      break;
    case 4: // switch_mode
      body = approved
        ? encodeField(1, WIRE_TYPE.LEN, new Uint8Array())
        : encodeField(2, WIRE_TYPE.LEN, encodeField(1, WIRE_TYPE.LEN, reason));
      break;
    case 3: // ask_question -> AskQuestionResult.rejected = 3
      body = encodeField(3, WIRE_TYPE.LEN, encodeField(1, WIRE_TYPE.LEN, reason));
      break;
    case 7: // create_plan -> CreatePlanResult.error = 2
      body = encodeField(2, WIRE_TYPE.LEN, encodeField(1, WIRE_TYPE.LEN, reason));
      break;
    case 8: // setup_vm -> SetupVmEnvironmentResult.success = 1 (only option)
      body = encodeField(1, WIRE_TYPE.LEN, new Uint8Array());
      break;
    default:
      return null;
  }
  const response = concatArrays(
    encodeField(1, WIRE_TYPE.VARINT, id || 0),
    encodeField(queryKind, WIRE_TYPE.LEN, body)
  );
  return wrapConnectRPCFrame(encodeField(6, WIRE_TYPE.LEN, response));
}

export function encodeAgentHeartbeat() {
  return wrapConnectRPCFrame(encodeField(7, WIRE_TYPE.LEN, new Uint8Array()));
}

export function encodeAgentStreamClose(id = 0) {
  const streamClose = encodeField(1, WIRE_TYPE.LEN, encodeField(1, WIRE_TYPE.VARINT, id));
  return wrapConnectRPCFrame(encodeField(5, WIRE_TYPE.LEN, streamClose));
}

export default {
  encodeVarint,
  encodeField,
  encodeAgentValue,
  decodeAgentValue,
  encodeMcpToolDefinition,
  encodeMcpTools,
  decodeMcpArgs,
  encodeMcpResultSuccess,
  encodeMcpResultError,
  encodeMcpResultToolNotFound,
  encodeMessage,
  buildChatRequest,
  wrapConnectRPCFrame,
  generateCursorBody,
  decodeVarint,
  decodeField,
  decodeMessage,
  parseConnectRPCFrame,
  extractTextFromResponse
};
