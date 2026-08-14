import { describe, expect, it } from "vitest";
import {
  decodeMessage,
  encodeField,
  encodeAgentValue,
  decodeAgentValue,
  encodeMcpToolDefinition,
  encodeMcpTools,
  decodeMcpArgs,
  encodeMcpResultSuccess,
  encodeMcpResultError,
  encodeMcpResultToolNotFound,
} from "../../open-sse/utils/cursorProtobuf.js";
import {
  isAgentCapableRequest,
  buildAgentRunFrame,
} from "../../open-sse/executors/cursor.js";
import {
  decodeExecServerEvent,
  encodeAgentNativeRejection,
  decodeAgentKvServerEvent,
  encodeAgentKvGetResult,
  encodeAgentHeartbeat,
  encodeAgentStreamClose,
  encodeAgentInteractionResponse,
  decodeAgentInteractionQuery,
  encodeAgentReadSuccess,
  encodeAgentGrepSuccess,
  encodeAgentLsSuccess,
  encodeAgentDiagnosticsSuccess,
  encodeAgentFetchSuccess,
  encodeAgentWriteSuccess,
  encodeAgentDeleteSuccess,
  encodeAgentShellSuccess,
  wrapConnectRPCFrame,
} from "../../open-sse/utils/cursorProtobuf.js";
const VARINT = 0;
const LEN = 2;
import { openaiToCursorRequest } from "../../open-sse/translator/request/openai-to-cursor.js";
import { claudeToOpenAIRequest } from "../../open-sse/translator/request/claude-to-openai.js";
import { buildCursorHeaders } from "../../open-sse/utils/cursorChecksum.js";
import { PROVIDERS } from "../../open-sse/config/providers.js";

// AgentService (agent.v1) codec tests — validate the production implementation
// in cursorProtobuf.js + the executor's frame builders. Pure round-trip, no network.
// Field numbers verified against Cursor's agent.proto (extracted via @oh-my-pi).

// McpArgs.args map entry { field1: key, field2: Value }
const entry = (k, v) => Buffer.concat([
  Buffer.from(encodeField(2, LEN,
    Buffer.concat([Buffer.from(encodeField(1, LEN, k)), Buffer.from(encodeField(2, LEN, encodeAgentValue(v)))])
  )),
]);

describe("Cursor AgentService codec (cursorProtobuf.js)", () => {
  describe("google.protobuf.Value round-trip", () => {
    const cases = [
      ["null", null],
      ["bool true", true],
      ["bool false", false],
      ["string", "hello"],
      ["integer", 42],
      ["float", 3.14],
      ["empty object", {}],
      ["flat object", { a: 1, b: "x", c: true }],
      ["nested object", { outer: { inner: [1, 2, "three"] } }],
      ["array of mixed", [1, "two", false, null]],
      ["deeply nested", { a: { b: { c: { d: 1 } } } }],
    ];
    for (const [label, value] of cases) {
      it(`encodes/decodes ${label}`, () => {
        expect(decodeAgentValue(encodeAgentValue(value))).toEqual(value);
      });
    }

    it("encodes doubles as protobuf little-endian fixed64", () => {
      expect(Buffer.from(encodeAgentValue(3.14))).toEqual(
        Buffer.from([0x11, 0x1f, 0x85, 0xeb, 0x51, 0xb8, 0x1e, 0x09, 0x40])
      );
    });
  });

  describe("McpToolDefinition", () => {
    it("encodes name, description, input_schema (Value), provider, tool_name", () => {
      const schema = { type: "object", properties: { city: { type: "string" } }, required: ["city"] };
      const def = encodeMcpToolDefinition({ function: { name: "get_weather", description: "Get weather", parameters: schema } });
      const msg = decodeMessage(def);
      expect(Buffer.from(msg.get(1)[0].value).toString("utf8")).toBe("get_weather");
      expect(Buffer.from(msg.get(2)[0].value).toString("utf8")).toBe("Get weather");
      expect(Buffer.from(msg.get(4)[0].value).toString("utf8")).toBe("9router");
      expect(Buffer.from(msg.get(5)[0].value).toString("utf8")).toBe("get_weather");
      expect(decodeAgentValue(msg.get(3)[0].value)).toEqual(schema);
    });

    it("preserves nested JSON-schema types", () => {
      const schema = {
        type: "object",
        properties: {
          query: { type: "string", description: "search query" },
          opts: { type: "array", items: { type: "string" } },
        },
        required: ["query"],
      };
      const def = encodeMcpToolDefinition({ function: { name: "search", parameters: schema } });
      const msg = decodeMessage(def);
      expect(decodeAgentValue(msg.get(3)[0].value)).toEqual(schema);
    });

    it("accepts flat tool shape (no .function wrapper)", () => {
      const def = encodeMcpToolDefinition({ name: "noop", description: "d", inputSchema: { type: "object" } });
      const msg = decodeMessage(def);
      expect(Buffer.from(msg.get(1)[0].value).toString("utf8")).toBe("noop");
    });
  });

  describe("encodeMcpTools", () => {
    it("produces empty bytes for no tools", () => {
      expect(encodeMcpTools([]).length).toBe(0);
      expect(encodeMcpTools().length).toBe(0);
    });

    it("wraps multiple tool defs as repeated field 1", () => {
      const tools = [
        { function: { name: "get_weather", parameters: { type: "object" } } },
        { function: { name: "calculate", parameters: { type: "object" } } },
      ];
      const mcpTools = encodeMcpTools(tools);
      const inner = decodeMessage(mcpTools);
      expect(inner.get(1).length).toBe(2);
    });
  });

  describe("McpArgs decode", () => {
    it("decodes name, toolName, toolCallId, and typed args map", () => {
      const argsBytes = Buffer.concat([
        entry("city", "Hanoi"),
        entry("count", 5),
        entry("flag", true),
        entry("nested", { a: [1, 2] }),
      ]);
      const mcpArgs = Buffer.concat([
        Buffer.from(encodeField(1, LEN, "get_weather")),
        argsBytes,
        Buffer.from(encodeField(3, LEN, "call_abc")),
        Buffer.from(encodeField(5, LEN, "get_weather")),
      ]);
      const decoded = decodeMcpArgs(mcpArgs);
      expect(decoded.name).toBe("get_weather");
      expect(decoded.toolName).toBe("get_weather");
      expect(decoded.toolCallId).toBe("call_abc");
      expect(decoded.args).toEqual({ city: "Hanoi", count: 5, flag: true, nested: { a: [1, 2] } });
    });

    it("handles empty args map", () => {
      const mcpArgs = Buffer.concat([
        Buffer.from(encodeField(1, LEN, "noop")),
        Buffer.from(encodeField(5, LEN, "noop")),
      ]);
      expect(decodeMcpArgs(mcpArgs).args).toEqual({});
    });
  });

  describe("McpResult success", () => {
    it("builds success with single text content", () => {
      const bytes = encodeMcpResultSuccess({ textItems: ['{"temp":32}'], isError: false });
      const msg = decodeMessage(bytes); // McpResult level
      expect(msg.has(1)).toBe(true); // success variant
      const success = decodeMessage(msg.get(1)[0].value);
      expect(success.get(1).length).toBe(1);
      expect(success.get(2)[0].value).toBe(0); // is_error=false
      const item = decodeMessage(success.get(1)[0].value);
      const textContent = decodeMessage(item.get(1)[0].value);
      expect(Buffer.from(textContent.get(1)[0].value).toString("utf8")).toBe('{"temp":32}');
    });

    it("builds success with multiple text items", () => {
      const bytes = encodeMcpResultSuccess({ textItems: ["line1", "line2"] });
      const success = decodeMessage(decodeMessage(bytes).get(1)[0].value);
      expect(success.get(1).length).toBe(2);
    });

    it("marks is_error=true", () => {
      const bytes = encodeMcpResultSuccess({ textItems: ["fail"], isError: true });
      const success = decodeMessage(decodeMessage(bytes).get(1)[0].value);
      expect(success.get(2)[0].value).toBe(1);
    });
  });

  describe("McpResult image content", () => {
    it("builds image item with raw bytes + mime type", () => {
      const imgBytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47]);
      const bytes = encodeMcpResultSuccess({ imageItems: [{ data: imgBytes, mimeType: "image/png" }] });
      const success = decodeMessage(decodeMessage(bytes).get(1)[0].value);
      const item = decodeMessage(success.get(1)[0].value);
      expect(item.has(2)).toBe(true); // image variant
      const img = decodeMessage(item.get(2)[0].value);
      expect(Buffer.from(img.get(1)[0].value)).toEqual(Buffer.from(imgBytes));
      expect(Buffer.from(img.get(2)[0].value).toString("utf8")).toBe("image/png");
    });

    it("builds mixed text + image content", () => {
      const imgBytes = new Uint8Array([1, 2, 3]);
      const bytes = encodeMcpResultSuccess({ textItems: ["see image"], imageItems: [{ data: imgBytes, mimeType: "image/jpeg" }] });
      const success = decodeMessage(decodeMessage(bytes).get(1)[0].value);
      expect(success.get(1).length).toBe(2);
      expect(decodeMessage(success.get(1)[0].value).has(1)).toBe(true); // text
      expect(decodeMessage(success.get(1)[1].value).has(2)).toBe(true); // image
    });
  });

  describe("McpResult error / toolNotFound", () => {
    it("builds error result (field 2)", () => {
      const bytes = encodeMcpResultError("tool crashed");
      const msg = decodeMessage(bytes);
      expect(msg.has(2)).toBe(true);
      const err = decodeMessage(msg.get(2)[0].value);
      expect(Buffer.from(err.get(1)[0].value).toString("utf8")).toBe("tool crashed");
    });

    it("builds toolNotFound result (field 5)", () => {
      const bytes = encodeMcpResultToolNotFound("missing_tool");
      const msg = decodeMessage(bytes);
      expect(msg.has(5)).toBe(true);
      const tnf = decodeMessage(msg.get(5)[0].value);
      expect(Buffer.from(tnf.get(1)[0].value).toString("utf8")).toBe("missing_tool");
    });
  });
});

describe("Cursor AgentService native exec bridge", () => {
  const nativeExec = (field, args = encodeField(1, LEN, "value"), id = 7, execId = "exec-7") => {
    const exec = Buffer.concat([
      Buffer.from(encodeField(1, VARINT, id)),
      Buffer.from(encodeField(field, LEN, args)),
      Buffer.from(encodeField(15, LEN, execId)),
    ]);
    return Buffer.from(wrapConnectRPCFrame(encodeField(2, LEN, exec)));
  };

  it("decodes every known native ExecServerMessage discriminator", () => {
    const cases = [
      [2, "exec_shell"], [3, "exec_write"], [4, "exec_delete"], [5, "exec_grep"],
      [7, "exec_read"], [8, "exec_ls"], [9, "exec_diagnostics"], [10, "exec_request_context"],
      [11, "exec_mcp"], [14, "exec_shell_stream"], [16, "exec_bg_shell"],
      [17, "exec_list_mcp_resources"], [18, "exec_read_mcp_resource"], [20, "exec_fetch"],
      [21, "exec_record_screen"], [22, "exec_computer_use"], [23, "exec_write_shell_stdin"],
      [27, "exec_execute_hook"],
    ];
    for (const [field, kind] of cases) {
      expect(decodeExecServerEvent(nativeExec(field).subarray(5))).toMatchObject({ kind, execMsgId: 7, execId: "exec-7" });
    }
  });

  it("emits the correct ExecClient result oneof for every native variant", () => {
    const cases = [
      ["exec_read", 7, 3], ["exec_write", 3, 6], ["exec_delete", 4, 6], ["exec_ls", 8, 3],
      ["exec_shell", 2, 4], ["exec_shell_stream", 14, 5], ["exec_bg_shell", 16, 3],
      ["exec_grep", 5, 2], ["exec_diagnostics", 9, 3], ["exec_fetch", 20, 2],
      ["exec_write_shell_stdin", 23, 2], ["exec_list_mcp_resources", 17, 3],
      ["exec_read_mcp_resource", 18, 3], ["exec_record_screen", 21, 4], ["exec_computer_use", 22, 2],
    ];
    for (const [kind, resultField, resultVariant] of cases) {
      const event = { kind, execMsgId: 7, execId: "exec-7", path: "/tmp/a", command: "pwd", workingDir: "/tmp", url: "https://example.test", uri: "file:///tmp/a" };
      const outer = decodeMessage(encodeAgentNativeRejection(event).subarray(5));
      const exec = decodeMessage(outer.get(2)[0].value);
      expect(exec.has(resultField), `${kind} result field`).toBe(true);
      const result = decodeMessage(exec.get(resultField)[0].value);
      if (resultVariant === null) expect(result.size).toBe(0);
      else expect(result.has(resultVariant), `${kind} result oneof`).toBe(true);
    }
  });


  it("encodes empty ClientHeartbeat and correctly nested stream close control", () => {
    const heartbeat = decodeMessage(encodeAgentHeartbeat().subarray(5));
    expect(heartbeat.has(7)).toBe(true);
    expect(decodeMessage(heartbeat.get(7)[0].value).size).toBe(0);
    const close = decodeMessage(encodeAgentStreamClose(9).subarray(5));
    const control = decodeMessage(close.get(5)[0].value);
    const streamClose = decodeMessage(control.get(1)[0].value);
    expect(streamClose.get(1)[0].value).toBe(9);
  });

  it("auto-responds to interaction queries with typed InteractionResponse", () => {
    // web_search query (kind 2, id 5) -> approved
    const q = Buffer.concat([
      Buffer.from(encodeField(1, VARINT, 5)),
      Buffer.from(encodeField(2, LEN, encodeField(1, LEN, "q"))),
    ]);
    const queryMsg = wrapConnectRPCFrame(encodeField(7, LEN, q));
    expect(decodeAgentInteractionQuery(queryMsg.subarray(5))).toEqual({ id: 5, kind: 2 });
    const resp = decodeMessage(encodeAgentInteractionResponse(5, 2, true).subarray(5));
    const body = decodeMessage(resp.get(6)[0].value);
    expect(body.get(1)[0].value).toBe(5); // id
    const webSearch = decodeMessage(body.get(2)[0].value);
    expect(webSearch.has(1)).toBe(true); // approved
    // ask_question query (kind 3) -> rejected
    const reject = decodeMessage(encodeAgentInteractionResponse(5, 3, false, "no ui").subarray(5));
    const rejectBody = decodeMessage(reject.get(6)[0].value);
    const askResult = decodeMessage(rejectBody.get(3)[0].value);
    expect(askResult.has(3)).toBe(true); // AskQuestionResult.rejected
  });

  it("encodes native tool success results on the official oneofs", () => {
    const read = decodeMessage(encodeAgentReadSuccess(1, "e", { path: "/a", content: "hi\n" }).subarray(5));
    const readExec = decodeMessage(read.get(2)[0].value);
    const readResult = decodeMessage(readExec.get(7)[0].value);
    expect(readResult.has(1)).toBe(true); // ReadResult.success
    const grep = decodeMessage(encodeAgentGrepSuccess(1, "e", { pattern: "x", matches: [{ file: "/a", lines: [{ lineNumber: 1, content: "x" }] }] }).subarray(5));
    const grepExec = decodeMessage(grep.get(2)[0].value);
    expect(decodeMessage(grepExec.get(5)[0].value).has(1)).toBe(true); // GrepResult.success
    const ls = decodeMessage(encodeAgentLsSuccess(1, "e", { path: "/a", files: ["f"], dirs: ["d"] }).subarray(5));
    expect(decodeMessage(decodeMessage(ls.get(2)[0].value).get(8)[0].value).has(1)).toBe(true); // LsResult.success
    const diag = decodeMessage(encodeAgentDiagnosticsSuccess(1, "e", "/a").subarray(5));
    expect(decodeMessage(decodeMessage(diag.get(2)[0].value).get(9)[0].value).has(1)).toBe(true); // DiagnosticsResult.success
    const fetch = decodeMessage(encodeAgentFetchSuccess(1, "e", { url: "https://x", content: "c" }).subarray(5));
    expect(decodeMessage(decodeMessage(fetch.get(2)[0].value).get(20)[0].value).has(1)).toBe(true); // FetchResult.success
    const write = decodeMessage(encodeAgentWriteSuccess(1, "e", { path: "/a" }).subarray(5));
    expect(decodeMessage(decodeMessage(write.get(2)[0].value).get(3)[0].value).has(1)).toBe(true); // WriteResult.success
    const del = decodeMessage(encodeAgentDeleteSuccess(1, "e", { path: "/a" }).subarray(5));
    expect(decodeMessage(decodeMessage(del.get(2)[0].value).get(4)[0].value).has(1)).toBe(true); // DeleteResult.success
    const shell = decodeMessage(encodeAgentShellSuccess(1, "e", { command: "ls", exitCode: 0, stdout: "o" }).subarray(5));
    const shellResult = decodeMessage(decodeMessage(shell.get(2)[0].value).get(2)[0].value);
    expect(shellResult.has(1)).toBe(true); // ShellResult.success
  });

  it("decodes and echoes KV request metadata", () => {
    const metadata = Buffer.from("opaque-correlation");
    const request = Buffer.from(wrapConnectRPCFrame(encodeField(4, LEN, Buffer.concat([
      Buffer.from(encodeField(1, VARINT, 3)),
      Buffer.from(encodeField(2, LEN, encodeField(1, LEN, Buffer.from([1, 2])))),
      Buffer.from(encodeField(4, LEN, metadata)),
    ]))));
    const decoded = decodeAgentKvServerEvent(request.subarray(5));
    expect(decoded).toMatchObject({ kind: "get", id: 3, metadata });
    const result = decodeMessage(encodeAgentKvGetResult(decoded.id, Buffer.from("blob"), decoded.metadata).subarray(5));
    expect(result.get(3)[0].value).toBeInstanceOf(Uint8Array);
    expect(Buffer.from(decodeMessage(result.get(3)[0].value).get(4)[0].value)).toEqual(metadata);
  });
});

describe("Cursor AgentService executor helpers (cursor.js)", () => {
  it("uses headers from the current supported Cursor client", () => {
    const headers = buildCursorHeaders("test-token", "a".repeat(64));
    expect(headers["x-cursor-client-version"]).toBe("3.13.25");
    expect(headers["x-cursor-client-commit"]).toBe("d5c0e77a0214208f36b56d42e8e787de88d02ea4");
    expect(PROVIDERS.cursor.clientVersion).toBe(headers["x-cursor-client-version"]);
  });

  describe("isAgentCapableRequest", () => {
    it("accepts plain text content", () => {
      expect(isAgentCapableRequest({ messages: [{ role: "user", content: "hi" }] })).toBe(true);
    });

    it("accepts array text content", () => {
      expect(isAgentCapableRequest({ messages: [{ role: "user", content: [{ type: "text", text: "hi" }] }] })).toBe(true);
    });

    it("accepts request with tools declared", () => {
      expect(isAgentCapableRequest({ messages: [{ role: "user", content: "hi" }], tools: [{ function: { name: "t" } }] })).toBe(true);
    });

    it("accepts history with assistant tool_calls + tool results", () => {
      expect(isAgentCapableRequest({
        messages: [
          { role: "user", content: "weather?" },
          { role: "assistant", content: null, tool_calls: [{ id: "c1", type: "function", function: { name: "get_weather", arguments: "{}" } }] },
          { role: "tool", tool_call_id: "c1", content: "sunny" },
          { role: "user", content: "thanks" },
        ],
      })).toBe(true);
    });

    it("rejects non-text (image) content", () => {
      expect(isAgentCapableRequest({ messages: [{ role: "user", content: [{ type: "image_url" }] }] })).toBe(false);
    });

    it("rejects missing messages", () => {
      expect(isAgentCapableRequest({})).toBe(false);
      expect(isAgentCapableRequest(null)).toBe(false);
    });
  });

  describe("buildAgentRunFrame", () => {
    // buildAgentRunFrame returns a wrapped Connect-RPC frame (5-byte header + AgentClientMessage).
    const unwrap = (frame) => frame.subarray(5);

    it("encodes a text-only run request with system + model", () => {
      const frame = unwrap(buildAgentRunFrame(
        [{ role: "system", content: "be brief" }, { role: "user", content: "hi" }],
        "gpt-5.2",
      ));
      const clientMsg = decodeMessage(frame);
      expect(clientMsg.has(1)).toBe(true); // run_request
      const run = decodeMessage(clientMsg.get(1)[0].value);
      expect(run.has(2)).toBe(true); // action
      expect(run.has(9)).toBe(true); // requested_model
    });

    it("encodes a canonical model and catalog-selected parameters", () => {
      const frame = unwrap(buildAgentRunFrame(
        [{ role: "user", content: "hi" }],
        "cursor-grok-4.5-high",
        [],
        {
          modelId: "grok-4.5",
          parameters: [
            { id: "effort", value: "high" },
            { id: "fast", value: "false" },
          ],
          builtInModel: true,
          isVariantStringRepresentation: false,
        },
      ));
      const run = decodeMessage(decodeMessage(frame).get(1)[0].value);
      const requested = decodeMessage(run.get(9)[0].value);

      expect(Buffer.from(requested.get(1)[0].value).toString("utf8")).toBe("grok-4.5");
      expect(requested.get(3)).toHaveLength(2);
      expect(requested.get(3).map(({ value }) => {
        const parameter = decodeMessage(value);
        return [
          Buffer.from(parameter.get(1)[0].value).toString("utf8"),
          Buffer.from(parameter.get(2)[0].value).toString("utf8"),
        ];
      })).toEqual([
        ["effort", "high"],
        ["fast", "false"],
      ]);
      expect(requested.get(7)[0].value).toBe(1);
      expect(requested.has(8)).toBe(false);
    });

    it("encodes mcp_tools (field 4) when tools are provided", () => {
      const tools = [{ function: { name: "get_weather", description: "weather", parameters: { type: "object", properties: { city: { type: "string" } } } } }];
      const frame = unwrap(buildAgentRunFrame([{ role: "user", content: "weather?" }], "gpt-5.2", tools));
      const run = decodeMessage(decodeMessage(frame).get(1)[0].value);
      expect(run.has(4)).toBe(true); // mcp_tools
      const mcpTools = decodeMessage(run.get(4)[0].value);
      expect(mcpTools.get(1).length).toBe(1);
    });

    it("omits mcp_tools when no tools provided", () => {
      const frame = unwrap(buildAgentRunFrame([{ role: "user", content: "hi" }], "gpt-5.2", []));
      const run = decodeMessage(decodeMessage(frame).get(1)[0].value);
      expect(run.has(4)).toBe(false);
    });

    it("encodes conversation_history from prior turns including tool calls/results", () => {
      const messages = [
        { role: "user", content: "weather in Tokyo?" },
        { role: "assistant", content: null, tool_calls: [{ id: "c1", type: "function", function: { name: "get_weather", arguments: '{"city":"Tokyo"}' } }] },
        { role: "tool", tool_call_id: "c1", content: "18C cloudy" },
        { role: "user", content: "thanks" },
      ];
      const frame = unwrap(buildAgentRunFrame(messages, "gpt-5.2", []));
      const run = decodeMessage(decodeMessage(frame).get(1)[0].value);
      const action = decodeMessage(run.get(2)[0].value);
      const userAction = decodeMessage(action.get(1)[0].value);
      expect(userAction.has(7)).toBe(true); // conversation_history (field 7)
      const history = decodeMessage(userAction.get(7)[0].value);
      expect(history.get(1).length).toBeGreaterThanOrEqual(2); // prior turns
    });
  });
});

describe("Cursor tool-result translation", () => {
  it("preserves OpenAI tool error status for AgentService continuation", () => {
    const translated = openaiToCursorRequest("gpt-5.2", {
      messages: [
        { role: "assistant", content: null, tool_calls: [{ id: "call_1", type: "function", function: { name: "failing_tool", arguments: "{}" } }] },
        { role: "tool", tool_call_id: "call_1", content: "failed", status: "error" },
      ],
    });
    expect(translated.messages[1].content).toContain("<is_error>true</is_error>");
  });

  it("preserves Claude tool-result error status", () => {
    const openai = claudeToOpenAIRequest("gpt-5.2", {
      messages: [
        { role: "assistant", content: [{ type: "tool_use", id: "call_1", name: "failing_tool", input: {} }] },
        { role: "user", content: [{ type: "tool_result", tool_use_id: "call_1", content: "failed", is_error: true }] },
      ],
    }, false);
    const translated = openaiToCursorRequest("gpt-5.2", openai, false);
    expect(translated.messages[1].content).toContain("<is_error>true</is_error>");
  });
});
