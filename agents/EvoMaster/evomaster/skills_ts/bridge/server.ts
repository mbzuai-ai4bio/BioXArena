/**
 * Bridge Server — stdio JSON-RPC server for executing Openclaw plugin tools.
 *
 * Protocol:
 *   → {"id":1,"method":"init","params":{"plugins":["feishu"]}}
 *   ← {"id":1,"result":{"tools":[{"name":"feishu_doc","description":"...","parameters":{...}}]}}
 *
 *   → {"id":2,"method":"execute","params":{"tool_name":"feishu_drive","args":{"action":"list"}}}
 *   ← {"id":2,"result":{"content":[{"type":"text","text":"..."}]}}
 *
 *   → {"id":3,"method":"shutdown"}
 *   ← {"id":3,"result":"ok"}
 */

import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createInterface } from "node:readline";
import { createWriteStream } from "node:fs";

// ---------------------------------------------------------------------------
// stdout protection: SDKs (e.g. Lark) may write non-JSON content to stdout
// via console.log, polluting the JSON-RPC protocol channel. Before importing
// any plugins, save stdout.write as a dedicated RPC channel and redirect
// console.log / process.stdout.write to stderr.
// ---------------------------------------------------------------------------
const _rpcWrite = process.stdout.write.bind(process.stdout);

// Redirect process.stdout.write to stderr (capture SDK's console.log output)
process.stdout.write = process.stderr.write.bind(process.stderr) as any;

// console.log/info/warn/error all go to stderr
console.log = (...args: any[]) => process.stderr.write(args.map(String).join(" ") + "\n");
console.info = console.log;
console.warn = (...args: any[]) => process.stderr.write("[warn] " + args.map(String).join(" ") + "\n");
console.error = (...args: any[]) => process.stderr.write("[error] " + args.map(String).join(" ") + "\n");

import { OpenClawPluginApiShim, buildConfigFromEnv } from "./openclaw-shim.js";
import { loadPlugin } from "./plugin-loader.js";
import type {
  BridgeRequest,
  BridgeResponse,
  BridgeToolInfo,
  InitParams,
  InitResult,
  ExecuteParams,
  ExecuteResult,
} from "./types.js";
import type { AnyAgentTool } from "openclaw/plugin-sdk/feishu";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PLUGINS_DIR = resolve(__dirname, "..", "plugins");

// Global tool registry (populated after init)
const tools = new Map<string, AnyAgentTool>();

// ---------------------------------------------------------------------------
// JSON-RPC I/O helpers — use the saved _rpcWrite to write directly to real stdout
// ---------------------------------------------------------------------------

/**
 * Send a JSON-RPC response to stdout.
 */
function send(response: BridgeResponse): void {
  _rpcWrite(JSON.stringify(response) + "\n");
}

/**
 * Send a JSON-RPC error response.
 */
function sendError(id: number, code: number, message: string): void {
  send({ id, error: { code, message } });
}

// ---------------------------------------------------------------------------
// Method handlers
// ---------------------------------------------------------------------------

/**
 * Handle the "init" method: load specified plugins and register their tools.
 */
async function handleInit(id: number, params: InitParams): Promise<void> {
  const pluginNames = params.plugins ?? [];
  if (pluginNames.length === 0) {
    sendError(id, -32602, "No plugins specified");
    return;
  }

  const config = buildConfigFromEnv();

  for (const name of pluginNames) {
    const pluginDir = resolve(PLUGINS_DIR, name);
    try {
      const pluginDef = await loadPlugin(pluginDir);
      const apiShim = new OpenClawPluginApiShim(pluginDef.id ?? name, config);

      // Register all tools/channels/hooks (only tools are captured)
      await pluginDef.register(apiShim);

      // Materialize tool instances
      const pluginTools = apiShim.materializeTools();
      for (const [toolName, tool] of pluginTools) {
        tools.set(toolName, tool);
        process.stderr.write(`[bridge] Registered tool: ${toolName}\n`);
      }
    } catch (err) {
      process.stderr.write(
        `[bridge] Failed to load plugin "${name}": ${err instanceof Error ? err.message : String(err)}\n`,
      );
    }
  }

  // Build tool info for the Python side
  const toolInfos: BridgeToolInfo[] = [];
  for (const [name, tool] of tools) {
    toolInfos.push({
      name,
      label: (tool as any).label,
      description: tool.description,
      parameters: tool.parameters,
    });
  }

  const result: InitResult = { tools: toolInfos };
  send({ id, result });
}

/**
 * Handle the "execute" method: execute a registered tool by name with the given arguments.
 */
async function handleExecute(id: number, params: ExecuteParams): Promise<void> {
  const { tool_name, args } = params;

  const tool = tools.get(tool_name);
  if (!tool) {
    sendError(id, -32602, `Unknown tool: ${tool_name}`);
    return;
  }

  try {
    const toolCallId = `bridge-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const result = await tool.execute(toolCallId, args);
    const executeResult: ExecuteResult = {
      content: result.content,
      details: result.details,
    };
    send({ id, result: executeResult });
  } catch (err) {
    const errorText = err instanceof Error ? err.message : String(err);
    const executeResult: ExecuteResult = {
      content: [{ type: "text", text: JSON.stringify({ error: errorText }) }],
    };
    send({ id, result: executeResult });
  }
}

/**
 * Handle the "shutdown" method: send acknowledgment and exit the process.
 */
function handleShutdown(id: number): void {
  send({ id, result: "ok" });
  process.exit(0);
}

// ---------------------------------------------------------------------------
// Main: read JSON lines from stdin, dispatch to handlers
// ---------------------------------------------------------------------------

const rl = createInterface({ input: process.stdin, terminal: false });

rl.on("line", async (line: string) => {
  const trimmed = line.trim();
  if (!trimmed) return;

  let request: BridgeRequest;
  try {
    request = JSON.parse(trimmed) as BridgeRequest;
  } catch {
    sendError(0, -32700, "Parse error");
    return;
  }

  const { id, method, params } = request;

  try {
    switch (method) {
      case "init":
        await handleInit(id, params as unknown as InitParams);
        break;
      case "execute":
        await handleExecute(id, params as unknown as ExecuteParams);
        break;
      case "shutdown":
        handleShutdown(id);
        break;
      default:
        sendError(id, -32601, `Unknown method: ${method}`);
    }
  } catch (err) {
    sendError(id, -32603, err instanceof Error ? err.message : String(err));
  }
});

rl.on("close", () => {
  process.exit(0);
});

process.stderr.write("[bridge] Server started, waiting for init...\n");
