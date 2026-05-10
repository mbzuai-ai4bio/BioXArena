/**
 * Minimal OpenClawPluginApi shim.
 *
 * Captures tool registrations from Openclaw plugins and materializes
 * tool instances for execution in the bridge context.
 */

import type {
  AnyAgentTool,
  OpenClawConfig,
  OpenClawPluginApi,
  OpenClawPluginToolContext,
  OpenClawPluginToolFactory,
  OpenClawPluginToolOptions,
  PluginLogger,
  PluginRuntime,
} from "openclaw/plugin-sdk/feishu";

// ---------------------------------------------------------------------------
// Config builder: constructs an Openclaw-style config from env vars
// ---------------------------------------------------------------------------

export function buildConfigFromEnv(): OpenClawConfig {
  const appId = process.env.FEISHU_APP_ID ?? "";
  const appSecret = process.env.FEISHU_APP_SECRET ?? "";

  return {
    channels: {
      feishu: {
        enabled: true,
        appId,
        appSecret,
        accounts: {
          default: {
            appId,
            appSecret,
            enabled: true,
            tools: { doc: true, wiki: true, drive: true, perm: true, chat: true, scopes: true },
          },
        },
        tools: { doc: true, wiki: true, drive: true, perm: true, chat: true, scopes: true },
      },
    },
  };
}

// ---------------------------------------------------------------------------
// Logger: writes to stderr so stdout stays clean for JSON-RPC
// ---------------------------------------------------------------------------

function createLogger(pluginId: string): PluginLogger {
  return {
    debug: (msg: string) => process.stderr.write(`[${pluginId}][debug] ${msg}\n`),
    info: (msg: string) => process.stderr.write(`[${pluginId}][info] ${msg}\n`),
    warn: (msg: string) => process.stderr.write(`[${pluginId}][warn] ${msg}\n`),
    error: (msg: string) => process.stderr.write(`[${pluginId}][error] ${msg}\n`),
  };
}

// ---------------------------------------------------------------------------
// Minimal PluginRuntime stub
// ---------------------------------------------------------------------------

function createMinimalRuntime(): PluginRuntime {
  return {
    channel: {
      media: {
        async fetchRemoteMedia(opts: { url: string; maxBytes: number }) {
          // Use native fetch to download media (Node 18+)
          const response = await fetch(opts.url);
          if (!response.ok) {
            throw new Error(`Failed to fetch media: ${response.status} ${response.statusText}`);
          }
          const arrayBuffer = await response.arrayBuffer();
          const buffer = Buffer.from(arrayBuffer);
          if (buffer.length > opts.maxBytes) {
            throw new Error(`Media exceeds limit: ${buffer.length} > ${opts.maxBytes}`);
          }
          return { buffer };
        },
      },
    },
  };
}

// ---------------------------------------------------------------------------
// OpenClawPluginApi Shim
// ---------------------------------------------------------------------------

type ToolRegistration = {
  factory: OpenClawPluginToolFactory;
  opts: OpenClawPluginToolOptions;
};

export class OpenClawPluginApiShim implements OpenClawPluginApi {
  id: string;
  name: string;
  version?: string;
  description?: string;
  source: string;
  config: OpenClawConfig;
  pluginConfig?: Record<string, unknown>;
  runtime: PluginRuntime;
  logger: PluginLogger;

  private toolRegistrations: ToolRegistration[] = [];

  constructor(pluginId: string, config: OpenClawConfig) {
    this.id = pluginId;
    this.name = pluginId;
    this.source = "bridge";
    this.config = config;
    this.runtime = createMinimalRuntime();
    this.logger = createLogger(pluginId);
  }

  registerTool(
    tool: AnyAgentTool | OpenClawPluginToolFactory,
    opts?: OpenClawPluginToolOptions,
  ): void {
    const factory: OpenClawPluginToolFactory =
      typeof tool === "function" ? tool : () => tool;
    this.toolRegistrations.push({ factory, opts: opts ?? {} });
  }

  /**
   * Materialize all registered tool factories into concrete tool instances.
   * Returns a Map of tool_name -> tool instance.
   */
  materializeTools(): Map<string, AnyAgentTool> {
    const ctx: OpenClawPluginToolContext = {
      config: this.config,
      agentAccountId: "default",
    };

    const tools = new Map<string, AnyAgentTool>();
    for (const reg of this.toolRegistrations) {
      const result = reg.factory(ctx);
      if (!result) continue;

      const instances = Array.isArray(result) ? result : [result];
      for (const tool of instances) {
        if (tool && tool.name) {
          tools.set(tool.name, tool);
        }
      }
    }
    return tools;
  }

  // Noop stubs for non-tool registration methods
  registerHook(): void {}
  registerHttpRoute(): void {}
  registerChannel(): void {}
  registerGatewayMethod(): void {}
  registerCli(): void {}
  registerService(): void {}
  registerProvider(): void {}
  registerCommand(): void {}
  registerContextEngine(): void {}
  resolvePath(input: string): string { return input; }
  on(): void {}
}
