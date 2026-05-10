/**
 * Narrowed plugin-sdk surface for the feishu plugin.
 * Only the symbols actually used by tool-related files are provided.
 * Channel/monitor/bot symbols are stubbed or omitted since we only run tools.
 */

import { emptyPluginConfigSchema } from "./config-schema.js";
import { DEFAULT_ACCOUNT_ID, normalizeAgentId } from "./account-id.js";

// Re-export values used by feishu tool files
export { emptyPluginConfigSchema };
export { DEFAULT_ACCOUNT_ID, normalizeAgentId };

// ---------------------------------------------------------------------------
// Minimal type definitions (replacing heavy upstream imports)
// ---------------------------------------------------------------------------

export type PluginLogger = {
  debug?: (message: string) => void;
  info?: (message: string) => void;
  warn: (message: string) => void;
  error: (message: string) => void;
};

/** Minimal OpenClawConfig — only the shape consumed by feishu tool code. */
export type OpenClawConfig = Record<string, unknown>;
export type ClawdbotConfig = OpenClawConfig;

/** Minimal PluginRuntime — stub for setFeishuRuntime(). */
export type PluginRuntime = {
  channel: {
    media: {
      fetchRemoteMedia: (opts: { url: string; maxBytes: number }) => Promise<{ buffer: Buffer }>;
    };
  };
  [key: string]: unknown;
};

/** Minimal RuntimeEnv — not used by tools, but imported by types.ts */
export type RuntimeEnv = Record<string, unknown>;

export type AnyAgentTool = {
  name: string;
  label?: string;
  description: string;
  parameters: unknown;
  execute: (toolCallId: string, params: unknown) => Promise<{
    content: Array<{ type: string; text: string }>;
    details?: unknown;
  }>;
};

export type OpenClawPluginToolContext = {
  config?: OpenClawConfig;
  workspaceDir?: string;
  agentDir?: string;
  agentId?: string;
  sessionKey?: string;
  sessionId?: string;
  messageChannel?: string;
  agentAccountId?: string;
  requesterSenderId?: string;
  senderIsOwner?: boolean;
  sandboxed?: boolean;
};

export type OpenClawPluginToolFactory = (
  ctx: OpenClawPluginToolContext,
) => AnyAgentTool | AnyAgentTool[] | null | undefined;

export type OpenClawPluginToolOptions = {
  name?: string;
  names?: string[];
  optional?: boolean;
};

export type OpenClawPluginApi = {
  id: string;
  name: string;
  version?: string;
  description?: string;
  source: string;
  config: OpenClawConfig;
  pluginConfig?: Record<string, unknown>;
  runtime: PluginRuntime;
  logger: PluginLogger;
  registerTool: (
    tool: AnyAgentTool | OpenClawPluginToolFactory,
    opts?: OpenClawPluginToolOptions,
  ) => void;
  registerHook: (...args: unknown[]) => void;
  registerHttpRoute: (...args: unknown[]) => void;
  registerChannel: (...args: unknown[]) => void;
  registerGatewayMethod: (...args: unknown[]) => void;
  registerCli: (...args: unknown[]) => void;
  registerService: (...args: unknown[]) => void;
  registerProvider: (...args: unknown[]) => void;
  registerCommand: (...args: unknown[]) => void;
  registerContextEngine: (...args: unknown[]) => void;
  resolvePath: (input: string) => string;
  on: (...args: unknown[]) => void;
};

/** Minimal BaseProbeResult — used by feishu types.ts */
export type BaseProbeResult<T = string> = {
  ok: boolean;
  error?: T;
  [key: string]: unknown;
};

// ---------------------------------------------------------------------------
// SecretInput helpers (used by feishu secret-input.ts)
// ---------------------------------------------------------------------------

export type SecretInput = string | { source: string; provider: string; id: string };

export function normalizeSecretInputString(value: unknown): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

export function normalizeResolvedSecretInputString(params: {
  value: unknown;
  refValue?: unknown;
  defaults?: unknown;
  path: string;
}): string | undefined {
  return normalizeSecretInputString(params.value);
}

export function hasConfiguredSecretInput(value: unknown, _defaults?: unknown): boolean {
  return normalizeSecretInputString(value) !== undefined;
}

/**
 * Stub for buildSecretInputSchema — returns a minimal schema-like object.
 * The original uses Zod; we provide a plain-object stub since config schemas
 * are not validated at runtime in the bridge context.
 */
export function buildSecretInputSchema(): unknown {
  return { _type: "secretInput" };
}
