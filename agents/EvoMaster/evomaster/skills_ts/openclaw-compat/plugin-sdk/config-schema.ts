/**
 * Minimal OpenClawPluginConfigSchema type and emptyPluginConfigSchema implementation.
 * Copied from openclaw/src/plugins/config-schema.ts with minimal dependencies.
 */

export type OpenClawPluginConfigSchema = {
  safeParse?: (value: unknown) => {
    success: boolean;
    data?: unknown;
    error?: {
      issues?: Array<{ path: Array<string | number>; message: string }>;
    };
  };
  parse?: (value: unknown) => unknown;
  validate?: (value: unknown) => { ok: boolean; errors?: string[] };
  uiHints?: Record<string, unknown>;
  jsonSchema?: Record<string, unknown>;
};

type SafeParseResult =
  | { success: true; data?: unknown }
  | { success: false; error: { issues: Array<{ path: Array<string | number>; message: string }> } };

function error(message: string): SafeParseResult {
  return { success: false, error: { issues: [{ path: [], message }] } };
}

export function emptyPluginConfigSchema(): OpenClawPluginConfigSchema {
  return {
    safeParse(value: unknown): SafeParseResult {
      if (value === undefined) {
        return { success: true, data: undefined };
      }
      if (!value || typeof value !== "object" || Array.isArray(value)) {
        return error("expected config object");
      }
      if (Object.keys(value as Record<string, unknown>).length > 0) {
        return error("config must be empty");
      }
      return { success: true, data: value };
    },
    jsonSchema: {
      type: "object",
      additionalProperties: false,
      properties: {},
    },
  };
}
