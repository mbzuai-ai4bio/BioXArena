/**
 * Shared types for the Bridge JSON-RPC protocol.
 */

/** JSON-RPC request envelope */
export type BridgeRequest = {
  id: number;
  method: "init" | "execute" | "shutdown";
  params?: Record<string, unknown>;
};

/** JSON-RPC response envelope */
export type BridgeResponse = {
  id: number;
  result?: unknown;
  error?: { code: number; message: string };
};

/** Tool information exposed to the Python side */
export type BridgeToolInfo = {
  name: string;
  label?: string;
  description: string;
  parameters: unknown; // JSON Schema from TypeBox
};

/** Init request params */
export type InitParams = {
  plugins: string[];
};

/** Init response result */
export type InitResult = {
  tools: BridgeToolInfo[];
};

/** Execute request params */
export type ExecuteParams = {
  tool_name: string;
  args: Record<string, unknown>;
};

/** Execute response result */
export type ExecuteResult = {
  content: Array<{ type: string; text: string }>;
  details?: unknown;
};
