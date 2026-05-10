/**
 * Generic plugin loader for Openclaw-compatible plugins.
 * Dynamically imports plugin directories that export a default plugin definition.
 */

import { resolve } from "node:path";

export type PluginDefinition = {
  id: string;
  name: string;
  description?: string;
  configSchema?: unknown;
  register: (api: unknown) => void | Promise<void>;
};

/**
 * Load a plugin from a directory. Expects the directory to contain an index.ts
 * with a default export matching the PluginDefinition shape.
 */
export async function loadPlugin(pluginDir: string): Promise<PluginDefinition> {
  const entryPath = resolve(pluginDir, "index.ts");
  const mod = await import(entryPath);
  const plugin = mod.default as PluginDefinition;

  if (!plugin || typeof plugin.register !== "function") {
    throw new Error(
      `Plugin at ${pluginDir} does not export a valid plugin definition (missing register function)`,
    );
  }

  return plugin;
}
