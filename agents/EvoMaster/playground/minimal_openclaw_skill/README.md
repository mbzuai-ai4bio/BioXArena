# Minimal OpenClaw Skill Playground

A minimal playground that demonstrates how to use **OpenClaw-format TypeScript skills** in EvoMaster. Based on [minimal](../minimal) playground, it adds support for skills implemented in TypeScript via the Node.js Bridge.

## Overview

Minimal OpenClaw Skill Playground is ideal for:

- Using OpenClaw plugins (e.g., Feishu/Lark) as EvoMaster skills
- Understanding the TypeScript skill integration flow
- Running agents with document, drive, wiki, and permission tools

The playground loads skills from `evomaster/skills_ts/` and runs the OpenClaw bridge to execute TypeScript tools.

## Prerequisites

1. **Node.js** (for the OpenClaw bridge)
2. **Feishu credentials** (if using Feishu skills): configure `FEISHU_APP_ID`, `FEISHU_APP_SECRET`, etc. in `.env`

## Quick Start

### 1. Install TypeScript Dependencies

```bash
cd evomaster/skills_ts && npm install
```

### 2. Configure

Edit `configs/minimal_openclaw_skill/config.yaml`:

- Set LLM provider (e.g., `local_sglang`, `openai`, `anthropic`)
- Ensure `openclaw.plugins` and `skills` match your setup

Add credentials to `.env` at project root:

```bash
# For Feishu plugin
FEISHU_APP_ID=your_app_id
FEISHU_APP_SECRET=your_app_secret
# ... other env vars as needed
```

### 3. Run

```bash
python run.py --agent minimal_openclaw_skill --config configs/minimal_openclaw_skill/config.yaml --task "Summarize the content of this Feishu document: <your_feishu_doc_url>"
```

### 4. View Results

Results are saved in `runs/`:

```
runs/minimal_openclaw_skill_{timestamp}/
├── trajectories/       # Agent execution trajectories
├── logs/               # Execution logs
└── workspace/          # Agent working files
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `agents.general.tools.openclaw.plugins` | OpenClaw plugins to load | `["feishu"]` |
| `agents.general.skills` | Skill names (SKILL.md directories) | `["feishu-doc", "feishu-drive", "feishu-perm", "feishu-wiki"]` |
| `session.local.working_dir` | Workspace directory | `./playground/minimal_openclaw_skill/workspace` |

## Example Tasks

### Summarize a Feishu Document
```bash
python run.py --agent minimal_openclaw_skill --config configs/minimal_openclaw_skill/config.yaml --task "Summarize the content of this Feishu document: https://xxx.feishu.cn/docx/ABC123"
```


## Directory Structure

```
playground/minimal_openclaw_skill/
├── core/
│   ├── __init__.py
│   └── playground.py    # Main playground implementation
├── prompts/
│   ├── system_prompt.txt
│   └── user_prompt.txt
├── README.md
└── README_CN.md
```

Skills and plugins live in `evomaster/skills_ts/`:

```
evomaster/skills_ts/
├── plugins/
│   └── feishu/         # Feishu OpenClaw plugin
├── feishu-doc/         # SKILL.md for feishu_doc tool
├── feishu-drive/
├── feishu-perm/
├── feishu-wiki/
└── bridge/             # Node.js bridge server
```

## Customization

- **Prompts**: Edit `prompts/system_prompt.txt` and `prompts/user_prompt.txt`
- **Skills**: Add or remove skills in `configs/minimal_openclaw_skill/config.yaml` under `agents.general.skills`
- **Plugins**: Add new OpenClaw plugins following the Migration Guide below

---

## OpenClaw Skill Migration Guide

This section describes how to migrate an OpenClaw plugin to EvoMaster. EvoMaster uses a Node.js Bridge to run TypeScript-based OpenClaw plugins.

### 1. Overview

Migrating an OpenClaw plugin involves:

1. Copy plugin source to `evomaster/skills_ts/plugins/`
2. Adapt the entry file (remove channel-related code)
3. Create SKILL.md files for each tool
4. Update configuration
5. Handle import paths (usually no changes needed)

### 2. Directory Structure

```
evomaster/skills_ts/
├── plugins/
│   └── {plugin-name}/              # Plugin source
│       ├── index.ts                # Entry file (required)
│       └── src/                    # Source directory
│           └── ...
├── {tool-name-1}/                  # One SKILL.md per tool
│   └── SKILL.md
├── {tool-name-2}/
│   └── SKILL.md
└── bridge/                         # Node.js bridge server
```

**Constraints:**

- Plugin entry `index.ts` must export `default` with format: `{ id: string, name: string, register(api: OpenClawPluginApi): void }`
- `register()` registers tools via `api.registerTool(factory, opts)`
- Each registered tool needs a corresponding SKILL.md

### 3. Migration Steps

#### Step 1: Copy Plugin Source

Copy tool-related files from the OpenClaw repo. **Do NOT copy:**

- `channel.ts` / `monitor.ts` / `send.ts` — message channel
- `media.ts` / `reactions.ts` / `mention.ts` — message handling
- `probe.ts` — health check
- `config-schema.ts` — Zod config validation (needs special handling if used)
- Test files (`__tests__/`, `*.test.ts`, `*.spec.ts`)

```bash
mkdir -p evomaster/skills_ts/plugins/{plugin-name}/src
cp -r openclaw/extensions/{plugin-name}/src/*.ts evomaster/skills_ts/plugins/{plugin-name}/src/
```

#### Step 2: Adapt Entry File

Create `plugins/{plugin-name}/index.ts`, remove channel-related imports and registration:

```typescript
// Before (original OpenClaw)
import { OpenClawPluginApi } from "openclaw/plugin-sdk/feishu";
import { registerMyTools } from "./src/my-tools.js";
import { registerMyChannel } from "./src/channel.js";  // ← remove

const plugin = {
  id: "my-plugin",
  name: "My Plugin",
  register(api: OpenClawPluginApi) {
    registerMyTools(api);
    registerMyChannel(api);  // ← remove
  },
};
export default plugin;
```

```typescript
// After (adapted for EvoMaster)
import type { OpenClawPluginApi } from "openclaw/plugin-sdk/feishu";
import { registerMyTools } from "./src/my-tools.js";

const plugin = {
  id: "my-plugin",
  name: "My Plugin",
  register(api: OpenClawPluginApi) {
    registerMyTools(api);
  },
};
export default plugin;
```

#### Step 3: Handle Dependencies

**Openclaw SDK imports:** `openclaw/plugin-sdk/*` is mapped via `package.json` imports and `tsconfig.json` paths to `openclaw-compat/`. Usually no changes needed.

Supported paths:
- `openclaw/plugin-sdk/feishu` → `openclaw-compat/plugin-sdk/feishu.ts`
- `openclaw/plugin-sdk/compat` → `openclaw-compat/plugin-sdk/compat.ts`
- `openclaw/plugin-sdk/account-id` → `openclaw-compat/plugin-sdk/account-id.ts`

For new paths, add a file under `openclaw-compat/plugin-sdk/`.

**Zod:** If the plugin uses Zod for runtime validation:
1. **Recommended**: Convert Zod schema to pure TypeScript types
2. **Alternative**: Add `zod` to `package.json` (`npm install zod`)

**External SDKs:** Add to `package.json`:
```bash
cd evomaster/skills_ts
npm install {package-name}
```

#### Step 4: Create SKILL.md

Each tool needs `evomaster/skills_ts/{tool-name}/SKILL.md`:

```yaml
---
name: {tool-name}
type: openclaw
tool_name: {actual_tool_name_in_code}
description: |
  One-line description of what this tool does.
  When to use: trigger conditions.
---

# Tool Name

Detailed usage documentation...

## Actions

### Action 1

    { "action": "action1", "param": "value" }

### Action 2
...
```

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Skill name for `use_skill(skill_name=...)` |
| `type` | Yes | Must be `"openclaw"` |
| `tool_name` | Yes | Tool name in code (`registerTool` opts.name) |
| `description` | Yes | Helps Agent decide when to use the skill |

#### Step 5: Update Configuration

Add Openclaw and Skills to your agent config:

```yaml
agents:
  default:
    tools:
      builtin: ["*"]
      openclaw:
        enabled: true
        skills_ts_dir: "./evomaster/skills_ts"
        plugins:
          - {plugin-name}

    skills:
      skill_dir: "./evomaster/skills_ts"
      skills:
        - {tool-name-1}
        - {tool-name-2}
```

#### Step 6: Configure Credentials

Add required env vars to `.env` at project root. The Bridge subprocess inherits them.

For custom config injection, modify `buildConfigFromEnv()` in `bridge/openclaw-shim.ts`.

#### Step 7: Test

```bash
cd evomaster/skills_ts && npm install

# Test bridge load
echo '{"id":1,"method":"init","params":{"plugins":["{plugin-name}"]}}' | npx tsx bridge/server.ts

# Verify tools appear in the response

# End-to-end
python run.py --agent minimal_openclaw_skill --config configs/minimal_openclaw_skill/config.yaml --task "Your test task"
```

### 4. Adding New Import Aliases

If a plugin uses `openclaw/plugin-sdk/{new-path}`:

1. Create `openclaw-compat/plugin-sdk/{new-path}.ts`
2. Add to `package.json` imports:
   ```json
   "openclaw/plugin-sdk/{new-path}": "./openclaw-compat/plugin-sdk/{new-path}.ts"
   ```
3. Add to `tsconfig.json` paths:
   ```json
   "openclaw/plugin-sdk/{new-path}": ["./openclaw-compat/plugin-sdk/{new-path}.ts"]
   ```

### 5. FAQ

**Q: Multiple tools in one plugin — need separate SKILL.md for each?**  
Yes. Each tool needs its own SKILL.md; the Agent selects skills by `name`.

**Q: How to debug bridge communication?**  
Bridge logs go to stderr. Set `logging.level: "DEBUG"` in config to see detailed logs.

**Q: Tools that need channel context?**  
In EvoMaster, context is passed explicitly via tool parameters. If a tool strongly depends on channel context and cannot accept it as parameters, it may not be suitable for migration.

**Q: How to customize buildConfigFromEnv() for new plugins?**  
`buildConfigFromEnv()` is currently tailored for Feishu. For new plugins, adapt it to your plugin’s config format, or extend it to support multiple plugins via env var prefixes.

---

## Related Documentation

- [EvoMaster Main README](../../README.md)
- [Minimal Playground](../minimal/README.md)
