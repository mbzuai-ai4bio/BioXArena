# Minimal OpenClaw Skill Playground

基于 [minimal](../minimal) playground 的最小化实现，演示如何在 EvoMaster 中使用 **OpenClaw 格式的 TypeScript 技能**。通过 Node.js Bridge 支持以 TypeScript 实现的技能。

## 概述

Minimal OpenClaw Skill Playground 适用于：

- 将 OpenClaw 插件（如飞书/Lark）作为 EvoMaster 技能使用
- 理解 TypeScript 技能集成流程
- 运行具备文档、云盘、知识库、权限等工具的 Agent

Playground 从 `evomaster/skills_ts/` 加载技能，并通过 OpenClaw bridge 执行 TypeScript 工具。

## 前置条件

1. **Node.js**（用于运行 OpenClaw bridge）
2. **飞书凭证**（若使用飞书技能）：在 `.env` 中配置 `FEISHU_APP_ID`、`FEISHU_APP_SECRET` 等

## 快速开始

### 1. 安装 TypeScript 依赖

```bash
cd evomaster/skills_ts && npm install
```

### 2. 配置

编辑 `configs/minimal_openclaw_skill/config.yaml`：

- 设置 LLM 提供者（如 `local_sglang`、`openai`、`anthropic`）
- 确认 `openclaw.plugins` 与 `skills` 与你的环境一致

在项目根目录的 `.env` 中添加凭证：

```bash
# 飞书插件
FEISHU_APP_ID=your_app_id
FEISHU_APP_SECRET=your_app_secret
# ... 其他所需环境变量
```

### 3. 运行

```bash
python run.py --agent minimal_openclaw_skill --config configs/minimal_openclaw_skill/config.yaml --task "总结这个飞书文档的内容：<你的飞书文档网址>"
```

### 4. 查看结果

执行完成后，结果保存在 `runs/` 目录：

```
runs/minimal_openclaw_skill_{timestamp}/
├── trajectories/       # Agent 执行轨迹
├── logs/               # 执行日志
└── workspace/          # Agent 工作文件
```

## 配置选项

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `agents.general.tools.openclaw.plugins` | 要加载的 OpenClaw 插件 | `["feishu"]` |
| `agents.general.skills` | 技能名称（SKILL.md 所在目录） | `["feishu-doc", "feishu-drive", "feishu-perm", "feishu-wiki"]` |
| `session.local.working_dir` | 工作目录 | `./playground/minimal_openclaw_skill/workspace` |

## 示例任务

### 总结飞书文档
```bash
python run.py --agent minimal_openclaw_skill --config configs/minimal_openclaw_skill/config.yaml --task "总结这个飞书文档的内容：https://xxx.feishu.cn/docx/ABC123"
```


## 目录结构

```
playground/minimal_openclaw_skill/
├── core/
│   ├── __init__.py
│   └── playground.py    # 主 playground 实现
├── prompts/
│   ├── system_prompt.txt
│   └── user_prompt.txt
├── README.md
└── README_CN.md
```

技能与插件位于 `evomaster/skills_ts/`：

```
evomaster/skills_ts/
├── plugins/
│   └── feishu/         # 飞书 OpenClaw 插件
├── feishu-doc/         # feishu_doc 工具的 SKILL.md
├── feishu-drive/
├── feishu-perm/
├── feishu-wiki/
└── bridge/             # Node.js bridge 服务
```

## 自定义

- **提示词**：编辑 `prompts/system_prompt.txt` 和 `prompts/user_prompt.txt`
- **技能**：在 `configs/minimal_openclaw_skill/config.yaml` 的 `agents.general.skills` 中增删技能
- **插件**：按下方迁移指南添加新的 OpenClaw 插件

---

## OpenClaw 技能迁移指南

本节说明如何将 OpenClaw 插件迁移到 EvoMaster。EvoMaster 通过 Node.js Bridge 运行基于 TypeScript 的 OpenClaw 插件。

### 1. 概述

迁移一个 OpenClaw 插件涉及：

1. 复制插件源码到 `evomaster/skills_ts/plugins/`
2. 适配入口文件（移除 channel 相关代码）
3. 为每个工具创建 SKILL.md 技能描述文件
4. 更新配置文件
5. 处理 import 路径（通常无需修改）

### 2. 目录结构要求

```
evomaster/skills_ts/
├── plugins/
│   └── {plugin-name}/              # 插件源码
│       ├── index.ts                # 入口文件（必须）
│       └── src/                    # 源码目录
│           └── ...
├── {tool-name-1}/                  # 每个工具一个 SKILL.md
│   └── SKILL.md
├── {tool-name-2}/
│   └── SKILL.md
└── bridge/                         # Node.js bridge 服务
```

**关键约束：**

- 插件入口 `index.ts` 必须导出 `default`，格式为 `{ id: string, name: string, register(api: OpenClawPluginApi): void }`
- `register()` 中通过 `api.registerTool(factory, opts)` 注册工具
- 每个注册的工具需要一个对应的 SKILL.md 文件

### 3. 迁移步骤

#### Step 1：复制插件源码

从 Openclaw 仓库复制工具相关文件。**不需要复制**的文件：

- `channel.ts` / `monitor.ts` / `send.ts` — 消息通道相关
- `media.ts` / `reactions.ts` / `mention.ts` — 消息处理相关
- `probe.ts` — 健康检查
- `config-schema.ts` — Zod 配置验证（如果使用 Zod，需特殊处理）
- 测试文件 (`__tests__/`, `*.test.ts`, `*.spec.ts`)

```bash
mkdir -p evomaster/skills_ts/plugins/{plugin-name}/src
cp -r openclaw/extensions/{plugin-name}/src/*.ts evomaster/skills_ts/plugins/{plugin-name}/src/
```

#### Step 2：适配入口文件

创建 `plugins/{plugin-name}/index.ts`，移除 channel 相关 import 和注册：

```typescript
// 原始 Openclaw 入口
import { OpenClawPluginApi } from "openclaw/plugin-sdk/feishu";
import { registerMyTools } from "./src/my-tools.js";
import { registerMyChannel } from "./src/channel.js";  // ← 删除

const plugin = {
  id: "my-plugin",
  name: "My Plugin",
  register(api: OpenClawPluginApi) {
    registerMyTools(api);
    registerMyChannel(api);  // ← 删除
  },
};
export default plugin;
```

适配后：

```typescript
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

#### Step 3：处理依赖

**Openclaw SDK imports：** `openclaw/plugin-sdk/*` 的 import 路径已通过 `package.json` 的 `imports` 字段和 `tsconfig.json` 的 `paths` 映射到本地 `openclaw-compat/` 目录，通常无需修改插件源码。

支持的 import 路径：
- `openclaw/plugin-sdk/feishu` → `openclaw-compat/plugin-sdk/feishu.ts`
- `openclaw/plugin-sdk/compat` → `openclaw-compat/plugin-sdk/compat.ts`
- `openclaw/plugin-sdk/account-id` → `openclaw-compat/plugin-sdk/account-id.ts`

如果新插件使用了不在兼容层中的 import 路径，需要在 `openclaw-compat/plugin-sdk/` 下创建对应文件。

**Zod 依赖：** 如果插件使用 Zod 进行运行时验证：
1. **推荐**：将 Zod schema 转换为纯 TypeScript 类型定义
2. **备选**：在 `package.json` 中添加 `zod` 依赖（`npm install zod`）

**外部 SDK：** 如果插件依赖外部 SDK（如 `@larksuiteoapi/node-sdk`），需添加到 `package.json`：

```bash
cd evomaster/skills_ts
npm install {package-name}
```

#### Step 4：创建 SKILL.md 文件

每个工具需要一个 SKILL.md 文件，放在 `evomaster/skills_ts/{tool-name}/SKILL.md`：

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

**关键字段说明：**

| 字段 | 必填 | 说明 |
|------|------|------|
| `name` | 是 | 技能名称，用于 `use_skill(skill_name=...)` |
| `type` | 是 | 必须为 `"openclaw"` |
| `tool_name` | 是 | Openclaw 中注册的工具名称（`registerTool` 的 `opts.name`） |
| `description` | 是 | 技能描述，帮助 Agent 决定是否使用该技能 |

#### Step 5：更新配置文件

在 Agent 配置中添加 Openclaw 和 Skills 配置：

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

#### Step 6：配置凭证

在项目根目录 `.env` 中添加插件所需的环境变量。Bridge 子进程会自动继承这些环境变量。

如果插件需要特定格式的配置注入（超出简单环境变量），需要修改 `bridge/openclaw-shim.ts` 中的 `buildConfigFromEnv()` 函数。

#### Step 7：测试验证

```bash
cd evomaster/skills_ts && npm install

# 测试 bridge 加载
echo '{"id":1,"method":"init","params":{"plugins":["{plugin-name}"]}}' | npx tsx bridge/server.ts

# 确认工具出现在返回的 tools 列表中

# 端到端测试
python run.py --agent minimal_openclaw_skill --config configs/minimal_openclaw_skill/config.yaml --task "测试指令"
```

### 4. 添加新的 import 别名

如果新插件使用了 `openclaw/plugin-sdk/{new-path}` 这样的 import 路径：

1. 在 `openclaw-compat/plugin-sdk/` 下创建 `{new-path}.ts`
2. 在 `package.json` 的 `imports` 中添加映射：
   ```json
   "openclaw/plugin-sdk/{new-path}": "./openclaw-compat/plugin-sdk/{new-path}.ts"
   ```
3. 在 `tsconfig.json` 的 `paths` 中添加映射：
   ```json
   "openclaw/plugin-sdk/{new-path}": ["./openclaw-compat/plugin-sdk/{new-path}.ts"]
   ```

### 5. 常见问题

**Q：插件有多个工具，需要分别创建 SKILL.md 吗？**  
是的。每个工具需要一个独立的 SKILL.md，因为 Agent 通过 SKILL.md 的 `name` 来选择和调用技能。

**Q：如何调试 bridge 通信？**  
Bridge 进程将日志输出到 stderr。在配置中设置 `logging.level: "DEBUG"` 可以看到 bridge 的详细日志。

**Q：如何处理需要 channel 上下文的工具？**  
部分 Openclaw 工具可能依赖 channel 上下文（如当前会话 ID）。在 EvoMaster 场景下，这些上下文通过工具参数显式传递，而不是从 channel 自动获取。如果工具强依赖 channel 上下文且无法通过参数替代，该工具可能不适合迁移。

**Q：buildConfigFromEnv() 如何为新插件定制？**  
当前 `buildConfigFromEnv()` 是为飞书插件设计的。迁移新插件时，需要根据该插件的配置格式修改此函数。建议的做法是将 `buildConfigFromEnv()` 改为支持多插件配置的通用方案，例如通过环境变量前缀区分不同插件的配置。

---

## 相关文档

- [EvoMaster 主 README](../../README.md)
- [Minimal Playground](../minimal/README_CN.md)
