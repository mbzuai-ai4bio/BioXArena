<p align="center">
  【<a href="./README.md">English</a> | <a href="./README-zh.md">简体中文</a>】
</p>

# EvoMaster 扩展（extensions）

本目录是 [EvoMaster](../README-zh.md) 的**可选扩展**：包含面向「支持 Skill 的智能体」（如 OpenClaw 等）的 **EvoMaster 技能包**，以及便于开发者直接接入的**预置自定义工具**，更多自定义工具即将上线，敬请期待。

## 目录结构

| 路径 | 说明 |
|------|------|
| [`skills/evomaster/`](skills/evomaster/) | Skill 元数据（`SKILL.md`）、参考文档、以及示例工具脚本，用于指导如何搭建与运行 EvoMaster |
| [`tools/`](tools/) | 便于开发者直接接入的预置自定义工具，可复制到任意 playground 使用 |

## `skills/evomaster` — EvoMaster 技能包

**evomaster** 技能是一份结构化指南：前置条件、环境安装、智能体形态（单智能体 / 多智能体顺序与并行 / 自演化）、工具与 Skill 的注册方式，以及按场景选择模式的决策说明。

- **`SKILL.md`** — 入口：框架能力、配置要点、文档导航。
- **`reference/`** — 专题说明，例如最小示例、Bohrium MCP、Kaggle 式循环、多智能体模式、[自定义工具](skills/evomaster/reference/custom_tools.md)、[配置说明](skills/evomaster/reference/configuration.md)、[Skill 使用](skills/evomaster/reference/skills.md)。
- **`scripts/`** — 示例 `BaseTool` 源码（`google_search.py`、`web_fetch.py`）。

### 在支持 Skill 的智能体中使用（如 OpenClaw）

将 `extensions/skills/evomaster` **安装或符号链接**到该产品要求的 **skills 目录**（具体路径以对应产品的文档为准）。安装后，智能体可按名称 **`evomaster`** 加载该技能，并阅读其中的 `SKILL.md` 与 `reference/`。

## `tools/` — 预置自定义工具

这些文件是 EvoMaster 的**自定义工具**：继承 `BaseTool` 的 Python 实现，可在 YAML 中直接注册，让开发者更方便地给自己的智能体添加更多能力。

| 文件 | 作用 |
|------|------|
| [`google_search.py`](tools/google_search.py) | 通过 **Serper** API 进行网页检索（环境变量 `SERPER_KEY_ID`） |
| [`web_fetch.py`](tools/web_fetch.py) | 通过 **Jina Reader** 抓取网页并可结合会话 LLM 抽取信息（按需配置 `JINA_API_KEY` 等，详见脚本内说明与长度限制） |

### 接入到某个 playground

1. 将 `.py` 文件复制或链接到 `playground/<你的_playground>/tools/`。
2. 在配置里注册（值为**不含** `.py` 的文件名）：

   ```yaml
   agents:
     my_agent:
       tools:
         builtin: ["*"]
         google_search: "google_search"
         web_fetch: "web_fetch"
   ```

3. 在项目根目录 **`.env`**（或 shell）中配置各脚本要求的 API 密钥。


## 延伸阅读

- 主项目说明：[README-zh.md](../README-zh.md)
- 技能入口：[skills/evomaster/SKILL.md](skills/evomaster/SKILL.md)
