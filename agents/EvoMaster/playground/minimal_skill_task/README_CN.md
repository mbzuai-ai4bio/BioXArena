# Minimal Skill Task Playground

基于技能的任务 playground，实现 分析 → 规划 → 搜索 → 总结 工作流。

## 概述

Minimal Skill Task Playground 展示了一个四智能体工作流，用于知识型任务：

- **Analyze Agent**: 分析任务并提取关键信息
- **Plan Agent**: 根据分析创建搜索计划
- **Search Agent**: 使用 RAG（检索增强生成）执行搜索
- **Summarize Agent**: 将搜索结果综合为最终答案

## 可用技能

技能系统提供模块化能力：

| 技能 | 类型 | 描述 |
|------|------|------|
| `rag` | Operator | RAG 系统，用于语义搜索和知识检索 |
| `pdf` | Operator | PDF 处理：提取文本/表格、创建、合并/拆分文档 |
| `mcp-builder` | Knowledge | MCP 服务器创建指南 |
| `skill-creator` | Knowledge | 新技能创建指南 |

技能位于 `evomaster/skills/`，可通过 `use_skill` 工具使用。

## 工作流程

```
┌─────────────────┐
│    任务输入     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Analyze Agent   │  提取关键信息
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Plan Agent    │  创建搜索计划
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Search Agent   │  执行 RAG 搜索（use_skill → rag）
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Summarize Agent  │  综合最终答案
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    最终答案     │
└─────────────────┘
```

## 快速开始

### 1. 准备知识库

Playground 使用 RAG 进行搜索。准备你的知识库：

```
# 知识库结构（在任务描述中引用）
knowledge_base/
├── vec_dir/          # 向量数据库（FAISS 索引）
├── nodes_data/       # 节点数据
└── model/            # 嵌入模型
```

### 2. 配置

编辑 `configs/minimal_skill_task/config.yaml`：

```yaml
llm:
  openai:
    provider: "openai"
    model: "gpt-4"
    api_key: "your-api-key"

agents:
  analyze:
    llm: "openai"
    enable_tools: true
  plan:
    llm: "openai"
    enable_tools: true
  search:
    llm: "openai"
    enable_tools: true
  summarize:
    llm: "openai"
    enable_tools: true

skills:
  enabled: true
  skills_root: "evomaster/skills"
```

### 3. 运行

```bash
# 带任务描述运行（包含知识库信息）
python run.py --agent minimal_skill_task --task "基于 /path/to/kb 的知识库，回答：产品的主要特性是什么？"

# 交互式模式
python run.py --agent minimal_skill_task --interactive
```

### 4. 查看结果

结果保存在：

```
runs/{task_id}/
├── logs/                            # 执行日志
└── trajectories/trajectory.json     # 实验轨迹
```

结果结构：

```python
{
    "status": "completed",
    "analyze_output": "...",      # 分析结果
    "search_results": [...],      # 搜索发现
    "summarize_output": "..."     # 最终综合答案
}
```

## 使用技能

### RAG 搜索（通过 use_skill 工具）

```python
# 获取技能信息
{"action": "get_info", "skill_name": "rag"}

# 运行搜索脚本
{"action": "run_script", "skill_name": "rag", "script_name": "search.py", "script_args": "--query '你的查询' --vec_dir /path/to/vec --nodes_data /path/to/nodes --model /path/to/model"}
```

### PDF 处理

```python
# 获取技能信息
{"action": "get_info", "skill_name": "pdf"}

# 获取参考文档
{"action": "get_reference", "skill_name": "pdf", "reference_name": "forms.md"}
```

## 配置选项

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `agents.analyze.max_turns` | 分析最大轮数 | `50` |
| `agents.search.max_turns` | 搜索最大轮数 | `50` |
| `skills.enabled` | 启用技能系统 | `true` |
| `skills.skills_root` | 技能目录 | `"evomaster/skills"` |

## 目录结构

```
playground/minimal_skill_task/
├── core/
│   ├── __init__.py
│   ├── playground.py      # 主 playground
│   ├── exp/
│   │   ├── analyze_exp.py # 分析实验
│   │   ├── search_exp.py  # 搜索实验
│   │   └── summarize_exp.py # 总结实验
│   └── utils/
│       └── rag_utils.py   # RAG 工具
├── prompts/               # Agent 提示词
└── workspace/             # 工作目录

evomaster/skills/
├── rag/                   # RAG 技能（Operator）
│   ├── SKILL.md
│   └── scripts/
│       ├── search.py
│       ├── encode.py
│       └── database.py
├── pdf/                   # PDF 技能（Operator）
│   ├── SKILL.md
│   └── references/
├── mcp-builder/           # MCP 构建器技能（Knowledge）
│   └── SKILL.md
└── skill-creator/         # 技能创建器技能（Knowledge）
    └── SKILL.md
```

## 相关文档

- [EvoMaster 主 README](../../README-zh.md)
- [技能文档](../../docs/zh/skills.md)
- [配置示例](../../configs/)
