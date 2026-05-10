# X-Master Playground

实现 X-Master 工作流的多智能体 playground，通过迭代优化解决复杂问题。

## 概述 

X-Master Playground 实现了一个四阶段工作流,支持并行执行：

- **Solve 阶段**: 生成初始解决方案（可并行运行多个 Solver 实例）
- **Critique 阶段**: 审查和修正解决方案（每个修正器可并行处理对应的初始方案）
- **Rewrite 阶段**: 综合改进解决方案（可并行运行多个 Rewriter 实例，每个实例整合所有修正方案）
- **Select 阶段**: 选择最佳最终方案（单 Agent 执行）
 
通过并行配置，您可以同时启动多个 Agent 实例，大幅提升执行效率，尤其适合复杂任务的探索与优化。

## 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│                      阶段 1: Solve                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                     │
│  │ Solver 1 │ │ Solver 2 │ │ Solver N │  (并行执行)         │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘                    │
│       │            │            │                          │
│       └────────────┼────────────┘                          │
│                    ▼                                        │
│                  多个初始方案                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     阶段 2: Critique                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                     │
│  │ Critic 1 │ │ Critic 2 │ │ Critic N │  (并行执行)         │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘                    │
│       │            │            │                          │
│       └────────────┼────────────┘                          │
│                    ▼                                        │
│                多个修正方案                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     阶段 3: Rewrite                         │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐                 │
│  │ Rewriter 1│ │ Rewriter 2│ │ Rewriter N│  (并行执行)      │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘                 │
│        │             │             │                       │
│        └─────────────┼─────────────┘                       │
│                      ▼                                      │
│                 多个重写方案                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      阶段 4: Select                         │
│                     ┌──────────┐                            │
│                     │ Selector │   (单 Agent)               │
│                     └────┬─────┘                            │
│                          ▼                                  │
│                      最终方案                                │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 配置

编辑 `configs/x_master/config.yaml`：

```yaml
xmaster:
  agent_num: 1        # 每个阶段启动的 Agent 数量（Solve/Critique/Rewriter）
  max_workers: 1      # 并行执行的最大工作进程数
  parallel: false     # 是否启用并行模式

llm:
  openai:
    provider: "openai"
    model: "gpt-4"
    api_key: "your-api-key"

agents:
  Solver:
    llm: "openai"
    max_turns: 50
    enable_tools: true
    system_prompt_file: "prompts/solver_prefix.txt"
    user_prompt_file: "prompts/solver_user.txt"

  Critic:
    llm: "openai"
    max_turns: 50
    enable_tools: true

  Rewriter:
    llm: "openai"
    max_turns: 50
    enable_tools: true

  Selector:
    llm: "openai"
    max_turns: 50
    enable_tools: true
```

### 2. 部署 MCP 服务

X-Master 需要两个 MCP 服务：mcp-sandbox（代码执行）和 search-tools（网络搜索）。

> **说明**：`mcp_sandbox` 基于 [sjtu-sai-agents/mcp_sandbox](https://github.com/sjtu-sai-agents/mcp_sandbox) 仓库修改，支持标准化的 MCP 协议调用。

#### 2.1 获取 Serper API Key

搜索工具依赖 [Serper](https://serper.dev/) 的 Google Search API，需要先申请 API Key：

1. 访问 [https://serper.dev/](https://serper.dev/)
2. 注册账号并获取 API Key
3. 将 Key 填入 `playground/x_master/mcp_sandbox/configs/web_agent.json`：

```json
{
    "serper_api_key": "your-serper-api-key",
    ...
}
```

#### 2.2 启动服务

**一键启动（推荐）：**

```bash
cd playground/x_master/mcp_sandbox
./start_all.sh          # 启动所有服务
./start_all.sh stop     # 停止所有服务
./start_all.sh status   # 检查服务状态
./start_all.sh restart  # 重启所有服务
```

默认端口：
- mcp-sandbox: 8001
- search-tools: 8002

自定义端口：
```bash
SANDBOX_PORT=8001 SEARCH_PORT=8002 ./start_all.sh
```

### 3. 运行

```bash
# 带任务描述运行
python run.py --agent x_master --task "Which condition of Arrhenius's sixth impossibility theorem do critical-level views violate?\n\nAnswer Choices:\nA. Egalitarian Dominance\nB. General Non-Extreme Priority\nC. Non-Elitism\nD. Weak Non-Sadism\nE. Weak Quality Addition"

# 使用自定义配置
python run.py --agent x_master --config configs/x_master/config.yaml --task "Which condition of Arrhenius's sixth impossibility theorem do critical-level views violate?\n\nAnswer Choices:\nA. Egalitarian Dominance\nB. General Non-Extreme Priority\nC. Non-Elitism\nD. Weak Non-Sadism\nE. Weak Quality Addition"
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
    "final_solution": "...",          # 最佳选中方案
    "phase_results": {
        "solver": [...],              # 初始方案
        "critic": [...],              # 修正方案
        "rewriter": [...],            # 重写方案
        "selector": "..."             # 选中的方案
    }
}
```

## 配置选项

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `xmaster.agent_num` | 每个阶段（Solve/Critique/Rewriter）启动的 Agent 数量 | `1` |
| `xmaster.max_workers` | 并行执行的最大工作进程数 | `1` |
| `xmaster.parallel` | 是否启用并行执行模式 | `false` |
| `agents.Solver.max_turns` | Solver 最大轮数 | `50` |
| `agents.Critic.max_turns` | Critic 最大轮数 | `50` |
| `agents.Rewriter.max_turns` | Rewriter 最大轮数 | `50` |
| `agents.Selector.max_turns` | Selector 最大轮数 | `50` |
| `mcp.enabled` | 启用 MCP 工具 | `true` |

## 可配置文件

| 路径 | 描述 |
|------|------|
| `configs/x_master/config.yaml` | 主配置文件 |
| `configs/x_master/mcp_config.json` | MCP 配置 |
| `playground/x_master/prompts/*` | Agent 提示词文件 |

## 使用示例

```bash
python run.py --agent x_master --task "Which condition of Arrhenius's sixth impossibility theorem do critical-level views violate?\n\nAnswer Choices:\nA. Egalitarian Dominance\nB. General Non-Extreme Priority\nC. Non-Elitism\nD. Weak Non-Sadism\nE. Weak Quality Addition"
```

## 目录结构

```
playground/x_master/
├── core/
│   ├── __init__.py
│   ├── playground.py       # 主 playground
│   └── exp/
│       ├── solve_exp.py    # Solve 实验
│       ├── critique_exp.py # Critique 实验
│       ├── rewrite_exp.py  # Rewrite 实验
│       ├── select_exp.py   # Select 实验
│       └── utils.py        # 工具函数
├── prompts/                # Agent 提示词
├── mcp_sandbox/            # MCP 工具和服务
└── workspace/              # 工作目录
```

## 自定义

### 不同阶段使用不同 LLM

```yaml
agents:
  Solver:
    llm: "openai"     # GPT-4 用于求解
  Critic:
    llm: "anthropic"  # Claude 用于评审
  Rewriter:
    llm: "openai"
  Selector:
    llm: "anthropic"
```

## 相关文档

- [EvoMaster 主 README](../../README-zh.md)
- [配置示例](../../configs/)
