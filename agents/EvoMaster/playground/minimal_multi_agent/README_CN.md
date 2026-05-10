# Multi-Agent Playground

展示 Planning Agent 和 Coding Agent 协作的多智能体 playground。

## 概述

Multi-Agent Playground 展示了多个 agent 如何协作完成复杂任务：

- **Planning Agent**: 分析任务并创建执行计划
- **Coding Agent**: 根据计划执行代码任务

这种模式适用于需要将规划和执行分离的任务。

## 工作流程

```
┌─────────────────┐
│    任务输入     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Planning Agent  │  分析任务，创建计划
└────────┬────────┘
         │ 计划
         ▼
┌─────────────────┐
│  Coding Agent   │  根据计划执行代码
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      结果       │
└─────────────────┘
```

## 快速开始

### 1. 配置

编辑 `configs/minimal_multi_agent/deepseek-v3.2-example.yaml`：

```yaml
  local_sglang:
    provider: "deepseek"
    model: "deepseek-v3.2"
    api_key: "dummy"  # 本地部署可使用占位符
    base_url: "http://192.168.2.110:18889/v1"
    temperature: 0.7
    max_tokens: 16384
    timeout: 300  
    max_retries: 3
    retry_delay: 1.0

  # ... 如果使用openai api格式，需要同时修改每个agent的LLM配置，如
  agents:
    draft:
      llm: "local_sglang" #如修改成openai
```

### 2. 运行

```bash
# 带任务描述运行
python run.py --agent minimal_multi_agent --config configs/minimal_multi_agent/deepseek-v3.2-example.yaml --task "Write a Python program that implements the following features: Read a text file (create a sample file if it doesn't exist). Count the occurrences of each word in the file. Sort the results by frequency in descending order. Save the results to a new file named word_count.txt. Output the top 10 most common words to the terminal."
```

### 3. 查看结果

结果保存在：

```
runs/minimal_multi_agent_{timestamp}/
├── trajectories/       # Agent 执行轨迹
├── logs/              # 执行日志
└── workspace/         # 生成的文件
```

## 配置选项

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `agents.planning.max_turns` | 规划最大轮数 | `10` |
| `agents.planning.enable_tools` | 启用规划工具 | `false` |
| `agents.coding.max_turns` | 编码最大轮数 | `50` |
| `agents.coding.enable_tools` | 启用编码工具 | `true` |
| `skills.enabled` | 启用技能系统 | `false` |

## 使用示例

### 写代码
```bash
python run.py --agent minimal_multi_agent --config configs/minimal_multi_agent/deepseek-v3.2-example.yaml --task "Write a Python program that implements the following features: Read a text file (create a sample file if it doesn't exist). Count the occurrences of each word in the file. Sort the results by frequency in descending order. Save the results to a new file named word_count.txt. Output the top 10 most common words to the terminal."
```

## 目录结构

```
playground/minimal_multi_agent/
├── core/
│   ├── __init__.py
│   ├── playground.py    # 主 playground
│   └── exp.py           # 多智能体实验
├── prompts/
│   ├── planning_system_prompt.txt
│   ├── planning_user_prompt.txt
│   ├── coding_system_prompt.txt
│   └── coding_user_prompt.txt
```

## 自定义

### 添加更多 Agent

要添加更多 agent，更新配置：

```yaml
agents:
  planning:
    # ...
  coding:
    # ...
  review:  # 新 agent
    llm: "openai"
    max_turns: 10
    enable_tools: false
    system_prompt_file: "prompts/review_system_prompt.txt"
```

然后修改 `playground.py` 和`exp.py` 将新 agent 加入工作流。

## 相关文档

- [EvoMaster 主 README](../../README.md)
- [Minimal Playground](../minimal/README_CN.md)
- [配置示例](../../configs/)
