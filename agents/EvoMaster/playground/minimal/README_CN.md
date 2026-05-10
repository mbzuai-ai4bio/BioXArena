# Minimal Playground

最简单的 playground 实现，展示如何使用 EvoMaster 的基础功能。

## 概述

Minimal Playground 是一个单智能体 playground，展示如何快速设置和运行 EvoMaster agent。适用于：

- EvoMaster 入门
- 简单任务执行
- 学习基础 playground 结构

## 快速开始

### 1. 配置

编辑 `configs/minimal/deepseek-v3.2-example.yaml`：

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
# 使用统一入口
python run.py --agent minimal --config configs/minimal/deepseek-v3.2-example.yaml --task "Discover a pattern: Given sequence 1, 4, 9, 16, 25... find the formula"
```

### 3. 查看结果

执行完成后，结果保存在 `runs/` 目录中：

```
runs/minimal_{timestamp}/
├── trajectories/       # Agent 执行轨迹
├── logs/              # 执行日志
└── workspace/         # Agent 工作文件
```

## 配置选项

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `llm.default` | 默认 LLM 提供者 | `"openai"` |
| `agents.general.max_turns` | 最大对话轮数 | `50` |
| `agents.general.enable_tools` | 启用工具调用 | `true` |
| `session.type` | Session 类型 (local/docker) | `"local"` |
| `mcp.enabled` | 启用 MCP 工具 | `true` |

## 示例任务

### 寻找数学规律
```bash
python run.py --agent minimal --config configs/minimal/deepseek-v3.2-example.yaml --task "Discover a pattern: Given sequence 1, 4, 9, 16, 25... find the formula"
```

## 目录结构

```
playground/minimal/
├── core/
│   ├── __init__.py
│   └── playground.py    # 主 playground 实现
├── prompts/
│   ├── system_prompt.txt
│   └── user_prompt.txt
```

## 自定义

要自定义 agent 的行为，编辑 `configs/minimal/prompts/` 中的提示词文件：

- `system_prompt.txt` - Agent 的角色和能力
- `user_prompt.txt` - 任务格式模板

## 相关文档

- [EvoMaster 主 README](../../README.md)
- [配置示例](../../configs/)
