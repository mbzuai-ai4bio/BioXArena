# Minimal Kaggle Playground

专为 Kaggle 竞赛自动化设计的多智能体 playground，支持迭代改进工作流。

## 概述

Minimal Kaggle Playground 实现了完整的 Kaggle 竞赛解决工作流，包含多个专业化的 agent：

- **Draft Agent**: 生成初始解决方案代码
- **Debug Agent**: 调试和修复代码错误
- **Improve Agent**: 迭代改进解决方案
- **Research Agent**: 探索新的改进方向
- **Knowledge Promotion Agent**: 提取和维护知识
- **Metric Agent**: 评估解决方案性能

## 工作流程

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│ Draft Agent │────▶│ Debug Agent │────▶│ Metric Agent │
└─────────────┘     └─────────────┘     └──────────────┘
                                               │
                    ┌──────────────────────────┘
                    ▼
            ┌───────────────┐
            │Research Agent │◀─────────────────┐
            └───────────────┘                  │
                    │                          │
                    ▼                          │
            ┌───────────────┐   ┌────────────┐ │
            │Improve Agent  │──▶│Debug Agent │─┘
            └───────────────┘   └────────────┘
                    │
                    ▼
            ┌───────────────┐
            │   最佳方案    │
            └───────────────┘
```

## 快速开始

### 1. 准备数据与环境

将 Kaggle 竞赛数据放入本地位置，后续在配置config.yaml后会自动将对应数据软链接到工作目录，示例任务已经提供了一个数据，位置在：

```
playground/minimal_kaggle/data/
├── private/
│   └── ...     # 测试数据
├── public/
│   └── ...    # 训练数据
```

如果需要运行示例机器学习任务，你需要根据`playground/minimal_kaggle/requirements.txt`在Evomaster基础环境上安装额外需要的环境

### 2. 配置

编辑 `configs/minimal_kaggle/deepseek-v3.2-example.yaml`：

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

### 3. 运行

```bash
# 示例已经在playground/minimal_kaggle/data下提供一个示例数据，如果使用自己的数据，需要修改config.yaml中的local下的软链接配置
python run.py --agent minimal_kaggle --config configs/minimal_kaggle/deepseek-v3.2-example.yaml --task playground/minimal_kaggle/data/public/description.md
```

### 4. 查看结果

结果保存在：

```
runs/minimal_kaggle/workspaces/
├── best_submission/
│   └── submission.csv      # 最佳提交文件
├── best_solution/
│   └── best_solution.py    # 最佳解决方案代码
├── submission/
│   └── submission_*.csv    # 所有提交
└── working/                # 工作文件
```

## 配置选项

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `agents.draft.max_turns` | Draft agent 最大轮数 | `50` |
| `agents.debug.max_turns` | Debug agent 最大轮数 | `50` |
| `agents.improve.max_turns` | Improve agent 最大轮数 | `50` |
| `session.local.working_dir` | 工作目录路径 | `"./playground/minimal_kaggle/workspace"` |

## 使用示例

### 文本检测 (detecting-insults-in-social-commentary)
```bash
python run.py --agent minimal_kaggle --config configs/minimal_kaggle/deepseek-v3.2-example.yaml --task playground/minimal_kaggle/data/public/description.md
```


## 目录结构

```
playground/minimal_kaggle/
├── core/
│   ├── __init__.py
│   ├── playground.py       # 主 playground
│   ├── exp/
│   │   ├── draft_exp.py    # Draft 实验
│   │   ├── improve_exp.py  # Improve 实验
│   │   └── research_exp.py # Research 实验
│   └── utils/
│       ├── code.py         # 代码工具
│       └── data_preview.py # 数据预览
├── prompts/                # Agent 提示词
└── workspace/              # 工作目录
```

## 相关文档

- [EvoMaster 主 README](../../README.md)
- [配置示例](../../configs/)
