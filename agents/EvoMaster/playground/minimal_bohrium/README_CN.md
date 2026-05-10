# Minimal Bohrium Playground

使用**玻尔科学平台（Bohrium）**算力的最小 playground 示例。

## 概述

Minimal Bohrium Playground 是一个单智能体 playground，展示如何在玻尔科学平台上运行 EvoMaster agent。适用于：

- 在 Bohrium 上入门 EvoMaster
- 需要 Bohrium OSS 和算力资源的任务
- 学习如何将 Bohrium 工具集成到 agent 中

## 前置条件

### 1. 环境安装

按照 [EvoMaster 主 README](../../README.md) 安装基础依赖：

```bash
pip install -r requirements.txt
```

### 2. 安装 Bohrium Agent SDK

安装 Bohrium agent SDK（本 playground 必需）：

```bash
pip install bohr-agent-sdk -i https://pypi.org/simple --upgrade
```

### 3. 配置环境变量

在项目根目录创建或编辑 `.env` 文件，配置以下 Bohrium 相关变量：

| 变量名 | 描述 |
|--------|------|
| `HTTP_PLUGIN_TYPE` | HTTP 插件类型（设为 `bohrium`） |
| `BOHRIUM_USER_ID` | 玻尔科学平台用户 ID |
| `BOHRIUM_PASSWORD` | 玻尔科学平台账号密码 |
| `BOHRIUM_PROJECT_ID` | 玻尔科学平台项目 ID |
| `BOHRIUM_ACCESS_KEY` | 玻尔科学平台访问密钥 |

示例 `.env` 配置：

```bash
# Bohrium（计算 MCP 存储）
HTTP_PLUGIN_TYPE=bohrium
BOHRIUM_USER_ID=your_user_id
BOHRIUM_PASSWORD=your_password
BOHRIUM_PROJECT_ID=your_project_id
BOHRIUM_ACCESS_KEY=your_access_key
```

## 快速开始

### 1. 配置文件

参考 `./configs/minimal_bohrium` 目录下的配置。

### 2. 运行

任务通过 `--task` 参数以字符串形式传入：

```bash
# 使用统一入口
python run.py --agent minimal_bohrium --config configs/minimal_bohrium/deepseek-v3.2-example.yaml --task "帮我分析这个分子结构
16
Generated_Zero_Day_Conformer_XYZ
C       1.500213    0.000142   -0.000051
N       0.964102    1.149331    0.440219
C      -0.286414    1.472881    0.880502
O      -1.311051    0.727192    1.320110
C      -1.400233   -0.537441    1.760892
N      -0.463881   -1.426115    2.200341
C       0.795120   -1.272901    2.640555
O       1.488334   -0.182413    3.080112
C       1.097561    1.023445    3.520899
N      -0.125212    1.494772    3.960441
C      -1.218553    0.877119    4.400213
O      -1.464882   -0.324551    4.840776
C      -0.634112   -1.359223    5.280145
N       0.634551   -1.359771    5.720882
C       1.449882   -0.388210    6.160113
O       1.213441    0.881552    6.600991"
```

### 3. 查看结果

执行完成后，结果保存在 `runs/` 目录中：

```
runs/minimal_bohrium_{timestamp}/
├── trajectories/       # Agent 执行轨迹
├── logs/              # 执行日志
└── workspace/         # Agent 工作文件
```

## 配置选项

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `llm.default` | 默认 LLM 提供者 | `"local_sglang"` |
| `agents.general.max_turns` | 最大对话轮数 | `50` |
| `agents.general.skills` | 技能（包含 `bohrium-oss`） | `["bohrium-oss"]` |
| `session.type` | Session 类型 (local/docker) | `"local"` |

## 目录结构

```
playground/minimal_bohrium/
├── core/
│   ├── __init__.py
│   └── playground.py    # 主 playground 实现
├── prompts/
│   ├── system_prompt.txt
│   └── user_prompt.txt
```

## 自定义

要自定义 agent 的行为，编辑 `configs/minimal_bohrium/prompts/` 中的提示词文件：

- `system_prompt.txt` - Agent 的角色和能力
- `user_prompt.txt` - 任务格式模板

## 相关文档

- [EvoMaster 主 README](../../README.md)
- [Bohrium OSS Skill](../../evomaster/skills/bohrium-oss/SKILL.md)
- [配置示例](../../configs/)
