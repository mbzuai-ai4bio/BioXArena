# Minimal Bohrium Playground

A minimal playground that demonstrates how to use EvoMaster with **Bohrium** (玻尔科学平台) computing power.

## Overview

Minimal Bohrium Playground is a single-agent playground that shows how to run an EvoMaster agent using Bohrium's cloud computing platform. It's ideal for:

- Getting started with EvoMaster on Bohrium
- Running tasks that require Bohrium OSS and computing resources
- Learning how to integrate Bohrium tools into your agent

## Prerequisites

### 1. Environment Setup

Follow the [EvoMaster Main README](../../README.md) to install base dependencies:

```bash
pip install -r requirements.txt
```

### 2. Install Bohrium Agent SDK

Install the Bohrium agent SDK (required for this playground):

```bash
pip install bohr-agent-sdk -i https://pypi.org/simple --upgrade
```

### 3. Configure Environment Variables

Create or edit the `.env` file in the project root and configure the following Bohrium-related variables:

| Variable | Description |
|----------|-------------|
| `HTTP_PLUGIN_TYPE` | HTTP plugin type (set to `bohrium`) |
| `BOHRIUM_USER_ID` | Your Bohrium user ID |
| `BOHRIUM_PASSWORD` | Your Bohrium account password |
| `BOHRIUM_PROJECT_ID` | Your Bohrium project ID |
| `BOHRIUM_ACCESS_KEY` | Your Bohrium access key |

Example `.env` snippet:

```bash
# Bohrium (calculation MCP storage)
HTTP_PLUGIN_TYPE=bohrium
BOHRIUM_USER_ID=your_user_id
BOHRIUM_PASSWORD=your_password
BOHRIUM_PROJECT_ID=your_project_id
BOHRIUM_ACCESS_KEY=your_access_key
```

## Quick Start

### 1. Configuration

Refer to the configuration files in `./configs/minimal_bohrium`.

### 2. Run

The task is passed as a string via the `--task` parameter:

```bash
# Using the unified entry point
python run.py --agent minimal_bohrium --config configs/minimal_bohrium/deepseek-v3.2-example.yaml --task "Analyze this molecule structure
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

### 3. View Results

After execution, results are saved in the `runs/` directory:

```
runs/minimal_bohrium_{timestamp}/
├── trajectories/       # Agent execution trajectories
├── logs/              # Execution logs
└── workspace/         # Agent working files
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `llm.default` | Default LLM provider | `"local_sglang"` |
| `agents.general.max_turns` | Maximum conversation turns | `50` |
| `agents.general.skills` | Skills (includes `bohrium-oss`) | `["bohrium-oss"]` |
| `session.type` | Session type (local/docker) | `"local"` |

## Directory Structure

```
playground/minimal_bohrium/
├── core/
│   ├── __init__.py
│   └── playground.py    # Main playground implementation
├── prompts/
│   ├── system_prompt.txt
│   └── user_prompt.txt
```

## Customization

To customize the agent's behavior, edit the prompt files in `configs/minimal_bohrium/prompts/`:

- `system_prompt.txt` - Agent's role and capabilities
- `user_prompt.txt` - Task formatting template

## Related Documentation

- [EvoMaster Main README](../../README.md)
- [Bohrium OSS Skill](../../evomaster/skills/bohrium-oss/SKILL.md)
- [Configuration Examples](../../configs/)
