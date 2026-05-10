# X-Master Playground

A multi-agent playground implementing the X-Master workflow for complex problem solving through iterative refinement.

## Overview

X-Master Playground implements a four-phase workflow with parallel execution:

- **Solver Phase**: Generate initial solution(multiple Solver instances can run in parallel)
- **Critic Phase**: Review and correct the solution (each Critic processes a corresponding initial solution in parallel)
- **Rewrite Phase**: Synthesize and improve the solution (multiple Rewriter instances run in parallel, each integrating all corrected solutions)
- **Select Phase**: Choose the best final solution(single agent)

By enabling parallel configuration, you can launch multiple agent instances simultaneously, significantly improving efficiency, especially for exploring and optimizing complex tasks.

## Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                      Step 1: Solve                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                     │
│  │ Solver 1 │ │ Solver 2 │ │ Solver N │                     │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘                     │
│       │            │            │                           │
│       └────────────┼────────────┘  (parallel execution)     │
│                    ▼                                        │
│        Multiple initial solutions                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Step 2: Critique                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                     │
│  │ Critic 1 │ │ Critic 2 │ │ Critic N │                     │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘                     │
│       │            │            │                           │
│       └────────────┼────────────┘                           │
│                    ▼               (parallel execution)     │
│          Multiple revised solutions                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Step 3: Rewrite                         │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐                  │
│  │ Rewriter 1│ │ Rewriter 2│ │ Rewriter N│                  │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘                  │
│        │             │             │                        │
│        └─────────────┼─────────────┘                        │
│                      ▼               (parallel execution)   │
│          Multiple rewritten solutions                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Step 4: Select                         │
│                     ┌──────────┐                            │
│                     │ Selector │   (single Agent)           │
│                     └────┬─────┘                            │
│                          ▼                                  │
│                      Final Solution                         │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Configure

Edit `configs/x_master/config.yaml`:

```yaml
xmaster:
  agent_num: 1        # Number of agents in Solve/Critique/Rewrite phases
  max_workers: 1      # Maximum parallel workers
  parallel: false     # Enable parallel execution (set to true to enable)

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

### 2. Deploy MCP Servers

X-Master requires two MCP services: mcp-sandbox (code execution) and search-tools (web search).

> **Note**: `mcp_sandbox` is based on [sjtu-sai-agents/mcp_sandbox](https://github.com/sjtu-sai-agents/mcp_sandbox), modified to support standardized MCP protocol calls.

#### 2.1 Get Serper API Key

The search tool relies on [Serper](https://serper.dev/)'s Google Search API. You need to obtain an API key first:

1. Visit [https://serper.dev/](https://serper.dev/)
2. Register an account and get your API Key
3. Add the key to `playground/x_master/mcp_sandbox/configs/web_agent.json`:

```json
{
    "serper_api_key": "your-serper-api-key",
    ...
}
```

#### 2.2 Start Services

**One-click start (recommended):**

```bash
cd playground/x_master/mcp_sandbox
./start_all.sh          # Start all services
./start_all.sh stop     # Stop all services
./start_all.sh status   # Check service status
./start_all.sh restart  # Restart all services
```

Default ports:
- mcp-sandbox: 8001
- search-tools: 8002

Custom ports:
```bash
SANDBOX_PORT=8001 SEARCH_PORT=8002 ./start_all.sh
```

### 3. Run

```bash
# Run with task description
python run.py --agent x_master --task "Which condition of Arrhenius's sixth impossibility theorem do critical-level views violate?\n\nAnswer Choices:\nA. Egalitarian Dominance\nB. General Non-Extreme Priority\nC. Non-Elitism\nD. Weak Non-Sadism\nE. Weak Quality Addition"

# Using Custom Configuration
python run.py --agent x_master --config configs/x_master/config.yaml --task "Which condition of Arrhenius's sixth impossibility theorem do critical-level views violate?\n\nAnswer Choices:\nA. Egalitarian Dominance\nB. General Non-Extreme Priority\nC. Non-Elitism\nD. Weak Non-Sadism\nE. Weak Quality Addition"
```

### 4. View Results

Results are saved in:

```
runs/{task_id}/
├── logs/                            # Execution logs
└── trajectories/trajectory.json     # Experiment trajectories
```

Result structure:

```python
{
    "status": "completed",
    "final_solution": "...",          # Best selected solution
    "phase_results": {
        "solver": [...],              # All initial solutions
        "critic": [...],              # All corrected solutions
        "rewriter": [...],            # All rewritten solutions
        "selector": "..."             # Selected solution
    }
}
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `xmaster.agent_num` | Number of agents in Solve/Critique/Rewrite phases | `1` |
| `xmaster.max_workers` | Maximum parallel workers | `1` |
| `xmaster.parallel` | Enable parallel execution mode | `false` |
| `agents.Solver.max_turns` | Solver max turns | `50` |
| `agents.Critic.max_turns` | Critic max turns | `50` |
| `agents.Rewriter.max_turns` | Rewriter max turns | `50` |
| `agents.Selector.max_turns` | Selector max turns | `50` |
| `mcp.enabled` | Enable MCP tools | `true` |

## Configurable Files

| Path | Description |
|------|-------------|
| `configs/x_master/config.yaml` | Main configuration |
| `configs/x_master/mcp_config.json` | MCP configuration |
| `playground/x_master/prompts/*` | Agent prompt files |

## Example Tasks

```bash
python run.py --agent x_master --task "Which condition of Arrhenius's sixth impossibility theorem do critical-level views violate?\n\nAnswer Choices:\nA. Egalitarian Dominance\nB. General Non-Extreme Priority\nC. Non-Elitism\nD. Weak Non-Sadism\nE. Weak Quality Addition"
```

## Directory Structure

```
playground/x_master/
├── core/
│   ├── __init__.py
│   ├── playground.py       # Main playground
│   └── exp/
│       ├── solve_exp.py    # Solve experiment
│       ├── critique_exp.py # Critique experiment
│       ├── rewrite_exp.py  # Rewrite experiment
│       ├── select_exp.py   # Select experiment
│       └── utils.py        # Utilities
├── prompts/                # Agent prompts
├── mcp_sandbox/            # MCP tools & services
└── workspace/              # Working directory
```

## Customization

### Use Different LLMs for Different Phases

```yaml
agents:
  Solver:
    llm: "openai"    
  Critic:
    llm: "anthropic"  
  Rewriter:
    llm: "openai"
  Selector:
    llm: "anthropic"
```

## Related

- [EvoMaster Main README](../../README.md)
- [Configuration Examples](../../configs/)
