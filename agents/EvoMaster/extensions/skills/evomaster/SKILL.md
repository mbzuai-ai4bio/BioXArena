---
name: evomaster
description: Build, configure, and run EvoMaster autonomous AI agents. Use when you need to create single or multi-agent systems with tool calling, MCP integration, skill extensions, or self-evolving workflows for scientific discovery, Kaggle competitions, or general automation tasks.
---

# EvoMaster Skill

EvoMaster is a framework for building autonomous AI agents that can think, use tools, and self-evolve through iterative experimentation. This skill guides you through setting up, configuring, and running EvoMaster-based agent systems.

## Prerequisites

Before running any EvoMaster agent, you need API keys for the LLM providers you plan to use. **Please ask the user to provide the following as needed:**

- **LLM API Key** (required): e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or a compatible provider key
- **LLM Base URL** (if using a proxy or custom endpoint): e.g., `GPT_BASE_URL`
- **LLM Model Name**: e.g., `GPT_CHAT_MODEL`
- **Search API Key** (optional, for web search tools): e.g., Google Search API key
- **MCP Service URLs** (optional, for MCP tool integrations): e.g., Bohrium platform endpoints
- **Bohrium Credentials** (optional, for scientific computing): `BOHRIUM_ACCESS_KEY`, `BOHRIUM_PROJECT_ID`
- **Kaggle Credentials** (optional, for Kaggle competitions): Kaggle API token

Store these in a `.env` file at the project root. EvoMaster automatically loads `.env` files and substitutes `${VAR_NAME}` placeholders in configuration YAML files.

## Environment Setup

### Clone and Install

```bash
git clone https://github.com/sjtu-sai-agents/EvoMaster.git
cd EvoMaster

# Install dependencies
pip install -r requirements.txt

# (Optional) Install additional dependencies for specific features
pip install faiss-cpu       # For RAG vector search
pip install bohr-agent-sdk  # For Bohrium integration
```

### Verify Installation

```bash
python -c "from evomaster.core import BasePlayground; print('EvoMaster ready')"
```

## Framework Capability Map

EvoMaster provides a comprehensive agent framework with the following capabilities:

### Agent Architectures

| Architecture | Description | Use Case |
|---|---|---|
| **Single Agent** | One agent with tools, running a think-act-observe loop | Simple automation, scientific discovery, general tasks |
| **Multi-Agent (Sequential)** | Multiple agents with different roles executing in sequence | Planning + execution workflows, research pipelines |
| **Multi-Agent (Parallel)** | Multiple agent copies executing the same or different tasks concurrently | Parallel experimentation, best-of-N optimization |
| **Self-Evolving Multi-Agent** | Agents that iterate through draft-research-improve cycles | Kaggle competitions, iterative optimization |

### Output Modes

| Mode | Description | Config |
|---|---|---|
| **Plain Text** | Agent responds with text only, no tool calls | `tools: { builtin: [] }` |
| **ReAct (Think + Act + Observe)** | Agent reasons, calls tools, observes results, and iterates | `tools: { builtin: ["*"] }` |

### Extension System

| Extension Type | Description |
|---|---|
| **Built-in Tools** | `execute_bash` (shell commands), `str_replace_editor` (file operations), `think` (reasoning), `finish` (task completion) |
| **Custom Tools** | User-defined tools inheriting from `BaseTool`, auto-discovered from `playground/{name}/tools/` |
| **MCP Tools** | Tools provided by MCP (Model Context Protocol) servers via stdio, HTTP, or SSE transport |
| **Skills** | Domain knowledge packages with metadata, documentation, and executable scripts |

## Tool & Skill Registration Mechanism

### Built-in Tools

Built-in tools are automatically registered when creating an agent. Control which tools are exposed via the config:

```yaml
agents:
  my_agent:
    tools:
      builtin: ["*"]                    # All built-in tools
      # builtin: ["execute_bash", "finish"]  # Only specific tools
      # builtin: []                     # No built-in tools (text-only agent)
```

### Custom Tools

1. Create a Python file in `playground/{your_playground}/tools/`:

```python
from evomaster.agent.tools.base import BaseTool, BaseToolParams
from pydantic import Field
from typing import ClassVar, Any

class MySearchParams(BaseToolParams):
    """Search the web for information."""
    name: ClassVar[str] = "my_search"
    query: str = Field(description="Search query")

class MySearchTool(BaseTool):
    name: ClassVar[str] = "my_search"
    params_class: ClassVar[type[BaseToolParams]] = MySearchParams

    def execute(self, session, args_json: str) -> tuple[str, dict[str, Any]]:
        params = self.parse_params(args_json)
        # Your tool logic here
        return f"Results for: {params.query}", {"status": "ok"}
```

2. Register in config:

```yaml
agents:
  my_agent:
    tools:
      builtin: ["*"]
      my_search: "my_search"   # key: tool_type, value: filename (without .py)
```

3. **Pre-built custom tools in this skill package**

   Agents using this skill can find **ready-made EvoMaster `BaseTool` implementations** under **`scripts/`** in the same directory as this `SKILL.md`. Copy or adapt these files into `playground/{your_playground}/tools/` and register them as in step 2 (filename without `.py` as the tool config value).

   | Script | Purpose |
   |---|---|
   | `scripts/google_search.py` | Google-style search via the **Serper** API (see script / env for API key variables). |
   | `scripts/web_fetch.py` | Fetch pages via **Jina Reader** and extract structured content with the session LLM (see script for env and limits). |


### MCP Tools

1. Create an MCP configuration file (e.g., `mcp_config.json`) in your config directory:

```json
{
  "mcpServers": {
    "my-server": {
      "command": "npx",
      "args": ["-y", "my-mcp-server"]
    },
    "my-http-server": {
      "transport": "http",
      "url": "http://localhost:8080/mcp"
    }
  }
}
```

2. Enable in config:

```yaml
agents:
  my_agent:
    tools:
      builtin: ["*"]
      mcp: "mcp_config.json"    # Path to MCP config (relative to config dir)
      # mcp: "*"                 # Use default mcp_config.json
```

MCP tools are automatically discovered, connected, and registered. Each MCP tool is named `{server_name}_{tool_name}`.

### Skills

Skills are domain knowledge packages located under `evomaster/skills/`. Each skill has a `SKILL.md` with YAML frontmatter (name, description) and optional `scripts/` and `reference/` directories.

1. Enable skills for an agent in config:

```yaml
agents:
  my_agent:
    skills: ["*"]           # Expose all available skills
    # skills: ["rag"]       # Expose only the RAG skill
    # skills: []            # No skills (default)
    skill_dir: "./evomaster/skills"  # Skills root directory
```

2. The agent uses skills via the `use_skill` tool with three actions:
   - `get_info`: Load the full SKILL.md content for a skill
   - `get_reference`: Load a specific reference document from the skill
   - `run_script`: Execute a script bundled with the skill

## Agent Workflow Decision Tree

When receiving a user request to build an EvoMaster agent, follow this decision process:

```
1. ANALYZE the business scenario
   ├── Simple automation / single task?
   │   └── Use Single Agent (see reference/minimal.md)
   │
   ├── Task requires planning + execution phases?
   │   └── Use Multi-Agent Sequential (see reference/minimal_multi_agent.md)
   │
   ├── Task benefits from parallel attempts?
   │   └── Use Multi-Agent Parallel (see reference/minimal_multi_agent_parallel.md)
   │
   ├── Task requires iterative self-improvement?
   │   └── Use Self-Evolving Multi-Agent (see reference/minimal_kaggle.md)
   │
   └── Task requires external services (scientific computing, web APIs)?
       └── Use MCP Integration (see reference/minimal_bohrium.md)

2. EVALUATE tool requirements
   ├── Only text reasoning needed? → builtin: []
   ├── Code execution + file editing? → builtin: ["*"]
   ├── External API access needed? → Add MCP tools
   ├── Domain knowledge needed? → Add Skills (see reference/skills.md)
   └── Custom functionality? → Create Custom Tools (see reference/custom_tools.md)

3. READ the appropriate reference documents
   └── Navigate to reference/ for detailed implementation guides
```

## Reference Documentation

Detailed guides are in the `reference/` directory. Load only what you need:

| Document | Description |
|---|---|
| `reference/minimal.md` | Single-agent ReAct pattern - the simplest EvoMaster setup |
| `reference/minimal_bohrium.md` | MCP tool integration with Bohrium scientific computing platform |
| `reference/minimal_kaggle.md` | Self-evolving multi-agent system for Kaggle competitions |
| `reference/minimal_multi_agent.md` | Sequential multi-agent collaboration (Planning + Coding) |
| `reference/minimal_multi_agent_parallel.md` | Parallel multi-agent execution with independent workspaces |
| `reference/minimal_skill_task.md` | Agents using Skills for knowledge retrieval (RAG) |
| `reference/custom_tools.md` | How to create, register, and use custom tools (including MCP) |
| `reference/skills.md` | How to create, register, and use Skills |
| `reference/configuration.md` | Complete configuration reference (YAML structure, environment variables) |

## Troubleshooting

### Dependency Errors

| Error | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'evomaster'` | Run from the project root, or add the root to `PYTHONPATH` |
| `ModuleNotFoundError: No module named 'dotenv'` | `pip install python-dotenv` |
| `ImportError: cannot import name 'create_llm'` | Ensure `evomaster/utils/` is intact; check `requirements.txt` |

### API Call Failures

| Error | Solution |
|---|---|
| `AuthenticationError` / 401 | Check `OPENAI_API_KEY` or equivalent in `.env`; verify the key is valid |
| `RateLimitError` / 429 | Reduce `max_retries` or add `retry_delay` in LLM config; use a different API key |
| `Timeout` errors | Increase `timeout` in LLM config (default 60s); check network connectivity |
| `ContextOverflowError` | Set `truncation_strategy: "latest_half"` or `"summary"` in agent context config |

### MCP Tool Issues

| Error | Solution |
|---|---|
| `MCP config file not found` | Verify the `mcp` path in tools config is relative to the config directory |
| `Failed to add MCP server` | Check the MCP server command/URL; ensure the server binary is installed |
| MCP tools not appearing | Verify `mcp_config.json` format; check server logs for connection errors |

### Multi-Agent Issues

| Error | Solution |
|---|---|
| Agents sharing context unexpectedly | Use `copy_agent()` to create independent agent copies for parallel execution |
| Parallel tasks interfering | Enable `split_workspace_for_exp: true` in session parallel config |
| Agent deadlock in multi-agent | Ensure agents are sequential (not waiting on each other); check `max_turns` limits |

### Session Issues

| Error | Solution |
|---|---|
| `working_dir does not exist` | Create the workspace directory, or let EvoMaster auto-create via `set_run_dir()` |
| Docker container not starting | Check Docker daemon is running; verify image exists; check resource limits |
| File not found in workspace | Verify `symlinks` config maps source data correctly; check `working_dir` path |

### Other Issues
Please refer to the core code of EvoMaster.