# Minimal Bohrium: MCP Tool Integration with Scientific Computing

## Design Pattern

The `minimal_bohrium` playground extends the single-agent pattern by integrating **MCP (Model Context Protocol)** tools for external service access. It connects to the Bohrium scientific computing platform, enabling the agent to submit computational jobs, generate molecular structures, and interact with cloud-based scientific tools.

### Architecture

```
User Task
    │
    ▼
┌───────────────────────────────┐
│  Single Agent (ReAct + MCP)   │
│  Tools:                       │
│   - Built-in (bash, editor)   │
│   - MCP: structure-generator  │
│   - Skill: bohrium-oss        │
└───────────────────────────────┘
    │              │
    ▼              ▼
 Local Env    Bohrium Platform
              (via MCP HTTP)
```

### Application Scenarios

- Scientific computing workflows (molecular simulation, DFT calculations)
- Tasks requiring external cloud API integration
- Any scenario where an agent needs tools provided by remote services via MCP

## Core Code Logic

### Playground Class

The playground class is structurally identical to `minimal` but registered under a different name:

```python
@register_playground("minimal_bohrium")
class MinimalBohriumPlayground(BasePlayground):
    def __init__(self, config_dir=None, config_path=None):
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "agent" / "minimal"
        super().__init__(config_dir=config_dir, config_path=config_path)
```

The differentiation comes entirely from the **configuration**: MCP tools and skills are enabled in the config file.

### MCP Integration Flow

1. During `_setup_tools()`, the framework detects `mcp: "mcp_config.json"` in the agent's tool config
2. `_setup_mcp_tools()` loads the JSON config, creates `MCPToolManager`, and initializes connections
3. Each MCP server is connected via its transport (stdio, HTTP, or SSE)
4. Tools are discovered from each server and registered with names like `{server}_{tool}`
5. The agent can call these MCP tools just like built-in tools

### MCP Configuration

```json
{
  "mcpServers": {
    "structure-generator": {
      "transport": "http",
      "url": "<GET_YOUR_URL_AT https://www.bohrium.com/apps/structure-generator>"
    }
  }
}
```

Supported transports:
- **stdio**: Launch a local process (e.g., `"command": "npx", "args": [...]`)
- **http** / **streamable_http**: Connect to an HTTP endpoint
- **sse**: Server-Sent Events transport

### Skills Integration

The config enables the `bohrium-oss` skill for OSS (Object Storage Service) uploads:

```yaml
agents:
  general:
    tools:
      builtin: ["*"]
      mcp: "mcp_config.json"
    skills: ["bohrium-oss"]
```

The system prompt instructs the agent to use the `bohrium-oss` skill when a tool requires network file paths (e.g., uploading local files to OSS before passing URLs to MCP tools).

### Environment Variables

The following must be set in `.env` for Bohrium integration:

```bash
BOHRIUM_ACCESS_KEY=your_access_key
BOHRIUM_PROJECT_ID=your_project_id
```

## How to Run

```bash
# Install Bohrium SDK
pip install bohr-agent-sdk

# Set environment variables
cp .env.example .env
# Edit .env with your Bohrium credentials

# Run
python run.py --agent minimal_bohrium \
  --config configs/minimal_bohrium/deepseek-v3.2-example.yaml \
  --task "Generate a water molecule structure and optimize it"
```

### Key Differences from Minimal

| Aspect | minimal | minimal_bohrium |
|---|---|---|
| MCP Tools | None | structure-generator (via HTTP) |
| Skills | None | bohrium-oss |
| External Services | None | Bohrium platform |
| System Prompt | Chinese, scientific discovery | English, general + MCP awareness |
| Config Complexity | Basic | Adds MCP config + skill config |
