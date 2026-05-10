# Minimal Skill Task Playground

A skill-based task playground implementing Analyze → Plan → Search → Summarize workflow.

## Overview

Minimal Skill Task Playground demonstrates a four-agent workflow for knowledge-based tasks:

- **Analyze Agent**: Analyzes the task and extracts key information
- **Plan Agent**: Creates a search plan based on analysis
- **Search Agent**: Executes searches using RAG (Retrieval-Augmented Generation)
- **Summarize Agent**: Synthesizes search results into a final answer

## Available Skills

The skill system provides modular capabilities:

| Skill | Type | Description |
|-------|------|-------------|
| `rag` | Operator | RAG system for semantic search and knowledge retrieval |
| `pdf` | Operator | PDF manipulation: extract text/tables, create, merge/split documents |
| `mcp-builder` | Knowledge | Guide for creating MCP servers |
| `skill-creator` | Knowledge | Guide for creating new skills |

Skills are located in `evomaster/skills/` and can be used via the `use_skill` tool.

## Workflow

```
┌─────────────────┐
│   Task Input    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Analyze Agent   │  Extract key information
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Plan Agent    │  Create search plan
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Search Agent   │  Execute RAG searches (use_skill → rag)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Summarize Agent  │  Synthesize final answer
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Final Answer  │
└─────────────────┘
```

## Quick Start

### 1. Prepare Knowledge Base

The playground uses RAG for searching. Prepare your knowledge base:

```
# Knowledge base structure (referenced in task description)
knowledge_base/
├── vec_dir/          # Vector database (FAISS index)
├── nodes_data/       # Node data
└── model/            # Embedding model
```

### 2. Configure

Edit `configs/minimal_skill_task/config.yaml`:

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

### 3. Run

```bash
# Run with task description (include knowledge base info)
python run.py --agent minimal_skill_task --task "Based on the knowledge base at /path/to/kb, answer: What are the main features of the product?"

# Interactive mode
python run.py --agent minimal_skill_task --interactive
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
    "analyze_output": "...",      # Analysis results
    "search_results": [...],      # Search findings
    "summarize_output": "..."     # Final synthesized answer
}
```

## Using Skills

### RAG Search (via use_skill tool)

```python
# Get skill info
{"action": "get_info", "skill_name": "rag"}

# Run search script
{"action": "run_script", "skill_name": "rag", "script_name": "search.py", "script_args": "--query 'your query' --vec_dir /path/to/vec --nodes_data /path/to/nodes --model /path/to/model"}
```

### PDF Processing

```python
# Get skill info
{"action": "get_info", "skill_name": "pdf"}

# Get reference documentation
{"action": "get_reference", "skill_name": "pdf", "reference_name": "forms.md"}
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `agents.analyze.max_turns` | Max analyze turns | `50` |
| `agents.search.max_turns` | Max search turns | `50` |
| `skills.enabled` | Enable skill system | `true` |
| `skills.skills_root` | Skills directory | `"evomaster/skills"` |

## Directory Structure

```
playground/minimal_skill_task/
├── core/
│   ├── __init__.py
│   ├── playground.py      # Main playground
│   ├── exp/
│   │   ├── analyze_exp.py # Analyze experiment
│   │   ├── search_exp.py  # Search experiment
│   │   └── summarize_exp.py # Summarize experiment
│   └── utils/
│       └── rag_utils.py   # RAG utilities
├── prompts/               # Agent prompts
└── workspace/             # Working directory

evomaster/skills/
├── rag/                   # RAG skill (Operator)
│   ├── SKILL.md
│   └── scripts/
│       ├── search.py
│       ├── encode.py
│       └── database.py
├── pdf/                   # PDF skill (Operator)
│   ├── SKILL.md
│   └── references/
├── mcp-builder/           # MCP builder skill (Knowledge)
│   └── SKILL.md
└── skill-creator/         # Skill creator skill (Knowledge)
    └── SKILL.md
```

## Related

- [EvoMaster Main README](../../README.md)
- [Skills Documentation](../../docs/skills.md)
- [Configuration Examples](../../configs/)
