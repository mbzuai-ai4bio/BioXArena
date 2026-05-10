# Minimal Skill Task: Agents Using Skills for Knowledge Retrieval

## Design Pattern

The `minimal_skill_task` playground demonstrates how agents use **Skills** — specifically the RAG (Retrieval-Augmented Generation) skill — to search a vector knowledge base and synthesize findings. It features a four-agent pipeline that analyzes, plans, searches, and summarizes.

### Architecture

```
User Task (with vec_dir, nodes_data paths)
    │
    ▼
┌──────────────────┐
│  Analyze Agent   │  Examines the knowledge base structure
└────────┬─────────┘
         │ analysis result
         ▼
┌──────────────────┐
│   Plan Agent     │  Creates a search plan (queries, parameters)
└────────┬─────────┘
         │ plan (JSON)
         ▼
┌──────────────────┐
│  Search Agent    │  Executes RAG searches using the `rag` skill
│  (uses use_skill │
│   tool)          │
└────────┬─────────┘
         │ search results
         ▼
┌──────────────────┐
│ Summarize Agent  │  Synthesizes findings into a final answer
└──────────────────┘
```

### Application Scenarios

- Knowledge base Q&A with semantic search
- Research tasks requiring evidence retrieval
- Any workflow where agents need to query domain-specific knowledge

## Core Code Logic

### Playground Class

```python
@register_playground("minimal_skill_task")
class MinimalSkillTaskPlayground(BasePlayground):
    def __init__(self, config_dir=None, config_path=None):
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.agents.declare(
            "analyze_agent", "plan_agent",
            "search_agent", "summarize_agent"
        )
```

### Multi-Phase Execution

The `run()` method orchestrates four sequential phases:

1. **Analyze Phase** (`AnalyzeExp`): The analyze agent examines the vector store structure and reports what knowledge is available.

2. **Plan Phase** (×2 rounds): The plan agent receives the analysis and task description, then outputs a structured search plan as JSON:
   ```json
   [
     {"query": "search query text", "top_k": 5, "threshold": 0.7},
     ...
   ]
   ```

3. **Search Phase** (`SearchExp`, ×2 rounds): The search agent executes the plan by calling the `rag` skill's `search.py` script via the `use_skill` tool:
   ```json
   {
     "skill_name": "rag",
     "action": "run_script",
     "script_name": "search.py",
     "script_args": "--vec_dir /path/to/vectorstore --query 'search text' --top_k 5"
   }
   ```

4. **Summarize Phase** (`SummarizeExp`): The summarize agent consolidates all search results into a coherent final answer.

### Skill Tool Usage

The search agent interacts with the RAG skill through the `use_skill` tool:

```python
# Agent calls use_skill with:
{
    "skill_name": "rag",
    "action": "get_info"          # First, read the skill documentation
}

{
    "skill_name": "rag",
    "action": "get_reference",
    "reference_name": "search.md" # Then, read the search reference
}

{
    "skill_name": "rag",
    "action": "run_script",
    "script_name": "search.py",
    "script_args": "--vec_dir ... --query ... --top_k 5"
}
```

### RAG Utility Functions

`core/utils/rag_utils.py` provides helper functions:

- `get_db_from_description(description)`: Extracts `vec_dir` and `nodes_data` paths from the task description
- `resolve_db_to_absolute_paths(vec_dir, nodes_data)`: Converts relative paths to absolute
- `parse_plan_output(plan_text)`: Parses the plan agent's JSON output into search queries
- `extract_agent_response(trajectory)`: Extracts the final agent response from a trajectory

### Configuration

```yaml
agents:
  analyze:
    llm: "openai"
    max_turns: 15
    tools:
      builtin: ["*"]
    skills: ["rag"]                    # Only the RAG skill is exposed
    system_prompt_file: "prompts/analyze_system_prompt.txt"
    user_prompt_file: "prompts/analyze_user_prompt.txt"

  plan:
    llm: "openai"
    max_turns: 3
    tools:
      builtin: []                      # Text-only planning
    system_prompt_file: "prompts/plan_system_prompt.txt"
    user_prompt_file: "prompts/plan_user_prompt.txt"

  search:
    llm: "openai"
    max_turns: 30
    tools:
      builtin: ["*"]
    skills: ["rag"]                    # RAG skill for vector search
    system_prompt_file: "prompts/search_system_prompt.txt"
    user_prompt_file: "prompts/search_user_prompt.txt"

  summarize:
    llm: "openai"
    max_turns: 5
    tools:
      builtin: []                      # Text-only summarization
    system_prompt_file: "prompts/summarize_system_prompt.txt"
    user_prompt_file: "prompts/summarize_user_prompt.txt"

# Embedding configuration for the RAG skill
embedding:
  type: "openai"
  openai:
    model: "text-embedding-3-small"
    dimensions: 512
```

### Vector Store Structure

The example includes a minimal vector store at `playground/minimal_skill_task/minimal_vectorstore/`:

```
minimal_vectorstore/
├── nodes.jsonl          # One JSON per line: {task_name, summary, ...}
├── embeddings.npy       # Pre-computed embedding matrix
└── prepare_code.json    # Task name → {summary, prepare_code}
```

## How to Run

```bash
python run.py --agent minimal_skill_task \
  --config configs/minimal_skill_task/deepseek-v3.2-example.yaml \
  --task "playground/minimal_skill_task/prompts/task_prompt.txt"
```

### Key Design Decisions

1. **Skills as progressive disclosure**: Agents first call `get_info` to understand the skill, then `get_reference` for details, then `run_script` to execute — matching the skill design philosophy.
2. **Plan-then-execute**: The plan agent's structured JSON output makes search queries explicit and reproducible.
3. **Selective skill exposure**: Only agents that need RAG (`analyze`, `search`) have `skills: ["rag"]`; others have no skill access.
4. **Embedding config forwarding**: The search prompt injects embedding model parameters so the RAG script uses the correct model.
