# Minimal Kaggle: Self-Evolving Multi-Agent for Competitions

## Design Pattern

The `minimal_kaggle` playground implements a **self-evolving multi-agent system** designed for iterative machine learning competition workflows. Multiple specialized agents collaborate through a cycle of drafting, researching, debugging, improving, and evaluating solutions to progressively achieve higher scores.

### Architecture

```
                    ┌─────────────────────────────────────────┐
                    │         MinimalKagglePlayground          │
                    │                                         │
                    │  Agents: draft, debug, improve,         │
                    │          research, knowledge_promotion,  │
                    │          metric                          │
                    └─────────────┬───────────────────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │      Initial Draft Phase     │
                    │  DraftExp: draft agent writes │
                    │  baseline → debug → evaluate │
                    └─────────────┬───────────────┘
                                  │
              ┌───────────────────▼───────────────────┐
              │         Research Phase (×N rounds)      │
              │  ResearchExp: research agent proposes   │
              │  improvement ideas from past results    │
              └───────────────────┬───────────────────┘
                                  │
              ┌───────────────────▼───────────────────┐
              │         Improve Phase (×N rounds)       │
              │  ImproveExp: improve agent implements   │
              │  ideas → debug → evaluate → compare     │
              └───────────────────┬───────────────────┘
                                  │
                          ┌───────▼───────┐
                          │  Best Result   │
                          └───────────────┘
```

### Application Scenarios

- Kaggle and machine learning competitions
- Any iterative optimization task requiring multiple attempts
- Workflows where research, implementation, and evaluation need distinct agent roles

## Core Code Logic

### Playground Class

The `MinimalKagglePlayground` declares six agent roles and orchestrates a multi-phase experiment loop:

```python
@register_playground("minimal_kaggle")
class MinimalKagglePlayground(BasePlayground):
    def __init__(self, config_dir=None, config_path=None):
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.agents.declare(
            "draft_agent", "debug_agent", "improve_agent",
            "research_agent", "knowledge_promotion_agent", "metric_agent"
        )
```

### Multi-Phase Execution Flow

The `run()` method implements the self-evolution loop:

1. **Draft Phase** (`DraftExp`):
   - The `draft` agent writes baseline code
   - The `debug` agent fixes bugs
   - The `metric` agent evaluates the submission
   - Loop until a valid submission is produced

2. **Research Phase** (`ResearchExp`, repeated N times):
   - The `research` agent analyzes previous results and proposes improvement ideas
   - Output is structured JSON with specific improvement plans

3. **Improve Phase** (`ImproveExp`, repeated N times):
   - The `improve` agent implements research ideas
   - The `debug` agent fixes any issues
   - The `metric` agent evaluates and compares against the best score
   - The best solution is tracked across iterations

### Agent Roles and Tool Configuration

| Agent | Purpose | Tools |
|---|---|---|
| `draft` | Write initial baseline solution | `builtin: ["*"]` |
| `debug` | Fix code bugs, retry execution | `builtin: ["*"]` |
| `improve` | Implement improvement ideas | `builtin: ["*"]` |
| `research` | Analyze results, propose ideas | `builtin: []` (text-only) |
| `knowledge_promotion` | Extract knowledge from past runs | `builtin: []` (text-only) |
| `metric` | Evaluate submission quality | `builtin: ["*"]` |

Note: The `research` and `knowledge_promotion` agents use `builtin: []` (no tools), meaning they operate as pure text reasoning agents. This is intentional — their role is analysis, not execution.

### Data Preparation

The config uses `symlinks` to make competition data available in the workspace:

```yaml
session:
  local:
    working_dir: "./playground/minimal_kaggle/workspace"
    symlinks:
      "./playground/minimal_kaggle/data/public": "input"
```

A `data_preview` utility generates a summary of the dataset structure (directory tree, CSV headers, sample rows) that is injected into prompts via `{data_preview}`.

### Prompt Template Variables

The prompts use rich template variables to carry state between agents:

- `{task_description}`: The competition description
- `{data_preview}`: Auto-generated dataset summary
- `{terminal_output}`: Output from code execution (for debug agent)
- `{buggy_code}`: Code that failed (for debug agent)
- `{previous_solution}`: Best solution so far (for improve agent)
- `{improve_idea}`: Research-generated improvement plan
- `{memory}`: Accumulated knowledge from past iterations

## How to Run

```bash
# Install ML dependencies
pip install -r playground/minimal_kaggle/requirements.txt

# Run with the example config
python run.py --agent minimal_kaggle \
  --config configs/minimal_kaggle/deepseek-v3.2-example.yaml \
  --task "playground/minimal_kaggle/data/public/description.md"
```

### Key Design Decisions

1. **Text-only research agents**: Separating reasoning from execution prevents the research agent from wasting turns on implementation details.
2. **Shared session**: All agents share the same `LocalSession` and workspace, enabling them to read each other's output files.
3. **Score tracking**: The `run()` method tracks `best_score` across iterations and preserves the best submission.
4. **Agent independence**: Each experiment phase creates fresh agent context to avoid context pollution between phases.
