# Minimal Kaggle Playground

A multi-agent playground designed for Kaggle competition automation with iterative improvement workflow.

## Overview

Minimal Kaggle Playground implements a complete Kaggle competition solving workflow with multiple specialized agents:

- **Draft Agent**: Generates initial solution code
- **Debug Agent**: Debugs and fixes code errors
- **Improve Agent**: Iteratively improves the solution
- **Research Agent**: Explores new improvement directions
- **Knowledge Promotion Agent**: Extracts and maintains knowledge
- **Metric Agent**: Evaluates solution performance

## Workflow

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
            │ Best Solution │
            └───────────────┘
```

## Quick Start

### 1. Prepare Data and Environment

Place your Kaggle competition data in a local location. After configuring config.yaml, the corresponding data will be automatically soft-linked to the working directory. The example task already provides sample data at:

```
playground/minimal_kaggle/data/
├── private/
│   └── ...     # Test data
├── public/
│   └── ...    # Training data
```

If you need to run the example machine learning task, you need to install additional required environment based on `playground/minimal_kaggle/requirements.txt` on top of the EvoMaster base environment.

### 2. Configure

Edit `configs/minimal_kaggle/deepseek-v3.2-example.yaml`:

```yaml
  local_sglang:
    provider: "deepseek"
    model: "deepseek-v3.2"
    api_key: "dummy"  # Use placeholder for local deployment
    base_url: "http://192.168.2.110:18889/v1"
    temperature: 0.7
    max_tokens: 16384
    timeout: 300  
    max_retries: 3
    retry_delay: 1.0

  # ... If using OpenAI API format, you need to modify each agent's LLM configuration accordingly, e.g.
  agents:
    draft:
      llm: "local_sglang" # Change to openai if needed
```

### 3. Run

```bash
# Example data is already provided in playground/minimal_kaggle/data. If using your own data, you need to modify the soft link configuration under local in config.yaml
python run.py --agent minimal_kaggle --config configs/minimal_kaggle/deepseek-v3.2-example.yaml --task playground/minimal_kaggle/data/public/description.md
```

### 4. View Results

Results are saved in:

```
runs/minimal_kaggle/workspaces/
├── best_submission/
│   └── submission.csv      # Best submission file
├── best_solution/
│   └── best_solution.py    # Best solution code
├── submission/
│   └── submission_*.csv    # All submissions
└── working/                # Working files
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `agents.draft.max_turns` | Max turns for draft agent | `50` |
| `agents.debug.max_turns` | Max turns for debug agent | `50` |
| `agents.improve.max_turns` | Max turns for improve agent | `50` |
| `session.local.working_dir` | Working directory path | `"./playground/minimal_kaggle/workspace"` |

## Example Usage

### Text Detection (detecting-insults-in-social-commentary)
```bash
python run.py --agent minimal_kaggle --config configs/minimal_kaggle/deepseek-v3.2-example.yaml --task playground/minimal_kaggle/data/public/description.md
```

## Directory Structure

```
playground/minimal_kaggle/
├── core/
│   ├── __init__.py
│   ├── playground.py       # Main playground
│   ├── exp/
│   │   ├── draft_exp.py    # Draft experiment
│   │   ├── improve_exp.py  # Improve experiment
│   │   └── research_exp.py # Research experiment
│   └── utils/
│       ├── code.py         # Code utilities
│       └── data_preview.py # Data preview
├── prompts/                # Agent prompts
└── workspace/              # Working directory
```

## Related Documentation

- [EvoMaster Main README](../../README.md)
- [Configuration Examples](../../configs/)
