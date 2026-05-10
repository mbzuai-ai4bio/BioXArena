# Skills: Creating and Using Domain Knowledge Packages

## Overview

Skills are EvoMaster's mechanism for packaging domain knowledge, workflows, and executable scripts into reusable, self-documenting units that agents can discover, learn, and use at runtime.

### Skill Architecture: Progressive Disclosure

Skills use a three-level progressive disclosure model to minimize context usage:

| Level | Content | Context Cost | When Loaded |
|---|---|---|---|
| **Level 1: Meta Info** | Name + description (~100 tokens) | Always in context | At agent startup |
| **Level 2: Full Info** | Complete SKILL.md body (500–2000 tokens) | On demand | When agent calls `get_info` |
| **Level 3: Scripts** | Executable code + reference docs | On demand | When agent calls `run_script` or `get_reference` |

## Creating a Skill

### Directory Structure

```
evomaster/skills/my-skill/
├── SKILL.md              # Required: metadata + instructions
├── scripts/              # Optional: executable scripts
│   ├── process.py
│   └── requirements.txt
├── reference/            # Optional: detailed documentation
│   ├── api.md
│   └── examples.md
└── LICENSE.txt           # Optional: license information
```

### SKILL.md Format

```markdown
---
name: my-skill
description: Brief description of what this skill does and when to use it. Include trigger conditions so agents know when to activate it.
license: MIT
---

# My Skill Overview

Explain the skill's purpose and typical use cases.

## Directory Structure

List the files in this skill and their roles.

## When to Use This Skill

Clear criteria for when an agent should use this skill.

## Quick Start

Show the most common usage patterns with command examples.

## Reference Documentation Navigation

Point to reference/ docs for detailed information.
```

Key requirements:
- YAML frontmatter with `name` and `description` is **mandatory**
- `description` should be specific enough for agents to decide relevance (~1-2 sentences)
- The body should guide agents on how to use the skill effectively
- Keep SKILL.md concise; push details into `reference/`

### Scripts

Place executable scripts in `scripts/`. Supported types:
- `.py` — Executed with `python {script_path} {args}`
- `.sh` — Executed with `bash {script_path} {args}`
- `.js` — Executed with `node {script_path} {args}`

Scripts are executed through the agent's session (`session.exec_bash`), so they run in the same environment as other bash commands.

### Reference Documents

Place detailed documentation in `reference/`. These are loaded on demand when the agent calls `use_skill` with `action: "get_reference"`.

Good reference documents include:
- Detailed parameter documentation for each script
- Input/output format specifications
- Usage examples with expected results
- Integration notes and best practices

## Registering Skills

### Automatic Discovery

Skills are automatically discovered from the skills root directory (default: `evomaster/skills/`). Any directory containing a `SKILL.md` file is loaded as a skill.

### Configuration

Control which skills are exposed to each agent:

```yaml
agents:
  my_agent:
    skills: ["*"]           # Expose all discovered skills
    # skills: ["rag", "pdf"]  # Expose only specific skills
    # skills: []             # No skills (default)
    skill_dir: "./evomaster/skills"  # Skills root directory (optional)
```

### Registration Flow

1. `BasePlayground._get_or_create_full_skill_registry()` scans `evomaster/skills/` and `evomaster/skills_ts/`
2. Each `SKILL.md` is parsed: frontmatter → `SkillMetaInfo`, body → `full_info`
3. A `SkillRegistry` is created containing all discovered skills
4. When creating agent tools, `SkillTool` is registered with the full registry
5. The `enabled_skills` parameter controls which skills' metadata is shown to the LLM
6. **All skills are always available for execution** — config only controls LLM visibility

### Skill Visibility vs. Execution

This is an important distinction:
- **Visibility**: Controlled by `agents.{name}.skills` in config. Only visible skills appear in the `use_skill` tool description.
- **Execution**: The full `SkillRegistry` is always used for execution. An agent that knows a skill name can still call it even if it's not in the visibility list (though the LLM won't see it in the tool description).

## Using Skills at Runtime

### The `use_skill` Tool

Agents interact with skills through a single tool called `use_skill`:

#### Action: `get_info`
Load the complete SKILL.md content for a skill.

```json
{
    "skill_name": "rag",
    "action": "get_info"
}
```

Returns the full body of SKILL.md (Level 2 information).

#### Action: `get_reference`
Load a specific reference document.

```json
{
    "skill_name": "rag",
    "action": "get_reference",
    "reference_name": "search.md"
}
```

The system searches for the file in:
1. `{skill_path}/{reference_name}`
2. `{skill_path}/references/{reference_name}`
3. `{skill_path}/reference/{reference_name}`

#### Action: `run_script`
Execute a script bundled with the skill.

```json
{
    "skill_name": "rag",
    "action": "run_script",
    "script_name": "search.py",
    "script_args": "--vec_dir /path/to/store --query 'search text' --top_k 5"
}
```

The script is executed via the session's bash tool. Output (stdout + stderr) is returned to the agent.

### Typical Agent Workflow with Skills

1. Agent sees skill meta_info in the tool description (Level 1)
2. Agent decides the skill is relevant and calls `get_info` (Level 2)
3. Agent reads the SKILL.md to understand capabilities and file structure
4. Agent calls `get_reference` for specific script documentation (Level 3)
5. Agent calls `run_script` with the correct arguments
6. Agent processes the script output and continues reasoning

## Example: The RAG Skill

The built-in `rag` skill demonstrates best practices:

```
evomaster/skills/rag/
├── SKILL.md                    # Overview + navigation
├── scripts/
│   ├── search.py               # Vector search + content retrieval
│   ├── encode.py               # Text embedding generation
│   ├── build_faiss.py          # FAISS index builder
│   ├── database.py             # Vector database interface
│   └── requirements.txt        # Python dependencies
└── reference/
    ├── search.md               # Detailed search.py documentation
    ├── encode.md               # Detailed encode.py documentation
    ├── build_faiss.md          # FAISS index building guide
    └── database.md             # Database interface design notes
```

Key design principles from this example:
- **SKILL.md is concise**: Focuses on "what", "when", and "how to navigate"
- **Details are in reference/**: Each script has its own reference document
- **Project-agnostic**: No business-specific fields; everything is configurable
- **Pluggable backends**: Supports local models and OpenAI embeddings through configuration

## Best Practices

1. **Write clear descriptions**: The `description` in frontmatter is the primary signal for agent skill selection. Make it specific about use cases and trigger conditions.

2. **Keep SKILL.md lean**: Push detailed parameter docs, examples, and edge cases into `reference/` documents.

3. **Script-first logic**: Implement functionality in scripts, not in prose. Documentation describes how to call scripts, not how to reimplement them.

4. **Design for agents**: Write documentation as if explaining to a colleague who will call your scripts. Include exact parameter names, types, and example values.

5. **Include requirements**: If scripts need additional packages, provide a `requirements.txt` in `scripts/`.

6. **Project-agnostic design**: Keep skills reusable by avoiding hardcoded paths, business-specific field names, or project-specific assumptions.
