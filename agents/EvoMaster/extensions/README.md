<p align="center">
  【<a href="./README.md">English</a> | <a href="./README-zh.md">简体中文</a>】
</p>

# EvoMaster Extensions (`extensions`)

This directory holds **optional extensions** for [EvoMaster](../README.md): an **EvoMaster Skill** package for skill-capable agents (such as OpenClaw), plus **pre-built custom tools** developers can wire in directly. More custom tools are on the way.

## Layout

| Path | Purpose |
|------|---------|
| [`skills/evomaster/`](skills/evomaster/) | Skill metadata (`SKILL.md`), reference docs, and example tool scripts that explain how to set up and run EvoMaster |
| [`tools/`](tools/) | Pre-built custom tools for direct integration; copy into any playground |

## `skills/evomaster` — EvoMaster Skill package

The **evomaster** skill is a structured guide: prerequisites, environment setup, agent shapes (single-agent / multi-agent sequential and parallel / self-evolving), how to register tools and Skills, and how to choose a mode for your scenario.

- **`SKILL.md`** — Entry point: framework capabilities, configuration highlights, documentation map.
- **`reference/`** — Topic guides, for example minimal examples, Bohrium MCP, Kaggle-style loops, multi-agent patterns, [custom tools](skills/evomaster/reference/custom_tools.md), [configuration](skills/evomaster/reference/configuration.md), and [skills usage](skills/evomaster/reference/skills.md).
- **`scripts/`** — Example `BaseTool` source (`google_search.py`, `web_fetch.py`).

### Using this skill with skill-capable agents (e.g. OpenClaw)

Install or symlink `extensions/skills/evomaster` into that product’s **skills directory** (exact paths are defined in that product’s documentation). After installation, the agent can load the skill by name **`evomaster`** and read `SKILL.md` and `reference/`.

## `tools/` — Pre-built custom tools

These are **EvoMaster custom tools**: Python classes inheriting `BaseTool`, ready to register in YAML so you can add capabilities to your agents more easily.

| File | Role |
|------|------|
| [`google_search.py`](tools/google_search.py) | Web search via the **Serper** API (environment variable `SERPER_KEY_ID`) |
| [`web_fetch.py`](tools/web_fetch.py) | Page fetch via **Jina Reader**, with optional session LLM extraction (set `JINA_API_KEY` and related options as needed; see the script for notes and length limits) |

### Wiring into a playground

1. Copy or symlink the `.py` files into `playground/<your_playground>/tools/`.
2. Register in your config (value is the filename **without** `.py`):

   ```yaml
   agents:
     my_agent:
       tools:
         builtin: ["*"]
         google_search: "google_search"
         web_fetch: "web_fetch"
   ```

3. Set the API keys each script expects in the project root **`.env`** (or your shell).

## See also

- Main project: [README.md](../README.md)
- Skill entry: [skills/evomaster/SKILL.md](skills/evomaster/SKILL.md)
