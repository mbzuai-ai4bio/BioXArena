# Skills Module

The Skills module provides the skill system for EvoMaster, enabling reusable capabilities that can be loaded on demand.

## Overview

```
evomaster/skills/
├── base.py           # BaseSkill, Skill, SkillRegistry
└── {skill_name}/
    ├── SKILL.md         # Skill definition (frontmatter + body)
    ├── scripts/         # Optional executable scripts
    │   ├── main.py
    │   └── helper.sh
    └── references/      # Optional reference documents
        └── api.md
```

## Unified Skill Model

In v0.0.2, the previous `KnowledgeSkill` and `OperatorSkill` subclasses have been removed. All skills are now represented by a single unified `Skill` class that supports all three levels of information:

- **Level 1 (meta_info)**: ~100 tokens, always in context
- **Level 2 (full_info)**: 500-2000 tokens, loaded on demand
- **Level 3 (scripts)**: Optional executable code, run via tool call

Whether a skill has scripts depends on whether a `scripts/` directory exists in the skill directory, not on a type field.

## SkillMetaInfo

Metadata parsed from SKILL.md frontmatter.

```python
class SkillMetaInfo(BaseModel):
    """Skill metadata (Level 1)

    Parsed from SKILL.md YAML frontmatter.
    Always in context to help Agent decide whether to use the skill.
    """
    name: str = Field(description="Skill name")
    description: str = Field(description="Skill description with usage scenarios")
    license: str | None = Field(default=None, description="License info")
    type: str | None = Field(default=None, description="Skill type, e.g. 'openclaw' for Openclaw plugin skills")
    tool_name: str | None = Field(default=None, description="Openclaw tool name, e.g. 'feishu_doc'")
```

## BaseSkill

Abstract base class for all skills.

```python
class BaseSkill(ABC):
    """Skill base class

    Skills are EvoMaster components containing:
    - Level 1 (meta_info): Skill metadata (~100 tokens), always in context
    - Level 2 (full_info): Complete info (500-2000 tokens), loaded on demand
    - Level 3 (scripts): Optional executable scripts
    """

    def __init__(self, skill_path: Path):
        """Initialize Skill

        Args:
            skill_path: Skill directory path
        """

    def get_full_info(self) -> str:
        """Get complete info (Level 2)

        If job_submit.md exists, returns its content; otherwise extracts from SKILL.md body.

        Returns:
            Complete skill info text
        """

    def get_reference(self, reference_name: str) -> str:
        """Get reference document content

        Args:
            reference_name: Reference name (e.g., "forms.md", "reference/api.md")

        Returns:
            Reference document content
        """

    @abstractmethod
    def to_context_string(self) -> str:
        """Convert to context string

        Returns string that should be added to Agent context.
        """
```

## Skill

The unified concrete implementation of `BaseSkill`. Replaces the previous `KnowledgeSkill` and `OperatorSkill` classes.

```python
class Skill(BaseSkill):
    """Skill concrete implementation

    Contains optional executable scripts:
    - Level 1: meta_info (always in context)
    - Level 2: full_info (loaded on demand)
    - Level 3: scripts (optional executable scripts)
    """

    def __init__(self, skill_path: Path):
        super().__init__(skill_path)
        self.scripts_dir = self.skill_path / "scripts"
        self.available_scripts = self._scan_scripts()

    def _scan_scripts(self) -> list[Path]:
        """Scan scripts directory for executable scripts

        Returns:
            List of script paths (.py, .sh, .js)
        """

    def get_script_path(self, script_name: str) -> Path | None:
        """Get script path by name

        Args:
            script_name: Script name

        Returns:
            Script path, or None if not exists
        """

    def to_context_string(self) -> str:
        """Returns meta_info with available scripts list"""
        scripts_info = ", ".join([s.name for s in self.available_scripts]) if self.available_scripts else "No scripts"
        return f"[Skill: {self.meta_info.name}] {self.meta_info.description} (Scripts: {scripts_info})"
```

## SkillRegistry

Skill registry for managing all available skills.

```python
class SkillRegistry:
    """Skill registry center

    Manages all available Skills, supporting:
    - Auto-discovery and loading
    - Filtering by name during loading
    - On-demand retrieval
    - Subset creation for per-agent skill views
    - Providing meta_info for Agent selection
    """

    def __init__(self, skills_root: Path, skills: list[str] | None = None):
        """Initialize SkillRegistry

        Args:
            skills_root: Skills root directory
            skills: Optional list of skill directory names to load; None loads all
        """

    def load_from_directory(self, directory: Path, skills: list[str] | None = None) -> None:
        """Load skills from an additional directory

        Can be called multiple times to load skills from multiple directories.

        Args:
            directory: Additional skills directory
            skills: Optional list of skill directory names to load; None loads all
        """

    def get_skill(self, name: str) -> Skill | None:
        """Get skill by name

        Args:
            name: Skill name

        Returns:
            Skill object, or None if not exists
        """

    def get_all_skills(self) -> list[Skill]:
        """Get all skills"""

    def get_meta_info_context(self) -> str:
        """Get all skills' meta_info for Agent context

        Returns:
            String containing all skills' meta_info
        """

    def create_subset(self, skill_names: list[str]) -> SkillRegistry:
        """Create a subset SkillRegistry containing only specified skills

        Used to create independent, filtered skill views for each Agent.

        Args:
            skill_names: List of skill names to keep

        Returns:
            New SkillRegistry instance with only the specified skills
        """

    def search_skills(self, query: str) -> list[Skill]:
        """Search skills by keyword

        Args:
            query: Search keyword

        Returns:
            List of matching skills
        """
```

## SKILL.md Format

### Frontmatter (YAML)

```yaml
---
name: skill-name
description: Brief description with usage scenarios and trigger conditions
license: MIT
---
```

Note: The `skill_type` field has been removed in v0.0.2. Whether a skill has scripts is determined by the presence of a `scripts/` directory.

### Body (Markdown)

The body contains the full_info (Level 2):

```markdown
# Skill Name

## Overview

Detailed description of what this skill does.

## Usage

When to use this skill:
- Scenario 1
- Scenario 2

## Details

Technical details, parameters, examples, etc.

## References

- [Reference 1](./references/ref1.md)
- [Reference 2](./references/ref2.md)
```

## Directory Structure

All skills follow the same directory structure:

```
evomaster/skills/
└── my_skill/
    ├── SKILL.md           # Skill definition
    ├── scripts/           # Optional executable scripts
    │   ├── main.py
    │   └── helper.sh
    └── references/        # Optional reference docs
        └── api.md
```

If the `scripts/` directory exists and contains files, the skill has executable scripts (Level 3). Otherwise, it only provides knowledge (Level 1 and Level 2).

## Usage Examples

### Loading Skills in Playground (v0.0.2)

Skills are now configured per-agent in the config file:

```yaml
# config.yaml
agents:
  search_agent:
    llm: "openai"
    skills:            # Load all skills for this agent
      - "*"
  summarize_agent:
    llm: "openai"
    skills:            # Load specific skills only
      - "rag"
      - "pdf"
  plan_agent:
    llm: "openai"
    # No skills key -> no skills loaded for this agent
```

### Programmatic Usage

```python
from evomaster.skills import SkillRegistry, Skill
from pathlib import Path

# Load all skills from root directory
registry = SkillRegistry(Path("evomaster/skills"))

# Load with name filtering
registry = SkillRegistry(Path("evomaster/skills"), skills=["rag", "pdf"])

# Load skills from additional directories
registry.load_from_directory(Path("extra_skills"))

# Get all skills
all_skills = registry.get_all_skills()

# Get meta_info for Agent context
context = registry.get_meta_info_context()

# Create a subset for a specific agent
subset = registry.create_subset(["rag", "pdf"])

# Search skills
results = registry.search_skills("rag")
```

### Using Skills via SkillTool

Agent can use skills through the `use_skill` tool:

```python
# Get skill info
{"action": "get_info", "skill_name": "rag"}

# Get reference doc
{"action": "get_reference", "skill_name": "rag", "reference_name": "api.md"}

# Run script
{"action": "run_script", "skill_name": "rag", "script_name": "search.py", "script_args": "--query 'search term'"}
```

### Creating a New Skill

1. Create skill directory:
```bash
mkdir -p evomaster/skills/my_skill
```

2. Create SKILL.md:
```markdown
---
name: my-skill
description: A skill that helps with XYZ tasks. Use when you need to do ABC.
license: MIT
---

# My Skill

## Overview

This skill provides knowledge about XYZ...

## When to Use

- When you need to understand ABC
- When working with DEF concepts

## Details

Detailed information here...
```

3. Add scripts (optional):
```bash
mkdir -p evomaster/skills/my_skill/scripts
cat > evomaster/skills/my_skill/scripts/run.py << 'EOF'
# Your executable script
EOF
```

4. Add references (optional):
```bash
mkdir -p evomaster/skills/my_skill/references
echo "# Reference Doc" > evomaster/skills/my_skill/references/guide.md
```

## Related Documentation

- [Architecture Overview](./architecture.md)
- [Tools Module](./tools.md)
- [Agent Module](./agent.md)
