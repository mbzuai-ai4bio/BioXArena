"""EvoMaster Skills base classes.

Provides the base abstractions and registration mechanism for Skills.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession


class SkillMetaInfo(BaseModel):
    """Skill metadata (Level 1).

    Parsed from the YAML frontmatter of SKILL.md.
    This information is always present in the context, helping the Agent decide whether to use the skill.
    """
    name: str = Field(description="Skill name")
    description: str = Field(description="Skill description, including use cases and trigger conditions")
    license: str | None = Field(default=None, description="License information")
    type: str | None = Field(default=None, description="Skill type, e.g., 'openclaw' for Openclaw plugin skills")
    tool_name: str | None = Field(default=None, description="Openclaw tool name, e.g., 'feishu_doc'")


class BaseSkill(ABC):
    """Skill base class.

    Skills are EvoMaster's skill components, containing:
    - Level 1 (meta_info): Skill metadata (~100 tokens), always in context
    - Level 2 (full_info): Full information (500-2000 tokens), loaded on demand
    - Level 3 (scripts): Executable scripts
    """

    def __init__(self, skill_path: Path):
        """Initialize the Skill.

        Args:
            skill_path: Skill directory path.
        """
        self.skill_path = skill_path
        self.logger = logging.getLogger(self.__class__.__name__)

        # Parse meta_info
        self.meta_info = self._parse_meta_info()

        # full_info cache (lazy loading)
        self._full_info_cache: str | None = None

    def _parse_meta_info(self) -> SkillMetaInfo:
        """Parse the SKILL.md frontmatter to obtain meta_info.

        Returns:
            SkillMetaInfo object.
        """
        skill_md_path = self.skill_path / "SKILL.md"
        if not skill_md_path.exists():
            raise FileNotFoundError(f"SKILL.md not found in {self.skill_path}")

        content = skill_md_path.read_text(encoding="utf-8")

        # Parse YAML frontmatter
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
        if not frontmatter_match:
            raise ValueError(f"Invalid SKILL.md format: no YAML frontmatter found in {skill_md_path}")

        frontmatter_text = frontmatter_match.group(1)

        # YAML parsing: supports key: value and key: | multiline block formats
        frontmatter_data = {}
        current_key = None
        multiline_lines: list[str] = []

        for line in frontmatter_text.split('\n'):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            # Check if this is a continuation line of a multiline block
            if current_key and (line.startswith('  ') or line.startswith('\t')):
                multiline_lines.append(stripped)
                continue

            # If a multiline value was being collected, save it
            if current_key and multiline_lines:
                frontmatter_data[current_key] = ' '.join(multiline_lines)
                current_key = None
                multiline_lines = []

            if ':' in stripped:
                key, value = stripped.split(':', 1)
                key = key.strip()
                value = value.strip()
                if value == '|' or value == '>':
                    # Start a multiline block
                    current_key = key
                    multiline_lines = []
                else:
                    frontmatter_data[key] = value

        # Handle the last multiline block
        if current_key and multiline_lines:
            frontmatter_data[current_key] = ' '.join(multiline_lines)

        # Create SkillMetaInfo
        return SkillMetaInfo(
            name=frontmatter_data.get('name', self.skill_path.name),
            description=frontmatter_data.get('description', ''),
            license=frontmatter_data.get('license'),
            type=frontmatter_data.get('type'),
            tool_name=frontmatter_data.get('tool_name'),
        )

    def get_full_info(self) -> str:
        """Get the full information (Level 2).

        Returns the content of job_submit.md if it exists; otherwise extracts from the body of SKILL.md.
        """
        if self._full_info_cache is not None:
            return self._full_info_cache

        job_submit_path = self.skill_path / "job_submit.md"
        if job_submit_path.exists():
            self._full_info_cache = job_submit_path.read_text(encoding="utf-8").strip()
            return self._full_info_cache

        skill_md_path = self.skill_path / "SKILL.md"
        content = skill_md_path.read_text(encoding="utf-8")
        body_match = re.search(r'^---\s*\n.*?\n---\s*\n(.*)$', content, re.DOTALL)
        if body_match:
            self._full_info_cache = body_match.group(1).strip()
        else:
            self._full_info_cache = content
        return self._full_info_cache

    def get_reference(self, reference_name: str) -> str:
        """Get the content of a reference document.

        Args:
            reference_name: Reference document name (e.g., "forms.md", "reference/api.md").

        Returns:
            Reference document content.
        """
        # Try multiple possible paths
        possible_paths = [
            self.skill_path / reference_name,
            self.skill_path / "references" / reference_name,
            self.skill_path / "reference" / reference_name,
        ]

        for ref_path in possible_paths:
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8")

        raise FileNotFoundError(f"Reference {reference_name} not found in {self.skill_path}")

    @abstractmethod
    def to_context_string(self) -> str:
        """Convert to a context string.

        Returns the string that should be added to the Agent's context.
        """
        pass


class Skill(BaseSkill):
    """Skill concrete implementation.

    A skill with executable scripts:
    - Level 1: meta_info (always in context)
    - Level 2: full_info (loaded on demand)
    - Level 3: scripts (executable scripts)
    """

    def __init__(self, skill_path: Path):
        super().__init__(skill_path)

        # Scan the scripts directory
        self.scripts_dir = self.skill_path / "scripts"
        self.available_scripts = self._scan_scripts()

    def _scan_scripts(self) -> list[Path]:
        """Scan the scripts directory for all executable scripts.

        Returns:
            List of script paths.
        """
        if not self.scripts_dir.exists():
            return []

        scripts = []
        for script_path in self.scripts_dir.iterdir():
            if script_path.is_file() and script_path.suffix in ['.py', '.sh', '.js']:
                scripts.append(script_path)

        return scripts

    def get_script_path(self, script_name: str) -> Path | None:
        """Get a script path.

        Args:
            script_name: Script name.

        Returns:
            Script path, or None if not found.
        """
        for script in self.available_scripts:
            if script.name == script_name:
                return script
        return None

    def to_context_string(self) -> str:
        """Convert to a context string.

        Returns the meta_info description and list of available scripts.
        """
        scripts_info = ", ".join([s.name for s in self.available_scripts]) if self.available_scripts else "No scripts"
        return f"[Skill: {self.meta_info.name}] {self.meta_info.description} (Scripts: {scripts_info})"


class SkillRegistry:
    """Skill registry.

    Manages all available Skills, supporting:
    - Automatic discovery and loading of skills
    - On-demand skill retrieval
    - Providing meta_info for Agent selection
    """

    def __init__(self, skills_root: Path, skills: list[str] | None = None):
        """Initialize the SkillRegistry.

        Args:
            skills_root: Skills root directory.
            skills: List of skill directory names to load; None means load all.
        """
        self.skills_root = skills_root
        self.logger = logging.getLogger(self.__class__.__name__)

        # Store all skills
        self._skills: dict[str, Skill] = {}

        # Auto-load skills
        self._load_skills(skills)

    def _load_skills(self, skills: list[str] | None = None) -> None:
        """Auto-load skills (optionally filtered by directory name)."""
        if not self.skills_root.exists():
            return

        selected_set = set(skills) if skills is not None else None
        for skill_dir in self.skills_root.iterdir():
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                if selected_set is not None and skill_dir.name not in selected_set:
                    continue
                try:
                    skill = Skill(skill_dir)
                    self._skills[skill.meta_info.name] = skill
                    self.logger.info(f"Loaded skill: {skill.meta_info.name}")
                except Exception as e:
                    self.logger.error(f"Failed to load skill from {skill_dir}: {e}")

    def load_from_directory(self, directory: Path, skills: list[str] | None = None) -> None:
        """Load skills from an additional directory.

        Can be called multiple times to load skills from multiple directories.

        Args:
            directory: Additional skills directory.
            skills: List of skill directory names to load; None means load all.
        """
        if not directory.exists():
            self.logger.debug(f"Skills directory not found, skipping: {directory}")
            return

        selected_set = set(skills) if skills is not None else None
        for skill_dir in directory.iterdir():
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                if selected_set is not None and skill_dir.name not in selected_set:
                    continue
                try:
                    skill = Skill(skill_dir)
                    if skill.meta_info.name in self._skills:
                        self.logger.warning(
                            f"Skill '{skill.meta_info.name}' already loaded, "
                            f"overwriting with {skill_dir}"
                        )
                    self._skills[skill.meta_info.name] = skill
                    self.logger.info(f"Loaded skill: {skill.meta_info.name} (from {directory})")
                except Exception as e:
                    self.logger.error(f"Failed to load skill from {skill_dir}: {e}")

    def get_skill(self, name: str) -> Skill | None:
        """Get a skill by name.

        Args:
            name: Skill name.

        Returns:
            Skill object, or None if not found.
        """
        return self._skills.get(name)

    def get_all_skills(self) -> list[Skill]:
        """Get all skills."""
        return list(self._skills.values())

    def get_meta_info_context(self) -> str:
        """Get the meta_info for all skills, for adding to the Agent context.

        Returns:
            String containing the meta_info of all skills.
        """
        lines = ["# Available Skills\n"]

        if self._skills:
            for skill in self._skills.values():
                lines.append(skill.to_context_string())
            lines.append("")

        return "\n".join(lines)

    def create_subset(self, skill_names: list[str]) -> "SkillRegistry":
        """Create a subset SkillRegistry containing only the specified skills.

        Used to create an independent, filtered skill view for each Agent.

        Args:
            skill_names: List of skill names to keep.

        Returns:
            New SkillRegistry instance containing only the specified skills.
        """
        subset = object.__new__(SkillRegistry)
        subset.skills_root = self.skills_root
        subset.logger = logging.getLogger(f"{self.__class__.__name__}[subset]")
        subset._skills = {
            name: skill for name, skill in self._skills.items()
            if name in skill_names
        }

        # Check for skill names not found in the registry
        not_found = set(skill_names) - set(self._skills.keys())
        if not_found:
            subset.logger.warning(f"Skills not found in registry: {not_found}")

        return subset

    def search_skills(self, query: str) -> list[Skill]:
        """Search for skills.

        Args:
            query: Search keyword.

        Returns:
            List of matching skills.
        """
        query_lower = query.lower()
        results = []

        for skill in self.get_all_skills():
            if (query_lower in skill.meta_info.name.lower() or
                query_lower in skill.meta_info.description.lower()):
                results.append(skill)

        return results
