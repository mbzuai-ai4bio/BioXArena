"""Skill Tool - Converts Skills into executable Tools.

This tool allows an Agent to use Skills.
"""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from .base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession
    from evomaster.agent.tools.openclaw_bridge import OpenclawBridge
    from evomaster.skills import Skill, SkillRegistry


class SkillToolParams(BaseToolParams):
    """Use a skill and perform related operations.

    A Skill is an EvoMaster extension capability containing domain knowledge and executable scripts.
    """

    name: ClassVar[str] = "use_skill"

    skill_name: str = Field(description="Skill name")
    action: str = Field(
        description="Action to perform: 'get_info' for full information, 'get_reference' for reference documentation, 'run_script' to run a script"
    )
    reference_name: str | None = Field(
        default=None,
        description="Reference document name (required when action='get_reference')"
    )
    script_name: str | None = Field(
        default=None,
        description="Script name (required when action='run_script')"
    )
    script_args: str | None = Field(
        default=None,
        description="Script arguments, space-separated (optional when action='run_script')"
    )


class SkillTool(BaseTool):
    """Skill tool.

    Allows an Agent to use Skills:
    - Get the full information of a skill (full_info)
    - Get reference documentation for a skill
    - Execute scripts from a skill
    """

    name: ClassVar[str] = "use_skill"
    params_class: ClassVar[type[BaseToolParams]] = SkillToolParams

    def __init__(
        self,
        skill_registry: SkillRegistry,
        bridge: OpenclawBridge | None = None,
        enabled_skills: list[str] | None = None,
    ):
        """Initialize SkillTool.

        Args:
            skill_registry: SkillRegistry instance (always contains all skills, used for execution).
            bridge: OpenclawBridge instance (optional, for executing Openclaw-type skills).
            enabled_skills: None=skills: ["*"] exposes all; []=exposes none; [x,y]=exposes only specified skills.
        """
        super().__init__()
        self.skill_registry = skill_registry
        self._bridge = bridge
        self.enabled_skills = enabled_skills

    def get_description(self) -> str:
        """Dynamically inject skills metadata into the tool description."""
        base_description = super().get_description()

        # Only expose skills configured in config to the agent; execution still uses the full registry
        # enabled_skills=None means skills: ["*"], expose all; [] means expose none; [x,y] means expose only specified skills
        if self.enabled_skills is None:
            skills_context = self.skill_registry.get_meta_info_context()
        elif len(self.enabled_skills) == 0:
            skills_context = ""  # Do not expose any skill
        else:
            registry_for_context = self.skill_registry.create_subset(self.enabled_skills)
            skills_context = registry_for_context.get_meta_info_context()
        if skills_context:
            self.logger.info("skills_context: %s", skills_context)
            return (
                f"{base_description}\n\n"
                f"{skills_context}\n"
                "You can use this tool to:\n"
                "1. Get detailed information about a skill: action='get_info'\n"
                "2. Get reference documentation: action='get_reference'\n"
                "3. Run scripts from skills: action='run_script'"
                "4. Example 1 of script_args: \"script_args\": \"{ \"action\": \"read\", \"doc_token\": \"<your doc token>\" }\""
                "5. Example 2 of script_args: \"script_args\": \"--top_k 1 --threshold 0.7 --output json\""
            )
        return base_description

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        """Execute a skill operation.

        Args:
            session: Environment session.
            args_json: Parameter JSON string.

        Returns:
            (observation, info) tuple.
        """
        try:
            params = self.parse_params(args_json)

            # Get the skill
            skill = self.skill_registry.get_skill(params.skill_name)
            if skill is None:
                return (
                    f"Error: Skill '{params.skill_name}' not found",
                    {"error": "skill_not_found"}
                )

            # Execute different operations based on action
            if params.action == "get_info":
                return self._get_info(skill)
            elif params.action == "get_reference":
                return self._get_reference(skill, params.reference_name)
            elif params.action == "run_script":
                return self._run_script(session, skill, params.script_name, params.script_args)
            else:
                return (
                    f"Error: Unknown action '{params.action}'",
                    {"error": "invalid_action"}
                )

        except Exception as e:
            self.logger.error(f"Skill tool execution failed: {e}", exc_info=True)
            return f"Error: {str(e)}", {"error": str(e)}

    def _get_info(self, skill: Skill) -> tuple[str, dict[str, Any]]:
        """Get the full information of a skill.

        Args:
            skill: Skill instance.

        Returns:
            (observation, info) tuple.
        """
        full_info = skill.get_full_info()
        return (
            f"# Skill: {skill.meta_info.name}\n\n{full_info}",
            {"action": "get_info", "skill_name": skill.meta_info.name}
        )

    def _get_reference(
        self,
        skill: Skill,
        reference_name: str | None
    ) -> tuple[str, dict[str, Any]]:
        """Get reference documentation for a skill.

        Args:
            skill: Skill instance.
            reference_name: Reference document name.

        Returns:
            (observation, info) tuple.
        """
        if not reference_name:
            return (
                "Error: reference_name is required for action='get_reference'",
                {"error": "missing_parameter"}
            )

        try:
            reference_content = skill.get_reference(reference_name)
            return (
                f"# Reference: {reference_name}\n\n{reference_content}",
                {
                    "action": "get_reference",
                    "skill_name": skill.meta_info.name,
                    "reference_name": reference_name
                }
            )
        except FileNotFoundError as e:
            return (
                f"Error: {str(e)}",
                {"error": "reference_not_found"}
            )

    def _run_script(
        self,
        session: BaseSession,
        skill: Skill,
        script_name: str | None,
        script_args: str | None
    ) -> tuple[str, dict[str, Any]]:
        """Run a script from a skill.

        For Openclaw-type skills, executes the tool through the bridge.
        For regular skills, executes the script through the session's bash tool.

        Args:
            session: Environment session.
            skill: Skill instance.
            script_name: Script name.
            script_args: Script arguments.

        Returns:
            (observation, info) tuple.
        """
        # Openclaw-type skill: execute through bridge
        if skill.meta_info.type == "openclaw":
            return self._run_openclaw_tool(skill, script_args)

        if not script_name:
            return (
                "Error: script_name is required for action='run_script'",
                {"error": "missing_parameter"}
            )

        # Get script path
        script_path = skill.get_script_path(script_name)
        if script_path is None:
            available_scripts = ", ".join([s.name for s in skill.available_scripts])
            return (
                f"Error: Script '{script_name}' not found in skill '{skill.meta_info.name}'. "
                f"Available scripts: {available_scripts}",
                {"error": "script_not_found"}
            )
        # Convert to absolute path
        script_path = script_path.resolve()
        # Build command
        if script_path.suffix == '.py':
            cmd = f"python {script_path}"
        elif script_path.suffix == '.sh':
            cmd = f"bash {script_path}"
        elif script_path.suffix == '.js':
            cmd = f"node {script_path}"
        else:
            return (
                f"Error: Unsupported script type: {script_path.suffix}",
                {"error": "unsupported_script_type"}
            )

        # Add arguments
        if script_args:
            cmd += f" {script_args}"

        # Execute the script using the session's bash tool
        try:
            result = session.exec_bash(cmd)
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
            exit_code = result.get("exit_code", 0)

            output = f"Script output:\n{stdout}"
            if stderr:
                output += f"\n\nStderr:\n{stderr}"
            if exit_code != 0:
                output += f"\n\nExit code: {exit_code}"

            return (
                output,
                {
                    "action": "run_script",
                    "skill_name": skill.meta_info.name,
                    "script_name": script_name,
                    "script_args": script_args,
                    "exit_code": exit_code,
                }
            )
        except Exception as e:
            return (
                f"Error executing script: {str(e)}",
                {"error": "script_execution_failed"}
            )

    def _run_openclaw_tool(
        self,
        skill: Skill,
        script_args: str | None,
    ) -> tuple[str, dict[str, Any]]:
        """Execute an Openclaw tool through the OpenclawBridge.

        Args:
            skill: Skill instance (type='openclaw').
            script_args: Tool arguments in JSON format.

        Returns:
            (observation, info) tuple.
        """
        if not self._bridge:
            return (
                "Error: Openclaw bridge not initialized. "
                "Enable openclaw in tool config to use this skill.",
                {"error": "bridge_not_initialized"}
            )

        tool_name = skill.meta_info.tool_name
        if not tool_name:
            return (
                f"Error: Skill '{skill.meta_info.name}' is marked as openclaw "
                "but has no tool_name configured.",
                {"error": "missing_tool_name"}
            )

        # Parse arguments
        try:
            args = json.loads(script_args) if isinstance(script_args, str) and script_args else {}
        except json.JSONDecodeError as e:
            return (
                f"Error: Invalid JSON in script_args: {e}",
                {"error": "invalid_args"}
            )

        # Execute through bridge
        try:
            result = self._bridge.execute_tool(tool_name, args)
            return (
                result,
                {
                    "action": "openclaw_execute",
                    "skill_name": skill.meta_info.name,
                    "tool_name": tool_name,
                }
            )
        except Exception as e:
            return (
                f"Error executing openclaw tool '{tool_name}': {str(e)}",
                {"error": "openclaw_execution_failed"}
            )
