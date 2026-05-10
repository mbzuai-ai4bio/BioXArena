"""ML Master 2 local environment implementation.

Inherits from evomaster's LocalEnv. Difference: when split_workspace_for_exp is enabled,
symlink creation in the main workspace is not skipped (still created).
"""

from __future__ import annotations

from pathlib import Path

from evomaster.env.local import LocalEnv, LocalEnvConfig


class MLMaster2LocalEnv(LocalEnv):
    """ML Master 2 dedicated local environment.

    Difference from base LocalEnv: in setup(), even if split_workspace_for_exp is enabled,
    symlink creation in the main workspace is not skipped and will always be created.
    """

    def setup(self) -> None:
        """Initialize the local environment.

        Unlike the base class: symlinks are always created in the main workspace
        regardless of whether split_workspace_for_exp is enabled.
        """
        if self._is_ready:
            self.logger.warning("Environment already setup")
            return

        self.logger.info("Setting up ML Master 2 local environment")

        # Ensure working directory exists
        workspace = Path(self.config.session_config.workspace_path)
        workspace.mkdir(parents=True, exist_ok=True)

        # Always create symlinks in main workspace (not skipped due to split_workspace_for_exp)
        session_config = self.config.session_config
        if hasattr(session_config, "symlinks") and session_config.symlinks:
            self._create_symlinks(workspace, session_config.symlinks)
            self.logger.info("Main workspace symlinks created")
        else:
            self.logger.debug("No symlinks configuration, skipping symlink creation")

        self._is_ready = True
        self.logger.info("ML Master 2 local environment setup complete")
