"""ML Master 2 local Session implementation.

Uses MLMaster2LocalEnv instead of the default LocalEnv to ensure
symlinks are created in the main workspace even when split_workspace_for_exp is enabled.
"""

from __future__ import annotations

from evomaster.env.local import LocalEnvConfig
from evomaster.agent.session.base import BaseSession
from evomaster.agent.session.local import LocalSession, LocalSessionConfig

from ...env.local import MLMaster2LocalEnv


class MLMaster2LocalSession(LocalSession):
    """ML Master 2 dedicated local Session.

    Uses MLMaster2LocalEnv instead of the default LocalEnv to ensure
    symlinks are created in the main workspace even when split_workspace_for_exp is enabled.
    """

    def __init__(self, config: LocalSessionConfig | None = None):
        BaseSession.__init__(self, config)
        self.config: LocalSessionConfig = config or LocalSessionConfig()
        env_config = LocalEnvConfig(session_config=self.config)
        self._env = MLMaster2LocalEnv(env_config)
