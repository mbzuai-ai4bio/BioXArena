"""EvoMaster Core - Generic Base Classes

Provides base implementations for Exp and Playground, for quickly building concrete playgrounds.

Usage examples:
    from evomaster.core import BasePlayground, BaseExp

    # Direct usage (simplest)
    playground = BasePlayground()
    result = playground.run("discover patterns")

    # Custom Exp
    class MyExp(BaseExp):
        def run(self, task_description, task_id="exp_001"):
            # Custom logic
            return super().run(task_description, task_id)

    # Custom Playground
    class MyPlayground(BasePlayground):
        def _create_exp(self):
            return MyExp(self.agent, self.config)

    playground = MyPlayground()
    result = playground.run("discover patterns")
"""

from .exp import BaseExp, extract_agent_response
from .playground import BasePlayground
from .registry import (
    register_playground,
    get_playground_class,
    list_registered_playgrounds,
    get_registry_info,
)

__all__ = [
    "BaseExp",
    "BasePlayground",
    "extract_agent_response",
    "register_playground",
    "get_playground_class",
    "list_registered_playgrounds",
    "get_registry_info",
]
