"""Minimal Playground Implementation

The simplest playground implementation, demonstrating how to use EvoMaster basic features.
"""

import logging
from pathlib import Path

from evomaster.core import BasePlayground, register_playground


@register_playground("minimal")
class MinimalPlayground(BasePlayground):
    """Minimal Playground

    The simplest playground implementation, demonstrating how to use EvoMaster basic features.
    Currently uses the default BasePlayground behavior; custom logic can be added in the future.

    Usage:
        # Via the unified entry point
        python run.py --agent minimal --task "task description"

        # Or via the standalone entry point
        python playground/minimal/main.py
    """

    def __init__(self, config_dir: Path = None, config_path: Path = None):
        """Initialize MinimalPlayground.

        Args:
            config_dir: Configuration directory path, defaults to configs/minimal/
            config_path: Full path to config file (overrides config_dir if provided)
        """
        if config_path is None and config_dir is None:
            # Default configuration directory
            config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "agent" / "minimal"

        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
