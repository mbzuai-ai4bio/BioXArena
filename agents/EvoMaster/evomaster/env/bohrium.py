"""Bohrium authentication and MCP calculation storage/executor configuration.

Used by the MCP calculation path adaptor. Reads BOHRIUM_* from environment variables (.env)
to generate HTTPS storage and inject executor authentication information.
Aligned with the private_callback in _tmp/MatMaster.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict


def get_bohrium_credentials() -> Dict[str, Any]:
    """Read Bohrium credentials from environment variables (.env or os.environ)."""
    access_key = os.getenv("BOHRIUM_ACCESS_KEY", "").strip()
    try:
        project_id = int(os.getenv("BOHRIUM_PROJECT_ID", "-1"))
    except (TypeError, ValueError):
        project_id = -1
    try:
        user_id = int(os.getenv("BOHRIUM_USER_ID", "-1"))
    except (TypeError, ValueError):
        user_id = -1
    return {
        "access_key": access_key,
        "project_id": project_id,
        "user_id": user_id,
    }


def get_bohrium_storage_config() -> Dict[str, Any]:
    """HTTPS storage for MCP calculation (type https + Bohrium plugin)."""
    cred = get_bohrium_credentials()
    return {
        "type": "https",
        "plugin": {
            "type": "bohrium",
            "access_key": cred["access_key"],
            "project_id": cred["project_id"],
            "app_key": "agent",
        },
    }


def inject_bohrium_executor(executor_template: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-copy an executor template and inject BOHRIUM_* credentials (aligned with MatMaster private_callback)."""
    executor = copy.deepcopy(executor_template)
    cred = get_bohrium_credentials()
    if executor.get("type") == "dispatcher":
        rp = executor.setdefault("machine", {}).setdefault("remote_profile", {})
        rp["access_key"] = cred["access_key"]
        rp["project_id"] = cred["project_id"]
        rp["real_user_id"] = cred["user_id"]
        resources = executor.setdefault("resources", {})
        envs = resources.setdefault("envs", {})
        envs["BOHRIUM_PROJECT_ID"] = cred["project_id"]
    return executor
