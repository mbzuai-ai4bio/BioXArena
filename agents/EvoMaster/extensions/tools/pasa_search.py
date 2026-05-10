"""MagiClaw PASA Search Tool

Searches academic papers via the PASA search engine sandbox.
"""

from __future__ import annotations

import ast
import json
import logging
import os
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import uuid4

import httpx
from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession

logger = logging.getLogger(__name__)

_DEFAULT_TOOL_URL = "https://scimaster-sandbox.uat.bohrium.com"


class PasaSearchToolParams(BaseToolParams):
    """Search academic papers using the PASA search engine.

    Returns a list of papers with titles, links, and abstracts.
    Use this when you need to find academic papers, research articles,
    or scientific publications on a specific topic.
    """

    name: ClassVar[str] = "pasa_search"

    query: str = Field(description="The search query for academic papers.")
    batch_size: int = Field(
        default=10,
        description="Number of papers to return. Defaults to 10.",
    )


class PasaSearchTool(BaseTool):
    """PASA academic paper search tool."""

    name: ClassVar[str] = "pasa_search"
    params_class: ClassVar[type[BaseToolParams]] = PasaSearchToolParams

    def __init__(self) -> None:
        super().__init__()

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        tool_url = os.environ.get("PASA_TOOL_URL", _DEFAULT_TOOL_URL)

        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"Parameter validation error: {e}", {"error": str(e)}

        assert isinstance(params, PasaSearchToolParams)

        self.logger.info("PASA search: query=%r, batch_size=%d", params.query, params.batch_size)

        papers, errors = self._fetch(params.query, params.batch_size, tool_url)

        if not papers and errors:
            err_msg = errors[0].get("error", "Unknown error")
            return (
                f"PASA search failed for '{params.query}': {err_msg}",
                {"query": params.query, "errors": errors},
            )

        if not papers:
            return (
                f"No papers found for '{params.query}'. Try a different or broader query.",
                {"query": params.query},
            )

        lines = [f"### PASA search for '{params.query}' found {len(papers)} papers:\n"]
        for idx, p in enumerate(papers, 1):
            title = p.get("title", "Untitled")
            link = p.get("link", "")
            abstract = p.get("abstract", "")
            entry = f"{idx}. [{title}]({link})"
            if abstract:
                entry += f"\n   {abstract[:300]}{'...' if len(abstract) > 300 else ''}"
            lines.append(entry)

        response = "\n\n".join(lines)
        return response, {"query": params.query, "count": len(papers)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch(
        self,
        query: str,
        batch_size: int,
        tool_url: str,
        timeout: int = 500,
    ) -> tuple[list[dict], list[dict]]:
        """Submit a PASA search to the sandbox and parse streamed results."""
        code = f"print(pasa_search({query!r}, {batch_size}))"
        session_id = str(uuid4())
        base = tool_url.rstrip("/")
        headers = {"Content-Type": "application/json"}
        payload = {"code": code, "session_id": session_id, "timeout": timeout}

        # --- submit ---
        try:
            with httpx.Client(timeout=60) as client:
                resp = client.post(f"{base}/submit", headers=headers, json=payload)
        except httpx.HTTPError as e:
            return [], [{"error": f"HTTP error on submit: {e}"}]

        if resp.status_code != 200:
            return [], [{"error": f"Submit HTTP {resp.status_code}: {resp.text[:500]}"}]
        if resp.json().get("status") == "fail":
            return [], [{"error": f"Submit failed: {resp.json()}"}]

        # --- stream results ---
        return_value: dict | None = None
        try:
            with httpx.Client(timeout=None) as stream_client:
                with stream_client.stream("GET", f"{base}/get_mcp_result/{session_id}") as stream:
                    for line in stream.iter_lines():
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        if data.get("main_stream_type") == "code_result":
                            return_value = {"output": data.get("content", "")}
                        if not data.get("sub_stream_type") and data.get("stream_state") == "end":
                            break
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            return [], [{"error": f"Stream error: {e}"}]

        if return_value is None:
            return [], [{"error": "No code_result received from stream"}]

        # --- parse output ---
        return self._parse_output(return_value.get("output", ""), query)

    @staticmethod
    def _parse_output(output: str, query: str) -> tuple[list[dict], list[dict]]:
        """Parse the raw sandbox output into paper dicts."""
        processed: list[dict] = []
        errors: list[dict] = []

        try:
            out = ast.literal_eval(output)
            raw_results = out["tool_result"]["result"]
        except (ValueError, SyntaxError) as e:
            return [], [{"error": f"Failed to parse output: {e}", "raw": output[:500]}]
        except KeyError as e:
            return [], [{"error": f"Missing key in output: {e}", "raw": output[:500]}]

        for item in raw_results:
            item = dict(item)
            item["method"] = "PASA"
            item["search_key"] = query
            if not item.get("link"):
                errors.append({**item, "error": "No link"})
                continue
            # Convert arxiv abstract pages to PDF links
            link = item["link"]
            if "arxiv" in link and "/abs/" in link:
                item["link"] = link.replace("/abs/", "/pdf/")
            processed.append(item)

        return processed, errors