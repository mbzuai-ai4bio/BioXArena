"""Chat Agent Google Search Tool

Performs Google searches via the Serper API, returning raw search result lists (title, link, snippet).
"""

from __future__ import annotations

import http.client
import json
import logging
import os
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession

logger = logging.getLogger(__name__)


class GoogleSearchToolParams(BaseToolParams):
    """Search Google for real-time web results.

    Returns top 10 organic search results with titles, URLs, snippets and dates.
    Use this when you need raw search result links to visit specific pages,
    or when you want to see multiple sources before diving deeper.

    For a synthesized AI-generated answer, use ai_search instead.
    """

    name: ClassVar[str] = "google_search"

    query: list[str] = Field(
        description="Array of search queries. Include multiple complementary queries for broader coverage."
    )


class GoogleSearchTool(BaseTool):
    """Google Search Tool (Serper API)."""

    name: ClassVar[str] = "google_search"
    params_class: ClassVar[type[BaseToolParams]] = GoogleSearchToolParams

    def __init__(self):
        super().__init__()

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        """Execute Google search."""
        api_key = os.environ.get("SERPER_KEY_ID")
        if not api_key:
            return (
                "google_search: SERPER_KEY_ID environment variable is not set. "
                "Please set it to use Google search.",
                {"error": "SERPER_KEY_ID not configured"},
            )

        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"Parameter validation error: {e}", {"error": str(e)}

        assert isinstance(params, GoogleSearchToolParams)
        queries = params.query

        self.logger.info("Google search queries: %s", queries)

        results = []
        for q in queries:
            results.append(self._search_single(q, api_key))

        response = "\n---\n".join(results)
        return response, {"queries": queries}

    def _search_single(self, query: str, api_key: str) -> str:
        """Execute a single-query Google search."""
        conn = http.client.HTTPSConnection("google.serper.dev")

        if self._contains_chinese(query):
            payload = json.dumps({
                "q": query,
                "location": "China",
                "gl": "cn",
                "hl": "zh-cn",
            })
        else:
            payload = json.dumps({
                "q": query,
                "location": "United States",
                "gl": "us",
                "hl": "en",
            })

        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        }

        for attempt in range(3):
            try:
                conn.request("POST", "/search", payload, headers)
                res = conn.getresponse()
                data = res.read()
                results = json.loads(data.decode("utf-8"))
                break
            except Exception as e:
                self.logger.warning("Google search attempt %d failed: %s", attempt + 1, e)
                if attempt == 2:
                    return f"Google search failed for '{query}'. Please try again later."
                continue

        if "organic" not in results:
            return f"No results found for '{query}'. Try with a more general query."

        web_snippets = []
        for idx, page in enumerate(results["organic"], 1):
            date_published = ""
            if "date" in page:
                date_published = f"\nDate published: {page['date']}"

            source = ""
            if "source" in page:
                source = f"\nSource: {page['source']}"

            snippet = ""
            if "snippet" in page:
                snippet = f"\n{page['snippet']}"

            entry = (
                f"{idx}. [{page.get('title', '')}]({page.get('link', '')})"
                f"{date_published}{source}{snippet}"
            )
            web_snippets.append(entry)

        return (
            f"### A Google search for '{query}' found {len(web_snippets)} results:\n\n"
            + "\n\n".join(web_snippets)
        )

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        """Detect whether the text contains Chinese characters."""
        return any("\u4E00" <= char <= "\u9FFF" for char in text)
