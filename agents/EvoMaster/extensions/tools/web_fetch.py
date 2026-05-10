"""Chat Agent Web Fetch Tool

Fetches web page content via the Jina Reader API and uses an LLM to extract key information based on user goals.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, ClassVar

import requests
from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession

logger = logging.getLogger(__name__)

# Maximum fetched content length (characters); truncated before sending to LLM
MAX_CONTENT_LENGTH = 80000

EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning**: Locate the **specific sections/data** directly related to the user's goal within the webpage content.
2. **Key Extraction**: Identify and extract the **most relevant information** from the content. Never miss any important information. Output the **full original context** as far as possible — it can be more than three paragraphs.
3. **Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judging the contribution of the information to the goal.

**Output as JSON with "evidence" and "summary" fields.**"""


class WebFetchToolParams(BaseToolParams):
    """Fetch and read webpage content from a URL, then extract key information based on your goal.

    Returns extracted evidence and summary from the page content.
    Use this to read articles, documentation, or any web page content.
    Pair with google_search to first find relevant URLs, then fetch their content.
    """

    name: ClassVar[str] = "web_fetch"

    url: list[str] = Field(
        description="URL(s) of the webpage(s) to fetch. Can be one or multiple URLs."
    )
    goal: str = Field(
        description="The goal or purpose of fetching the page(s). Used to extract the most relevant information."
    )


class WebFetchTool(BaseTool):
    """Web content fetch + LLM summary extraction tool (Jina Reader API + LLM)."""

    name: ClassVar[str] = "web_fetch"
    params_class: ClassVar[type[BaseToolParams]] = WebFetchToolParams

    def __init__(self):
        super().__init__()
        self._llm = None

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        """Fetch web page content and extract key information using LLM."""
        jina_api_key = os.environ.get("JINA_API_KEY")

        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"Parameter validation error: {e}", {"error": str(e)}

        assert isinstance(params, WebFetchToolParams)
        urls = params.url
        goal = params.goal

        self.logger.info("Web fetch URLs: %s, goal: %s", urls, goal)

        results = []
        with ThreadPoolExecutor(max_workers=min(len(urls), 5)) as pool:
            future_to_url = {
                pool.submit(self._fetch_and_extract, u, goal, jina_api_key): u
                for u in urls
            }
            for future in as_completed(future_to_url, timeout=300):
                url = future_to_url[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append(f"[web_fetch] Failed {url}: {e}")

        response = "\n---\n".join(results)
        return response, {"urls": urls, "goal": goal}

    def _fetch_and_extract(self, url: str, goal: str, jina_api_key: str | None) -> str:
        """Fetch and extract information from a single page using LLM."""
        content = self._fetch_single(url, jina_api_key)

        if content.startswith("[web_fetch]"):
            return content

        # Truncate overly long content
        if len(content) > MAX_CONTENT_LENGTH:
            content = content[:MAX_CONTENT_LENGTH]

        # LLM extraction
        extracted = self._extract_with_llm(content, url, goal)
        return extracted

    def _fetch_single(self, url: str, jina_api_key: str | None) -> str:
        """Fetch a single page using Jina Reader API."""
        headers = {}
        if jina_api_key:
            headers["Authorization"] = f"Bearer {jina_api_key}"

        for attempt in range(3):
            try:
                response = requests.get(
                    f"https://r.jina.ai/{url}",
                    headers=headers,
                    timeout=50,
                )
                if response.status_code == 200:
                    content = response.text
                    if not content or not content.strip():
                        return f"[web_fetch] Empty content from {url}"
                    return content
                else:
                    self.logger.warning(
                        "Jina API returned %d for %s", response.status_code, url
                    )
                    if attempt == 2:
                        return f"[web_fetch] Failed to fetch {url}: HTTP {response.status_code}"
            except Exception as e:
                self.logger.warning("Web fetch attempt %d failed for %s: %s", attempt + 1, url, e)
                if attempt < 2:
                    time.sleep(0.5)
                else:
                    return f"[web_fetch] Failed to fetch {url}: {e}"

        return f"[web_fetch] Failed to fetch {url}"

    def _extract_with_llm(self, content: str, url: str, goal: str) -> str:
        """Use LLM to extract goal-relevant information from web page content."""
        if self._llm is None:
            self.logger.warning("No LLM set for web_fetch extraction, returning raw content")
            if len(content) > MAX_CONTENT_LENGTH:
                content = content[:MAX_CONTENT_LENGTH] + "\n\n[web_fetch] Content truncated."
            return content

        from evomaster.utils.types import Dialog, UserMessage

        try:
            # Retry summary up to 3 times, progressively truncating content
            raw = ""
            summary_content = content
            for retry in range(3):
                prompt = EXTRACTOR_PROMPT.format(webpage_content=summary_content, goal=goal)
                dialog = Dialog(
                    messages=[UserMessage(content=prompt)],
                    tools=[],
                )
                try:
                    response = self._llm.query(dialog)
                    raw = response.content or ""
                except Exception as e:
                    self.logger.warning("LLM extraction attempt %d failed: %s", retry + 1, e)
                    raw = ""

                if len(raw) >= 10:
                    break

                # Truncate content and retry
                truncate_length = int(0.7 * len(summary_content)) if retry < 2 else 25000
                self.logger.info(
                    "[web_fetch] Summary for %s attempt %d/3, truncating to %d chars",
                    url, retry + 1, truncate_length,
                )
                summary_content = summary_content[:truncate_length]

            # Parse JSON
            parsed = self._parse_json(raw)

            if parsed:
                useful = f"The useful information in {url} for user goal \"{goal}\" as follows:\n\n"
                useful += f"Evidence in page:\n{parsed.get('evidence', 'N/A')}\n\n"
                useful += f"Summary:\n{parsed.get('summary', 'N/A')}\n\n"
                return useful
            else:
                # JSON parse failed, return raw LLM output
                return f"The useful information in {url} for user goal \"{goal}\" as follows:\n\n{raw}"

        except Exception as e:
            self.logger.error("LLM extraction failed for %s: %s", url, e)
            if len(content) > MAX_CONTENT_LENGTH:
                content = content[:MAX_CONTENT_LENGTH] + "\n\n[web_fetch] Content truncated."
            return content

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """Attempt to parse JSON from LLM output."""
        if not text:
            return None

        text = text.strip()
        # Remove markdown code blocks
        if text.startswith("```"):
            text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON wrapped in {}
            left = text.find("{")
            right = text.rfind("}")
            if left != -1 and right != -1 and left < right:
                try:
                    return json.loads(text[left:right + 1])
                except json.JSONDecodeError:
                    pass
        return None