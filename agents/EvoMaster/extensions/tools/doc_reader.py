"""Feishu document reading tool

Read the text content of Feishu documents / Wiki pages via the Feishu Open API.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession

logger = logging.getLogger(__name__)

# URL regex: supports /wiki/XXX, /docx/XXX, /docs/XXX
_URL_PATTERNS = [
    (re.compile(r"/wiki/([A-Za-z0-9]+)"), "wiki"),
    (re.compile(r"/docx/([A-Za-z0-9]+)"), "docx"),
    (re.compile(r"/docs/([A-Za-z0-9]+)"), "docs"),
]


def _parse_feishu_url(url: str) -> tuple[str, str] | None:
    """Extract token and type from a Feishu URL.

    Returns:
        (token, url_type) or None.
    """
    for pattern, url_type in _URL_PATTERNS:
        m = pattern.search(url)
        if m:
            return m.group(1), url_type
    return None


class FeishuDocReadToolParams(BaseToolParams):
    """Read the content of a Feishu (Lark) document or wiki page.

    Use this tool when the user shares a Feishu URL (containing /wiki/, /docx/, or /docs/ in the path)
    and wants to know the content of that document. The tool fetches the document text via the Feishu Open API.
    """

    name: ClassVar[str] = "feishu_doc_read"

    url: str = Field(
        description="The Feishu document URL, e.g. https://xxx.feishu.cn/wiki/XXX or https://xxx.feishu.cn/docx/XXX"
    )


class FeishuDocReadTool(BaseTool):
    """Feishu document reading tool."""

    name: ClassVar[str] = "feishu_doc_read"
    params_class: ClassVar[type[BaseToolParams]] = FeishuDocReadToolParams

    def __init__(self, app_id: str, app_secret: str, domain: str = "https://open.feishu.cn"):
        """Initialize the Feishu document reader.

        Args:
            app_id: Feishu application App ID.
            app_secret: Feishu application App Secret.
            domain: Feishu API domain.
        """
        super().__init__()
        self.app_id = app_id
        self.app_secret = app_secret
        self.domain = domain
        self._client = None

    def _get_client(self):
        """Lazily create or retrieve the cached Feishu Client."""
        if self._client is None:
            from ..messaging.client import create_feishu_client

            self._client = create_feishu_client(
                app_id=self.app_id,
                app_secret=self.app_secret,
                domain=self.domain,
            )
        return self._client

    def _resolve_wiki_token(self, node_token: str) -> tuple[str, str]:
        """Resolve a wiki node_token to (obj_token, title).

        Raises:
            RuntimeError: On API call failure.
        """
        from lark_oapi.api.wiki.v2 import GetNodeSpaceRequest

        client = self._get_client()
        req = GetNodeSpaceRequest.builder().token(node_token).build()
        resp = client.wiki.v2.space.get_node(req)

        if not resp.success():
            raise RuntimeError(
                f"Failed to resolve wiki node: code={resp.code}, msg={resp.msg}"
            )

        node = resp.data.node
        return node.obj_token, node.title or ""

    def _read_document(self, doc_token: str) -> str:
        """Read the plain text content of a document.

        Raises:
            RuntimeError: On API call failure.
        """
        from lark_oapi.api.docx.v1 import RawContentDocumentRequest

        client = self._get_client()
        req = RawContentDocumentRequest.builder().document_id(doc_token).build()
        resp = client.docx.v1.document.raw_content(req)

        if not resp.success():
            raise RuntimeError(
                f"Failed to read document: code={resp.code}, msg={resp.msg}"
            )

        return resp.data.content or ""

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        """Read the content of a Feishu document."""
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"Parameter validation error: {e}", {"error": str(e)}

        assert isinstance(params, FeishuDocReadToolParams)
        url = params.url.strip()

        parsed = _parse_feishu_url(url)
        if parsed is None:
            return (
                "Could not parse the Feishu URL. Supported formats: /wiki/XXX, /docx/XXX",
                {"error": "unsupported_url", "url": url},
            )

        token, url_type = parsed
        self.logger.info("Feishu doc read: type=%s, token=%s", url_type, token)

        try:
            title = ""

            if url_type == "wiki":
                # Wiki: first resolve node_token -> obj_token
                doc_token, title = self._resolve_wiki_token(token)
                self.logger.info(
                    "Wiki resolved: node=%s -> doc=%s, title=%s",
                    token, doc_token, title,
                )
            elif url_type == "docs" and token.startswith("doccn"):
                # Legacy doc format not supported
                return (
                    "Legacy doc format (doccn) is not supported. "
                    "Please convert the document to the new docx format.",
                    {"error": "legacy_doc", "token": token},
                )
            else:
                # docx or non-legacy docs
                doc_token = token

            content = self._read_document(doc_token)

            if not content.strip():
                return "The document is empty.", {"token": doc_token, "title": title}

            # Assemble result
            header = f"Document: {title}\n\n" if title else ""
            result = f"{header}{content}"

            self.logger.info(
                "Feishu doc read completed, content length: %d", len(content)
            )
            return result, {"token": doc_token, "title": title, "url_type": url_type}

        except Exception as e:
            self.logger.error("Feishu doc read failed: %s", e)
            return f"Failed to read Feishu document: {e}", {"error": str(e), "url": url}