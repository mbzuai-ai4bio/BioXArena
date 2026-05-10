"""Feishu Document Write Tool

Wraps FeishuDocumentWriter as an Agent-callable tool,
supporting creating documents, appending headings/text/code/dividers, etc.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Optional

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession
    from evomaster.interface.feishu.messaging.document import FeishuDocumentWriter

logger = logging.getLogger(__name__)


class FeishuDocWriteToolParams(BaseToolParams):
    """Create, write, or modify content in a Feishu (Lark) document.

    Actions (recommended — use markdown for batch writing):
    - "create_and_write": Create a new document and write markdown content in one call.
      Preferred over separate "create" + multiple "append_*" calls.
      Requires: title, content (markdown string).
    - "write": Replace entire document content with markdown.
      Requires: content (markdown string).
    - "append": Append markdown content to end of document.
      Requires: content (markdown string).

    Actions (legacy, for fine-grained single-block control):
    - "create": Create a new document with the given title. Returns the document URL.
    - "append_heading": Append a heading (level 1-4) to the end of document.
    - "append_text": Append a text paragraph to the end of document.
    - "append_code": Append a code block to the end of document.
    - "append_divider": Append a divider line to the end of document.
    - "append_image": Append an image to the end of document. Requires file_path to a local image file.
    - "list_blocks": List all blocks in the document with their index, type, and text preview.
                     Use this first to identify which blocks to modify.
    - "update_block": Update the text content of an existing block (by block_id from list_blocks).
                      Works for text, heading, and code blocks.
    - "delete_blocks": Delete a range of blocks by start_index and end_index (from list_blocks).
    - "insert_blocks": Insert a heading/text/code/divider/image block at a specific position (by index).

    Workflow for modifying existing content:
    1. Call "list_blocks" to see all blocks and their indices
    2. To update a single block's text: use "update_block" with the block_id
    3. To rewrite a section: use "delete_blocks" to remove, then "insert_blocks" to add new content

    You must call "create" or "create_and_write" first before using any other actions.
    """

    name: ClassVar[str] = "feishu_doc_write"

    action: Literal[
        "create_and_write", "write", "append",
        "create", "append_heading", "append_text", "append_code", "append_divider",
        "append_image",
        "list_blocks", "update_block", "delete_blocks", "insert_blocks",
    ] = Field(
        description=(
            'The action to perform. '
            'Recommended: "create_and_write" to create doc and write markdown in one step, '
            '"write" to replace doc content with markdown, '
            '"append" to append markdown content. '
            'Legacy: "create", "append_heading", "append_text", "append_code", etc.'
        )
    )
    title: Optional[str] = Field(
        default=None,
        description='Document title (required for "create" action)'
    )
    content: Optional[str] = Field(
        default=None,
        description='Text/markdown content. For "write", "append", "create_and_write": full markdown string. For append_*, update_block, insert_blocks: plain text.'
    )
    level: Optional[int] = Field(
        default=2,
        description='Heading level 1-4 (for "append_heading" and "insert_blocks" with block_type="heading")'
    )
    language: Optional[str] = Field(
        default="plaintext",
        description='Code language: "plaintext", "python", "json" (for append_code and insert_blocks with block_type="code")'
    )
    file_path: Optional[str] = Field(
        default=None,
        description='Local file path for image upload (required for "append_image" and "insert_blocks" with block_type="image")'
    )
    block_id: Optional[str] = Field(
        default=None,
        description='Target block ID (required for "update_block", from list_blocks output)'
    )
    start_index: Optional[int] = Field(
        default=None,
        description='Start index inclusive (required for "delete_blocks", from list_blocks output)'
    )
    end_index: Optional[int] = Field(
        default=None,
        description='End index exclusive (required for "delete_blocks", from list_blocks output)'
    )
    insert_index: Optional[int] = Field(
        default=None,
        description='Insert position index (required for "insert_blocks", from list_blocks output)'
    )
    block_type: Optional[str] = Field(
        default="text",
        description='Block type for "insert_blocks": "heading", "text", "code", "divider", "image"'
    )


class FeishuDocWriteTool(BaseTool):
    """Feishu Document Write Tool

    Dynamically creates Feishu documents and writes structured content during agent execution.
    The FeishuDocumentWriter instance is injected by the dispatcher.
    """

    name: ClassVar[str] = "feishu_doc_write"
    params_class: ClassVar[type[BaseToolParams]] = FeishuDocWriteToolParams

    def __init__(
        self,
        document_writer: FeishuDocumentWriter,
        sender_open_id: str | None = None,
    ):
        super().__init__()
        self._writer = document_writer
        self._sender_open_id = sender_open_id
        # Currently active document (set after create)
        self._current_doc_id: str | None = None
        self._current_doc_url: str | None = None

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        """Execute document write operation."""
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"Parameter validation error: {e}", {"error": str(e)}

        assert isinstance(params, FeishuDocWriteToolParams)
        action = params.action

        try:
            if action == "create":
                return self._do_create(params)
            elif action == "create_and_write":
                return self._do_create_and_write(params)
            elif action == "write":
                return self._do_write(params)
            elif action == "append":
                return self._do_append_markdown(params)
            elif action == "append_heading":
                return self._do_append_heading(params)
            elif action == "append_text":
                return self._do_append_text(params)
            elif action == "append_code":
                return self._do_append_code(params)
            elif action == "append_divider":
                return self._do_append_divider()
            elif action == "append_image":
                return self._do_append_image(params)
            elif action == "list_blocks":
                return self._do_list_blocks()
            elif action == "update_block":
                return self._do_update_block(params)
            elif action == "delete_blocks":
                return self._do_delete_blocks(params)
            elif action == "insert_blocks":
                return self._do_insert_blocks(params)
            else:
                return f"Unknown action: {action}", {"error": "unknown_action"}
        except Exception as e:
            self.logger.error("feishu_doc_write failed: action=%s, error=%s", action, e)
            return f"Failed to execute {action}: {e}", {"error": str(e)}

    def _do_create(self, params: FeishuDocWriteToolParams) -> tuple[str, dict[str, Any]]:
        """Create a new document."""
        title = params.title or params.content
        if not title:
            return "Title is required for 'create' action.", {"error": "missing_title"}

        doc_id = self._writer.create_document(title)
        if not doc_id:
            return "Failed to create document.", {"error": "create_failed"}

        # 设置公开可读
        self._writer.set_public_editable(doc_id)

        # Transfer ownership to the sender
        if self._sender_open_id:
            self._writer.transfer_ownership(doc_id, self._sender_open_id)

        self._current_doc_id = doc_id
        self._current_doc_url = self._writer.get_document_url(doc_id)

        self.logger.info("Created document: %s -> %s", title, self._current_doc_url)
        return (
            f"Document created successfully.\n"
            f"URL: {self._current_doc_url}\n"
            f"Document ID: {doc_id}\n"
            f"You can now use append_* actions to add content.",
            {"document_id": doc_id, "url": self._current_doc_url},
        )

    def _require_doc(self) -> tuple[str, dict[str, Any]] | None:
        """Check whether a document has been created."""
        if not self._current_doc_id:
            return (
                "No active document. Please use 'create' or 'create_and_write' action first.",
                {"error": "no_document"},
            )
        return None

    def _do_write(self, params: FeishuDocWriteToolParams) -> tuple[str, dict[str, Any]]:
        """Replace document content with markdown."""
        err = self._require_doc()
        if err:
            return err
        if not params.content:
            return "content is required for 'write'.", {"error": "missing_content"}
        ok, count = self._writer.write_markdown(self._current_doc_id, params.content)
        if not ok:
            return "Failed to write document.", {"error": "write_failed"}
        return f"Document written successfully ({count} blocks).", {"action": "write", "blocks_added": count}

    def _do_append_markdown(self, params: FeishuDocWriteToolParams) -> tuple[str, dict[str, Any]]:
        """Append markdown content to document."""
        err = self._require_doc()
        if err:
            return err
        if not params.content:
            return "content is required for 'append'.", {"error": "missing_content"}
        ok, count = self._writer.append_markdown(self._current_doc_id, params.content)
        if not ok:
            return "Failed to append content.", {"error": "append_failed"}
        return f"Content appended successfully ({count} blocks).", {"action": "append", "blocks_added": count}

    def _do_create_and_write(self, params: FeishuDocWriteToolParams) -> tuple[str, dict[str, Any]]:
        """Create document and write markdown in one call."""
        title = params.title
        if not title:
            return "title is required for 'create_and_write'.", {"error": "missing_title"}
        if not params.content:
            return "content is required for 'create_and_write'.", {"error": "missing_content"}
        # Reuse existing create logic
        result_text, result_meta = self._do_create(params)
        if "error" in result_meta:
            return result_text, result_meta
        # Then write markdown
        ok, count = self._writer.write_markdown(self._current_doc_id, params.content)
        if not ok:
            return "Document created but failed to write content.", {"error": "write_failed"}
        return (
            f"Document created and written successfully.\n"
            f"URL: {self._current_doc_url}\n"
            f"Blocks: {count}",
            {"document_id": self._current_doc_id, "url": self._current_doc_url, "blocks_added": count},
        )

    def _do_append_heading(self, params: FeishuDocWriteToolParams) -> tuple[str, dict[str, Any]]:
        """Append a heading."""
        err = self._require_doc()
        if err:
            return err

        content = params.content
        if not content:
            return "Content is required for 'append_heading'.", {"error": "missing_content"}

        level = max(1, min(4, params.level or 2))
        ok = self._writer.append_heading(self._current_doc_id, content, level=level)
        if not ok:
            return "Failed to append heading.", {"error": "append_failed"}
        return f"Heading (level {level}) appended.", {"action": "append_heading"}

    def _do_append_text(self, params: FeishuDocWriteToolParams) -> tuple[str, dict[str, Any]]:
        """Append text."""
        err = self._require_doc()
        if err:
            return err

        content = params.content
        if not content:
            return "Content is required for 'append_text'.", {"error": "missing_content"}

        ok = self._writer.append_text(self._current_doc_id, content)
        if not ok:
            return "Failed to append text.", {"error": "append_failed"}
        return "Text appended.", {"action": "append_text"}

    def _do_append_code(self, params: FeishuDocWriteToolParams) -> tuple[str, dict[str, Any]]:
        """Append a code block."""
        err = self._require_doc()
        if err:
            return err

        content = params.content
        if not content:
            return "Content is required for 'append_code'.", {"error": "missing_content"}

        language = params.language or "plaintext"
        ok = self._writer.append_code_block(self._current_doc_id, content, language=language)
        if not ok:
            return "Failed to append code block.", {"error": "append_failed"}
        return f"Code block ({language}) appended.", {"action": "append_code"}

    def _do_append_divider(self) -> tuple[str, dict[str, Any]]:
        """Append a divider."""
        err = self._require_doc()
        if err:
            return err

        ok = self._writer.append_divider(self._current_doc_id)
        if not ok:
            return "Failed to append divider.", {"error": "append_failed"}
        return "Divider appended.", {"action": "append_divider"}

    def _do_append_image(self, params: FeishuDocWriteToolParams) -> tuple[str, dict[str, Any]]:
        """Append an image."""
        err = self._require_doc()
        if err:
            return err

        file_path = params.file_path
        if not file_path:
            return "file_path is required for 'append_image'.", {"error": "missing_file_path"}

        file_token = self._writer.upload_and_append_image(self._current_doc_id, file_path)
        if not file_token:
            return f"Failed to upload or append image: {file_path}", {"error": "image_failed"}
        return (
            f"Image appended successfully (file_token={file_token}).",
            {"action": "append_image", "file_token": file_token},
        )

    # ---- Block editing handlers ----

    def _do_list_blocks(self) -> tuple[str, dict[str, Any]]:
        """List all blocks in the document."""
        err = self._require_doc()
        if err:
            return err

        blocks = self._writer.list_blocks(self._current_doc_id)
        if blocks is None:
            return "Failed to list document blocks.", {"error": "list_failed"}

        # Skip the page block (type=1, always index 0)
        content_blocks = [b for b in blocks if b["block_type"] != 1]

        if not content_blocks:
            return "Document is empty (no content blocks).", {"blocks": []}

        lines = [f"Document has {len(content_blocks)} content blocks:\n"]
        for b in content_blocks:
            preview = b["text_content"][:80]
            if len(b["text_content"]) > 80:
                preview += "..."
            lines.append(
                f'  [{b["index"]}] {b["block_type_name"]} '
                f'(id={b["block_id"]}): "{preview}"'
            )

        return "\n".join(lines), {"blocks": content_blocks}

    def _do_update_block(self, params: FeishuDocWriteToolParams) -> tuple[str, dict[str, Any]]:
        """Update the text content of a specified block."""
        err = self._require_doc()
        if err:
            return err

        if not params.block_id:
            return "block_id is required for 'update_block'.", {"error": "missing_block_id"}
        if not params.content:
            return "content is required for 'update_block'.", {"error": "missing_content"}

        ok = self._writer.update_block_text(
            self._current_doc_id, params.block_id, params.content
        )
        if not ok:
            return f"Failed to update block {params.block_id}.", {"error": "update_failed"}
        return f"Block {params.block_id} updated successfully.", {"action": "update_block"}

    def _do_delete_blocks(self, params: FeishuDocWriteToolParams) -> tuple[str, dict[str, Any]]:
        """Delete a specified range of blocks."""
        err = self._require_doc()
        if err:
            return err

        if params.start_index is None or params.end_index is None:
            return (
                "start_index and end_index are required for 'delete_blocks'.",
                {"error": "missing_index"},
            )

        ok = self._writer.delete_blocks(
            self._current_doc_id, params.start_index, params.end_index
        )
        if not ok:
            return (
                f"Failed to delete blocks [{params.start_index}, {params.end_index}).",
                {"error": "delete_failed"},
            )
        return (
            f"Blocks [{params.start_index}, {params.end_index}) deleted successfully.",
            {"action": "delete_blocks"},
        )

    def _do_insert_blocks(self, params: FeishuDocWriteToolParams) -> tuple[str, dict[str, Any]]:
        """Insert a block at a specified position."""
        err = self._require_doc()
        if err:
            return err

        if params.insert_index is None:
            return "insert_index is required for 'insert_blocks'.", {"error": "missing_index"}

        block_type = params.block_type or "text"

        if block_type == "heading":
            if not params.content:
                return "content is required for heading.", {"error": "missing_content"}
            level = max(1, min(4, params.level or 2))
            ok = self._writer.insert_heading(
                self._current_doc_id, params.content, params.insert_index, level=level
            )
        elif block_type == "text":
            if not params.content:
                return "content is required for text.", {"error": "missing_content"}
            ok = self._writer.insert_text(
                self._current_doc_id, params.content, params.insert_index
            )
        elif block_type == "code":
            if not params.content:
                return "content is required for code.", {"error": "missing_content"}
            language = params.language or "plaintext"
            ok = self._writer.insert_code_block(
                self._current_doc_id, params.content, params.insert_index, language=language
            )
        elif block_type == "divider":
            ok = self._writer.insert_divider(
                self._current_doc_id, params.insert_index
            )
        elif block_type == "image":
            if not params.file_path:
                return "file_path is required for image.", {"error": "missing_file_path"}
            from evomaster.interface.feishu.messaging.document import upload_media_for_doc
            file_token = upload_media_for_doc(
                self._writer._client, params.file_path, self._current_doc_id
            )
            if not file_token:
                return f"Failed to upload image: {params.file_path}", {"error": "upload_failed"}
            ok = self._writer.insert_image(
                self._current_doc_id, file_token, params.insert_index
            )
        else:
            return f"Unknown block_type: {block_type}", {"error": "unknown_block_type"}

        if not ok:
            return (
                f"Failed to insert {block_type} at index {params.insert_index}.",
                {"error": "insert_failed"},
            )
        return (
            f"{block_type.capitalize()} block inserted at index {params.insert_index}.",
            {"action": "insert_blocks"},
        )