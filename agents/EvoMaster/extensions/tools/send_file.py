"""Feishu file/image sending tool

Agent calls this tool to send files (images, PDFs, CSVs, etc.) from the workspace directly to the Feishu chat.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


class SendFileToolParams(BaseToolParams):
    """Send a file from the workspace to the user via Feishu chat.

    Use this tool when you have generated a file and want to share it directly
    with the user in the conversation. Supports all file types.

    - Images (.png, .jpg, .jpeg, .gif, .bmp, .webp) are sent as image messages with preview.
    - Other files (.pdf, .csv, .xlsx, .py, etc.) are sent as file attachments the user can download.

    The file_path must be an absolute path to an existing file on disk.
    """

    name: ClassVar[str] = "send_file"

    file_path: str = Field(
        description="Absolute path to the file to send (e.g. /path/to/chart.png or /path/to/data.csv)"
    )
    caption: Optional[str] = Field(
        default=None,
        description="Optional text message to send along with the file",
    )


class SendFileTool(BaseTool):
    """Feishu file/image sending tool.

    Sends generated files to the Feishu chat during agent execution.
    Images use image messages (with preview), other files use file messages (downloadable).
    The feishu client and chat_id are injected by the dispatcher.
    """

    name: ClassVar[str] = "send_file"
    params_class: ClassVar[type[BaseToolParams]] = SendFileToolParams

    def __init__(self, client, chat_id: str):
        """Initialize the file sending tool.

        Args:
            client: Feishu Client instance.
            chat_id: Target Feishu chat ID.
        """
        super().__init__()
        self._client = client
        self._chat_id = chat_id

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        """Send a file to the Feishu chat."""
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"Parameter validation error: {e}", {"error": str(e)}

        assert isinstance(params, SendFileToolParams)
        file_path = params.file_path.strip()
        caption = params.caption

        # Validate file exists
        p = Path(file_path)
        if not p.exists():
            return f"File not found: {file_path}", {"error": "file_not_found"}
        if not p.is_file():
            return f"Not a file: {file_path}", {"error": "not_a_file"}

        suffix = p.suffix.lower()

        if suffix in _IMAGE_EXTENSIONS:
            return self._send_as_image(p, file_path, caption)
        else:
            return self._send_as_file(p, file_path, caption)

    def _send_as_image(
        self, p: Path, file_path: str, caption: str | None
    ) -> tuple[str, dict[str, Any]]:
        """Send as an image message (with inline preview)."""
        from ..messaging.sender import upload_image, send_image_message, send_text_message

        self.logger.info("Uploading image to Feishu: %s", file_path)
        image_key = upload_image(self._client, file_path)
        if not image_key:
            return "Failed to upload image to Feishu.", {"error": "upload_failed"}

        message_id = send_image_message(self._client, self._chat_id, image_key)
        if not message_id:
            return (
                f"Image uploaded (key={image_key}) but failed to send message.",
                {"error": "send_failed", "image_key": image_key},
            )

        if caption and caption.strip():
            send_text_message(self._client, self._chat_id, caption.strip())

        self.logger.info(
            "Image sent to chat %s: %s (image_key=%s)",
            self._chat_id, p.name, image_key,
        )
        return (
            f"Image sent successfully to the user.\n"
            f"File: {p.name}\n"
            f"Image key: {image_key}",
            {"image_key": image_key, "message_id": message_id},
        )

    def _send_as_file(
        self, p: Path, file_path: str, caption: str | None
    ) -> tuple[str, dict[str, Any]]:
        """Send as a file message (downloadable attachment) for non-image files."""
        from ..messaging.sender import upload_file, send_file_message, send_text_message

        self.logger.info("Uploading file to Feishu: %s", file_path)
        file_key = upload_file(self._client, file_path)
        if not file_key:
            return "Failed to upload file to Feishu.", {"error": "upload_failed"}

        message_id = send_file_message(self._client, self._chat_id, file_key)
        if not message_id:
            return (
                f"File uploaded (key={file_key}) but failed to send message.",
                {"error": "send_failed", "file_key": file_key},
            )

        if caption and caption.strip():
            send_text_message(self._client, self._chat_id, caption.strip())

        self.logger.info(
            "File sent to chat %s: %s (file_key=%s)",
            self._chat_id, p.name, file_key,
        )
        return (
            f"File sent successfully to the user.\n"
            f"File: {p.name}\n"
            f"File key: {file_key}",
            {"file_key": file_key, "message_id": message_id},
        )