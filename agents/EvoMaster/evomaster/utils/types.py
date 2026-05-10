"""EvoMaster Agent type definitions

Defines the core data types used in the Agent system, including messages, dialogs, trajectories, etc.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Message role enumeration"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FunctionCall(BaseModel):
    """Function call definition"""
    name: str = Field(description="Function name")
    arguments: str = Field(description="Function arguments in JSON string format")


class ToolCall(BaseModel):
    """Tool call definition"""
    id: str = Field(description="Unique identifier for the tool call")
    type: Literal["function"] = "function"
    function: FunctionCall = Field(description="Function call details")


class BaseMessage(BaseModel):
    """Base message class"""
    role: MessageRole = Field(description="Message role")
    content: str | list[dict[str, Any]] | None = Field(default=None, description="Message content, can be a string or a list of multimodal content blocks")
    meta: dict[str, Any] = Field(default_factory=dict, description="Metadata")


class SystemMessage(BaseMessage):
    """System message"""
    role: MessageRole = MessageRole.SYSTEM


class UserMessage(BaseMessage):
    """User message

    content supports two formats:
    - str: Plain text message
    - list[dict]: Multimodal content block list, e.g.:
        [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]
    """
    role: MessageRole = MessageRole.USER


class AssistantMessage(BaseMessage):
    """Assistant message"""
    role: MessageRole = MessageRole.ASSISTANT
    tool_calls: list[ToolCall] | None = Field(default=None, description="List of tool calls")


class ToolMessage(BaseMessage):
    """Tool response message"""
    role: MessageRole = MessageRole.TOOL
    tool_call_id: str = Field(description="Corresponding tool call ID")
    name: str = Field(description="Tool name")


# Message union type
Message = SystemMessage | UserMessage | AssistantMessage | ToolMessage


class ToolSpec(BaseModel):
    """Tool specification definition, used for LLM function calling"""
    type: Literal["function"] = "function"
    function: FunctionSpec = Field(description="Function specification")


class FunctionSpec(BaseModel):
    """Function specification definition"""
    name: str = Field(description="Function name")
    description: str = Field(description="Function description")
    parameters: dict[str, Any] = Field(description="Parameter JSON Schema")
    strict: bool | None = Field(default=None, description="Whether to use strict mode")


class Dialog(BaseModel):
    """Dialog definition, containing a message list and available tools"""
    messages: list[Message] = Field(default_factory=list, description="Message list")
    tools: list[ToolSpec] = Field(default_factory=list, description="Available tools list")
    meta: dict[str, Any] = Field(default_factory=dict, description="Metadata")

    def add_message(self, message: Message) -> None:
        """Add a message to the dialog"""
        self.messages.append(message)

    def get_messages_for_api(self) -> list[dict[str, Any]]:
        """Get messages formatted for API calls

        Supports multimodal content: when content is a list, the content block list (containing text and image_url blocks) is passed directly.
        """
        result = []
        for msg in self.messages:
            msg_dict: dict[str, Any] = {"role": msg.role.value}
            content = msg.content

            # Some APIs (e.g., Claude/OpenRouter) require non-empty text content blocks
            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                if content is None or (isinstance(content, str) and not content.strip()):
                    content = " "
            elif isinstance(msg, ToolMessage):
                if content is None or (isinstance(content, str) and not content.strip()):
                    content = " "

            if content is not None:
                msg_dict["content"] = content
            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                msg_dict["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
            if isinstance(msg, ToolMessage):
                msg_dict["tool_call_id"] = msg.tool_call_id
                msg_dict["name"] = msg.name
            result.append(msg_dict)
        return result


class StepRecord(BaseModel):
    """Single step execution record"""
    step_id: int = Field(description="Step number")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp")
    assistant_message: AssistantMessage | None = Field(default=None, description="Assistant message")
    tool_responses: list[ToolMessage] = Field(default_factory=list, description="Tool response list")
    meta: dict[str, Any] = Field(default_factory=dict, description="Metadata")


class Trajectory(BaseModel):
    """Task execution trajectory, recording the complete execution process"""
    task_id: str = Field(description="Task ID")
    dialogs: list[Dialog] = Field(default_factory=list, description="Dialog list")
    steps: list[StepRecord] = Field(default_factory=list, description="Step records")
    start_time: datetime = Field(default_factory=datetime.now, description="Start time")
    end_time: datetime | None = Field(default=None, description="End time")
    status: Literal["running", "completed", "failed", "cancelled", "waiting_for_input"] = Field(
        default="running", description="Execution status"
    )
    result: dict[str, Any] = Field(default_factory=dict, description="Execution result")
    meta: dict[str, Any] = Field(default_factory=dict, description="Metadata")

    def add_step(self, step: StepRecord) -> None:
        """Add an execution step"""
        self.steps.append(step)

    def finish(self, status: Literal["completed", "failed", "cancelled", "waiting_for_input"], result: dict[str, Any] | None = None) -> None:
        """Finish the trajectory recording"""
        self.end_time = datetime.now()
        self.status = status
        if result:
            self.result = result


class TaskInstance(BaseModel):
    """Task instance definition"""
    task_id: str = Field(description="Unique task identifier")
    task_type: str = Field(default="general", description="Task type")
    description: str = Field(default="", description="Task description")
    input_data: dict[str, Any] = Field(default_factory=dict, description="Input data")
    images: list[str] = Field(default_factory=list, description="Image file path list (supports PNG/JPG)")
    meta: dict[str, Any] = Field(default_factory=dict, description="Metadata")

