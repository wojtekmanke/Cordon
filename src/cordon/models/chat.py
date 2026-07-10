"""Conversion between Inspect's chat types and what open chat models expect.

Local models take the conversation as a list of dicts rendered by the model's
own chat template, and answer in plain text. Tool calls come back as JSON
inside `<tool_call>` tags, the convention Qwen and other Hermes-style models
are trained on.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageTool
from inspect_ai.tool import ToolCall, ToolInfo

_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|\Z)", re.DOTALL)


def hf_tools(tools: list[ToolInfo]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters.model_dump(exclude_none=True),
            },
        }
        for tool in tools
    ]


def hf_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for message in messages:
        entry: dict[str, Any] = {"role": message.role, "content": message.text}
        if isinstance(message, ChatMessageAssistant) and message.tool_calls:
            entry["tool_calls"] = [
                {"type": "function", "function": {"name": c.function, "arguments": c.arguments}}
                for c in message.tool_calls
            ]
        if isinstance(message, ChatMessageTool):
            entry["name"] = message.function
        rendered.append(entry)
    return rendered


def parse_reply(text: str) -> tuple[str, list[ToolCall]]:
    """Split a model's reply into its prose and the tool calls it contains.

    A call that is not valid JSON is kept, with `parse_error` set, so the
    agent loop can tell the model what went wrong.
    """
    calls: list[ToolCall] = []
    for match in _TOOL_CALL.finditer(text):
        body = match.group(1)
        name, arguments, error = "unknown", {}, None
        try:
            parsed = json.loads(body)
            name = parsed["name"]
            arguments = parsed.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise TypeError("name must be a string and arguments an object")
        except (json.JSONDecodeError, KeyError, TypeError) as err:
            name, arguments, error = "unknown", {}, f"could not read the tool call: {err}"
        calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:10]}",
                function=name,
                arguments=arguments,
                parse_error=error,
            )
        )
    prose = _TOOL_CALL.sub("", text).strip()
    return prose, calls
