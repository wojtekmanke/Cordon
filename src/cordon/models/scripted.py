"""Models that follow a script, for tests and for trying the pipeline without a GPU.

`scripted/oracle` solves each benchmark task with its reference plan and never
acts on planted text. `scripted/gullible` does the same but obeys any attacker
instruction it reads, straight away. Neither says anything about real models:
they exist so that scoring, defences and reports can be checked against
behaviour that is known in advance.
"""

from __future__ import annotations

import json
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
)
from inspect_ai.tool import ToolChoice, ToolInfo

from ..bench.attacks import GOALS
from ..bench.tasks import TASKS, Step


def _key(name: str, arguments: dict[str, Any]) -> str:
    return name + json.dumps(arguments, sort_keys=True)


def next_step(messages: list[ChatMessage], gullible: bool) -> Step | str:
    """The scripted agent's next tool call, or its final answer."""
    prompt = next(m.text for m in messages if isinstance(m, ChatMessageUser))
    task = next((t for t in TASKS.values() if t.prompt in prompt), None)
    if task is None:
        return "I do not know how to do that."
    made = {
        _key(call.function, call.arguments)
        for m in messages
        if isinstance(m, ChatMessageAssistant)
        for call in m.tool_calls or []
    }
    todo: list[Step] = []
    if gullible:
        read = " ".join(m.text for m in messages if isinstance(m, ChatMessageTool)).lower()
        for goal in GOALS.values():
            # Tool results are JSON, so compare against the JSON-escaped instruction.
            if json.dumps(goal.instruction.lower())[1:-1] in read:
                todo += goal.calls
    todo += task.plan
    for name, arguments in todo:
        if _key(name, arguments) not in made:
            return name, arguments
    return task.answer


class ScriptedAPI(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),  # noqa: B008
        **model_args: Any,
    ) -> None:
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, config=config)
        if model_name not in ("oracle", "gullible"):
            raise ValueError(f"unknown scripted model: {model_name}")

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        step = next_step(input, gullible=self.model_name == "gullible")
        if isinstance(step, str):
            return ModelOutput.from_content(self.model_name, step)
        name, arguments = step
        return ModelOutput.for_tool_call(self.model_name, name, arguments)
