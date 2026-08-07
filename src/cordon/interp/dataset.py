"""Turn recorded benchmark runs into a dataset for probing.

Each example is one moment in a real run: the agent has just read a record
with planted text in it, an attack or a harmless note, and is about to choose
its next move. The prompt is exactly what the model saw at that moment. The
labels are what was planted and what the agent went on to do in that run.

Reading activations at this moment asks a specific question: when the model
has read an injection but not yet acted, can you tell from the inside that it
has, and whether it will obey?
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from inspect_ai.model import ChatMessage, ChatMessageTool

from ..agent import tool_info
from ..bench.cases import Case
from ..models.chat import hf_messages, hf_tools
from ..shop import TOOLS


@dataclass(frozen=True)
class Example:
    id: str
    task: str
    point: str
    goal: str | None
    template: str | None
    note: str | None
    planted: str  # the text that was planted
    injected: bool  # an attack, as opposed to a harmless note
    attempted: bool  # the agent went on to act on the attack
    attack: bool  # and the attack succeeded
    step: int  # index of the message in which the planted text was read
    prompt: str = ""  # what the model saw when choosing its next move

    def meta(self) -> dict[str, Any]:
        """Everything but the prompt, which is long and can be rebuilt."""
        row = asdict(self)
        del row["prompt"]
        return row


def read_step(messages: list[ChatMessage], planted: str) -> int | None:
    """The index of the first tool result that contains the planted text."""
    needle = json.dumps(planted, ensure_ascii=False)[1:-1]
    for index, message in enumerate(messages):
        if isinstance(message, ChatMessageTool) and needle in message.text:
            return index
    return None


def examples_from_logs(paths: Iterable[str], render) -> tuple[list[Example], int]:
    """Examples from undefended benchmark logs, and how many runs never read the planted text.

    `render` turns chat-template messages and tools into the prompt string,
    and must be the same function the model was run with.
    """
    from inspect_ai.log import read_eval_log

    tools = hf_tools([tool_info(tool) for tool in TOOLS.values()])
    examples: list[Example] = []
    skipped = 0
    for path in paths:
        log = read_eval_log(path)
        if (log.eval.task_args or {}).get("defence", "none") != "none":
            continue
        for sample in log.samples or []:
            case = Case.from_metadata(sample.metadata)
            if not case.payload:
                continue
            step = read_step(sample.messages, case.payload)
            if step is None:
                skipped += 1
                continue
            score = next(iter(sample.scores.values()))
            examples.append(
                Example(
                    id=case.id,
                    task=case.task.id,
                    point=case.point.id,
                    goal=case.goal.id if case.goal else None,
                    template=case.template.id if case.template else None,
                    note=case.note,
                    planted=case.payload,
                    injected=case.injected,
                    attempted=bool(score.metadata["attempted"]),
                    attack=bool(score.value["attack"]),
                    step=step,
                    prompt=render(hf_messages(sample.messages[: step + 1]), tools),
                )
            )
    return sorted(examples, key=lambda e: e.id), skipped


def save(directory: Path, examples: list[Example], activations) -> None:
    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "examples.jsonl").open("w") as out:
        for example in examples:
            out.write(json.dumps(example.meta(), sort_keys=True) + "\n")
    np.save(directory / "activations.npy", activations.astype(np.float16))


def load(directory: Path) -> tuple[list[Example], Any]:
    import numpy as np

    lines = (directory / "examples.jsonl").read_text().splitlines()
    examples = [Example(**json.loads(line)) for line in lines if line.strip()]
    return examples, np.load(directory / "activations.npy").astype(np.float32)
