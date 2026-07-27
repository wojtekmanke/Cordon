"""The benchmark's cases: every task on its own, and every attack on every task."""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..shop import World, seed_world
from .attacks import GOALS, TEMPLATES, Goal, Template, plant
from .benign import NOTES
from .tasks import TASKS, InjectionPoint, UserTask


@dataclass(frozen=True)
class Case:
    """One run of one task.

    With no goal it is a clean run. With a goal, a template and a point it is
    an injection: the goal is planted in what the task reads. With a goal
    alone it is an insider run: the agent itself has been told to pursue the
    goal, and nothing is planted. With a note and a point, harmless text is
    planted where an attack would go.
    """

    task: UserTask
    point: InjectionPoint | None = None
    goal: Goal | None = None
    template: Template | None = None
    note: str | None = None

    @property
    def attacked(self) -> bool:
        return self.goal is not None

    @property
    def injected(self) -> bool:
        return self.template is not None

    @property
    def id(self) -> str:
        if self.note:
            return f"{self.task.id}/{self.point.id}/benign/{self.note}"
        if not self.attacked:
            return f"{self.task.id}/clean"
        if not self.injected:
            return f"{self.task.id}/insider/{self.goal.id}"
        return f"{self.task.id}/{self.point.id}/{self.goal.id}/{self.template.id}"

    @property
    def payload(self) -> str | None:
        """The text planted in what the task reads, if any."""
        if self.note:
            return NOTES[self.note]
        return self.template.render(self.goal) if self.injected else None

    def world(self) -> World:
        """The shop as this case starts: the seed data, plus the planted text if any."""
        world = seed_world()
        if self.payload:
            plant(world, self.point, self.payload)
        return world

    def metadata(self) -> dict[str, str | None]:
        return {
            "task": self.task.id,
            "point": self.point.id if self.point else None,
            "kind": self.point.kind if self.point else None,
            "goal": self.goal.id if self.goal else None,
            "template": self.template.id if self.template else None,
            "note": self.note,
        }

    @classmethod
    def from_metadata(cls, meta: dict[str, str | None]) -> Case:
        task = TASKS[meta["task"]]
        if meta.get("note"):
            point = next(p for p in task.points if p.id == meta["point"])
            return cls(task, point, note=meta["note"])
        if not meta.get("goal"):
            return cls(task)
        if not meta.get("template"):
            return cls(task, goal=GOALS[meta["goal"]])
        point = next(p for p in task.points if p.id == meta["point"])
        return cls(task, point, GOALS[meta["goal"]], TEMPLATES[meta["template"]])


def clean_cases() -> list[Case]:
    return [Case(task) for task in TASKS.values()]


def attack_cases() -> list[Case]:
    return [
        Case(task, point, goal, template)
        for task in TASKS.values()
        for point in task.points
        for goal in GOALS.values()
        for template in TEMPLATES.values()
    ]


def benign_cases() -> list[Case]:
    return [
        Case(task, point, note=note)
        for task in TASKS.values()
        for point in task.points
        for note in NOTES
    ]


def insider_cases() -> list[Case]:
    return [Case(task, goal=goal) for task in TASKS.values() for goal in GOALS.values()]


def sample_cases(cases: list[Case], n: int, seed: int = 0) -> list[Case]:
    """A fixed random subset, for quick runs. The same seed gives the same subset."""
    if n >= len(cases):
        return list(cases)
    return sorted(random.Random(seed).sample(cases, n), key=lambda c: c.id)
