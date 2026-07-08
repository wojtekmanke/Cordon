"""The benchmark's cases: every task on its own, and every attack on every task."""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..shop import World, seed_world
from .attacks import GOALS, TEMPLATES, Goal, Template, plant
from .tasks import TASKS, InjectionPoint, UserTask


@dataclass(frozen=True)
class Case:
    task: UserTask
    point: InjectionPoint | None = None
    goal: Goal | None = None
    template: Template | None = None

    @property
    def attacked(self) -> bool:
        return self.goal is not None

    @property
    def id(self) -> str:
        if not self.attacked:
            return f"{self.task.id}/clean"
        return f"{self.task.id}/{self.point.id}/{self.goal.id}/{self.template.id}"

    @property
    def payload(self) -> str | None:
        return self.template.render(self.goal) if self.attacked else None

    def world(self) -> World:
        """The shop as this case starts: the seed data, plus the planted text if any."""
        world = seed_world()
        if self.attacked:
            plant(world, self.point, self.payload)
        return world

    def metadata(self) -> dict[str, str | None]:
        return {
            "task": self.task.id,
            "point": self.point.id if self.point else None,
            "kind": self.point.kind if self.point else None,
            "goal": self.goal.id if self.goal else None,
            "template": self.template.id if self.template else None,
        }

    @classmethod
    def from_metadata(cls, meta: dict[str, str | None]) -> Case:
        task = TASKS[meta["task"]]
        if not meta.get("goal"):
            return cls(task)
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


def sample_cases(cases: list[Case], n: int, seed: int = 0) -> list[Case]:
    """A fixed random subset, for quick runs. The same seed gives the same subset."""
    if n >= len(cases):
        return list(cases)
    return sorted(random.Random(seed).sample(cases, n), key=lambda c: c.id)
