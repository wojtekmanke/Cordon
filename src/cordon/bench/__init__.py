"""A prompt-injection benchmark for agents that work on the shop."""

from .attacks import GOALS, TEMPLATES, Goal, Template, plant
from .cases import Case, attack_cases, clean_cases, sample_cases
from .tasks import TASKS, InjectionPoint, UserTask

__all__ = [
    "GOALS",
    "TASKS",
    "TEMPLATES",
    "Case",
    "Goal",
    "InjectionPoint",
    "Template",
    "UserTask",
    "attack_cases",
    "clean_cases",
    "plant",
    "sample_cases",
]
