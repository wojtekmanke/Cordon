"""A prompt-injection benchmark for agents that work on the shop."""

from .attacks import GOALS, TEMPLATES, Goal, Template, plant
from .benign import NOTES
from .cases import Case, attack_cases, benign_cases, clean_cases, insider_cases, sample_cases
from .tasks import TASKS, InjectionPoint, UserTask

__all__ = [
    "GOALS",
    "NOTES",
    "TASKS",
    "TEMPLATES",
    "Case",
    "Goal",
    "InjectionPoint",
    "Template",
    "UserTask",
    "attack_cases",
    "benign_cases",
    "clean_cases",
    "insider_cases",
    "plant",
    "sample_cases",
]
