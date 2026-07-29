"""A control evaluation for a fleet of agents with split privileges."""

from .roles import ROLES, SOLO, Role
from .run import Fleet, FleetConfig

__all__ = ["ROLES", "SOLO", "Fleet", "FleetConfig", "Role"]
