"""Registers this package's model providers and tasks with Inspect.

Inspect imports this module through the `inspect_ai` entry point, so
`inspect eval cordon/shop_attacks --model mlx/<repo>` works from the shell.
"""

from inspect_ai.model import modelapi

from .bench.eval import shop_attacks, shop_benign, shop_clean, shop_near_miss
from .fleet.eval import fleet_clean, fleet_injection, fleet_insider

__all__ = [
    "fleet_clean",
    "fleet_injection",
    "fleet_insider",
    "shop_attacks",
    "shop_benign",
    "shop_clean",
    "shop_near_miss",
]


@modelapi(name="mlx")
def mlx():
    from .models.local import MLXAPI

    return MLXAPI


@modelapi(name="torch")
def torch():
    from .models.local import TorchAPI

    return TorchAPI


@modelapi(name="scripted")
def scripted():
    from .models.scripted import ScriptedAPI

    return ScriptedAPI
