"""Intervals for the rates the reports quote.

Two kinds, for two questions. A Wilson interval treats every case as an
independent trial, which is fair for a breakdown within one run. But the
benchmark's cases are not independent: they reuse a handful of tasks, and a
model that mishandles one task mishandles every attack planted in it. For
headline numbers the interval comes from resampling whole tasks instead.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Rate:
    hits: int
    n: int
    low: float
    high: float

    @property
    def value(self) -> float:
        return self.hits / self.n if self.n else float("nan")

    def __str__(self) -> str:
        if not self.n:
            return "n/a"
        return f"{100 * self.value:.0f}% ({100 * self.low:.0f}-{100 * self.high:.0f})"


def wilson(hits: int, n: int, z: float = 1.96) -> Rate:
    """The Wilson score interval for `hits` successes in `n` independent trials."""
    if n == 0:
        return Rate(0, 0, float("nan"), float("nan"))
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return Rate(hits, n, max(0.0, centre - half), min(1.0, centre + half))


def clustered(
    values: Sequence[float],
    clusters: Sequence[str],
    draws: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> Rate:
    """A percentile bootstrap interval that resamples whole clusters.

    `values` are 0 or 1 and `clusters` names the group each belongs to.
    """
    if not values:
        return Rate(0, 0, float("nan"), float("nan"))
    groups: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values, clusters, strict=True):
        groups[cluster].append(value)
    names = sorted(groups)
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        total = count = 0.0
        for name in rng.choices(names, k=len(names)):
            total += sum(groups[name])
            count += len(groups[name])
        means.append(total / count)
    means.sort()
    tail = (1 - level) / 2
    low = means[int(tail * draws)]
    high = means[min(draws - 1, int((1 - tail) * draws))]
    return Rate(int(sum(values)), len(values), low, high)


def rate(rows: Iterable[dict], key: str, cluster: str | None = None) -> Rate:
    """The rate of `key` over `rows`, clustered by the `cluster` column if given."""
    rows = list(rows)
    values = [float(bool(row[key])) for row in rows]
    if cluster is None:
        return wilson(int(sum(values)), len(values))
    return clustered(values, [str(row[cluster]) for row in rows])
