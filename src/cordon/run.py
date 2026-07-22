"""Run a suite and keep one small record per case.

Inspect's own logs hold the full transcripts and are what you open to see why
a case went the way it did. They are large, so they stay out of the
repository. What is committed is one JSON line per case under `results/`,
which is enough to rebuild every table in the reports.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog

from . import _registry  # noqa: F401  registers the model providers
from .bench.eval import shop_attacks, shop_clean


def slug(model: str) -> str:
    """A directory name for a model, from the last part of its id."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model.split("/")[-1]).strip("-").lower()


def rows_from_log(log: EvalLog, scorer: str, **extra: Any) -> list[dict[str, Any]]:
    rows = []
    for sample in log.samples or []:
        score = sample.scores[scorer]
        rows.append(
            {
                "id": sample.id,
                **extra,
                **sample.metadata,
                **{key: int(value) for key, value in score.value.items()},
                **(score.metadata or {}),
            }
        )
    return rows


def run_task(
    task: Task,
    model: str,
    scorer: str,
    trusted: str | None = None,
    log_dir: str = "logs",
    display: str = "none",
    **extra: Any,
) -> list[dict[str, Any]]:
    (log,) = inspect_eval(
        task,
        model=model,
        model_roles={"trusted": trusted} if trusted else None,
        log_dir=log_dir,
        display=display,
        # One model on one GPU: more samples in flight only makes logs harder to read.
        max_samples=4,
    )
    if log.status != "success":
        raise RuntimeError(f"run did not finish: {log.status}: {log.error}")
    return rows_from_log(log, scorer, model=model, trusted=trusted, **extra)


def run_bench(
    model: str,
    defence: str = "none",
    suite: str = "attacks",
    cases: int | None = None,
    seed: int = 0,
    trusted: str | None = None,
    log_dir: str = "logs",
    display: str = "none",
) -> list[dict[str, Any]]:
    task = shop_clean(defence) if suite == "clean" else shop_attacks(defence, cases, seed)
    return run_task(
        task, model, "shop_scorer", trusted, log_dir, display, defence=defence, suite=suite
    )


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        for row in sorted(rows, key=lambda r: str(r["id"])):
            out.write(json.dumps(row, sort_keys=True) + "\n")


def read_rows(root: Path, pattern: str = "**/*.jsonl") -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob(pattern)):
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows
