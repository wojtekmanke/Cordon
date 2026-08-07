"""The probing experiments, from saved activations to numbers.

Two questions, asked of the moment the agent has read planted text and not yet
acted:

* `injected`: can you tell an attack from a harmless note?
* `attempted`: among attacks, can you tell which ones the agent will act on?
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .dataset import Example
from .probes import (
    auroc,
    caught_at,
    groups,
    held_out_scores,
    logistic,
    mean_difference,
    one_hot,
    text_ngrams,
)

SPLITS = ("template", "goal", "task")


def select(examples: Sequence[Example], acts: np.ndarray, label: str):
    """The rows and labels for one question."""
    if label == "injected":
        keep = np.ones(len(examples), dtype=bool)
        y = np.array([e.injected for e in examples], dtype=int)
    elif label == "attempted":
        keep = np.array([e.injected for e in examples])
        y = np.array([e.attempted for e in examples], dtype=int)
    else:
        raise ValueError(f"unknown label: {label}")
    chosen = [e for e, k in zip(examples, keep, strict=True) if k]
    return chosen, acts[keep], y[keep]


def layer_sweep(examples: Sequence[Example], acts: np.ndarray, label: str, by: str = "template"):
    """Held-out AUROC of a logistic probe at every layer."""
    chosen, x, y = select(examples, acts, label)
    group = groups(chosen, by)
    rows = []
    for layer in range(x.shape[1]):
        score = auroc(y, held_out_scores(x[:, layer], y, group, logistic()), draws=200)
        rows.append(
            {"layer": layer + 1, "auroc": score.value, "low": score.low, "high": score.high}
        )
    return rows


def compare(
    examples: Sequence[Example], acts: np.ndarray, label: str, layer: int
) -> dict[str, dict[str, Any]]:
    """Probes at one layer against text-only baselines, under each way of holding out.

    `layer` counts from 1.
    """
    chosen, x, y = select(examples, acts, label)
    x = x[:, layer - 1]
    index = np.arange(len(chosen), dtype=float).reshape(-1, 1)
    methods = {
        "probe: logistic regression": (x, logistic()),
        "probe: difference of means": (x, mean_difference()),
        "text: character n-grams": (index, text_ngrams([e.planted for e in chosen])),
    }
    if label == "attempted":
        fields = ("task", "goal", "template")
        methods["labels: task, goal and template"] = (one_hot(chosen, fields), logistic(1.0))
    out: dict[str, dict[str, Any]] = {}
    for by in SPLITS:
        group = groups(chosen, by)
        out[by] = {}
        for name, (features, fit) in methods.items():
            scores = held_out_scores(features, y, group, fit)
            result = auroc(y, scores)
            out[by][name] = {
                "auroc": result.value,
                "low": result.low,
                "high": result.high,
                "caught_at_5pct": caught_at(y, scores, 0.05),
                "n": result.n,
                "positives": result.positives,
            }
    return out


def early_warning(examples: Sequence[Example], acts: np.ndarray, layer: int) -> dict[str, Any]:
    """The `injected` probe used as a monitor at the moment of reading.

    Scores come from models that never saw the example's template. The
    threshold is set on harmless notes; the question is what share of the
    attacks that went on to succeed were flagged before the agent acted.
    """
    x = acts[:, layer - 1]
    y = np.array([e.injected for e in examples], dtype=int)
    scores = held_out_scores(x, y, groups(examples, "template"), logistic())
    benign = scores[y == 0]
    succeeded = np.array([e.attack for e in examples])
    out: dict[str, Any] = {"layer": layer, "succeeded": int(succeeded.sum()), "points": []}
    for rate in (0.01, 0.05, 0.10):
        threshold = np.quantile(benign, 1 - rate)
        out["points"].append(
            {
                "false_alarm_rate": rate,
                "attacks_flagged": float((scores[y == 1] > threshold).mean()),
                "successful_attacks_flagged": float((scores[succeeded] > threshold).mean())
                if succeeded.any()
                else float("nan"),
            }
        )
    return out
