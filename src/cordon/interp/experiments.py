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
    shuffled_within,
    text_ngrams,
)

SPLITS = ("template", "goal", "task")


def select(examples: Sequence[Example], acts: np.ndarray, label: str):
    """The rows and labels for one question."""
    if label == "injected":
        keep = np.ones(len(examples), dtype=bool)
        y = np.array([e.injected for e in examples], dtype=int)
    elif label == "injected_vs_near_miss":
        # Attacks against only the harmless notes that mention what attacks mention.
        keep = np.array([e.injected or (e.note or "").startswith("near_") for e in examples])
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
    # A control for the compliance question. Some tasks are obeyed far more often
    # than others, and the activations surely say which task this is. Scrambling
    # the labels inside each task leaves that route open and closes every other.
    scrambled = None
    if label == "attempted":
        methods["control: probe, labels shuffled within task"] = (x, logistic())
        scrambled = shuffled_within(y, np.array([e.task for e in chosen]))
    out: dict[str, dict[str, Any]] = {}
    for by in SPLITS:
        group = groups(chosen, by)
        out[by] = {}
        for name, (features, fit) in methods.items():
            train = scrambled if name.startswith("control:") else None
            scores = held_out_scores(features, y, group, fit, train)
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
    """The `attempted` probe used as a monitor at the moment of reading.

    The probe is fitted on attacks only: acted on, or resisted. Scores come
    from probes that never saw the example's template. The threshold is set
    by how many resisted attacks it flags; the questions are what share of the
    attacks the agent acted on it catches before the agent acts, and how
    often it fires on harmless text, which it was never trained on.
    """
    x = acts[:, layer - 1]
    injected = np.array([e.injected for e in examples])
    acted = np.array([e.attempted for e in examples])
    succeeded = np.array([e.attack for e in examples])
    near = np.array([bool(e.note) and e.note.startswith("near_") for e in examples])
    plain = ~injected & ~near
    group = groups(examples, "template")
    scores = np.full(len(examples), np.nan)
    for name in np.unique(group):
        train = injected & (group != name)
        if len(np.unique(acted[train])) < 2:
            continue
        scores[group == name] = logistic()(x[train], acted[train].astype(int))(x[group == name])
    resisted = scores[injected & ~acted]
    out: dict[str, Any] = {
        "layer": layer,
        "acted": int(acted.sum()),
        "succeeded": int(succeeded.sum()),
        "points": [],
    }

    def share(mask: np.ndarray, threshold: float) -> float:
        return float((scores[mask] > threshold).mean()) if mask.any() else float("nan")

    for rate in (0.01, 0.05, 0.10, 0.20):
        threshold = float(np.nanquantile(resisted, 1 - rate))
        out["points"].append(
            {
                "resisted_flagged": rate,
                "acted_flagged": share(acted, threshold),
                "succeeded_flagged": share(succeeded, threshold),
                "plain_notes_flagged": share(plain, threshold),
                "near_misses_flagged": share(near, threshold),
            }
        )
    return out
