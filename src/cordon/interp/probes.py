"""Linear probes on the residual stream, and the baselines they have to beat.

A probe that scores well on examples like its training set has shown very
little: planted attacks share wording, and a classifier can learn the wording.
So every score here is on held-out groups. Leave out one attack template and
the probe is tested on a wording it never saw; leave out one goal and it is
tested on a demand it never saw; leave out one task and it is tested in a
conversation it never saw.

The baselines read only the planted text. If a bag of character n-grams does
as well as the activations, the probe has found nothing the text did not
already say.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .dataset import Example

# A fitter takes training rows and returns a function that scores test rows.
Fitter = Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], np.ndarray]]


def logistic(c: float = 0.1) -> Fitter:
    """Logistic regression on standardised features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def fit(x: np.ndarray, y: np.ndarray):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
        model.fit(x, y)
        return lambda rows: model.decision_function(rows)

    return fit


def mean_difference() -> Fitter:
    """Project onto the line between the two class means. Nothing to tune."""

    def fit(x: np.ndarray, y: np.ndarray):
        direction = x[y == 1].mean(axis=0) - x[y == 0].mean(axis=0)
        return lambda rows: rows @ direction

    return fit


def text_ngrams(texts: Sequence[str], c: float = 1.0):
    """A baseline that reads the planted text: TF-IDF of character n-grams."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    texts = np.array(texts, dtype=object)

    def fit(train: np.ndarray, y: np.ndarray):
        vectoriser = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
        model = LogisticRegression(C=c, max_iter=2000)
        model.fit(vectoriser.fit_transform(texts[train.astype(int).ravel()]), y)
        return lambda rows: model.decision_function(
            vectoriser.transform(texts[rows.astype(int).ravel()])
        )

    return fit


def one_hot(examples: Sequence[Example], fields: Sequence[str]) -> np.ndarray:
    """A baseline that knows only which task, goal or template a case is."""
    columns = []
    for name in fields:
        values = [str(getattr(e, name)) for e in examples]
        for value in sorted(set(values)):
            columns.append([float(v == value) for v in values])
    return np.array(columns).T


def groups(examples: Sequence[Example], by: str) -> np.ndarray:
    """The group each example is held out with.

    Attacks are grouped by their `by` value. Harmless notes have no template
    and no goal, so each note is assigned to one of the attack groups in
    rotation: holding out a group then also holds out notes never trained on.
    """
    if by == "task":
        return np.array([e.task for e in examples])
    names = sorted({str(getattr(e, by)) for e in examples if e.injected})
    notes = sorted({e.note for e in examples if e.note})
    of_note = {note: names[i % len(names)] for i, note in enumerate(notes)}
    return np.array([str(getattr(e, by)) if e.injected else of_note[e.note] for e in examples])


def held_out_scores(x: np.ndarray, y: np.ndarray, group: np.ndarray, fit: Fitter) -> np.ndarray:
    """Score every example with a model that never saw its group."""
    scores = np.full(len(y), np.nan)
    for name in np.unique(group):
        test = group == name
        if len(np.unique(y[~test])) < 2:
            continue
        scores[test] = fit(x[~test], y[~test])(x[test])
    return scores


@dataclass(frozen=True)
class Auroc:
    value: float
    low: float
    high: float
    n: int
    positives: int

    def __str__(self) -> str:
        return f"{self.value:.2f} ({self.low:.2f}-{self.high:.2f})"


def auroc(y: np.ndarray, scores: np.ndarray, draws: int = 1000, seed: int = 0) -> Auroc:
    """Area under the ROC curve with a bootstrap interval over examples."""
    from sklearn.metrics import roc_auc_score

    keep = ~np.isnan(scores)
    y, scores = y[keep], scores[keep]
    if len(np.unique(y)) < 2:
        return Auroc(float("nan"), float("nan"), float("nan"), len(y), int(y.sum()))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        pick = rng.integers(0, len(y), len(y))
        if len(np.unique(y[pick])) == 2:
            samples.append(roc_auc_score(y[pick], scores[pick]))
    low, high = np.percentile(samples, [2.5, 97.5])
    return Auroc(float(roc_auc_score(y, scores)), float(low), float(high), len(y), int(y.sum()))


def caught_at(y: np.ndarray, scores: np.ndarray, false_alarm_rate: float) -> float:
    """The share of positives scored above the level that flags this share of negatives."""
    keep = ~np.isnan(scores)
    y, scores = y[keep], scores[keep]
    negatives = np.sort(scores[y == 0])
    if not len(negatives) or not (y == 1).any():
        return float("nan")
    threshold = np.quantile(negatives, 1 - false_alarm_rate)
    return float((scores[y == 1] > threshold).mean())


def direction(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The difference between the class means: the simplest candidate for a causal direction."""
    return x[y == 1].mean(axis=0) - x[y == 0].mean(axis=0)


def random_like(vector: np.ndarray, seed: int = 0) -> np.ndarray:
    """A random direction of the same length, as a control for an intervention."""
    noise = np.random.default_rng(seed).standard_normal(vector.shape)
    return (noise / np.linalg.norm(noise) * np.linalg.norm(vector)).astype(vector.dtype)
