"""Figures for the README, drawn from the same records as the tables.

Every figure is written twice, for light and dark pages, with colours chosen
for each surface. The colours are the first four slots of a palette validated
for colour-vision deficiency on both surfaces; identity never rests on colour
alone, because each series is also labelled where it ends. The tables in
`docs/results/` hold every plotted value.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .report import group
from .run import read_rows
from .stats import rate

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
    },
}

DEFENCE_NAMES = {
    "none": "No defence",
    "spotlight": "Spotlight (delimiters)",
    "datamark": "Datamark",
    "reminder": "Reminder",
    "sanitise": "Sanitise control tokens",
    "detector": "Detector (1.5B model)",
    "least_privilege": "Least privilege",
    "taint": "Taint",
    "grounding": "Grounding",
    "ablate-comply": "Ablate compliance direction",
    "ablate-random": "Ablate random direction",
    "steer-against-comply": "Steer against compliance",
}


MONITOR_ORDER = ["ungrounded", "llm-7b+reads", "llm-7b", "llm-1.5b", "llm-1.5b+reads", "tainted"]


def _axes(theme: dict[str, Any], width: float, height: float, columns: int = 1, **kwargs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "text.color": theme["ink"],
            "axes.titlecolor": theme["ink"],
            "axes.labelcolor": theme["secondary"],
        }
    )
    figure, axes = plt.subplots(1, columns, figsize=(width, height), **kwargs)
    figure.patch.set_facecolor(theme["surface"])
    for axis in [axes] if columns == 1 else axes:
        axis.set_facecolor(theme["surface"])
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(theme["axis"])
            axis.spines[side].set_linewidth(0.8)
        axis.tick_params(colors=theme["muted"], length=0, labelsize=8.5)
        axis.set_axisbelow(True)
    return figure, axes


def _dot(axis, theme, x: float, y: float, colour: str) -> None:
    """A marker with a ring in the surface colour, so it stays legible over a line."""
    axis.plot(
        [x], [y], "o", ms=7.5, color=colour, mec=theme["surface"], mew=1.5, zorder=3, clip_on=False
    )


def _legend(axis, theme) -> None:
    """A legend in a row between the title and the plot, clear of the data.

    A single series gets none: the title already says what is plotted.
    """
    if len(axis.get_legend_handles_labels()[0]) < 2:
        return
    legend = axis.legend(
        loc="lower left",
        bbox_to_anchor=(-0.01, 1.0),
        ncols=4,
        frameon=False,
        fontsize=8.5,
        handlelength=1.4,
        columnspacing=1.6,
    )
    for text in legend.get_texts():
        text.set_color(theme["secondary"])


def defence_figure(rows: list[dict[str, Any]], theme: dict[str, Any]):
    """Attack success and task success for each defence, with intervals."""
    by_defence = group(rows, "defence")
    entries = []
    for name, cell in by_defence.items():
        if name not in DEFENCE_NAMES:
            continue
        attacks = [r for r in cell if r["suite"] == "attacks"]
        if attacks:
            entries.append(
                (name, rate(attacks, "attack", "task"), rate(attacks, "utility", "task"))
            )
    entries.sort(key=lambda e: (e[0] == "none", e[1].value))
    figure, (left, right) = _axes(theme, 8.4, 0.42 * len(entries) + 1.3, 2, sharey=True)
    colour = theme["series"][0]
    panels = [
        (left, 1, "Attack succeeded", 0.0),
        (right, 2, "Task still solved, under attack", 0.0),
    ]
    for axis, column, title, _ in panels:
        for y, entry in enumerate(entries):
            result = entry[column]
            axis.plot(
                [100 * result.low, 100 * result.high], [y, y], color=colour, lw=1.6, alpha=0.45
            )
            _dot(axis, theme, 100 * result.value, y, colour)
            axis.annotate(
                f"{100 * result.value:.0f}%",
                (100 * result.high, y),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                color=theme["secondary"],
                fontsize=8.5,
            )
        axis.set_title(title)
        axis.set_xlim(0, 100)
        axis.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
        axis.grid(axis="x", color=theme["grid"], lw=0.6)
        axis.spines["left"].set_visible(False)
    left.set_yticks(range(len(entries)), [DEFENCE_NAMES[e[0]] for e in entries])
    left.tick_params(axis="y", colors=theme["ink"], labelsize=9)
    left.set_ylim(-0.6, len(entries) - 0.4)
    figure.tight_layout(w_pad=2.5)
    return figure


def sweep_figure(result: dict[str, Any], theme: dict[str, Any]):
    """Held-out probe accuracy at every layer, for the two questions."""
    figure, axis = _axes(theme, 7.2, 3.4)
    names = {"injected": "Attack or harmless note?", "attempted": "Will the agent act on it?"}
    ends = []
    for index, (label, name) in enumerate(names.items()):
        sweep = result["sweep"][label]
        layers = [row["layer"] for row in sweep]
        colour = theme["series"][index]
        axis.fill_between(
            layers,
            [r["low"] for r in sweep],
            [r["high"] for r in sweep],
            color=colour,
            alpha=0.1,
            lw=0,
        )
        axis.plot(layers, [r["auroc"] for r in sweep], color=colour, lw=1.6, label=name)
        _dot(axis, theme, layers[-1], sweep[-1]["auroc"], colour)
        ends.append((sweep[-1]["auroc"], name))
    last = result["layers"]
    axis.axhline(0.5, color=theme["axis"], lw=0.8)
    axis.annotate("chance", (last, 0.5), xytext=(8, 0), textcoords="offset points", va="center",
                  color=theme["muted"], fontsize=8.5)  # fmt: skip
    # Label each line where it ends, nudged apart if the ends are close.
    ends.sort()
    if len(ends) == 2 and ends[1][0] - ends[0][0] < 0.035:
        middle = (ends[0][0] + ends[1][0]) / 2
        ends = [(middle - 0.018, ends[0][1]), (middle + 0.018, ends[1][1])]
    for y, name in ends:
        axis.annotate(name, (last, y), xytext=(8, 0), textcoords="offset points", va="center",
                      color=theme["secondary"], fontsize=8.5)  # fmt: skip
    axis.set_xlim(1, last)
    axis.set_ylim(0.4, 1.0)
    axis.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axis.set_xlabel("Layer")
    axis.set_title("Probe AUROC on attack wordings held out of training", pad=26)
    axis.grid(axis="y", color=theme["grid"], lw=0.6)
    _legend(axis, theme)
    figure.tight_layout()
    figure.subplots_adjust(right=0.74)
    return figure


def monitor_figure(counts: dict[str, int], rows: list[dict[str, Any]], theme: dict[str, Any]):
    """For each monitor: successful attacks caught against the share of honest events audited."""
    from .monitor import operating_points

    figure, axis = _axes(theme, 6.4, 3.6)
    by_name = group(rows, "monitor")
    # A fixed order, so a monitor keeps its colour whichever others are present.
    order = [name for name in MONITOR_ORDER if name in by_name]
    order += sorted(name for name in by_name if name not in MONITOR_ORDER)
    for index, name in enumerate(order[:4]):
        scored = by_name[name]
        colour = theme["series"][index]
        points = [p for p in operating_points(scored, counts["attacked"]) if p.caught == p.caught]
        points.sort(key=lambda p: (p.audit_rate, p.caught))
        xs = [100 * p.audit_rate for p in points]
        ys = [100 * p.caught for p in points]
        axis.plot(xs, ys, color=colour, lw=1.6, label=name)
        for x, y in zip(xs, ys, strict=True):
            _dot(axis, theme, x, y, colour)
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 100)
    axis.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    axis.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    axis.set_xlabel("Honest events a person would have to audit")
    axis.set_title("Successful attacks with a flagged event", pad=26)
    axis.grid(color=theme["grid"], lw=0.6)
    _legend(axis, theme)
    figure.tight_layout()
    return figure


def write_figures(results: Path, out: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    written = []

    def save(name: str, draw) -> None:
        for mode, theme in THEMES.items():
            figure = draw(theme)
            path = out / f"{name}-{mode}.png"
            figure.savefig(path, dpi=200, facecolor=theme["surface"])
            plt.close(figure)
            written.append(path)

    if (results / "bench").exists():
        for model, rows in group(read_rows(results / "bench"), "model").items():
            slug = Path(next(results.glob(f"bench/*{model.split('/')[-1].lower()}*"))).name
            if any(r["defence"] != "none" and r["suite"] == "attacks" for r in rows):
                save(f"defences-{slug}", lambda theme, rows=rows: defence_figure(rows, theme))
    for path in sorted(results.glob("interp/*/probes.json")):
        result = json.loads(path.read_text())
        save(f"probes-{path.parent.name}", lambda theme, r=result: sweep_figure(r, theme))
    for directory in sorted(results.glob("monitor/*")):
        if directory.is_dir():
            counts = json.loads((directory / "counts.json").read_text())
            rows = read_rows(directory)
            if counts["succeeded"]:
                save(
                    f"monitors-{directory.name}",
                    lambda theme, c=counts, r=rows: monitor_figure(c, r, theme),
                )
    return written
