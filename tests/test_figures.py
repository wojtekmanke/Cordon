"""The figures are drawn without error from records shaped like the real ones."""

import json

import pytest

pytest.importorskip("matplotlib")

from cordon.figures import write_figures  # noqa: E402
from cordon.run import write_rows  # noqa: E402


def test_every_figure_is_written_for_light_and_dark(tmp_path):
    results = tmp_path / "results"
    bench = []
    for defence, wins in [("none", 6), ("grounding", 1), ("spotlight", 4)]:
        for i in range(12):
            bench.append(
                {
                    "id": f"c{i}",
                    "model": "mlx/org/Tiny-Model",
                    "defence": defence,
                    "suite": "attacks",
                    "task": f"t{i % 4}",
                    "utility": int(i % 3 > 0),
                    "attack": int(i < wins),
                    "attempted": int(i < wins),
                }  # fmt: skip
            )
        write_rows(results / "bench" / "tiny-model" / f"{defence}.attacks.jsonl", bench[-12:])

    sweep = [
        {"layer": n, "auroc": 0.5 + n / 20, "low": 0.45 + n / 20, "high": 0.55 + n / 20}
        for n in range(1, 9)
    ]
    probes = {"layers": 8, "sweep": {"injected": sweep, "attempted": sweep}}
    (results / "interp" / "tiny-model").mkdir(parents=True)
    (results / "interp" / "tiny-model" / "probes.json").write_text(json.dumps(probes))

    events = []
    for i in range(20):
        attacked = i >= 12
        events.append(
            {
                "id": f"e{i}",
                "run": "r",
                "case": f"c{i}",
                "index": 0,
                "monitor": "llm",
                "score": float(i % 10),
                "attacked": attacked,
                "serves_attack": attacked,
                "attack_succeeded": attacked,
            }  # fmt: skip
        )
    write_rows(results / "monitor" / "set" / "llm.jsonl", events)
    counts = {"honest": 12, "attacked": 8, "succeeded": 8}
    (results / "monitor" / "set" / "counts.json").write_text(json.dumps(counts))

    written = write_figures(results, tmp_path / "figures")
    names = sorted(path.name for path in written)
    assert names == [
        "defences-tiny-model-dark.png",
        "defences-tiny-model-light.png",
        "monitors-set-dark.png",
        "monitors-set-light.png",
        "probes-tiny-model-dark.png",
        "probes-tiny-model-light.png",
    ]
    assert all(path.stat().st_size > 5_000 for path in written)
    print(tmp_path / "figures")
