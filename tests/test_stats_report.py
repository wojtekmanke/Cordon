import math

from cordon.report import bench_report, fleet_report, interp_report, monitor_report, table
from cordon.run import read_rows, slug, write_rows
from cordon.stats import clustered, rate, wilson


def test_wilson_matches_known_values():
    r = wilson(5, 10)
    assert r.value == 0.5 and math.isclose(r.low, 0.2366, abs_tol=1e-3)
    assert math.isclose(r.high, 0.7634, abs_tol=1e-3)
    zero = wilson(0, 20)
    assert zero.low == 0.0 and 0.1 < zero.high < 0.2
    assert str(wilson(0, 0)) == "n/a"


def test_clustered_interval_is_wider_when_clusters_disagree():
    values = [1] * 50 + [0] * 50
    spread = clustered(values, ["a"] * 50 + ["b"] * 50)
    mixed = clustered(values, [str(i % 10) for i in range(100)])
    assert spread.value == mixed.value == 0.5
    assert spread.high - spread.low > mixed.high - mixed.low
    assert clustered(values, ["a"] * 100) == clustered(values, ["a"] * 100)


def test_rate_counts_truthy_values():
    rows = [{"attack": 1, "task": "a"}, {"attack": 0, "task": "a"}, {"attack": True, "task": "b"}]
    assert rate(rows, "attack").hits == 2 and rate(rows, "attack", "task").n == 3


def test_rows_round_trip_and_slug(tmp_path):
    rows = [{"id": "b", "x": 1}, {"id": "a", "x": 2}]
    write_rows(tmp_path / "m" / "none.clean.jsonl", rows)
    assert [r["id"] for r in read_rows(tmp_path)] == ["a", "b"]
    assert slug("mlx/mlx-community/Qwen2.5-7B-Instruct-4bit") == "qwen2.5-7b-instruct-4bit"


def test_bench_report_compares_a_defence_with_the_baseline_on_the_same_cases():
    def row(i, defence, attack, suite="attacks"):
        return {
            "id": f"c{i}", "model": "m", "defence": defence, "suite": suite, "task": f"t{i % 3}",
            "template": "direct", "goal": "refund", "kind": "ticket",
            "utility": 1, "attack": attack, "attempted": attack,
        }  # fmt: skip

    rows = [row(i, "none", 1) for i in range(12)]
    rows += [row(i, "grounding", 0) for i in range(6)]
    rows += [row(i, "none", 0, "clean") for i in range(3)]
    text = bench_report(rows)
    lines = {
        line.split("|")[1].strip(): line for line in text.splitlines() if line.startswith("| `")
    }
    assert "100% (100-100)" in lines["`none`"] and "| 12 |" in lines["`none`"]
    assert "0% (0-0)" in lines["`grounding`"] and "100% (100-100)" in lines["`grounding`"]
    assert "3/3" in lines["`none`"]


def test_table_is_markdown():
    assert table(["a", "b"], [[1, 2]]) == "| a | b |\n|---|---|\n| 1 | 2 |"


def test_fleet_report_shows_arms_and_how_injections_travel():
    def row(i, arm, suite, attack=0, relayed=False):
        return {
            "id": f"c{i}", "model": "m", "arm": arm, "defence": "none", "suite": suite,
            "task": f"t{i % 3}", "goal": "refund", "utility": 1, "attack": attack,
            "attempted": attack, "relayed": relayed,
        }  # fmt: skip

    rows = [row(i, "solo", "injection", attack=1) for i in range(6)]
    rows += [row(i, "fleet", "injection", attack=int(i < 2), relayed=i < 3) for i in range(6)]
    rows += [row(i, "fleet", "clean") for i in range(3)]
    rows += [row(i, "fleet", "insider", attack=1) for i in range(4)]
    text = fleet_report(rows)
    assert "| `fleet` | `none` | 3/3 | 33%" in text and "| `solo` | `none` | n/a | 100%" in text
    # Three of six were passed on, and two of those three succeeded.
    assert "| `fleet` | 50% (19-81) | 67% (21-94) | 0% (0-56) |" in text


def test_monitor_report_lists_operating_points():
    def event(i, score, attacked, serves=False, won=False):
        return {
            "run": "r", "case": f"c{i}", "index": 0, "monitor": "m1", "score": score,
            "attacked": attacked, "serves_attack": serves, "attack_succeeded": won,
        }  # fmt: skip

    rows = [event(i, 0.0, False) for i in range(9)] + [event(9, 9.0, False)]
    rows += [event(10, 9.0, True, True, True), event(11, 0.0, True, True, True)]
    text = monitor_report({"runs": ({"honest": 10, "attacked": 4, "succeeded": 2}, rows)})
    assert "| >= 9 | 10% | 50% | 25.0% |" in text and "| never | 0% | 0% | 50.0% |" in text


def test_interp_report_renders_a_probe_result():
    cell = {"auroc": 0.9, "low": 0.8, "high": 0.95, "caught_at_5pct": 0.5, "n": 10, "positives": 4}
    splits = {split: {"probe": cell, "text": cell} for split in ("template", "goal", "task")}
    result = {
        "model": "mlx/x", "never_read": 3, "layers": 4, "width": 8,
        "counts": {"examples": 10, "injected": 6, "attempted": 2, "succeeded": 1},
        "sweep": {k: [{"layer": 1, "auroc": 0.6}, {"layer": 2, "auroc": 0.9}] for k in ("injected", "attempted")},
        "compare": {k: {"layer": 2, "splits": splits} for k in ("injected", "attempted")},
        "early_warning": {"layer": 2, "succeeded": 1, "points": [
            {"false_alarm_rate": 0.05, "attacks_flagged": 0.5, "successful_attacks_flagged": 1.0}]},
    }  # fmt: skip
    text = interp_report({"x": result})
    assert "| one template | 0.90 (0.80-0.95) | 0.90 (0.80-0.95) |" in text
    assert "| 5% | 50% | 100% |" in text and "10 decision points" in text
