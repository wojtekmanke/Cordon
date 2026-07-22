import math

from cordon.report import bench_report, table
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
