"""Published result stamps must survive a fresh checkout of the current history."""

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_recorded_code_versions_are_in_current_history():
    if not (ROOT / ".git").exists():
        pytest.skip("Result provenance requires a Git checkout, not a source archive")

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    assert git("rev-parse", "--is-shallow-repository") == "false", (
        "Result provenance requires full history; use git fetch --unshallow "
        "or actions/checkout with fetch-depth: 0"
    )
    paths = git("ls-files", "-z", "--", "results").split("\0")
    stamps = {}
    for name in paths:
        path = ROOT / name
        if path.suffix not in {".json", ".jsonl"}:
            continue
        text = path.read_text()
        records = (
            [json.loads(line) for line in text.splitlines() if line.strip()]
            if path.suffix == ".jsonl"
            else [json.loads(text)]
        )
        for index, record in enumerate(records, 1):
            location = f"{name}:{index}"
            if name.startswith(("results/bench/", "results/fleet/")):
                assert "code" in record, f"{location}: missing code stamp"
            if "code" not in record:
                # The initial monitor and probe files did not record versions.
                continue
            stamp = record["code"]
            assert isinstance(stamp, str) and re.fullmatch(r"[0-9a-f]{7,40}(\+changes)?", stamp), (
                f"{location}: invalid or unknown code stamp {stamp!r}"
            )
            stamps.setdefault(stamp.removesuffix("+changes"), location)

    assert stamps, "No result code stamps were checked"
    errors = []
    for commit, location in sorted(stamps.items()):
        try:
            full = git("rev-parse", "--verify", f"{commit}^{{commit}}")
            git("merge-base", "--is-ancestor", full, "HEAD")
        except subprocess.CalledProcessError:
            errors.append(f"{location}: {commit} is missing or not an ancestor of HEAD")
    assert not errors, "Result provenance is broken:\n" + "\n".join(errors)

    mappings = json.loads((ROOT / "docs/result-commit-map.json").read_text())["commits"]
    assert mappings, "The result commit migration must retain its mapping"
    for entry in mappings:
        assert git("rev-parse", f"{entry['current']}^{{tree}}") == entry["tree"]
        git("merge-base", "--is-ancestor", entry["current"], "HEAD")
