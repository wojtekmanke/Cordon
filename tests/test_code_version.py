import subprocess

import pytest

from cordon import run


@pytest.fixture
def source_checkout(tmp_path, monkeypatch):
    source = tmp_path / "src" / "cordon" / "run.py"
    source.parent.mkdir(parents=True)
    source.write_text("# initial source\n")
    monkeypatch.setattr(run, "__file__", str(source))

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, capture_output=True, text=True, check=True
        ).stdout.strip()

    git("init", "-q")
    git("add", "src")
    git(
        "-c", "user.name=Provenance test", "-c", "user.email=test@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-qm", "Initial source",
    )  # fmt: skip
    return source, git


def test_version_records_the_full_commit_without_unrelated_changes(source_checkout, tmp_path):
    _, git = source_checkout
    (tmp_path / "README.md").write_text("Uncommitted documentation\n")
    version = run._code_version()
    assert version == git("rev-parse", "HEAD") and len(version) == 40


@pytest.mark.parametrize("change", ["modified", "staged", "untracked"])
def test_version_marks_source_changes(source_checkout, change):
    source, git = source_checkout
    if change == "untracked":
        source.with_name("new_module.py").write_text("# new module\n")
    else:
        source.write_text("# changed source\n")
        if change == "staged":
            git("add", "src")
    assert run._code_version() == git("rev-parse", "HEAD") + "+changes"


def test_version_is_unknown_when_git_status_fails(source_checkout, monkeypatch):
    original = subprocess.run

    def failed_status(args, **kwargs):
        if args[:2] == ["git", "status"]:
            raise subprocess.CalledProcessError(128, args)
        return original(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", failed_status)
    assert run._code_version() == "unknown"


def test_version_is_unknown_without_a_repository(tmp_path, monkeypatch):
    source = tmp_path / "src" / "cordon" / "run.py"
    source.parent.mkdir(parents=True)
    monkeypatch.setattr(run, "__file__", str(source))
    assert run._code_version() == "unknown"
