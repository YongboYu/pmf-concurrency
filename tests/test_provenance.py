"""The evaluation manifest names the code that ran, even when it is not committed.

The commit alone does not name uncommitted code, so the fingerprint also hashes the diff
against it and the untracked files under ``src/`` and ``scripts/``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from pmf_concurrency.provenance import code_fingerprint, machine, versions


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    (tmp_path / "notes.txt").write_text("not code\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_a_clean_tree_is_named_by_its_commit(repo) -> None:
    first = code_fingerprint(repo)

    assert len(first["git_commit"]) == 40
    assert code_fingerprint(repo) == first


def test_an_edited_source_file_changes_the_fingerprint(repo) -> None:
    clean = code_fingerprint(repo)
    (repo / "src" / "a.py").write_text("x = 2\n")

    edited = code_fingerprint(repo)

    assert edited["git_commit"] == clean["git_commit"]
    assert edited["git_diff_sha256"] != clean["git_diff_sha256"]


def test_an_untracked_script_changes_the_fingerprint(repo) -> None:
    clean = code_fingerprint(repo)
    (repo / "scripts" / "new.py").write_text("print()\n")

    assert code_fingerprint(repo) != clean


def test_files_outside_the_code_do_not(repo) -> None:
    clean = code_fingerprint(repo)
    (repo / "notes.txt").write_text("edited\n")

    assert code_fingerprint(repo) == clean


def test_outside_git_the_files_themselves_are_hashed(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    before = code_fingerprint(tmp_path)
    (tmp_path / "src" / "a.py").write_text("x = 2\n")

    assert before["git_commit"] is None
    assert code_fingerprint(tmp_path) != before


def test_machine_and_versions_are_filled() -> None:
    assert all(machine().values())
    assert set(versions()) == {"python", "pm4py"}
