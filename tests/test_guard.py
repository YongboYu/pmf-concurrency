"""The budget guard records memory blowups, timeouts and crashes as statuses.

Healthy cells run in a forked child with pure functions, so their results are bit-
identical to an unguarded run.
"""

from __future__ import annotations

import sys

import pytest

from pmf_concurrency.guard import resolve_budgets, run_guarded


def _double(x: int) -> int:
    return x * 2


def _boom() -> None:
    raise ValueError("boom")


def _sleep_forever() -> None:
    import time

    time.sleep(600)


def _memory_bomb() -> None:
    hoard = []
    while True:
        hoard.append(bytearray(64 * 2**20))  # 64 MB per step, never freed


def test_ok_returns_result() -> None:
    assert run_guarded(_double, 21, mem_gb=1, time_min=1) == ("ok", 42)


def test_kwargs_pass_through() -> None:
    assert run_guarded(_double, x=4, mem_gb=1, time_min=1) == ("ok", 8)


def test_cell_exception_becomes_error_status() -> None:
    status, payload = run_guarded(_boom, mem_gb=1, time_min=1)
    assert status == "error"
    assert "boom" in payload


def test_timeout_kills_cell() -> None:
    status, payload = run_guarded(_sleep_forever, mem_gb=0, time_min=0.05)
    assert status == "timeout"
    assert payload is None


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS does not enforce RLIMIT_AS")
def test_memory_bomb_becomes_oom_status() -> None:
    status, payload = run_guarded(_memory_bomb, mem_gb=1, time_min=1)
    assert status == "oom"
    assert payload is None


def test_inline_fallback_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PMF_GUARD", "0")
    assert run_guarded(_double, 21) == ("ok", 42)
    status, payload = run_guarded(_boom)
    assert status == "error"
    assert "boom" in payload


def test_default_budgets_are_the_papers(monkeypatch: pytest.MonkeyPatch) -> None:
    """12 GB and 20 min per unit, the budget the paper reports."""
    monkeypatch.delenv("PMF_GUARD_MEM_GB", raising=False)
    monkeypatch.delenv("PMF_GUARD_TIME_MIN", raising=False)
    assert resolve_budgets() == (12.0, 20.0)


def test_budgets_come_from_arguments_then_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PMF_GUARD_MEM_GB", "48")
    monkeypatch.setenv("PMF_GUARD_TIME_MIN", "5")
    assert resolve_budgets() == (48.0, 5.0)
    # 0 is a budget (no limit), not a missing argument.
    assert resolve_budgets(mem_gb=0, time_min=1) == (0.0, 1.0)
