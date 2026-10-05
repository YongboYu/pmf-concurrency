"""The budget guard records memory blowups, timeouts and crashes as statuses.

Healthy cells run in a forked child with pure functions, so their results are bit-
identical to an unguarded run. The parent alone reaps the child, so the exit status and
the child's peak RSS are known for every unit, including one killed at its deadline.
"""

from __future__ import annotations

import math
import os
import signal
import sys

import pytest

from pmf_concurrency import guard
from pmf_concurrency.guard import UnitOutcome, classify, resolve_budgets, run_guarded


def _double(x: int) -> int:
    return x * 2


def _boom() -> None:
    raise ValueError("boom")


def _no_memory() -> None:
    raise MemoryError


def _sleep_forever() -> None:
    import time

    time.sleep(600)


def _segfault() -> None:
    import faulthandler

    faulthandler.disable()  # pytest's handler would print the child's traceback
    os.kill(os.getpid(), signal.SIGSEGV)


def _exit_without_result() -> None:
    os._exit(3)


def _memory_bomb() -> None:
    hoard = []
    while True:
        hoard.append(bytearray(64 * 2**20))  # 64 MB per step, never freed


def _unpicklable() -> object:
    return lambda: None


def test_ok_returns_result() -> None:
    outcome = run_guarded(_double, 21, mem_gb=1, time_min=1)

    assert outcome.status == "ok"
    assert outcome.result == 42
    assert outcome.exit_code == 0
    assert outcome.failure_category == ""
    assert outcome.peak_rss_gb > 0


def test_kwargs_pass_through() -> None:
    assert run_guarded(_double, x=4, mem_gb=1, time_min=1).result == 8


def test_cell_exception_becomes_error_status() -> None:
    outcome = run_guarded(_boom, mem_gb=1, time_min=1)

    assert outcome.status == "error"
    assert outcome.result is None
    assert outcome.failure_category == "ValueError"
    assert outcome.reason == "ValueError('boom')"


def test_a_caught_memory_error_is_oom() -> None:
    outcome = run_guarded(_no_memory, mem_gb=1, time_min=1)

    assert outcome.status == "oom"
    assert outcome.failure_category == "memory_budget"
    assert outcome.reason == "MemoryError()"


def test_an_unpicklable_result_is_an_error_not_a_crash() -> None:
    outcome = run_guarded(_unpicklable, mem_gb=1, time_min=1)

    assert outcome.status == "error"
    assert outcome.exit_code == 0


def test_timeout_kills_cell_and_still_measures_it() -> None:
    outcome = run_guarded(_sleep_forever, mem_gb=0, time_min=0.05)

    assert outcome.status == "timeout"
    assert outcome.result is None
    assert outcome.failure_category == "time_budget"
    assert outcome.exit_code == -signal.SIGKILL
    assert outcome.peak_rss_gb > 0


def test_a_native_crash_is_crashed_with_its_signal() -> None:
    outcome = run_guarded(_segfault, mem_gb=1, time_min=1)

    assert outcome.status == "crashed"
    assert outcome.failure_category == "signal_SIGSEGV"
    assert outcome.exit_code == -signal.SIGSEGV
    assert outcome.reason == "signal SIGSEGV"


def test_an_exit_without_a_result_is_crashed_with_its_code() -> None:
    outcome = run_guarded(_exit_without_result, mem_gb=1, time_min=1)

    assert outcome.status == "crashed"
    assert outcome.failure_category == "no_result"
    assert outcome.exit_code == 3
    assert outcome.reason == "exit code 3 without a result"


def test_the_parent_records_its_own_size_at_fork() -> None:
    outcome = run_guarded(_double, 1, mem_gb=1, time_min=1)

    if sys.platform == "linux":
        assert outcome.base_rss_gb > 0
        assert outcome.base_vsz_gb >= outcome.base_rss_gb
    else:
        assert math.isnan(outcome.base_rss_gb)


def test_a_hundred_units_leave_no_child_behind() -> None:
    units = [_double, _boom, _segfault, _exit_without_result, _no_memory]
    for i in range(100):
        run_guarded(units[i % len(units)], *([i] if i % 5 == 0 else []), time_min=1)

    with pytest.raises(ChildProcessError):
        os.waitpid(-1, os.WNOHANG)


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS enforces no memory limit")
def test_memory_bomb_becomes_oom_status() -> None:
    outcome = run_guarded(_memory_bomb, mem_gb=1, time_min=1)

    assert outcome.status == "oom"
    assert outcome.failure_category == "memory_budget"
    assert outcome.result is None


def test_inline_fallback_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PMF_GUARD", "0")
    assert run_guarded(_double, 21).result == 42
    outcome = run_guarded(_boom)
    assert outcome.status == "error"
    assert "boom" in outcome.reason


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


# Classification of a child that sent nothing. Only the cgroup's own counters attribute
# a kill to the unit's memory limit.


def test_a_kill_at_the_units_own_limit_is_oom() -> None:
    status, category, _ = classify(
        message=None, timed_out=False, exit_code=-9, oom=1, oom_kill=1
    )
    assert (status, category) == ("oom", "memory_budget")


def test_a_kill_by_the_global_oom_killer_is_host_oom() -> None:
    status, category, _ = classify(
        message=None, timed_out=False, exit_code=-9, oom=0, oom_kill=1
    )
    assert (status, category) == ("host_oom", "host_memory")


def test_a_kill_without_oom_evidence_stays_crashed() -> None:
    status, category, reason = classify(
        message=None, timed_out=False, exit_code=-9, oom=None, oom_kill=None
    )
    assert (status, category, reason) == ("crashed", "signal_SIGKILL", "signal SIGKILL")


def test_a_kill_near_the_limit_is_only_a_suspected_cause() -> None:
    outcome = guard._outcome(
        message=None,
        timed_out=False,
        exit_code=-9,
        memory={"memory_peak_gb": 11.9, "memory_max_gb": 12.0},
    )
    assert outcome.status == "crashed"
    assert outcome.suspected_cause == "near memory limit"


def test_the_row_carries_every_measurement() -> None:
    row = UnitOutcome(status="ok").row()

    assert set(row) == set(guard.OUTCOME_COLUMNS)
    assert row["guard_status"] == "ok"


# Enforcement


def test_without_delegation_linux_falls_back_to_rlimit(monkeypatch) -> None:
    monkeypatch.delenv("PMF_GUARD_ENFORCE", raising=False)
    monkeypatch.delenv("PMF_GUARD", raising=False)
    expected = "rlimit_as" if sys.platform == "linux" else "time_only"

    assert guard.enforcement().mechanism == expected


def test_disabled_guard_reports_inline(monkeypatch) -> None:
    monkeypatch.setenv("PMF_GUARD", "0")

    assert guard.enforcement() == guard.Enforcement("inline", in_force=False)


def test_a_requested_cgroup_that_is_unavailable_stops(monkeypatch, tmp_path) -> None:
    """Asking for the cgroup and silently getting RLIMIT_AS would mislabel a run."""
    monkeypatch.setenv("PMF_GUARD_ENFORCE", "cgroup")
    monkeypatch.delenv("PMF_GUARD", raising=False)
    monkeypatch.setattr(guard, "CGROUP_MOUNT", tmp_path / "no-cgroup")
    monkeypatch.setattr(guard, "_cgroups", None)

    with pytest.raises(guard.GuardSetupError):
        guard.enforcement()


# The per-unit cgroup, on a Linux box whose campaign scope is delegated (run the suite
# under scripts/local/delegated_scope.sh).

cgroup_only = pytest.mark.skipif(
    os.environ.get("PMF_GUARD_ENFORCE") != "cgroup",
    reason="needs a delegated cgroup scope (PMF_GUARD_ENFORCE=cgroup)",
)


@cgroup_only
def test_the_cgroup_caps_the_unit_and_attributes_the_kill() -> None:
    outcome = run_guarded(_memory_bomb, mem_gb=1, time_min=1)

    assert outcome.status == "oom"
    assert outcome.exit_code == -signal.SIGKILL
    assert outcome.memcg_oom >= 1
    assert outcome.memcg_oom_kill >= 1
    assert outcome.memory_max_gb == 1.0
    assert outcome.memory_peak_gb == pytest.approx(1.0, abs=0.01)


@cgroup_only
def test_the_cgroup_reports_peak_and_cap_of_a_healthy_unit() -> None:
    outcome = run_guarded(_double, 2, mem_gb=1, time_min=1)

    assert outcome.status == "ok"
    assert outcome.memory_max_gb == 1.0
    assert 0 < outcome.memory_peak_gb < 1.0
    assert (outcome.memcg_oom, outcome.memcg_oom_kill) == (0, 0)


@cgroup_only
def test_a_crash_under_the_cgroup_stays_crashed() -> None:
    outcome = run_guarded(_segfault, mem_gb=1, time_min=1)

    assert outcome.status == "crashed"
    assert outcome.failure_category == "signal_SIGSEGV"


@cgroup_only
def test_units_leave_no_cgroup_behind() -> None:
    for unit in (_double, _segfault, _memory_bomb):
        run_guarded(unit, *([1] if unit is _double else []), mem_gb=1, time_min=0.05)
    run_guarded(_sleep_forever, mem_gb=1, time_min=0.02)

    cgroups = guard.enforcement().cgroups
    assert cgroups is not None
    assert not list(cgroups.root.glob("unit_*"))
