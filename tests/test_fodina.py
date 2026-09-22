"""Edge cases of pmf_concurrency.fodina."""

from __future__ import annotations

from pmf_concurrency.fodina import DependencyGraph, TaskMiner


def _dg(traces: list[list[str]]) -> DependencyGraph:
    tm = TaskMiner(
        event_log=traces,
        collapse_repeats=True,
        count_collapsed=True,
        backward_context=1,
        forward_context=1,
        tDUP=0.90,
    )
    tm.mine()
    dg = DependencyGraph(
        tm=tm,
        tD=0.5,
        tL1L=0.5,
        tL2L=0.5,
        prevent_l2l_with_l1l=True,
        prefer_and_to_l2l=True,
        all_connected=True,
        long_dependencies=False,
    )
    dg.mine()
    return dg


def test_assure_connected_terminates_on_degenerate_day() -> None:
    """Mining terminates when no connecting arc exists.

    Two length-1 traces of different activities give two isolated tasks and no
    succession evidence. A regression here hangs the test.
    """
    dg = _dg([["A"], ["B"]])
    # Terminates with a finite arc set; the residual tasks stay unconnected.
    assert isinstance(dg.DG, set)


def test_assure_connected_terminates_single_activity() -> None:
    dg = _dg([["A"], ["A"]])
    assert set() == dg.DG  # one task, no arcs


def test_all_connected_normal_log_unchanged() -> None:
    """A well-formed log still mines its directly-follows arcs."""
    dg = _dg([["A", "B", "C"]] * 5 + [["A", "C"]])
    # DG holds task indices; map them back to activity names.
    named = {(dg.tm.tau(a), dg.tm.tau(b)) for a, b in dg.DG}
    assert ("A", "B") in named
    assert ("B", "C") in named
