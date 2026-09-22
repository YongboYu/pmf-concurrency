"""The forecast reconstruction runs Fodina's eight-step path.

Most tests check the resulting model from its input counts. The tPAT rule is tested
directly on the binding filter, with thresholds small enough to compute by hand, so
step 6's re-add can be shown apart from step 7's fallback.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from pmf_concurrency import reconstruction
from pmf_concurrency.data import END_SYMBOL, START_SYMBOL
from pmf_concurrency.reconstruction import (
    apply_binding_filter,
    assert_reconstruction_sound,
    assure_connected,
    build_causal_net_from_forecast,
    build_dependency_graph_from_forecast,
    dependency_measure,
)

S, E = START_SYMBOL, END_SYMBOL


def _kept(counts, neighbours, activity_count, tPAT):
    """The bindings the tPAT filter keeps for one side of a task."""
    return apply_binding_filter(counts, neighbours, activity_count, tPAT).kept


# =============================================================================
# Helpers
# =============================================================================


def _binding_targets(
    graph: dict[str, set[str]], binding_nodes: set[str], node: str
) -> set[str]:
    """Task nodes a binding node ultimately feeds."""
    tasks = set(graph) - binding_nodes
    seen: set[str] = set()
    todo = [node]
    reached: set[str] = set()
    while todo:
        cur = todo.pop()
        for nxt in graph.get(cur, set()):
            if nxt in seen:
                continue
            seen.add(nxt)
            if nxt in tasks:
                reached.add(nxt)
            else:
                todo.append(nxt)
    return reached


def _assert_no_dead_ends(graph: dict[str, set[str]], binding_nodes: set[str]) -> None:
    """Every binding node has at least one incoming and one outgoing arc."""
    for node in sorted(binding_nodes):
        assert graph.get(node), f"binding node {node!r} has no outgoing arc"
        assert any(node in targets for targets in graph.values()), (
            f"binding node {node!r} has no incoming arc"
        )


# A ▶ → A → {B, C} → ■ shape where C is the low-volume branch.
_COUNT_A = {S: 10.0, "A": 10.0, "B": 10.0, "C": 2.0, E: 12.0}
_COUNT_AB = {
    f"{S}→A": 10.0,
    "A→B": 10.0,
    "A→C": 2.0,
    f"B→{E}": 10.0,
    f"C→{E}": 2.0,
}


# =============================================================================
# Step 3: the connectedness repair removes dead ends
# =============================================================================


def test_low_volume_activity_leaves_no_dead_end() -> None:
    """A low-volume activity still gets a sound model.

    C has no output pattern and ■'s input patterns never mention it. Step 6's singleton
    re-add fills both sides, so no dead-end place remains.
    """
    graph, binding_nodes, _, _ = build_causal_net_from_forecast(
        _COUNT_A,
        _COUNT_AB,
        {},
        # ■'s inputs never mention C.
        {f"A:{{{S}}}": 10.0, "B:{A}": 10.0, "C:{A}": 2.0, f"{E}:{{B}}": 12.0},
        # C has no output pattern whatsoever.
        {f"{S}:{{A}}": 10.0, "A:{B}": 10.0, "A:{C}": 2.0, f"B:{{{E}}}": 10.0},
    )

    _assert_no_dead_ends(graph, binding_nodes)
    # C is still wired through to ■ rather than dropped.
    assert "C" in graph
    assert graph["C"], "C has no outgoing arc"
    assert E in _binding_targets(graph, binding_nodes, next(iter(graph["C"])))


# =============================================================================
# Step 2: anchoring on the declared markers
# =============================================================================


def test_anchors_on_markers_despite_contrary_evidence() -> None:
    """Anchors stay on the declared markers even when the counts disagree.

    A spurious ■→▶ succession makes ■ look like a mid-process activity. The repaired
    model must still have no arc into ▶ and none out of ■.
    """
    count_A = {S: 10.0, "A": 10.0, E: 10.0}
    count_AB = {f"{S}→A": 10.0, f"A→{E}": 10.0, f"{E}→{S}": 20.0}

    # The spurious arc is strong enough to enter the dependency graph.
    assert (E, S) in build_dependency_graph_from_forecast(count_A, count_AB, {})

    graph, binding_nodes, _, _ = build_causal_net_from_forecast(
        count_A,
        count_AB,
        {},
        {f"A:{{{S}}}": 10.0, f"{E}:{{A}}": 10.0},
        {f"{S}:{{A}}": 10.0, f"A:{{{E}}}": 10.0},
    )

    assert graph.get(E) == set(), "■ must have no outgoing arc"
    for node, targets in graph.items():
        assert S not in targets, f"{node!r} has an arc into ▶"
    _assert_no_dead_ends(graph, binding_nodes)


def test_marker_absent_fallback_builds_a_sound_model() -> None:
    """A window without ▶ falls back to count-based anchors and stays sound.

    A window only carries ▶ if a trace starts inside it, so this path is used on real
    logs. The fallback ranks activities by how often they start and end traces, which
    keeps the two anchors distinct.
    """
    graph, binding_nodes, _, _ = build_causal_net_from_forecast(
        {"A": 10.0, "B": 8.0, "C": 6.0},
        {"A→B": 8.0, "B→C": 6.0},
        {},
        {"B:{A}": 8.0, "C:{B}": 6.0},
        {"A:{B}": 8.0, "B:{C}": 6.0},
    )

    assert {"A", "B", "C"} <= set(graph)
    _assert_no_dead_ends(graph, binding_nodes)
    # A keeps its exit and C its entry, so the anchors did not collapse into one.
    assert graph["A"], "A has no outgoing arc: anchors collapsed"
    assert any("C" in _binding_targets(graph, binding_nodes, n) for n in binding_nodes)


# =============================================================================
# Step 3: termination on degenerate input
# =============================================================================


def test_degenerate_isolated_activities_terminate() -> None:
    """Two activities and no succession: the build returns instead of looping."""
    graph, binding_nodes, freqs, stats = build_causal_net_from_forecast(
        {"A": 1.0, "B": 1.0}, {}, {}, {}, {}
    )
    assert (graph, binding_nodes, freqs) == ({}, set(), {})
    # Steps 3 to 7 had nothing to act on, which the counters report as zeros.
    assert stats.bindings == 0
    assert stats.repair_arcs_added == 0
    assert stats.singleton_fallback_fires == 0


def test_no_admissible_edge_terminates() -> None:
    """The repair stops when no admissible arc is left.

    Both anchors fall back to X, step 2 removes X's self-loop, and the only candidate
    arc for Y is (Y, Y), which already exists. Without the stop this would loop forever.
    """
    graph, binding_nodes, _, _ = build_causal_net_from_forecast(
        {"X": 10.0, "Y": 5.0}, {"X→X": 10.0, "Y→Y": 5.0}, {}, {}, {}
    )

    assert {"X", "Y"} == set(graph) - binding_nodes
    # Residual tasks are left unconnected, but no dead-end place is produced.
    _assert_no_dead_ends(graph, binding_nodes)


def test_assure_connected_reports_incomplete_repair() -> None:
    """The repair returns False, so step 8 skips its reachability check."""
    DG = {("Y", "Y")}
    assert assure_connected(DG, {"X", "Y"}, {("X", "X"): 10.0}, "X", "X") is False


# =============================================================================
# Step 6: the tPAT rule
# =============================================================================


def test_tPAT_minus_one_removes_nothing() -> None:
    """tPAT = -1.0 sets the threshold to zero, so every binding survives."""
    counts = {frozenset({"X", "Y"}): 2.0, frozenset({"X"}): 8.0}

    kept = _kept(counts, {"X", "Y"}, 10.0, tPAT=-1.0)

    assert set(kept) == set(counts)
    assert kept == counts


def test_tPAT_zero_keeps_above_mean_and_readds_orphans() -> None:
    """Hand-computed. pat_measure(t, x) = x / count_A[t] with count_A[t] = 10:

        {X,Y} -> 2/10 = 0.2      {X} -> 8/10 = 0.8
        thresholdBase = (0.2 + 0.8) / 2 = 0.5
        thresholdAdd  = 0.0 * 0.5 = 0.0        (tPAT <= 0)
        threshold     = 0.5

    {X} is kept and the AND binding {X,Y} is dropped. That orphans Y, which the re-add
    restores as a singleton, so the AND binding becomes a choice.
    """
    counts = {frozenset({"X", "Y"}): 2.0, frozenset({"X"}): 8.0}

    kept = _kept(counts, {"X", "Y"}, 10.0, tPAT=0.0)

    assert set(kept) == {frozenset({"X"}), frozenset({"Y"})}
    assert kept[frozenset({"X"})] == 8.0
    # Y was never predicted alone, so its re-added singleton has no count.
    assert kept[frozenset({"Y"})] == 0.0


def test_tPAT_positive_uses_fodina_threshold_expression() -> None:
    """For tPAT > 0 the threshold is thresholdBase + tPAT * (1 - thresholdBase):

    thresholdBase = 0.5,  thresholdAdd = 0.5 * (1 - 0.5) = 0.25
    threshold     = 0.75   ->  only {X} at 0.8 clears it
    """
    counts = {frozenset({"X", "Y"}): 2.0, frozenset({"X"}): 8.0}

    kept = _kept(counts, {"X", "Y"}, 10.0, tPAT=0.5)

    assert set(kept) == {frozenset({"X"}), frozenset({"Y"})}


def test_step6_readd_fires_while_side_is_non_empty() -> None:
    """Step 6 restores orphans even when step 7 could not fire.

    The task side is still non-empty after the threshold, so the restored neighbour Y
    can only come from step 6.
    """
    counts = {frozenset({"X", "Y"}): 2.0, frozenset({"X"}): 8.0}

    kept = _kept(counts, {"X", "Y"}, 10.0, tPAT=0.0)

    assert kept, "side is non-empty, so the step-7 fallback cannot apply"
    assert frozenset({"Y"}) in kept


def test_step6_leaves_no_side_empty_so_step7_is_inert() -> None:
    """Step 6 never leaves a side empty, even with no predicted pattern."""
    assert _kept({}, {"X"}, 5.0, tPAT=0.0) == {frozenset({"X"}): 0.0}
    assert _kept({}, set(), 5.0, tPAT=0.0) == {}


def test_tPAT_is_live_in_the_reconstruction() -> None:
    """tPAT changes the reconstructed net.

    At -1.0 A keeps its AND binding and one binding node feeds both B and C. At
    0.0 the filter turns it into a choice and no binding node feeds both.
    """
    forecast_ib = {
        f"A:{{{S}}}": 10.0,
        "B:{A}": 10.0,
        "C:{A}": 2.0,
        f"{E}:{{B,C}}": 2.0,
        f"{E}:{{B}}": 10.0,
    }
    forecast_ob = {
        f"{S}:{{A}}": 10.0,
        "A:{B,C}": 2.0,
        "A:{B}": 8.0,
        f"B:{{{E}}}": 10.0,
        f"C:{{{E}}}": 2.0,
    }

    def and_bindings(tPAT: float) -> int:
        graph, binding_nodes, _, _ = build_causal_net_from_forecast(
            _COUNT_A, _COUNT_AB, {}, forecast_ib, forecast_ob, tPAT=tPAT
        )
        _assert_no_dead_ends(graph, binding_nodes)
        return sum(
            1
            for n in binding_nodes
            if {"B", "C"} <= _binding_targets(graph, binding_nodes, n)
        )

    assert and_bindings(-1.0) >= 1, "AND binding should survive at tPAT = -1.0"
    assert and_bindings(0.0) == 0, "AND binding should collapse at tPAT = 0.0"


# =============================================================================
# Step 8: the post-condition check
# =============================================================================


def test_post_condition_fires_on_dead_end_place() -> None:
    graph = {"A": {"o_A_0"}, "o_A_0": set(), "B": set()}
    with pytest.raises(AssertionError, match="dead-end place"):
        assert_reconstruction_sound(
            graph, {"o_A_0"}, {"A", "B"}, "A", "B", fully_connected=True
        )


def test_post_condition_fires_on_unsourced_place() -> None:
    graph = {"A": set(), "o_A_0": {"B"}, "B": set()}
    with pytest.raises(AssertionError, match="unsourced place"):
        assert_reconstruction_sound(
            graph, {"o_A_0"}, {"A", "B"}, "A", "B", fully_connected=True
        )


def test_post_condition_fires_on_unreachable_activity() -> None:
    """B reaches the end anchor but nothing reaches B from the start anchor."""
    graph = {"A": {"C"}, "B": {"C"}, "C": set()}
    with pytest.raises(AssertionError, match="unreachable activity: 'B'"):
        assert_reconstruction_sound(
            graph, set(), {"A", "B", "C"}, "A", "C", fully_connected=True
        )


def test_post_condition_fires_on_dead_end_activity() -> None:
    """B is reachable from the start anchor but cannot reach the end anchor."""
    graph = {"A": {"B", "C"}, "B": set(), "C": set()}
    with pytest.raises(AssertionError, match="dead-end activity: 'B'"):
        assert_reconstruction_sound(
            graph, set(), {"A", "B", "C"}, "A", "C", fully_connected=True
        )


def test_post_condition_skips_reachability_on_incomplete_repair() -> None:
    """An incomplete repair skips the reachability check but not the dead-end check."""
    graph = {"A": set(), "B": set()}
    assert_reconstruction_sound(
        graph, set(), {"A", "B"}, "A", "B", fully_connected=False
    )

    with pytest.raises(AssertionError, match="dead-end place"):
        assert_reconstruction_sound(
            {"A": {"o_A_0"}, "o_A_0": set()},
            {"o_A_0"},
            {"A"},
            "A",
            "A",
            fully_connected=False,
        )


# =============================================================================
# One dependency measure
# =============================================================================


def test_dependency_measure_matches_fodina() -> None:
    """ab / (ab + ba + 1), as in fodina.DependencyGraph.dep_measure."""
    ab_counts = {("A", "B"): 6.0, ("B", "A"): 3.0}
    assert dependency_measure(ab_counts, "A", "B") == 6.0 / 10.0
    assert dependency_measure(ab_counts, "B", "A") == 3.0 / 10.0
    assert dependency_measure({}, "A", "B") == 0.0


def test_dependency_graph_and_repair_share_one_definition() -> None:
    """The graph build and the repair's edge ranking use the same dependency measure."""
    real = reconstruction.dependency_measure

    for entry in ("build", "repair"):
        calls: list[tuple[str, str]] = []

        def spy(ab_counts, a, b, _calls=calls):  # type: ignore[no-untyped-def]
            _calls.append((a, b))
            return real(ab_counts, a, b)

        with patch.object(reconstruction, "dependency_measure", spy):
            if entry == "build":
                build_dependency_graph_from_forecast(
                    {"A": 5.0, "B": 5.0}, {"A→B": 5.0}, {}
                )
            else:
                assure_connected(
                    {("A", "B")}, {"A", "B", "C"}, {("A", "B"): 5.0}, "A", "B"
                )

        assert calls, f"{entry} did not use the module-level dependency measure"
