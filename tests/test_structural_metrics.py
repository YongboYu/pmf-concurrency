"""Structural statistics for every model family, and their CSV round trip.

Arity is counted on binding frozensets. One test checks that counting node degree on the
persisted graph gives a different, inflated answer. The round trip must keep an
unmeasured counter missing rather than zero.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pmf_concurrency.causal_net import mine_fodina
from pmf_concurrency.data import END_SYMBOL, START_SYMBOL
from pmf_concurrency.families import MODEL_FAMILIES
from pmf_concurrency.reconstruction import (
    apply_binding_filter,
    build_causal_net_from_forecast,
)
from pmf_concurrency.structural_metrics import (
    STRUCTURAL_METRICS_COLUMNS,
    STRUCTURAL_METRICS_FILENAME,
    STRUCTURAL_METRICS_KEY,
    StructuralStats,
    read_structural_metrics,
    stats_from_bindings,
    stats_from_fodina,
    structural_metrics_frame,
    structural_metrics_row,
    write_structural_metrics,
)

S, E = START_SYMBOL, END_SYMBOL


# =============================================================================
# Fixtures: the ▶ → A → {B, C} → ■ net
# =============================================================================

_COUNT_A = {S: 10.0, "A": 10.0, "B": 10.0, "C": 2.0, E: 12.0}
_COUNT_AB = {
    f"{S}→A": 10.0,
    "A→B": 10.0,
    "A→C": 2.0,
    f"B→{E}": 10.0,
    f"C→{E}": 2.0,
}
# A's output binding {B,C} and ■'s input binding {B,C} are the two AND bindings.
_FORECAST_IB = {
    f"A:{{{S}}}": 10.0,
    "B:{A}": 10.0,
    "C:{A}": 2.0,
    f"{E}:{{B,C}}": 2.0,
    f"{E}:{{B}}": 10.0,
}
_FORECAST_OB = {
    f"{S}:{{A}}": 10.0,
    "A:{B,C}": 2.0,
    "A:{B}": 8.0,
    f"B:{{{E}}}": 10.0,
    f"C:{{{E}}}": 2.0,
}


def _reconstruct(tPAT: float):
    return build_causal_net_from_forecast(
        _COUNT_A, _COUNT_AB, {}, _FORECAST_IB, _FORECAST_OB, tPAT=tPAT
    )


def _naive_indegree_and_share(
    graph: dict[str, set[str]], binding_nodes: set[str]
) -> tuple[int, float]:
    """AND share from raw node in-degree, which the real measurement must not match."""
    indegree: dict[str, int] = defaultdict(int)
    for targets in graph.values():
        for target in targets:
            indegree[target] += 1
    count = sum(1 for node in binding_nodes if indegree[node] >= 2)
    share = 100.0 * count / len(binding_nodes) if binding_nodes else float("nan")
    return count, share


# =============================================================================
# Arity comes from the binding frozensets
# =============================================================================


def test_and_share_counts_arity_from_binding_frozensets() -> None:
    """Hand-counted: 5 bindings, 2 of arity ≥ 2, so 40%."""
    stats = stats_from_bindings(
        tasks=3,
        arcs=4,
        IB={"B": {frozenset({"A"})}, "C": {frozenset({"A", "B"})}},
        OB={"A": {frozenset({"B", "C"}), frozenset({"B"})}, "B": {frozenset({"C"})}},
    )

    assert stats.bindings == 5
    assert stats.and_bindings == 2
    assert stats.and_share_pct == pytest.approx(40.0)


def test_raw_and_count_is_recorded_alongside_the_share() -> None:
    """The share moves with singleton counts, so the raw AND count is recorded too."""
    few_singletons = stats_from_bindings(
        tasks=2, arcs=1, IB={}, OB={"A": {frozenset({"B", "C"})}}
    )
    many_singletons = stats_from_bindings(
        tasks=4,
        arcs=3,
        IB={},
        OB={
            "A": {frozenset({"B", "C"})},
            "B": {frozenset({"C"})},
            "C": {frozenset({"D"})},
            "D": {frozenset({"E"})},
        },
    )

    # Same AND count, very different share, so both are recorded.
    assert few_singletons.and_bindings == many_singletons.and_bindings == 1
    assert few_singletons.and_share_pct == pytest.approx(100.0)
    assert many_singletons.and_share_pct == pytest.approx(25.0)


def test_input_output_split_accounts_for_every_binding() -> None:
    stats = stats_from_bindings(
        tasks=3,
        arcs=4,
        IB={"B": {frozenset({"A"})}, "C": {frozenset({"A", "B"})}},
        OB={"A": {frozenset({"B", "C"}), frozenset({"B"})}, "B": {frozenset({"C"})}},
    )

    assert stats.input_bindings == 2
    assert stats.input_and_bindings == 1
    assert stats.output_bindings == 3
    assert stats.output_and_bindings == 1
    assert stats.input_bindings + stats.output_bindings == stats.bindings
    assert stats.input_and_bindings + stats.output_and_bindings == stats.and_bindings


def test_a_net_with_no_bindings_has_no_share_rather_than_zero() -> None:
    """A net with no bindings has no share; 0% would claim a pure XOR net."""
    stats = stats_from_bindings(tasks=0, arcs=0, IB={}, OB={})

    assert stats.bindings == 0
    assert stats.and_bindings == 0
    assert stats.and_share_pct != stats.and_share_pct, "expected NaN, not 0.0"


# =============================================================================
# The node-degree trap
# =============================================================================


def test_and_share_disagrees_with_a_naive_node_degree_count() -> None:
    """Node degree reaches the right count from the wrong set.

    The persisted graph chains T → o_T_k → i_X_j → X, so B's singleton input binding has
    in-degree 2. The real measurement finds 2 AND bindings in 10 (20%). Naive in-degree
    finds 2 in 5 surviving binding nodes (40%) by counting a singleton and missing a
    real AND binding.
    """
    graph, binding_nodes, _, stats = _reconstruct(tPAT=-1.0)

    assert stats.bindings == 10
    assert stats.and_bindings == 2
    assert stats.and_share_pct == pytest.approx(20.0)

    naive_count, naive_share = _naive_indegree_and_share(graph, binding_nodes)
    assert naive_share == pytest.approx(40.0)
    assert naive_share > stats.and_share_pct, (
        "the naive count must be shown inflating the share, or this test guards nothing"
    )
    # Right count, wrong set: the denominator is graph nodes and a singleton is counted.
    assert naive_count == stats.and_bindings
    assert len(binding_nodes) != stats.bindings


def test_and_share_tracks_the_filter_collapsing_and_to_xor() -> None:
    """At tPAT = 0.0 both AND bindings become singletons, so the share is 0%."""
    _, _, _, stats = _reconstruct(tPAT=0.0)

    assert stats.bindings == 10
    assert stats.and_bindings == 0
    assert stats.and_share_pct == pytest.approx(0.0)


# =============================================================================
# Repair and filter counters stay separate
# =============================================================================


def test_filter_counter_and_readd_counter_are_separately_recorded() -> None:
    """The filter and the re-add are counted separately.

    At tPAT = 0.0 the threshold drops A's {B, C} and ■'s {B, C}, and the re-add restores
    the orphaned neighbours as singletons.
    """
    _, _, _, stats = _reconstruct(tPAT=0.0)

    assert stats.filter_bindings_removed == 2
    assert stats.singleton_readds == 2
    # Both re-adds are threshold orphans: the dropped AND bindings named B and C.
    assert stats.singleton_readds_orphaned == 2
    assert stats.singleton_readds_uncovered == 0

    _, _, _, unfiltered = _reconstruct(tPAT=-1.0)
    assert unfiltered.filter_bindings_removed == 0
    assert unfiltered.singleton_readds == 0


def test_repair_arc_counter_is_zero_on_an_already_connected_net() -> None:
    _, _, _, stats = _reconstruct(tPAT=0.0)
    assert stats.repair_arcs_added == 0


def test_repair_arc_counter_counts_the_arcs_the_repair_added() -> None:
    """Step 3 adds an arc to reach D, and that arc is counted apart from the filter."""
    count_A = {S: 10.0, "A": 10.0, "D": 6.0, E: 10.0}
    count_AB = {f"{S}→A": 10.0, f"A→{E}": 10.0, "D→A": 6.0}

    _, _, _, stats = build_causal_net_from_forecast(
        count_A,
        count_AB,
        {},
        {f"A:{{{S}}}": 10.0, f"{E}:{{A}}": 10.0},
        {f"{S}:{{A}}": 10.0, f"A:{{{E}}}": 10.0},
    )

    assert stats.repair_arcs_added is not None
    assert stats.repair_arcs_added >= 1


def test_singleton_fallback_does_not_fire_when_step_six_is_faithful() -> None:
    """Step 6 already covers every neighbour, so the step 7 fallback never fires."""
    for tPAT in (-1.0, 0.0):
        _, _, _, stats = _reconstruct(tPAT=tPAT)
        assert stats.singleton_fallback_fires == 0


def test_apply_pattern_filter_separates_the_threshold_from_the_readd() -> None:
    """A binding dropped and then restored as a singleton counts in both halves.

    Comparing the filter's input and output from outside would miss it, which is why the
    filter keeps its own counters.
    """
    counts = {frozenset({"X", "Y"}): 2.0, frozenset({"X"}): 8.0}

    outcome = apply_binding_filter(counts, {"X", "Y"}, 10.0, tPAT=0.0)

    assert set(outcome.kept) == {frozenset({"X"}), frozenset({"Y"})}
    assert outcome.removed == 1
    assert outcome.readded == 1


def test_apply_pattern_filter_removes_nothing_at_tpat_minus_one() -> None:
    counts = {frozenset({"X", "Y"}): 2.0, frozenset({"X"}): 8.0}

    outcome = apply_binding_filter(counts, {"X", "Y"}, 10.0, tPAT=-1.0)

    assert outcome.kept == counts
    assert outcome.removed == 0
    assert outcome.readded == 0


# =============================================================================
# The Fodina families report the same statistics
# =============================================================================

_TRACES = [
    ["a", "b", "c", "d"],
    ["a", "c", "b", "d"],
    ["a", "b", "c", "d"],
    ["a", "c", "b", "d"],
]


def test_fodina_families_report_the_same_statistics() -> None:
    """Direct-mined and Static are measured from Fodina's own IB/OB."""
    mined = mine_fodina(_TRACES)
    assert mined is not None
    stats = stats_from_fodina(mined)

    assert isinstance(stats, StructuralStats)
    assert stats.tasks > 0
    assert stats.arcs > 0
    assert stats.bindings > 0
    assert stats.input_bindings + stats.output_bindings == stats.bindings
    assert 0.0 <= stats.and_share_pct <= 100.0
    assert stats.and_bindings == pytest.approx(
        stats.and_share_pct * stats.bindings / 100.0
    )


def test_fodina_process_counters_are_absent_rather_than_zero() -> None:
    """Counters of Fodina's uninstrumented repair and filter are missing, not 0."""
    mined = mine_fodina(_TRACES)
    assert mined is not None
    stats = stats_from_fodina(mined)

    assert stats.repair_arcs_added is None
    assert stats.filter_bindings_removed is None
    assert stats.singleton_readds is None
    assert stats.singleton_fallback_fires is None


def test_fodina_arity_is_counted_on_frozensets_not_task_ids() -> None:
    """Fodina keys bindings by task index; arity is still the size of the binding."""
    pm = mine_fodina(_TRACES)
    assert pm is not None
    expected = sum(
        1
        for mapping in (pm.IB, pm.OB)
        for bindings in mapping.values()
        for binding in bindings
        if len(binding) >= 2
    )

    assert stats_from_fodina(pm).and_bindings == expected


# =============================================================================
# One CSV per campaign, keyed by evaluation window and model family
# =============================================================================


def _row(window: int, family: str, stats: StructuralStats) -> dict:
    return structural_metrics_row(window, family, stats, dataset="Sepsis", tPAT=0.0)


def _rows() -> list[dict]:
    _, _, _, reconstructed = _reconstruct(tPAT=0.0)
    mined = mine_fodina(_TRACES)
    assert mined is not None
    fodina = stats_from_fodina(mined)
    # Window 2's Re-mined net has no bindings, so its share is NaN.
    degenerate = stats_from_bindings(
        tasks=0,
        arcs=0,
        IB={},
        OB={},
        repair_arcs_added=0,
        filter_bindings_removed=0,
        singleton_readds_orphaned=0,
        singleton_readds_uncovered=0,
        singleton_fallback_fires=0,
    )
    return [
        _row(1, "forecast", reconstructed),
        _row(1, "re_mined", reconstructed),
        _row(1, "direct_mined", fodina),
        _row(1, "static", fodina),
        _row(2, "forecast", reconstructed),
        _row(2, "re_mined", degenerate),
    ]


def test_structural_metrics_frame_has_the_declared_schema() -> None:
    frame = structural_metrics_frame(_rows())

    assert list(frame.columns) == list(STRUCTURAL_METRICS_COLUMNS)
    assert STRUCTURAL_METRICS_COLUMNS[: len(STRUCTURAL_METRICS_KEY)] == (
        STRUCTURAL_METRICS_KEY
    )
    assert len(frame) == 6


def test_structural_metrics_is_keyed_by_window_and_model_family() -> None:
    frame = structural_metrics_frame(_rows())

    key = frame[["window", "model_family"]]
    assert not key.duplicated().any(), "window and family must identify a row"
    assert set(frame.loc[frame["window"] == 1, "model_family"]) == {
        "forecast",
        "re_mined",
        "direct_mined",
        "static",
    }


def test_structural_metrics_round_trips_through_the_csv(tmp_path) -> None:
    frame = structural_metrics_frame(_rows())

    path = write_structural_metrics(str(tmp_path), _rows())

    assert path.endswith(STRUCTURAL_METRICS_FILENAME)
    assert_frame_equal(read_structural_metrics(path), frame)


def test_unmeasured_counters_round_trip_as_missing_not_zero(tmp_path) -> None:
    """Unmeasured counters on Fodina rows read back as missing, not 0."""
    path = write_structural_metrics(str(tmp_path), _rows())

    read_back = read_structural_metrics(path)
    fodina_rows = read_back[read_back["model_family"].isin(["direct_mined", "static"])]

    assert len(fodina_rows) == 2
    for column in (
        "repair_arcs_added",
        "filter_bindings_removed",
        "singleton_readds",
        "singleton_fallback_fires",
    ):
        assert fodina_rows[column].isna().all(), f"{column} must be missing, not 0"

    reconstruction_rows = read_back[read_back["model_family"] == "forecast"]
    assert (reconstruction_rows["filter_bindings_removed"] == 2).all()


def test_structural_metrics_counts_stay_integers_through_the_csv(tmp_path) -> None:
    """Counts stay nullable integers, so a missing value does not make them floats."""
    path = write_structural_metrics(str(tmp_path), _rows())

    read_back = read_structural_metrics(path)

    assert read_back["bindings"].dtype == "Int64"
    assert read_back["repair_arcs_added"].dtype == "Int64"
    assert read_back["and_share_pct"].dtype == "Float64"


def test_reading_a_frame_without_the_declared_columns_is_an_error(tmp_path) -> None:
    path = tmp_path / STRUCTURAL_METRICS_FILENAME
    pd.DataFrame([{"window": 1, "and_share_pct": 20.0}]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="model_family"):
        read_structural_metrics(str(path))


def test_an_empty_campaign_writes_a_header_only_csv(tmp_path) -> None:
    """A campaign with no models still writes the file, with only a header."""
    path = write_structural_metrics(str(tmp_path), [])

    read_back = read_structural_metrics(path)
    assert read_back.empty
    assert list(read_back.columns) == list(STRUCTURAL_METRICS_COLUMNS)


def test_a_missing_share_round_trips_as_missing(tmp_path) -> None:
    """A missing share reads back as missing, not as 0.0."""
    path = write_structural_metrics(str(tmp_path), _rows())

    read_back = read_structural_metrics(path)
    degenerate = read_back[
        (read_back["window"] == 2) & (read_back["model_family"] == "re_mined")
    ]

    assert len(degenerate) == 1
    assert degenerate["and_share_pct"].isna().all()
    assert (degenerate["bindings"] == 0).all()
    # A net with bindings but no AND binding reads back as 0.0.
    xor_net = read_back[read_back["model_family"] == "forecast"]
    assert (xor_net["and_share_pct"] == 0.0).all()
    assert xor_net["and_share_pct"].notna().all()


# =============================================================================
# Only threshold orphans are filter effects
# =============================================================================


def test_readds_forced_by_the_repair_are_not_booked_to_the_filter() -> None:
    """Re-adds caused by the repair are not counted as filter effects.

    At tPAT = -1.0 the threshold drops nothing. Step 3 attaches D with a new arc and the
    re-add covers that neighbour, which is not a threshold orphan.
    """
    count_A = {S: 10.0, "A": 10.0, "D": 6.0, E: 10.0}
    count_AB = {f"{S}→A": 10.0, f"A→{E}": 10.0, "D→A": 6.0}

    _, _, _, stats = build_causal_net_from_forecast(
        count_A,
        count_AB,
        {},
        {f"A:{{{S}}}": 10.0, f"{E}:{{A}}": 10.0},
        {f"{S}:{{A}}": 10.0, f"A:{{{E}}}": 10.0},
        tPAT=-1.0,
    )

    assert stats.repair_arcs_added is not None and stats.repair_arcs_added >= 1
    assert stats.filter_bindings_removed == 0, "tPAT = -1.0 removes nothing"
    assert stats.singleton_readds_orphaned == 0, (
        "the threshold removed nothing, so it cannot have orphaned a neighbour"
    )
    assert stats.singleton_readds_uncovered is not None
    assert stats.singleton_readds_uncovered >= 1
    assert stats.singleton_readds == stats.singleton_readds_uncovered


def test_filter_outcome_separates_the_two_readd_reasons() -> None:
    """Y is orphaned by the threshold; Z was never named by any binding."""
    counts = {frozenset({"X", "Y"}): 2.0, frozenset({"X"}): 8.0}

    outcome = apply_binding_filter(counts, {"X", "Y", "Z"}, 10.0, tPAT=0.0)

    assert outcome.readded_orphaned == 1
    assert outcome.readded_uncovered == 1
    assert outcome.readded == 2


def test_a_side_the_threshold_could_not_run_on_is_marked_not_zero() -> None:
    """A side without a positive activity count is recorded as skipped, not as zero."""
    counts = {frozenset({"X", "Y"}): 2.0, frozenset({"X"}): 8.0}

    skipped = apply_binding_filter(counts, {"X", "Y"}, 0.0, tPAT=0.0)
    ran = apply_binding_filter(counts, {"X", "Y"}, 10.0, tPAT=-1.0)

    assert skipped.removed == ran.removed == 0
    assert skipped.threshold_skipped is True
    assert ran.threshold_skipped is False
    # A side with nothing to threshold is not "skipped": there was no work.
    assert apply_binding_filter({}, {"X"}, 0.0, tPAT=0.0).threshold_skipped is False


def test_every_reconstruction_task_has_a_positive_activity_count() -> None:
    """Every reconstructed task has a positive activity count.

    Step 1 takes the task set from count_A, so an activity named only in count_AB never
    becomes a task. This keeps filter_bindings_removed == 0 unambiguous.
    """
    graph, binding_nodes, _, stats = build_causal_net_from_forecast(
        {S: 10.0, "A": 10.0, E: 10.0},
        {f"{S}→A": 10.0, "A→B": 10.0, f"B→{E}": 10.0, f"A→{E}": 10.0},
        {},
        {"B:{A}": 4.0, f"{E}:{{B}}": 4.0},
        {"A:{B}": 4.0, f"B:{{{E}}}": 4.0},
    )

    assert "B" not in set(graph) - binding_nodes, (
        "B has no count_A, so it is not a task"
    )
    assert stats.tasks == 3
    assert stats.filter_bindings_removed is not None


# =============================================================================
# Each row records the campaign that produced it
# =============================================================================


def test_rows_carry_the_dataset_and_the_filter_setting() -> None:
    """Each row carries the dataset and tPAT, which the directory name does not."""
    _, _, _, stats = _reconstruct(tPAT=-1.0)
    row = structural_metrics_row(3, "re_mined", stats, dataset="BPI2019", tPAT=-1.0)

    assert row["dataset"] == "BPI2019"
    assert row["tPAT"] == -1.0
    assert set(STRUCTURAL_METRICS_KEY) <= set(row)


def test_duplicate_window_and_family_is_rejected() -> None:
    """A duplicated (window, family) key is an error: it would double a point."""
    _, _, _, stats = _reconstruct(tPAT=0.0)
    rows = [_row(1, "re_mined", stats), _row(1, "re_mined", stats)]

    with pytest.raises(ValueError, match="unique per"):
        structural_metrics_frame(rows)


def test_model_family_keeps_the_declared_order_not_the_alphabet() -> None:
    """Rows follow the paper's family order, not alphabetical order."""
    frame = structural_metrics_frame(_rows())

    families = list(frame.loc[frame["window"] == 1, "model_family"])
    assert families == [f for f in MODEL_FAMILIES if f in families]
    assert families[0] == "forecast"


def test_the_history_families_keep_their_names_and_sort_last(tmp_path) -> None:
    _, _, _, stats = _reconstruct(tPAT=0.0)
    rows = [
        _row(1, "expanding", stats),
        _row(1, "persistence", stats),
        _row(1, "static", stats),
        _row(1, "forecast", stats),
    ]
    path = write_structural_metrics(str(tmp_path), rows)

    for frame in (structural_metrics_frame(rows), read_structural_metrics(path)):
        assert list(frame["model_family"].astype(str)) == [
            "forecast",
            "static",
            "persistence",
            "expanding",
        ]


def test_tpat_round_trips_through_the_csv(tmp_path) -> None:
    path = write_structural_metrics(str(tmp_path), _rows())

    read_back = read_structural_metrics(path)
    assert (read_back["tPAT"] == 0.0).all()
    assert (read_back["dataset"] == "Sepsis").all()


# =============================================================================
# One task count, so the families stay comparable
# =============================================================================


def test_fodina_and_reconstruction_count_tasks_the_same_way() -> None:
    """Both paths count dependency-graph endpoints as tasks."""
    pm = mine_fodina(_TRACES)
    assert pm is not None

    stats = stats_from_fodina(pm)
    dg_endpoints = {a for a, _ in pm.dg.DG} | {b for _, b in pm.dg.DG}

    assert stats.tasks == len(dg_endpoints)
    assert stats.tasks <= len(pm.tm.TC), "TC can carry a task the arc set does not"
