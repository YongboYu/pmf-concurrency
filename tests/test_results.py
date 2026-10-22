"""The results directory the paper is built from, and the numbers read from it.

Nothing enters the paper from a file the manifest does not list with its hash, or from
a campaign whose gate did not pass. The Table 1 and Section 5 quantities are checked on
small frames whose answers are known.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from pmf_concurrency.data import END_SYMBOL, START_SYMBOL, sort_events
from pmf_concurrency.petri_net import unfinishable_places
from pmf_concurrency.results import (
    MANIFEST,
    ResultsError,
    describe_log,
    names_and_bytes_sha256,
    over_budget_share,
    pool,
    read_campaign,
    read_manifest,
    reduction_effect,
    replay_mode_ranges,
    sha256_file,
    window_composition,
)

S, E = START_SYMBOL, END_SYMBOL


# The manifest


def _results(tmp_path: Path) -> Path:
    root = tmp_path / "results"
    (root / "c1").mkdir(parents=True)
    (root / "c1" / "a.csv").write_text("x\n1\n")
    (root / "b.json").write_text("{}\n")
    files = {
        p.relative_to(root).as_posix(): sha256_file(p)
        for p in root.rglob("*")
        if p.is_file()
    }
    (root / MANIFEST).write_text(yaml.safe_dump({"files": files}))
    return root


def test_a_manifest_whose_files_are_unchanged_is_read(tmp_path):
    assert set(read_manifest(_results(tmp_path))["files"]) == {"c1/a.csv", "b.json"}


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (lambda r: (r / "c1" / "a.csv").write_text("x\n2\n"), "changed: c1/a.csv"),
        (lambda r: (r / "b.json").unlink(), "missing: b.json"),
        (lambda r: (r / "c.csv").write_text("y\n"), "not in the manifest: c.csv"),
    ],
)
def test_a_changed_missing_or_unlisted_file_is_refused(tmp_path, damage, message):
    root = _results(tmp_path)
    damage(root)
    with pytest.raises(ResultsError, match=message):
        read_manifest(root)


def test_hidden_files_such_as_finder_metadata_are_ignored(tmp_path):
    root = _results(tmp_path)
    (root / ".DS_Store").write_bytes(b"\0")
    read_manifest(root)


def test_the_campaign_fingerprint_hashes_each_name_and_its_bytes(tmp_path):
    a, b = tmp_path / "window_1_static.json", tmp_path / "window_2_static.json"
    a.write_text("1")
    b.write_text("2")
    renamed = tmp_path / "window_3_static.json"
    renamed.write_text("1")
    assert names_and_bytes_sha256([a, b]) != names_and_bytes_sha256([b, a])
    assert names_and_bytes_sha256([a]) != names_and_bytes_sha256([renamed])


# A campaign


def _unit(window: int, status: str = "ok", **values) -> dict:
    return {
        "dataset": "sepsis",
        "tpat": 0.0,
        "window": window,
        "model_type": "forecast",
        "mode": "case_complete",
        "evaluator": "alignment",
        "reduce_level": "tau",
        "guard_status": status,
        "wall_s": 1.0,
        **values,
    }


def _campaign(tmp_path: Path, recorded: list[dict], passed: bool = True) -> Path:
    d = tmp_path / "campaign"
    d.mkdir()
    expected = pd.DataFrame([_unit(1), _unit(2)])[
        ["dataset", "tpat", "window", "model_type", "mode", "evaluator", "reduce_level"]
    ].assign(start_date="20150101", end_date="20150107", cohort="table2")
    expected.to_csv(d / "expected_units.csv", index=False)
    pd.DataFrame(recorded).to_csv(d / "evaluation.csv", index=False)
    (d / "gate.json").write_text(json.dumps({"passed": passed}))
    return d


def test_a_campaign_is_read_when_its_gate_passed_and_it_matches_its_grid(tmp_path):
    units = read_campaign(_campaign(tmp_path, [_unit(1), _unit(2)]))
    assert list(units["window"]) == [1, 2]


def test_a_campaign_whose_gate_failed_is_refused(tmp_path):
    with pytest.raises(ResultsError, match="passed gate"):
        read_campaign(_campaign(tmp_path, [_unit(1), _unit(2)], passed=False))


def test_a_campaign_that_differs_from_its_grid_is_refused(tmp_path):
    with pytest.raises(ResultsError, match="1 missing"):
        read_campaign(_campaign(tmp_path, [_unit(1)]))


# Table 1


def _log(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows, columns=["case:concept:name", "concept:name", "time:timestamp"]
    )
    frame["time:timestamp"] = pd.to_datetime(frame["time:timestamp"], format="ISO8601")
    return sort_events(frame)


def test_variants_follow_the_log_order_of_tied_events():
    # c1 and c2 hold A and B at one instant, in opposite file order: two variants.
    log = _log(
        [
            ("c1", S, "2021-01-01 08:00"),
            ("c1", "A", "2021-01-01 09:00"),
            ("c1", "B", "2021-01-01 09:00"),
            ("c2", S, "2021-01-02 08:00"),
            ("c2", "B", "2021-01-02 09:00"),
            ("c2", "A", "2021-01-02 09:00"),
            ("c3", S, "2021-01-03 08:00"),
            ("c3", "A", "2021-01-03 09:00"),
            ("c3", "B", "2021-01-03 09:00"),
        ]
    )
    stats = describe_log(log)
    assert (stats["cases"], stats["events"], stats["activities"]) == (3, 9, 3)
    assert stats["variants"] == 2


def test_the_median_case_duration_is_in_days():
    log = _log(
        [
            ("c1", "A", "2021-01-01"),
            ("c1", "B", "2021-01-02"),
            ("c2", "A", "2021-01-01"),
            ("c2", "B", "2021-01-04"),
            ("c3", "A", "2021-01-01"),
            ("c3", "B", "2021-01-11"),
        ]
    )
    assert describe_log(log)["median_case_days"] == pytest.approx(3.0)


def test_active_cases_are_classed_by_their_first_and_last_events_in_the_log():
    window = pd.DataFrame(
        {"window": [1], "start_date": ["20210110"], "end_date": ["20210116"]}
    )
    log = _log(
        [
            ("inside", "A", "2021-01-10 00:00"),
            ("inside", "B", "2021-01-16 23:59:59.999999999"),
            ("prefix", "A", "2021-01-05"),
            ("prefix", "B", "2021-01-12"),
            ("suffix", "A", "2021-01-12"),
            ("suffix", "B", "2021-01-20"),
            ("middle", "A", "2021-01-05"),
            ("middle", "B", "2021-01-12"),
            ("middle", "C", "2021-01-20"),
            ("outside", "A", "2021-01-05"),
            ("outside", "B", "2021-01-17"),
        ]
    )
    row = window_composition(log, window).iloc[0]
    assert row["n_cases"] == 4
    assert [
        row["n_complete"],
        row["n_prefix_truncated"],
        row["n_suffix_truncated"],
        row["n_middle_crossing"],
    ] == [1, 1, 1, 1]


def test_composition_reads_timezone_aware_logs_on_their_own_wall_clock():
    window = pd.DataFrame(
        {"window": [1], "start_date": ["20210110"], "end_date": ["20210110"]}
    )
    log = _log([("c1", "A", "2021-01-10 23:30")])
    log["time:timestamp"] = log["time:timestamp"].dt.tz_localize("Europe/Brussels")
    assert window_composition(log, window).iloc[0]["n_complete"] == 1


# Table 2 and Section 5


def test_a_cell_pools_the_finite_values_of_the_units_that_finished():
    units = pd.DataFrame(
        [
            _unit(1, log_fitness=0.9, precision=0.8),
            _unit(2, log_fitness=0.0, precision=float("nan")),  # an empty model
            _unit(3, "timeout", log_fitness=float("nan"), precision=float("nan")),
            _unit(4, "oom", log_fitness=0.5, precision=0.5),
        ]
    )
    fitness, precision = pool(units, "log_fitness"), pool(units, "precision")
    assert (fitness.n_units, fitness.n_values) == (4, 2)
    assert fitness.mean == pytest.approx(0.45)
    assert (precision.n_values, precision.mean) == (1, pytest.approx(0.8))
    assert precision.sd != precision.sd  # one value has no sample SD


def test_over_budget_counts_timeouts_and_ooms_only():
    units = pd.DataFrame(
        [_unit(1), _unit(2, "timeout"), _unit(3, "oom"), _unit(4, "unavailable")]
    )
    assert over_budget_share(units) == pytest.approx(0.5)


def _levels(pairs: list[tuple[str, float, float, str, float, float]]):
    """(status, wall, fitness) unreduced and reduced for windows 1, 2, ..."""
    none, tau = [], []
    for w, (s0, t0, f0, s1, t1, f1) in enumerate(pairs, start=1):
        none.append(_unit(w, s0, reduce_level="none", wall_s=t0, log_fitness=f0))
        tau.append(_unit(w, s1, reduce_level="tau", wall_s=t1, log_fitness=f1))
    for frame in (none, tau):
        for row in frame:
            row["precision"] = row["er"] = row["log_fitness"]
    return pd.DataFrame(none), pd.DataFrame(tau)


def test_reduction_counts_rescued_and_lost_units_and_the_time_saved():
    none, tau = _levels(
        [
            ("ok", 10.0, 0.9, "ok", 4.0, 0.9),
            ("ok", 30.0, 0.8, "ok", 6.0, 0.8001),
            ("timeout", 1200.0, float("nan"), "ok", 600.0, 0.7),
            ("oom", 160.0, float("nan"), "oom", 190.0, float("nan")),
            ("unavailable", 1.0, float("nan"), "unavailable", 1.0, float("nan")),
        ]
    )
    effect = reduction_effect(none, tau)
    assert (effect.n_units, effect.n_over_budget_unreduced) == (5, 2)
    assert (effect.n_rescued, effect.n_lost) == (1, 0)
    assert effect.share_rescued == pytest.approx(0.5)
    assert effect.time_saved_all == pytest.approx(1 - 801 / 1401)
    assert effect.time_saved_both_ok == pytest.approx(1 - 10 / 40)
    fitness = effect.invariance[("alignment", "log_fitness")]
    assert (fitness.n, fitness.share_equal) == (2, pytest.approx(0.5))
    assert fitness.max_abs_diff == pytest.approx(1e-4)


def test_reduction_refuses_levels_with_different_units():
    none, tau = _levels([("ok", 1.0, 0.9, "ok", 1.0, 0.9)] * 2)
    with pytest.raises(ResultsError, match="different units"):
        reduction_effect(none, tau.iloc[:1])


def test_replay_modes_give_the_range_of_the_arms_means():
    units = pd.DataFrame(
        [
            _unit(1, mode="window_raw", model_type=m, log_fitness=f, perc_fit_traces=p)
            for m, f, p in [
                ("forecast", 0.70, 1.0),
                ("forecast", 0.74, 3.0),
                ("static", 0.80, 4.0),
            ]
        ]
    )
    row = replay_mode_ranges(units).iloc[0]
    assert (row["n_arms"], row["n_not_ok"]) == (2, 0)
    assert (row["fit_min"], row["fit_max"]) == (pytest.approx(0.72), 0.80)
    assert (row["perf_min"], row["perf_max"]) == (2.0, 4.0)


# The unfinishable-net audit


def _model(activities: list[str], bindings: dict[str, tuple[str, str]]) -> dict:
    """A causal-net JSON: each binding node joins its source to its target."""
    nodes = [
        {"id": a, "data": {"label": a, "is_binding": False, "frequency": 1.0}}
        for a in activities
    ] + [
        {"id": b, "data": {"label": b, "is_binding": True, "frequency": 1.0}}
        for b in bindings
    ]
    edges = [
        {"source": src, "target": b, "data": {}} for b, (src, _) in bindings.items()
    ] + [{"source": b, "target": tgt, "data": {}} for b, (_, tgt) in bindings.items()]
    return {
        "metadata": {"type": "sliding_forecast"},
        "graph": {"directed": True, "nodes": nodes, "edges": edges},
    }


THROUGH = {"o_s": (S, "A"), "o_a": ("A", E)}


def test_a_net_whose_every_place_reaches_the_final_place_is_finishable():
    assert unfinishable_places(_model([S, "A", E], THROUGH)) == 0


def test_a_branch_that_never_reaches_the_final_place_is_unfinishable():
    # A may also hand over to B, which leads nowhere, so a token past B is stranded.
    model = _model([S, "A", "B", E], {**THROUGH, "o_b": ("A", "B")})
    assert unfinishable_places(model) > 0


def test_activities_without_an_arc_to_the_final_place_are_unfinishable():
    assert unfinishable_places(_model(["Release A", E], {})) > 0
