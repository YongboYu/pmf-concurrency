"""The evaluation campaign: its expected grid, the model check and the gate.

The expected units come from the protocol (the logs' calendar windows, the Table 2 arms
and evaluators), never from the files on disk, so a missing model or a missing result
is caught instead of shrinking the grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pmf_concurrency.campaign import (
    EXPECTED_COLUMNS,
    check_models,
    expected_units,
    near_budget,
    oom_attribution,
    reconcile,
    windows_of,
)
from pmf_concurrency.families import MODEL_FAMILIES


def _events(first: str, last: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case:concept:name": ["c1", "c2"],
            "concept:name": ["A", "B"],
            "time:timestamp": pd.to_datetime([first, last]),
        }
    )


def test_windows_are_seven_calendar_days_sliding_by_one_after_the_cutoff_day():
    # 50 days; 80 % ends on 2021-02-10, so the test period is 02-11 to 02-20 (10
    # days) and holds 10 - 7 + 1 = 4 windows.
    windows = windows_of(_events("2021-01-01", "2021-02-20"), train_ratio=0.8)

    assert list(windows["window"]) == [1, 2, 3, 4]
    assert windows.iloc[0][["start_date", "end_date"]].tolist() == [
        "20210211",
        "20210217",
    ]
    assert windows.iloc[-1][["start_date", "end_date"]].tolist() == [
        "20210214",
        "20210220",
    ]


def test_windows_count_calendar_days_without_events():
    events = pd.concat(
        [_events("2021-01-01", "2021-02-11"), _events("2021-02-20", "2021-02-20")]
    )
    assert len(windows_of(events, train_ratio=0.8)) == 4


@pytest.fixture
def two_logs() -> dict[str, pd.DataFrame]:
    def grid(n: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "window": range(1, n + 1),
                "start_date": [f"202103{d:02d}" for d in range(1, n + 1)],
                "end_date": [f"202103{d + 6:02d}" for d in range(1, n + 1)],
            }
        )

    return {"bpi2017": grid(3), "sepsis": grid(2)}


def test_campaign_1_is_the_table_2_grid_on_the_reduced_nets(two_logs):
    grid = expected_units(two_logs, campaign=1)

    assert list(grid.columns) == list(EXPECTED_COLUMNS)
    # 5 windows x (4 families at tPAT 0 + 2 at tPAT -1) x 3 evaluators.
    assert len(grid) == 5 * 6 * 3
    assert set(grid["reduce_level"]) == {"tau"}
    assert set(grid["evaluator"]) == {"token_etc", "alignment", "er"}
    assert set(grid.loc[grid["evaluator"] == "er", "mode"]) == {"window"}
    assert set(grid.loc[grid["evaluator"] != "er", "mode"]) == {"case_complete"}
    assert set(grid.loc[grid["tpat"] == 0.0, "model_type"]) == set(MODEL_FAMILIES)
    assert set(grid.loc[grid["tpat"] == -1.0, "model_type"]) == {
        "forecast",
        "re_mined",
    }
    assert not grid.duplicated().any()


def test_campaign_2_is_the_unreduced_grid_plus_the_bpi2017_replay_modes(two_logs):
    grid = expected_units(two_logs, campaign=2)

    base = grid[grid["cohort"] == "table2"]
    assert len(base) == 5 * 6 * 3
    assert set(base["reduce_level"]) == {"none"}

    extra = grid[grid["cohort"] == "replay_modes"]
    # 6 evaluator/mode/level combinations x 3 BPI2017 windows x 6 arms.
    assert len(extra) == 6 * 3 * 6
    assert set(extra["dataset"]) == {"bpi2017"}
    assert set(zip(extra["evaluator"], extra["mode"], strict=True)) == {
        ("token_etc", "window_raw"),
        ("token_etc", "window_no_remain"),
        ("alignment", "window_raw"),
    }
    assert set(extra["reduce_level"]) == {"tau", "none"}
    key = [
        "dataset",
        "tpat",
        "window",
        "model_type",
        "mode",
        "evaluator",
        "reduce_level",
    ]
    assert not grid.duplicated(subset=key).any()


def test_campaign_3_is_the_history_families_at_tpat_0_on_the_reduced_nets(two_logs):
    grid = expected_units(two_logs, campaign=3)

    assert list(grid.columns) == list(EXPECTED_COLUMNS)
    # Windows x 2 history families x 3 evaluators, per log.
    assert grid.groupby("dataset").size().to_dict() == {
        "bpi2017": 3 * 2 * 3,
        "sepsis": 2 * 2 * 3,
    }
    assert set(grid["model_type"]) == {"persistence", "expanding"}
    assert set(grid["tpat"]) == {0.0}
    assert set(grid["reduce_level"]) == {"tau"}
    assert set(grid["cohort"]) == {"history"}
    assert set(grid.loc[grid["evaluator"] == "er", "mode"]) == {"window"}
    assert set(grid.loc[grid["evaluator"] != "er", "mode"]) == {"case_complete"}
    sepsis = grid[grid["dataset"] == "sepsis"]
    assert sorted(set(zip(sepsis["window"], sepsis["start_date"], strict=True))) == [
        (1, "20210301"),
        (2, "20210302"),
    ]
    assert not grid.duplicated().any()


# The model check


def _write_run(root: Path, tpat: float, windows: pd.DataFrame, families) -> Path:
    root.mkdir(parents=True)
    (root / "experiment_summary.json").write_text(
        json.dumps({"thresholds": {"tPAT": tpat}})
    )
    weekly = root / "weekly_models"
    weekly.mkdir()
    for w in windows.itertuples():
        for family in families:
            meta = {
                "window": w.window,
                "start_date": w.start_date,
                "end_date": w.end_date,
            }
            (weekly / f"window_{w.window}_{family}.json").write_text(
                json.dumps({"metadata": meta, "graph": {}})
            )
    return weekly


@pytest.fixture
def campaign_1(tmp_path, two_logs):
    grid = expected_units({"bpi2017": two_logs["bpi2017"]}, campaign=1)
    dirs = {
        ("bpi2017", 0.0): _write_run(
            tmp_path / "t0", 0.0, two_logs["bpi2017"], MODEL_FAMILIES
        ),
        ("bpi2017", -1.0): _write_run(
            tmp_path / "tm1", -1.0, two_logs["bpi2017"], ("forecast", "re_mined")
        ),
    }
    return grid, dirs


def test_complete_models_pass_the_check(campaign_1):
    grid, dirs = campaign_1
    assert check_models(grid, dirs) == []


def test_a_missing_model_is_a_problem(campaign_1):
    grid, dirs = campaign_1
    (dirs[("bpi2017", 0.0)] / "window_2_static.json").unlink()

    problems = check_models(grid, dirs)
    assert len(problems) == 1
    assert "window_2_static.json" in problems[0]


def test_a_model_that_does_not_parse_is_a_problem(campaign_1):
    grid, dirs = campaign_1
    (dirs[("bpi2017", -1.0)] / "window_1_forecast.json").write_text("{")

    assert any("window_1_forecast.json" in p for p in check_models(grid, dirs))


def test_a_model_with_other_window_dates_is_a_problem(campaign_1):
    grid, dirs = campaign_1
    path = dirs[("bpi2017", 0.0)] / "window_3_forecast.json"
    model = json.loads(path.read_text())
    model["metadata"]["start_date"] = "20210304"
    path.write_text(json.dumps(model))

    problems = check_models(grid, dirs)
    assert len(problems) == 1
    assert "20210304" in problems[0]


def test_models_from_the_other_threshold_are_a_problem(campaign_1):
    grid, dirs = campaign_1
    summary = dirs[("bpi2017", -1.0)].parent / "experiment_summary.json"
    summary.write_text(json.dumps({"thresholds": {"tPAT": 0.0}}))

    assert any("tPAT" in p for p in check_models(grid, dirs))


def test_an_arm_without_a_model_directory_is_a_problem(campaign_1):
    grid, dirs = campaign_1
    del dirs[("bpi2017", -1.0)]

    assert any("-1.0" in p for p in check_models(grid, dirs))


# The gate


def _recorded(grid: pd.DataFrame, **fields) -> pd.DataFrame:
    rows = grid.drop(columns=["start_date", "end_date", "cohort"]).copy()
    rows["guard_status"] = "ok"
    for k, v in fields.items():
        rows[k] = v
    return rows


def test_recorded_units_equal_to_the_grid_pass(two_logs):
    grid = expected_units(two_logs, campaign=1)
    result = reconcile(grid, _recorded(grid))
    assert result.passed
    assert result.n_missing == result.n_duplicated == result.n_extra == 0


def test_missing_duplicated_and_extra_units_fail(two_logs):
    grid = expected_units(two_logs, campaign=1)
    recorded = _recorded(grid)
    extra = recorded.iloc[[0]].assign(window=99)
    recorded = pd.concat([recorded.iloc[2:], recorded.iloc[[2]], extra])

    result = reconcile(grid, recorded)
    assert not result.passed
    assert (result.n_missing, result.n_duplicated, result.n_extra) == (2, 1, 1)


def test_units_that_differ_only_in_tpat_are_distinct(two_logs):
    grid = expected_units(two_logs, campaign=1)
    recorded = _recorded(grid)
    forecast_t0 = (recorded["model_type"] == "forecast") & (recorded["tpat"] == 0.0)
    # Recording the tPAT 0 Forecast units as tPAT -1 duplicates the -1 rows and leaves
    # the 0 rows missing.
    recorded.loc[forecast_t0, "tpat"] = -1.0

    result = reconcile(grid, recorded)
    assert result.n_missing == result.n_duplicated == forecast_t0.sum()


def test_finished_units_near_the_budget_are_counted():
    recorded = pd.DataFrame(
        {
            "dataset": ["sepsis"] * 4,
            "guard_status": ["ok", "ok", "unavailable", "timeout"],
            "wall_s": [16 * 60, 60, 60, 20 * 60],
            "memory_peak_gb": [1.0, 9.5, 1.0, 1.0],
        }
    )
    near = near_budget(recorded, mem_gb=12.0, time_min=20.0)

    row = near.set_index("dataset").loc["sepsis"]
    # The timeout is not a finished unit.
    assert (row["n_finished"], row["n_over_75pct_time"]) == (3, 1)
    assert (row["n_over_75pct_memory"], row["n_near_budget"]) == (1, 2)


def test_every_global_oom_kill_matched_by_unit_kills_passes():
    recorded = pd.DataFrame(
        {"guard_status": ["ok", "oom", "oom"], "memcg_oom_kill": [0, 2, 1]}
    )
    result = oom_attribution(global_rise=3, recorded=recorded)
    assert result.passed and result.unattributed == 0


def test_an_unmatched_global_oom_kill_is_a_host_oom():
    recorded = pd.DataFrame({"guard_status": ["oom"], "memcg_oom_kill": [1]})
    result = oom_attribution(global_rise=2, recorded=recorded)
    assert not result.passed and result.unattributed == 1


def test_a_host_oom_unit_fails_the_gate():
    recorded = pd.DataFrame({"guard_status": ["host_oom"], "memcg_oom_kill": [1]})
    result = oom_attribution(global_rise=1, recorded=recorded)
    assert not result.passed and result.n_host_oom == 1
