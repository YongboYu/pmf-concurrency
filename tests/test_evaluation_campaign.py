"""scripts/evaluation_campaign.py: the batches of a campaign and its gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from pmf_concurrency.campaign import expected_units, read_expected

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # dataclasses look their module up in sys.modules.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def driver():
    return _load("evaluation_campaign")


@pytest.fixture
def grid() -> pd.DataFrame:
    windows = pd.DataFrame(
        {
            "window": [1, 2],
            "start_date": ["20210301", "20210302"],
            "end_date": ["20210307", "20210308"],
        }
    )
    return expected_units({"bpi2017": windows, "bpi2019": windows}, campaign=1)


def test_the_grid_command_writes_campaign_3_once(driver, monkeypatch, tmp_path):
    # Two events 50 days apart stand in for the log. Writing an XES file with pm4py
    # would leave a child process running, which test_guard.py then finds.
    events = pd.DataFrame(
        {
            "case:concept:name": ["c1", "c2"],
            "concept:name": ["A", "B"],
            "time:timestamp": pd.to_datetime(["2021-01-01", "2021-02-20"]),
        }
    )
    monkeypatch.setattr(
        driver, "resolve_dataset", lambda key: (f"{key}.xes", tmp_path / key, None)
    )
    monkeypatch.setattr(driver, "load_event_log", lambda _path: events)
    out = tmp_path / "c3" / "expected_units.csv"
    argv = ["grid", "--campaign", "3", "--out", str(out), "--datasets", "sepsis"]
    monkeypatch.setattr(sys, "argv", ["evaluation_campaign.py", *argv])

    assert driver.main() == 0
    grid = read_expected(out)
    # 50 days at the configured 80 % and 7-day horizon leave 4 windows.
    assert len(grid) == 4 * 2 * 3
    assert set(grid["model_type"]) == {"persistence", "expanding"}
    assert set(grid["cohort"]) == {"history"}

    written = out.read_bytes()
    with pytest.raises(SystemExit, match="written once"):
        driver.main()
    assert out.read_bytes() == written


def test_the_grid_command_writes_campaign_4(driver, monkeypatch, tmp_path):
    events = pd.DataFrame(
        {
            "case:concept:name": ["c1", "c2"],
            "concept:name": ["A", "B"],
            "time:timestamp": pd.to_datetime(["2021-01-01", "2021-02-20"]),
        }
    )
    monkeypatch.setattr(
        driver, "resolve_dataset", lambda key: (f"{key}.xes", tmp_path / key, None)
    )
    monkeypatch.setattr(driver, "load_event_log", lambda _path: events)
    out = tmp_path / "c4" / "expected_units.csv"
    argv = ["grid", "--campaign", "4", "--out", str(out), "--datasets", "sepsis"]
    monkeypatch.setattr(sys, "argv", ["evaluation_campaign.py", *argv])

    assert driver.main() == 0
    grid = read_expected(out)
    assert len(grid) == 4 * 2 * 3
    assert set(grid["model_type"]) == {"persistence", "expanding"}
    assert set(grid["reduce_level"]) == {"none"}
    assert set(grid["cohort"]) == {"history"}


def test_one_batch_per_log_tpat_family_evaluator_mode_and_level(driver, grid):
    batches = driver.plan_batches(grid)

    # 2 logs x 6 arms x 3 evaluators, each over both windows.
    assert len(batches) == 2 * 6 * 3
    assert all(batch.n_units == 2 for batch in batches)
    assert len({batch.name for batch in batches}) == len(batches)


def test_the_heaviest_batches_start_first(driver, grid):
    batches = driver.plan_batches(grid)

    assert (batches[0].dataset, batches[0].evaluator) == ("bpi2019", "er")
    assert (batches[-1].dataset, batches[-1].evaluator) == ("bpi2017", "token_etc")


def test_a_batch_runs_evaluate_on_its_units_only(driver, grid, tmp_path):
    batch = next(
        j
        for j in driver.plan_batches(grid)
        if (j.dataset, j.tpat, j.model_type, j.evaluator)
        == ("bpi2017", -1.0, "re_mined", "alignment")
    )
    argv = driver.batch_command(
        batch,
        campaign_dir=tmp_path,
        weekly_dir=tmp_path / "w",
        concurrency=3,
        development=False,
    )

    assert argv[:2] == ["bash", "scripts/local/delegated_scope.sh"]
    joined = " ".join(argv)
    for part in (
        "--dataset bpi2017",
        f"--weekly-dir {tmp_path / 'w'}",
        f"--expected {tmp_path / 'expected_units.csv'}",
        "--model-types re_mined",
        "--evaluators alignment",
        "--reduce-levels tau",
        "--mode case_complete",
        "--concurrency 3",
    ):
        assert part in joined
    assert "--development" not in argv
    assert str(batch.out_dir(tmp_path)) in joined


def test_the_oom_kill_rise_is_summed_per_session(driver):
    samples = pd.DataFrame(
        {
            "session": [1, 1, 1, 2, 2],
            "oom_kill": [10, 11, 12, 15, 16],
        }
    )
    # Kills between sessions happened while no unit ran.
    assert driver.oom_kill_rise(samples) == 2 + 1


def test_a_resume_with_another_concurrency_is_refused(driver, tmp_path):
    driver.check_campaign_manifest(tmp_path, {"concurrency": 3, "git_commit": "a"})

    with pytest.raises(SystemExit, match="concurrency"):
        driver.check_campaign_manifest(tmp_path, {"concurrency": 2, "git_commit": "a"})
    driver.check_campaign_manifest(tmp_path, {"concurrency": 3, "git_commit": "a"})


# The gate over a finished campaign


SHARED = {"git_commit": "abc", "concurrency": 3, "mem_gb": 12.0, "time_min": 20.0}


def _finish(driver, grid, campaign_dir: Path, *, drop: int = 0, **manifest) -> None:
    evaluate = _load("evaluate")
    campaign_dir.mkdir(exist_ok=True)
    grid.to_csv(campaign_dir / "expected_units.csv", index=False)
    (campaign_dir / "campaign_manifest.json").write_text(json.dumps(SHARED))
    pd.DataFrame({"session": [1, 1], "oom_kill": [5, 6]}).to_csv(
        campaign_dir / "oom_kill_samples.csv", index=False
    )
    for batch in driver.plan_batches(grid):
        out = batch.out_dir(campaign_dir)
        out.mkdir(parents=True)
        units = batch.units.drop(columns=["start_date", "end_date", "cohort"]).assign(
            guard_status="ok", wall_s=1.0, memory_peak_gb=0.5, memcg_oom_kill=0
        )
        units.reindex(columns=evaluate.CSV_COLUMNS).to_csv(
            out / "evaluation.csv", index=False
        )
        (out / "evaluation_manifest.json").write_text(
            json.dumps(
                {
                    **SHARED,
                    "enforcement": "cgroup",
                    "enforcement_in_force": True,
                    **manifest,
                }
            )
        )
    # One unit was OOM-killed by its own cgroup, which explains the global rise.
    first = driver.plan_batches(grid)[0].out_dir(campaign_dir) / "evaluation.csv"
    rows = pd.read_csv(first)
    rows.loc[0, ["guard_status", "memcg_oom_kill"]] = ["oom", 1]
    rows.iloc[drop:].to_csv(first, index=False)


def test_a_complete_campaign_passes_the_gate(driver, grid, tmp_path):
    _finish(driver, grid, tmp_path)

    report = driver.gate(tmp_path)

    assert report["passed"], report
    assert report["units"] == {"missing": 0, "duplicated": 0, "extra": 0}
    assert report["status_counts"] == {"ok": len(grid) - 1, "oom": 1}
    assert (tmp_path / "evaluation.csv").exists()
    assert (tmp_path / "near_budget.csv").exists()


def test_a_missing_unit_fails_the_gate(driver, grid, tmp_path):
    _finish(driver, grid, tmp_path, drop=1)

    report = driver.gate(tmp_path)

    assert not report["passed"]
    assert report["units"]["missing"] == 1


def test_a_batch_without_the_cgroup_fails_the_gate(driver, grid, tmp_path):
    _finish(driver, grid, tmp_path, enforcement="rlimit_as")

    report = driver.gate(tmp_path)

    assert not report["passed"]
    assert any("enforcement" in p for p in report["manifest_problems"])


def test_a_batch_from_other_code_fails_the_gate(driver, grid, tmp_path):
    _finish(driver, grid, tmp_path, git_commit="def")

    report = driver.gate(tmp_path)

    assert not report["passed"]
    assert any("git_commit" in p for p in report["manifest_problems"])


def test_an_unattributed_oom_kill_fails_the_gate(driver, grid, tmp_path):
    _finish(driver, grid, tmp_path)
    pd.DataFrame({"session": [1, 1], "oom_kill": [5, 7]}).to_csv(
        tmp_path / "oom_kill_samples.csv", index=False
    )

    report = driver.gate(tmp_path)

    assert not report["passed"]
    assert report["oom"]["unattributed"] == 1


def test_a_unit_recorded_twice_is_reported_not_raised(driver, grid, tmp_path):
    _finish(driver, grid, tmp_path)
    csv = driver.plan_batches(grid)[-1].out_dir(tmp_path) / "evaluation.csv"
    rows = pd.read_csv(csv)
    pd.concat([rows, rows.iloc[[0]]]).to_csv(csv, index=False)

    report = driver.gate(tmp_path)

    assert not report["passed"]
    assert report["units"]["duplicated"] == 1


def test_a_campaign_with_nothing_recorded_yet_has_every_unit_missing(
    driver, grid, tmp_path
):
    _finish(driver, grid, tmp_path)
    for batch in driver.plan_batches(grid):
        (batch.out_dir(tmp_path) / "evaluation.csv").unlink()

    report = driver.gate(tmp_path)

    assert report["units"]["missing"] == len(grid)
    assert not report["passed"]


def test_the_summary_counts_expected_units_not_recorded_rows(driver, grid, tmp_path):
    _finish(driver, grid, tmp_path, drop=1)

    driver.gate(tmp_path)

    summary = pd.read_csv(tmp_path / "evaluation_summary.csv")
    assert summary["n_expected"].sum() == len(grid)
    assert "cohort" in summary.columns
    assert "cohort" in pd.read_csv(tmp_path / "near_budget.csv").columns


def test_lanes_must_fit_with_their_parents_holding_the_log(driver):
    """Each parent holds its log before the unit starts; the cap comes on top."""
    # 3 x (4.53 + 12) + reserve > 46.8 GiB, the office box.
    with pytest.raises(SystemExit, match="bpi2019"):
        driver.check_concurrency(["bpi2019", "sepsis"], 3, mem_gb=12.0, ram_gb=46.8)
    driver.check_concurrency(["bpi2019", "sepsis"], 2, mem_gb=12.0, ram_gb=46.8)
    driver.check_concurrency(["bpi2017"], 3, mem_gb=12.0, ram_gb=46.8)


def test_a_log_without_a_measured_parent_counts_as_the_largest(driver):
    with pytest.raises(SystemExit):
        driver.check_concurrency(["new_log"], 3, mem_gb=12.0, ram_gb=46.8)
