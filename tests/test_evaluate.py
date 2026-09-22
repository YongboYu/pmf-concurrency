"""scripts/evaluate.py: resuming a run and reading its units back.

A resumed run skips the units already recorded, but only if the log, the models and the
budgets are the ones the run started with. A unit recorded twice, or a family name
other than the four, stops the reader rather than being pooled.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load_evaluate():
    spec = importlib.util.spec_from_file_location("evaluate", SCRIPTS / "evaluate.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _unit(model_type: str, **fields) -> dict:
    return {
        "dataset": "bpi2017",
        "window": 1,
        "model_type": model_type,
        "mode": "case_complete",
        "evaluator": "token_etc",
        "reduce_level": "tau",
        "guard_status": "ok",
        "wall_s": 1.0,
        "log_fitness": 0.9,
        "precision": 0.8,
        **fields,
    }


def _write_units(path: Path, units: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = _load_evaluate().CSV_COLUMNS
    pd.DataFrame(units).reindex(columns=columns).to_csv(path, index=False)
    return path


@pytest.fixture
def run_evaluate(tmp_path, monkeypatch):
    """Run the script's ``main`` on one window with a stub evaluator.

    Returns a runner that takes extra CLI arguments and gives back the families it
    evaluated, in order.
    """
    mod = _load_evaluate()
    weekly = tmp_path / "weekly_models"
    weekly.mkdir()
    for family in ("forecast", "static"):
        meta = {"start_date": "20210301", "end_date": "20210307", "type": family}
        (weekly / f"window_1_{family}.json").write_text(json.dumps({"metadata": meta}))
    out = tmp_path / "out"
    log = tmp_path / "log.xes"
    log.write_text("log")
    events = pd.DataFrame(
        {
            "case:concept:name": ["c1", "c1"],
            "concept:name": ["A", "B"],
            "time:timestamp": pd.to_datetime(["2021-03-02 08:00", "2021-03-02 09:00"]),
        }
    )

    evaluated: list[str] = []
    remaining_checked: list[bool] = []

    def _evaluate_cell(
        _evaluator, model_json, _events, _traces, consider_remaining, *_
    ):
        evaluated.append(model_json["metadata"]["type"])
        remaining_checked.append(consider_remaining)
        return "ok", {"log_fitness": 1.0, "precision": 1.0}

    monkeypatch.setattr(mod, "resolve_dataset", lambda _key: (str(log), out, weekly))
    monkeypatch.setattr(mod, "load_event_log", lambda _path: events)
    monkeypatch.setattr(mod, "enable_determinism", lambda: None)
    monkeypatch.setattr(mod, "_evaluate_cell", _evaluate_cell)

    def _run(*args: str) -> list[str]:
        monkeypatch.setattr(sys, "argv", ["evaluate.py", "--dataset", "bpi2017", *args])
        mod.main()
        return evaluated

    _run.res_csv = out / "evaluation.csv"  # type: ignore[attr-defined]
    _run.manifest = out / "evaluation_manifest.json"  # type: ignore[attr-defined]
    _run.weekly = weekly  # type: ignore[attr-defined]
    _run.remaining_checked = remaining_checked  # type: ignore[attr-defined]
    return _run


_ONE_UNIT = ("--evaluators", "token_etc", "--reduce-levels", "none")


def test_resume_skips_units_already_recorded(run_evaluate):
    run_evaluate("--model-types", "static", *_ONE_UNIT)

    evaluated = run_evaluate("--model-types", "forecast", "static", *_ONE_UNIT)

    assert evaluated == ["static", "forecast"]
    families = pd.read_csv(run_evaluate.res_csv)["model_type"]
    assert sorted(families) == ["forecast", "static"]


def test_the_manifest_records_the_inputs_and_the_resolved_budgets(
    run_evaluate, monkeypatch
):
    monkeypatch.delenv("PMF_GUARD_MEM_GB", raising=False)
    monkeypatch.delenv("PMF_GUARD_TIME_MIN", raising=False)
    run_evaluate("--model-types", "static", *_ONE_UNIT)

    manifest = json.loads(run_evaluate.manifest.read_text())
    assert manifest["mem_gb"] == 12.0
    assert manifest["time_min"] == 20.0
    assert len(manifest["log_sha256"]) == len(manifest["models_sha256"]) == 64


def test_resume_refuses_changed_models(run_evaluate):
    run_evaluate("--model-types", "static", *_ONE_UNIT)
    model = run_evaluate.weekly / "window_1_static.json"
    model.write_text(model.read_text() + " ")

    with pytest.raises(SystemExit, match="models_sha256"):
        run_evaluate("--model-types", "static", *_ONE_UNIT)


def test_resume_refuses_a_changed_budget(run_evaluate):
    run_evaluate("--model-types", "static", *_ONE_UNIT)

    with pytest.raises(SystemExit, match="time_min"):
        run_evaluate("--model-types", "static", "--time-min", "40", *_ONE_UNIT)
    assert len(pd.read_csv(run_evaluate.res_csv)) == 1


def test_resume_refuses_results_without_a_manifest(run_evaluate):
    _write_units(run_evaluate.res_csv, [_unit("static", reduce_level="none")])

    with pytest.raises(SystemExit, match=r"evaluation_manifest\.json"):
        run_evaluate("--model-types", "static", *_ONE_UNIT)


def test_a_repeated_argument_evaluates_each_unit_once(run_evaluate):
    evaluated = run_evaluate(
        "--model-types",
        "forecast",
        "--evaluators",
        "token_etc",
        "token_etc",
        "--reduce-levels",
        "none",
    )

    assert evaluated == ["forecast"]
    assert len(pd.read_csv(run_evaluate.res_csv)) == 1


@pytest.mark.parametrize(
    ("mode", "checked"),
    [("case_complete", True), ("window_raw", True), ("window_no_remain", False)],
)
def test_only_window_no_remain_ignores_remaining_tokens(run_evaluate, mode, checked):
    """A trace with remaining tokens is not a perfect fit, except in that one mode."""
    run_evaluate("--mode", mode, "--model-types", "forecast", "--reduce-levels", "none")

    assert set(run_evaluate.remaining_checked) == {checked}


def test_a_run_that_evaluates_nothing_says_so(run_evaluate, capsys):
    run_evaluate("--model-types", "direct_mined")

    assert "No units evaluated." in capsys.readouterr().out


def test_resume_rejects_unknown_families(run_evaluate):
    run_evaluate("--model-types", "static", *_ONE_UNIT)
    _write_units(run_evaluate.res_csv, [_unit("training", reduce_level="none")])

    with pytest.raises(ValueError, match="training"):
        run_evaluate("--model-types", "static", "--reduce-levels", "none")


def test_reader_rejects_a_unit_recorded_twice(tmp_path):
    """Two rows for one unit would count one window twice."""
    csv = _write_units(
        tmp_path / "evaluation.csv",
        [_unit("static", wall_s=1.0), _unit("static", wall_s=2.0), _unit("forecast")],
    )

    with pytest.raises(ValueError, match="static"):
        _load_evaluate().read_units(csv)


def test_reader_accepts_distinct_units(tmp_path):
    csv = _write_units(
        tmp_path / "evaluation.csv",
        [_unit("static"), _unit("static", window=2), _unit("forecast")],
    )

    assert len(_load_evaluate().read_units(csv)) == 3


def test_threads_default_to_eight_and_the_environment_wins(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "4")

    _load_evaluate()

    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert os.environ["MKL_NUM_THREADS"] == "4"


def test_a_window_ends_after_the_last_instant_of_its_last_day():
    start, end = _load_evaluate().window_bounds(
        {"start_date": "20210301", "end_date": "20210307"}
    )

    assert start == pd.Timestamp("2021-03-01")
    assert pd.Timestamp("2021-03-07 23:59:59.5") <= end < pd.Timestamp("2021-03-08")
