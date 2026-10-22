"""The history models, built end to end on a small synthetic log.

The log is the pipeline tests' log plus one case that crosses midnight into the first
evaluation window. Only the Graphviz render is stubbed out.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pm4py
import pytest

from pmf_concurrency import forecasting, history_models, pipeline, weekly_models
from pmf_concurrency.campaign import check_models, windows_of
from pmf_concurrency.data import load_event_log
from pmf_concurrency.families import model_json_path
from pmf_concurrency.paths import OUTPUT_BASE_ENV
from pmf_concurrency.structural_metrics import (
    STRUCTURAL_METRICS_FILENAME,
    read_structural_metrics,
)
from tests.test_pipeline import (
    DATASET_DIR,
    END,
    HORIZON,
    ROOT,
    START,
    TRAIN_RATIO,
    _fake_chronos_forecast,
    _synthetic_events,
)

#: An activity only the crossing case has, after midnight.
CROSSING_ONLY = "X"


def _events_with_a_case_crossing_into_window_1() -> pd.DataFrame:
    events = _synthetic_events()
    first_day = pd.Timestamp(
        windows_of(events, TRAIN_RATIO, HORIZON)["start_date"].iloc[0]
    )
    crossing = [
        ("crossing", activity, first_day + pd.Timedelta(hours=hours))
        for activity, hours in (
            (START, -2),
            ("A", -1),
            (CROSSING_ONLY, 1),
            ("D", 2),
            (END, 3),
        )
    ]
    return pd.concat(
        [events, pd.DataFrame(crossing, columns=events.columns)], ignore_index=True
    )


@pytest.fixture(scope="module")
def history_log(tmp_path_factory) -> Path:
    """The log as an XES file, written once per module."""
    path = tmp_path_factory.mktemp("log") / "Synth.xes"
    events = _events_with_a_case_crossing_into_window_1()
    pm4py.write_xes(pm4py.format_dataframe(events), str(path))
    return path


def _stub_render(monkeypatch) -> None:
    monkeypatch.setattr(
        weekly_models,
        "render_petri_json_to_pdf",
        lambda _src, pdf_path: bool(Path(pdf_path).write_text("pdf")),
    )


@pytest.fixture
def build(history_log, monkeypatch, tmp_path):
    """Build the history models into ``tmp_path/<label>/Synth_pmf``."""
    _stub_render(monkeypatch)

    def _build(label: str = "history", **kwargs: Any) -> tuple[dict[str, Any], Path]:
        output_dir = tmp_path / label / DATASET_DIR
        summary = history_models.build_history_models(
            log_path=str(history_log),
            output_dir=str(output_dir),
            train_ratio=TRAIN_RATIO,
            horizon=HORIZON,
            **kwargs,
        )
        return summary, output_dir

    return _build


def test_every_window_gets_an_expanding_net_the_campaign_check_accepts(
    build, history_log
) -> None:
    summary, output_dir = build()

    windows = windows_of(load_event_log(str(history_log)), TRAIN_RATIO, HORIZON)
    expected = windows.assign(dataset="synth", tpat=0.0, model_type="expanding")
    weekly = output_dir / "weekly_models"
    assert check_models(expected, {("synth", 0.0): weekly}) == []

    metrics = read_structural_metrics(str(output_dir / STRUCTURAL_METRICS_FILENAME))
    assert list(metrics["model_family"].astype(str)) == ["expanding"] * len(windows)
    assert list(metrics["window"]) == list(windows["window"])
    assert summary["thresholds"]["tPAT"] == 0.0
    assert summary["structural_metrics"]["families"] == ["expanding"]


def _model(output_dir: Path, window: int, family: str) -> dict[str, Any]:
    path = model_json_path(output_dir / "weekly_models", window, family)
    return json.loads(path.read_text())


def test_expanding_in_window_1_is_static_and_the_periods_are_the_pipelines(
    build, history_log, monkeypatch, tmp_path
) -> None:
    summary, history_dir = build()
    monkeypatch.setattr(pipeline, "chronos_sliding_forecast", _fake_chronos_forecast)
    pipeline_dir = tmp_path / "pipeline" / DATASET_DIR
    pipeline_summary = pipeline.run_pmf_experiment(
        log_path=str(history_log),
        output_dir=str(pipeline_dir),
        train_ratio=TRAIN_RATIO,
        horizon=HORIZON,
    )

    expanding = _model(history_dir, 1, "expanding")
    static = _model(pipeline_dir, 1, "static")
    assert expanding["graph"] == static["graph"]
    assert expanding["metadata"]["traces"] == static["metadata"]["training_traces"]

    periods = ("train_period", "test_period", "test_days")
    assert [summary[k] for k in periods] == [pipeline_summary[k] for k in periods]


def _tasks(model: dict[str, Any]) -> set[str]:
    return {n["id"] for n in model["graph"]["nodes"] if not n["data"]["is_binding"]}


def test_no_history_reaches_into_its_window(build, history_log) -> None:
    _summary, output_dir = build()
    first_day = load_event_log(str(history_log))["date"].min().strftime("%Y%m%d")

    models = sorted((output_dir / "weekly_models").glob("window_*_expanding.json"))
    assert models
    for path in models:
        meta = json.loads(path.read_text())["metadata"]
        day_before = (pd.Timestamp(meta["start_date"]) - pd.Timedelta(days=1)).strftime(
            "%Y%m%d"
        )
        assert (meta["history_start"], meta["history_end"]) == (first_day, day_before)

    # The crossing case brings only its events before midnight into window 1's history.
    assert CROSSING_ONLY not in _tasks(_model(output_dir, 1, "expanding"))
    assert CROSSING_ONLY in _tasks(_model(output_dir, 2, "expanding"))


def test_a_second_build_into_the_same_directory_is_refused(build) -> None:
    _summary, output_dir = build()
    before = {p: p.read_bytes() for p in output_dir.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError, match="not empty"):
        build()
    assert {p: p.read_bytes() for p in output_dir.rglob("*") if p.is_file()} == before


def test_the_build_takes_the_pipeline_configuration_and_runs_no_forecast(
    history_log, monkeypatch, tmp_path
) -> None:
    """Through the Hydra entry point, with Chronos-2 unusable."""
    from hydra import compose, initialize_config_dir

    _stub_render(monkeypatch)

    def _no_chronos(*_args, **_kwargs):
        raise AssertionError("Chronos-2 was used")

    monkeypatch.setattr(forecasting, "load_chronos", _no_chronos)
    monkeypatch.setattr(pipeline, "chronos_sliding_forecast", _no_chronos)

    conf = tmp_path / "conf"
    shutil.copytree(ROOT / "conf", conf)
    (conf / "dataset" / "synth.yaml").write_text(
        f'name: Synth\nlog_path: "{history_log}"\n'
    )
    monkeypatch.setenv(OUTPUT_BASE_ENV, "outputs_history")
    monkeypatch.chdir(tmp_path)
    with initialize_config_dir(config_dir=str(conf), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset=synth",
                f"experiment.train_ratio={TRAIN_RATIO}",
                f"experiment.horizon={HORIZON}",
                "experiment.tPAT=-1.0",
            ],
        )
    history_models.main.__wrapped__(cfg)

    output_dir = tmp_path / "outputs_history" / DATASET_DIR
    summary = json.loads((output_dir / "experiment_summary.json").read_text())
    assert summary["thresholds"]["tPAT"] == -1.0
    assert summary["structural_metrics"]["windows"] == len(
        windows_of(load_event_log(str(history_log)), TRAIN_RATIO, HORIZON)
    )
    assert not (tmp_path / "outputs").exists(), "leaked into the default root"
