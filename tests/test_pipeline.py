"""The pipeline, run end to end on a small synthetic log.

A run ends at the weekly models and ``structural_metrics.csv``. It builds no net for
the whole test period, computes no forecast error and runs no evaluation. Only the
Chronos-2 forecast and the Graphviz render are stubbed out.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pm4py
import pytest

from pmf_concurrency import (
    conformance,
    entropic_relevance,
    guard,
    pipeline,
    weekly_models,
)
from pmf_concurrency.paths import OUTPUT_BASE_ENV
from pmf_concurrency.structural_metrics import (
    STRUCTURAL_METRICS_FILENAME,
    StructuralStats,
    read_structural_metrics,
    structural_metrics_row,
    write_structural_metrics,
)

ROOT = Path(__file__).resolve().parent.parent

#: Every log the pipeline loads starts and ends each case with these markers.
START, END = "▶", "■"

#: Small enough to mine in under a second, long enough to leave several evaluation
#: windows after the split.
TRAIN_RATIO, HORIZON = 0.75, 2

#: The dataset label comes from the output directory's name.
DATASET_DIR = "Synth_pmf"

#: Outputs a run does not write.
_NOT_WRITTEN = (
    "median_over_levels",
    "forecasted_model.json",
    "direct_mined_model.json",
    "static_model.json",
    "forecast_metrics.json",
    "forecast_metrics_per_type.csv",
    "forecast_metrics_summary.csv",
    "forecast_AXB",
    "forecast_AXB_summary.csv",
)


def _synthetic_events(days: int = 20, cases_per_day: int = 3) -> pd.DataFrame:
    """A log with real concurrency: B and C swap order from case to case.

    This makes the AND bindings non-zero, so the tests can tell written metrics from
    written zeros.
    """
    rows: list[tuple[str, str, pd.Timestamp]] = []
    day0 = pd.Timestamp("2021-03-01")
    for day_offset in range(days):
        day = day0 + pd.Timedelta(days=day_offset)
        for case_offset in range(cases_per_day):
            middle = (
                ["A", "B", "C"]
                if (day_offset + case_offset) % 2 == 0
                else ["A", "C", "B"]
            )
            for i, activity in enumerate([START, *middle, "D", END]):
                rows.append(
                    (
                        f"c{day_offset:02d}-{case_offset}",
                        activity,
                        day + pd.Timedelta(hours=8 + i),
                    )
                )
    return pd.DataFrame(
        rows, columns=["case:concept:name", "concept:name", "time:timestamp"]
    )


@pytest.fixture(scope="module")
def synthetic_log(tmp_path_factory) -> Path:
    """The log as an XES file, written once per module."""
    path = tmp_path_factory.mktemp("log") / "Synth.xes"
    pm4py.write_xes(pm4py.format_dataframe(_synthetic_events()), str(path))
    return path


def _fake_chronos_forecast(
    train_series: pd.DataFrame,
    test_series: pd.DataFrame,
    horizon: int = 7,
    device: str = "cpu",
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """A flat-mean forecast with Chronos-2's return shape, deterministic and instant."""
    if train_series.empty or test_series.empty:
        return {}
    common = sorted(set(train_series.columns) & set(test_series.columns))
    n_sequences = len(test_series) - horizon + 1
    if n_sequences <= 0:
        return {}
    return {
        column: np.full((n_sequences, horizon), float(train_series[column].mean()))
        for column in common
    }


def _stub_forecast_and_render(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "chronos_sliding_forecast", _fake_chronos_forecast)
    monkeypatch.setattr(
        weekly_models,
        "render_petri_json_to_pdf",
        lambda _src, pdf_path: bool(Path(pdf_path).write_text("pdf")),
    )


@pytest.fixture
def run_pipeline(synthetic_log, monkeypatch, tmp_path):
    """Run the real pipeline into ``tmp_path/<label>/Synth_pmf``."""
    _stub_forecast_and_render(monkeypatch)

    def _run(label: str, **kwargs: Any) -> tuple[dict[str, Any], Path]:
        output_dir = tmp_path / label / DATASET_DIR
        summary = pipeline.run_pmf_experiment(
            log_path=str(synthetic_log),
            output_dir=str(output_dir),
            train_ratio=TRAIN_RATIO,
            horizon=HORIZON,
            **kwargs,
        )
        return summary, output_dir

    return _run


def test_a_run_ends_at_the_weekly_models_and_their_metrics(run_pipeline) -> None:
    summary, output_dir = run_pipeline("run")

    metrics = read_structural_metrics(str(output_dir / STRUCTURAL_METRICS_FILENAME))
    assert set(metrics["model_family"].astype(str)) == {
        "forecast",
        "re_mined",
        "direct_mined",
        "static",
    }
    assert (
        metrics.loc[metrics["model_family"] == "forecast", "and_bindings"] > 0
    ).all()

    weekly = output_dir / "weekly_models"
    assert any(weekly.glob("window_*_forecast.json"))
    for name in _NOT_WRITTEN:
        assert not (output_dir / name).exists(), f"unexpected output: {name}"

    assert set(summary) == {
        "train_period",
        "test_period",
        "test_days",
        "thresholds",
        "structural_metrics",
    }
    assert summary["structural_metrics"]["rows"] == len(metrics)
    assert summary["structural_metrics"]["windows"] == metrics["window"].nunique()
    # A window of HORIZON days starts on every test day but the last HORIZON - 1.
    assert summary["test_days"] == metrics["window"].nunique() + HORIZON - 1
    written = json.loads((output_dir / "experiment_summary.json").read_text())
    assert written == summary


@pytest.mark.parametrize("tPAT", [0.0, -1.0])
def test_tpat_comes_from_the_caller(run_pipeline, tPAT) -> None:
    _summary, output_dir = run_pipeline("run", tPAT=tPAT)
    metrics = read_structural_metrics(str(output_dir / STRUCTURAL_METRICS_FILENAME))
    assert (metrics["tPAT"] == tPAT).all()

    forecast = metrics[metrics["model_family"] == "forecast"]
    if tPAT == -1.0:
        # -1.0 keeps every binding.
        assert (forecast["filter_bindings_removed"] == 0).all()
    else:
        assert (forecast["filter_bindings_removed"] > 0).any()


def test_a_second_run_into_the_same_directory_is_refused(run_pipeline) -> None:
    """Old window files would otherwise survive next to the new ones."""
    _summary, output_dir = run_pipeline("run")
    before = sorted(p.name for p in output_dir.rglob("*"))
    with pytest.raises(FileExistsError, match="not empty"):
        run_pipeline("run")
    assert sorted(p.name for p in output_dir.rglob("*")) == before


def test_an_empty_existing_directory_is_accepted(run_pipeline, tmp_path) -> None:
    (tmp_path / "run" / DATASET_DIR).mkdir(parents=True)
    summary, _output_dir = run_pipeline("run")
    assert summary["structural_metrics"]["rows"] > 0


def test_no_conformance_er_or_guarded_unit_runs(run_pipeline, monkeypatch) -> None:
    """Evaluation runs in a separate script, never inside the pipeline."""

    def _forbidden(name: str):
        def _raise(*_args, **_kwargs):
            raise AssertionError(f"{name} ran inside the pipeline")

        return _raise

    for module, attribute in (
        (conformance, "conformance_metrics"),
        (entropic_relevance, "compute_er"),
        (guard, "run_guarded"),
    ):
        monkeypatch.setattr(module, attribute, _forbidden(attribute))

    _summary, output_dir = run_pipeline("run")
    assert not list(output_dir.glob("*conformance*"))
    assert not list(output_dir.glob("*entropic*"))


def _hydra_call(monkeypatch, *overrides: str) -> dict[str, Any]:
    """The keyword arguments ``pipeline.main()`` passes to ``run_pmf_experiment``."""
    from hydra import compose, initialize_config_dir

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        pipeline, "run_pmf_experiment", lambda **kwargs: captured.update(kwargs)
    )
    monkeypatch.chdir(ROOT)

    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=list(overrides))
    pipeline.main.__wrapped__(cfg)
    return captured


def _fake_forecast_with_middle(scale: dict[str, float]):
    """The flat-mean forecast, recording ``scale[series key]`` times it as the sorted
    middle (1 where the key is not named)."""

    def _forecast(
        train, test, horizon=7, device="cpu", seed=42, median_over_levels=None
    ):
        point = _fake_chronos_forecast(train, test, horizon, device, seed)
        if median_over_levels is not None:
            for key, rows in point.items():
                median_over_levels[key] = rows * scale.get(key, 1.0)
        return point

    return _forecast


def test_the_sorted_middle_is_saved_only_when_asked(run_pipeline, monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline, "chronos_sliding_forecast", _fake_forecast_with_middle({})
    )
    _summary, output_dir = run_pipeline("run", record_median_over_levels=True)

    middle = output_dir / "median_over_levels"
    assert (middle / "forecast_A_summary.csv").exists()
    point = sorted(p.name for p in (output_dir / "forecast_A").glob("*.npy"))
    assert sorted(p.name for p in (middle / "forecast_A").glob("*.npy")) == point


def test_median_choice_counts_what_the_point_forecast_changes(
    run_pipeline, monkeypatch
) -> None:
    """Decision 1.6: days, weekly counts, and nets whose topology or activity
    frequencies differ between the 0.5 level and the sorted middle."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "median_choice", ROOT / "scripts" / "median_choice.py"
    )
    assert spec and spec.loader
    median_choice = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(median_choice)

    monkeypatch.setattr(
        pipeline, "chronos_sliding_forecast", _fake_forecast_with_middle({})
    )
    _summary, same = run_pipeline("same", record_median_over_levels=True)
    counts = median_choice.median_choice_counts(same)
    assert counts["n_windows"] > 0
    for key in (
        "n_values_differ",
        "n_weekly_counts_differ",
        "n_nets_topology_differ",
        "n_nets_frequencies_differ",
    ):
        assert counts[key] == 0, key

    # Doubling one activity's count moves only the frequencies.
    monkeypatch.setattr(
        pipeline, "chronos_sliding_forecast", _fake_forecast_with_middle({"B": 2.0})
    )
    _summary, freq = run_pipeline("freq", record_median_over_levels=True)
    counts = median_choice.median_choice_counts(freq)
    assert counts["n_values_differ"] > 0
    assert counts["max_weekly_relative_change"] == pytest.approx(1.0)
    assert counts["n_nets_frequencies_differ"] == counts["n_windows"]
    assert counts["n_nets_topology_differ"] == 0
    assert counts["n_weekly_counts_enter_or_leave"] == 0

    # Zeroing every relation count empties the nets.
    zero = dict.fromkeys(_relation_keys(freq), 0.0)
    monkeypatch.setattr(
        pipeline, "chronos_sliding_forecast", _fake_forecast_with_middle(zero)
    )
    _summary, topo = run_pipeline("topo", record_median_over_levels=True)
    counts = median_choice.median_choice_counts(topo)
    assert counts["n_nets_topology_differ"] == counts["n_windows"]
    # A series whose count drops to 0 leaves the net.
    assert counts["n_weekly_counts_enter_or_leave"] > 0


def _relation_keys(output_dir: Path) -> list[str]:
    return pd.read_csv(output_dir / "forecast_AB_summary.csv")["series_key"].tolist()


def test_the_summary_lists_the_history_families_a_run_holds(tmp_path) -> None:
    stats = StructuralStats(
        tasks=3,
        arcs=2,
        input_bindings=2,
        input_and_bindings=0,
        output_bindings=2,
        output_and_bindings=1,
    )
    rows = [
        structural_metrics_row(1, family, stats, dataset="Synth", tPAT=0.0)
        for family in ("expanding", "persistence")
    ]
    write_structural_metrics(str(tmp_path), rows)

    summary = pipeline.write_summary(str(tmp_path), {}, {"tPAT": 0.0})

    assert summary["structural_metrics"]["families"] == ["persistence", "expanding"]


def test_tpat_on_the_command_line_reaches_the_pipeline(monkeypatch) -> None:
    captured = _hydra_call(monkeypatch, "dataset=sepsis", "experiment.tPAT=-1.0")
    assert captured["tPAT"] == -1.0


def test_a_run_writes_under_the_campaign_root(monkeypatch) -> None:
    monkeypatch.setenv(OUTPUT_BASE_ENV, "outputs_tpatm1")
    captured = _hydra_call(monkeypatch, "dataset=sepsis")
    assert captured["output_dir"] == str(Path("outputs_tpatm1/Sepsis_pmf"))


def test_the_metrics_file_lands_inside_that_root(
    synthetic_log, monkeypatch, tmp_path
) -> None:
    """Through the Hydra entry point, the CSV lands under ``$PMF_OUTPUT_BASE``."""
    conf = tmp_path / "conf"
    shutil.copytree(ROOT / "conf", conf)
    dataset_yaml = conf / "dataset" / "synth.yaml"
    dataset_yaml.write_text(f'name: Synth\nlog_path: "{synthetic_log}"\n')

    monkeypatch.setenv(OUTPUT_BASE_ENV, "outputs_structural")
    monkeypatch.chdir(tmp_path)
    _stub_forecast_and_render(monkeypatch)

    from hydra import compose, initialize_config_dir

    if str(ROOT / "src") not in sys.path:  # pragma: no cover - editable install
        sys.path.insert(0, str(ROOT / "src"))
    with initialize_config_dir(config_dir=str(conf), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset=synth",
                f"experiment.train_ratio={TRAIN_RATIO}",
                f"experiment.horizon={HORIZON}",
            ],
        )
    pipeline.main.__wrapped__(cfg)

    output_dir = tmp_path / "outputs_structural" / "Synth_pmf"
    metrics_path = output_dir / STRUCTURAL_METRICS_FILENAME
    assert metrics_path.exists()
    assert not read_structural_metrics(str(metrics_path)).empty
    assert not (output_dir / "forecasted_model.json").exists()
    assert not (tmp_path / "outputs").exists(), "leaked into the default root"
