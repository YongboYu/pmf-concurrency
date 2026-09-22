"""A few evaluation windows of BPI2017 and Sepsis, checked byte for byte.

The expected model files and structural-metrics rows come from a full reference run at
both tPAT values: the campaign behind the paper's results, on the preprocessed logs with
the pinned Chronos-2 revision. The inputs are that run's Chronos-2 forecasts for the
fixture windows and the log's observed daily counts. Windows outside the fixture get
zeros and are not checked. The fixture should change only with an intended change of
results.

BPI2017 has events on every day of its test period. The Sepsis windows are a dense week
(1), the first week with an empty day (51) and a week with a single active day (62), and
they include length-2 loops.

Without ``data/`` only Forecast and Re-mined are checked, from the stored counts. With
``data/`` the whole pipeline runs on the log, with only Chronos-2 replaced by the stored
forecasts, and all four families are checked.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pmf_concurrency import pipeline, weekly_models
from pmf_concurrency.structural_metrics import STRUCTURAL_METRICS_FILENAME

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"

DATASETS = ("BPI2017", "Sepsis")
TAUS = (0.0, -1.0)
SERIES_TYPES = ("A", "AB", "ABA", "IB", "OB")
FAMILIES = ("forecast", "re_mined", "direct_mined", "static")
THRESHOLDS = {"tD": 0.5, "tL1L": 0.5, "tL2L": 0.5}


@cache
def _fixture(dataset: str) -> dict[str, Any]:
    path = FIXTURES / f"{dataset}_windows.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _forecast_arrays(
    fixture: dict[str, Any], series_type: str, n_sequences: int
) -> dict[str, np.ndarray]:
    """Chronos-2's output shape, with the stored rows and zeros elsewhere."""
    arrays = {}
    for key, rows in fixture["forecast"][series_type].items():
        array = np.zeros((n_sequences, fixture["horizon"]), dtype=np.float32)
        for window, values in rows.items():
            array[int(window) - 1] = values
        arrays[key] = array
    return arrays


def _observed_frames(fixture: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Daily test-period counts from the first day to the last fixture window."""
    n_days = max(fixture["windows"]) + fixture["horizon"] - 1
    index = pd.date_range(fixture["test_start"], periods=n_days, freq="D")
    frames = {}
    for series_type in SERIES_TYPES:
        days = fixture["observed"][series_type]
        columns = sorted({key for counts in days.values() for key in counts})
        frame = pd.DataFrame(0, index=index, columns=columns)
        for day, counts in days.items():
            for key, count in counts.items():
                frame.at[index[int(day)], key] = count
        frames[series_type] = frame
    return frames


def _stored_forecast(fixture: dict[str, Any]) -> Callable[..., dict[str, np.ndarray]]:
    """Stands in for Chronos-2 and checks that the series match the reference run."""

    def forecast(
        train_series: pd.DataFrame,
        test_series: pd.DataFrame,
        horizon: int = 7,
        device: str = "cpu",
        seed: int = 42,
    ) -> dict[str, np.ndarray]:
        if train_series.empty or test_series.empty:
            return {}
        columns = sorted(set(train_series.columns) & set(test_series.columns))
        matches = [t for t in SERIES_TYPES if sorted(fixture["forecast"][t]) == columns]
        assert len(matches) == 1, f"no stored series set matches {columns[:3]}..."
        n_sequences = len(test_series) - horizon + 1
        assert n_sequences == fixture["n_windows"], (
            f"{matches[0]}: {n_sequences} evaluation windows, "
            f"the reference run has {fixture['n_windows']}"
        )
        return _forecast_arrays(fixture, matches[0], n_sequences)

    return forecast


def _skip_render(monkeypatch) -> None:
    monkeypatch.setattr(weekly_models, "render_petri_json_to_pdf", lambda *_: False)


def _metric_rows(text: str, windows: list[int], families: tuple[str, ...]) -> list:
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    window, family = header.index("window"), header.index("model_family")
    rows = [r for r in reader if int(r[window]) in windows and r[family] in families]
    return [header, *rows]


def _assert_matches_reference(
    fixture: dict[str, Any], tau: float, output_dir: Path, families: tuple[str, ...]
) -> None:
    expected = fixture["expected"][str(tau)]
    windows = fixture["windows"]
    for window in windows:
        for family in families:
            name = f"window_{window}_{family}.json"
            actual = (output_dir / "weekly_models" / name).read_text()
            assert actual == expected["weekly_models"][name], name
    actual_rows = _metric_rows(
        (output_dir / STRUCTURAL_METRICS_FILENAME).read_text(), windows, families
    )
    expected_rows = _metric_rows(expected["structural_metrics.csv"], windows, families)
    assert len(actual_rows) == 1 + len(windows) * len(families)
    assert actual_rows == expected_rows


@pytest.mark.parametrize("tau", TAUS)
@pytest.mark.parametrize("dataset", DATASETS)
def test_reconstruction_matches_reference(dataset, tau, tmp_path, monkeypatch) -> None:
    fixture = _fixture(dataset)
    _skip_render(monkeypatch)
    n_sequences = max(fixture["windows"])
    output_dir = tmp_path / f"{dataset}_pmf"

    weekly_models.build_weekly_models(
        {t: _forecast_arrays(fixture, t, n_sequences) for t in SERIES_TYPES},
        _observed_frames(fixture),
        str(output_dir),
        **THRESHOLDS,
        tPAT=tau,
        horizon=fixture["horizon"],
    )

    _assert_matches_reference(fixture, tau, output_dir, ("forecast", "re_mined"))


@pytest.mark.parametrize("tau", TAUS)
@pytest.mark.parametrize("dataset", DATASETS)
def test_pipeline_matches_reference(dataset, tau, tmp_path, monkeypatch) -> None:
    fixture = _fixture(dataset)
    log = ROOT / "data" / fixture["log"]
    if not log.exists():
        pytest.skip(f"needs data/{fixture['log']}")
    _skip_render(monkeypatch)
    monkeypatch.setattr(pipeline, "chronos_sliding_forecast", _stored_forecast(fixture))
    output_dir = tmp_path / f"{dataset}_pmf"

    pipeline.run_pmf_experiment(
        log_path=str(log),
        output_dir=str(output_dir),
        train_ratio=0.8,
        **THRESHOLDS,
        tPAT=tau,
        horizon=fixture["horizon"],
    )

    _assert_matches_reference(fixture, tau, output_dir, FAMILIES)
