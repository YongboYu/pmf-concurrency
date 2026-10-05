"""The pipeline, from an event log to the models of every evaluation window.

    load_and_split       split the log by time into training and test periods
    daily_series         daily relation (A, AB, ABA) and binding (IB, OB) counts
    mine_static_model    Fodina on the training log, reused in every window
    forecast_series      Chronos-2 forecasts of every series over the test period
    build_weekly_models  the four families per window, and structural_metrics.csv

Run it through Hydra: ``uv run python -m pmf_concurrency.pipeline dataset=sepsis``.
"""

from __future__ import annotations

import os

# Before numpy and torch are imported, which fixes their thread pools. The thread count
# can change floating-point results.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from .bindings import extract_binding_timeseries
from .causal_net import fodina_to_json, mine_fodina
from .data import (
    extract_traces,
    load_event_log,
    slice_log_into_windows,
    split_train_test,
)
from .families import MODEL_FAMILIES
from .forecasting import chronos_sliding_forecast
from .paths import dataset_output_dir
from .relations import extract_relation_timeseries
from .reproducibility import enable_determinism
from .structural_metrics import (
    STRUCTURAL_METRICS_FILENAME,
    read_structural_metrics,
    stats_from_fodina,
)
from .weekly_models import SERIES, StaticModel, build_weekly_models

#: Where a run saves the sorted-middle forecasts when asked to.
MEDIAN_OVER_LEVELS_DIR = "median_over_levels"


@dataclass(frozen=True)
class SplitLog:
    """An event log split by time into training and test periods."""

    events: pd.DataFrame
    train_events: pd.DataFrame
    test_events: pd.DataFrame
    train_start: date
    train_end: date
    test_end: date


def load_and_split(log_path: str, train_ratio: float) -> SplitLog:
    events = load_event_log(log_path)
    return SplitLog(events, *split_train_test(events, train_ratio))


def daily_series(
    log: SplitLog, tD: float, tL1L: float, tL2L: float
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Training and test parts of every daily series, keyed by :data:`SERIES`.

    The series are counted on the whole log, every day from the first to the last
    event, and split at the end of the training period.
    """
    A, AB, ABA = extract_relation_timeseries(log.events)
    IB, OB = extract_binding_timeseries(
        slice_log_into_windows(log.events), tD, tL1L, tL2L
    )
    cut = pd.Timestamp(log.train_end)
    train: dict[str, pd.DataFrame] = {}
    test: dict[str, pd.DataFrame] = {}
    for name, frame in zip(SERIES, (A, AB, ABA, IB, OB), strict=True):
        train[name] = frame[frame.index <= cut]
        test[name] = frame[frame.index > cut]
        print(
            f"  {name}: {len(frame.columns)} series, "
            f"{len(train[name])} training and {len(test[name])} test days"
        )
    return train, test


def mine_static_model(
    log: SplitLog, tD: float, tL1L: float, tL2L: float, tPAT: float
) -> StaticModel | None:
    """Fodina on the training log. None if Fodina returns no model."""
    traces = extract_traces(log.train_events)
    pm = mine_fodina(traces, tD, tL1L, tL2L, tPAT)
    if pm is None:
        print("  Warning: Static model mining failed: Fodina returned no model")
        return None
    metadata: dict[str, Any] = {
        "type": "static",
        "method": "fodina_original",
        "period": f"{log.train_start} to {log.train_end}",
        "traces": len(traces),
        "thresholds": {"tD": tD, "tL1L": tL1L, "tL2L": tL2L, "tPAT": tPAT},
    }
    model_json = fodina_to_json(pm, metadata)
    stats = stats_from_fodina(pm)
    print(
        f"  Static model from {len(traces)} traces: AND-binding share "
        f"{stats.and_share_pct:.1f}% ({stats.and_bindings}/{stats.bindings} bindings)"
    )
    return StaticModel(model_json, stats)


def forecast_series(
    train: dict[str, pd.DataFrame],
    test: dict[str, pd.DataFrame],
    horizon: int,
    device: str,
    seed: int,
    median_over_levels: dict[str, dict[str, np.ndarray]] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Chronos-2 forecasts of every series, one row per evaluation window.

    ``median_over_levels``, if given, receives the middle of the sorted quantiles per
    series, for ``scripts/median_choice.py``.
    """
    forecasts = {}
    for name in SERIES:
        print(f"  Forecasting {name}...")
        extra: dict[str, Any] = {}
        if median_over_levels is not None:
            extra["median_over_levels"] = median_over_levels.setdefault(name, {})
        forecasts[name] = chronos_sliding_forecast(
            train[name], test[name], horizon, device, seed, **extra
        )
    return forecasts


def _file_key(series_key: str) -> str:
    """File name of a series key; lossy, so the summary CSV keeps the keys."""
    return (
        series_key.replace("/", "_")
        .replace(":", "_")
        .replace("{", "")
        .replace("}", "")
        .replace(",", "_")
    )


def save_forecasts(
    forecasts: dict[str, dict[str, np.ndarray]], output_dir: str
) -> None:
    """``forecast_<series>/<key>.npy`` per series, and ``forecast_<series>_summary.csv``
    with each key and its forecast total.
    """
    for name, arrays in forecasts.items():
        if not arrays:
            continue
        array_dir = os.path.join(output_dir, f"forecast_{name}")
        os.makedirs(array_dir, exist_ok=True)
        for key, predictions in arrays.items():
            if predictions.size > 0:
                np.save(os.path.join(array_dir, f"{_file_key(key)}.npy"), predictions)
        totals = [
            (key, float(predictions.sum())) for key, predictions in arrays.items()
        ]
        pd.DataFrame(totals, columns=["series_key", "forecasted_total"]).to_csv(
            os.path.join(output_dir, f"forecast_{name}_summary.csv"), index=False
        )


def _periods(log: SplitLog, test_days: pd.DatetimeIndex) -> dict[str, Any]:
    """Training period of the log, and the test days the series and windows cover."""
    return {
        "train_period": {"start": str(log.train_start), "end": str(log.train_end)},
        "test_period": {
            "start": str(test_days[0].date()) if len(test_days) else None,
            "end": str(test_days[-1].date()) if len(test_days) else None,
        },
        "test_days": len(test_days),
    }


def write_summary(
    output_dir: str, periods: dict[str, Any], thresholds: dict[str, float]
) -> dict[str, Any]:
    """Write ``experiment_summary.json``: the periods, thresholds and metrics file."""
    metrics_path = os.path.join(output_dir, STRUCTURAL_METRICS_FILENAME)
    metrics = read_structural_metrics(metrics_path)
    present = set(metrics["model_family"].dropna().astype(str))

    summary: dict[str, Any] = {
        **periods,
        "thresholds": thresholds,
        "structural_metrics": {
            "path": metrics_path,
            "rows": len(metrics),
            # int(): nunique returns numpy int64, which json.dump cannot encode.
            "windows": int(metrics["window"].nunique()),
            "families": [f for f in MODEL_FAMILIES if f in present],
        },
    }
    with open(os.path.join(output_dir, "experiment_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def run_pmf_experiment(
    log_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    tD: float = 0.5,
    tL1L: float = 0.5,
    tL2L: float = 0.5,
    tPAT: float = 0.0,
    horizon: int = 7,
    device: str = "cpu",
    seed: int = 42,
    record_median_over_levels: bool = False,
) -> dict[str, Any]:
    """Run the pipeline for one event log.

    Writes the forecast arrays, the weekly models, ``structural_metrics.csv`` and
    ``experiment_summary.json`` to ``output_dir``.

    Args:
        log_path:    Path to the XES event log, with ▶/■ markers.
        output_dir:  Directory for all outputs, ``<Name>_pmf``.
        train_ratio: Share of the timeline used for training.
        tD, tL1L, tL2L, tPAT: Fodina thresholds.
        horizon:     Forecast horizon and window length in days.
        device:      Torch device for Chronos-2 ("cpu", "cuda", "mps").
        seed:        Random seed for Chronos-2.
        record_median_over_levels: Also save the middle of the sorted quantiles
                     under ``median_over_levels/``, to compare it with the 0.5 level
                     (``scripts/median_choice.py``). The models do not use it.

    Returns:
        The run summary, as written to ``experiment_summary.json``.

    Raises:
        FileExistsError: ``output_dir`` is not empty. A new run would leave the old
            run's windows next to its own, and evaluation would pick them up.
    """
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        raise FileExistsError(
            f"{output_dir} is not empty: move it aside or choose another output base"
        )
    enable_determinism(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("\nLoading and splitting the log...")
    log = load_and_split(log_path, train_ratio)
    print("\nCounting daily relations and bindings...")
    train, test = daily_series(log, tD, tL1L, tL2L)
    print("\nMining the Static model...")
    static = mine_static_model(log, tD, tL1L, tL2L, tPAT)
    print("\nForecasting with Chronos-2...")
    middle: dict[str, dict[str, np.ndarray]] | None = (
        {} if record_median_over_levels else None
    )
    forecasts = forecast_series(train, test, horizon, device, seed, middle)
    save_forecasts(forecasts, output_dir)
    if middle is not None:
        save_forecasts(middle, os.path.join(output_dir, MEDIAN_OVER_LEVELS_DIR))
    print("\nBuilding the weekly models...")
    build_weekly_models(
        forecasts,
        test,
        output_dir,
        tD,
        tL1L,
        tL2L,
        tPAT,
        horizon,
        test_df=log.test_events,
        static=static,
    )

    summary = write_summary(
        output_dir,
        _periods(log, test["A"].index),
        {"tD": tD, "tL1L": tL1L, "tL2L": tL2L, "tPAT": tPAT},
    )
    metrics = summary["structural_metrics"]
    print(
        f"\nDone: {output_dir}/ ({metrics['windows']} windows, "
        f"{metrics['rows']} structural-metrics rows)"
    )
    return summary


# config_path is relative to this file: src/pmf_concurrency/ -> ../../conf
@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point; the dataset and thresholds come from conf/."""
    run_pmf_experiment(
        log_path=cfg.dataset.log_path,
        output_dir=str(
            dataset_output_dir(cfg.dataset.name, cfg.experiment.output_base)
        ),
        train_ratio=cfg.experiment.train_ratio,
        tD=cfg.experiment.tD,
        tL1L=cfg.experiment.tL1L,
        tL2L=cfg.experiment.tL2L,
        tPAT=cfg.experiment.tPAT,
        horizon=cfg.experiment.horizon,
        device=cfg.experiment.device,
        seed=cfg.experiment.get("seed", 42),
        record_median_over_levels=cfg.experiment.get(
            "record_median_over_levels", False
        ),
    )


if __name__ == "__main__":
    main()
