"""The history models of every evaluation window, built apart from the pipeline.

    persistence  Fodina on the events of the horizon days before the window, cut by
                 date as a Direct-mined window is, so from window horizon + 1 on it
                 is the Direct-mined net of the window one horizon earlier
    expanding    Fodina on every event from the log's first day to the day before the
                 window, cut by date as the training log is, so open cases contribute
                 their prefixes

The models go to a root of their own with a pipeline run's layout: the models in
``weekly_models/``, ``structural_metrics.csv`` and ``experiment_summary.json``. The
build makes no forecast and needs no GPU.

Run it through Hydra with the pipeline's configuration:
``PMF_OUTPUT_BASE=outputs_history_tpat0 uv run python -m pmf_concurrency.history_models
dataset=sepsis``.
"""

from __future__ import annotations

import os

# Before numpy is imported, as in the pipeline. The thread count can change
# floating-point results.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

from datetime import date
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig

from .campaign import windows_of
from .causal_net import fodina_to_json, mine_fodina
from .data import extract_traces
from .families import FAMILY_LABELS
from .paths import dataset_name_from_output_dir, dataset_output_dir
from .pipeline import load_and_split, summary_periods, write_summary
from .reproducibility import enable_determinism
from .structural_metrics import (
    StructuralStats,
    stats_from_fodina,
    structural_metrics_row,
    write_structural_metrics,
)
from .weekly_models import save_model


def _mine_history(
    events: pd.DataFrame,
    first: date,
    last: date,
    family: str,
    window_meta: dict[str, Any],
    thresholds: tuple[float, float, float, float],
) -> tuple[dict[str, Any], StructuralStats] | None:
    """Fodina on the events dated ``first`` to ``last``. None, with a warning, if the
    history holds no events or Fodina returns no model.
    """
    window = window_meta["window"]
    label = FAMILY_LABELS[family]
    history = events[(events["date"] >= first) & (events["date"] <= last)]
    if history.empty:
        print(
            f"    Warning: no events in the history of window {window}, "
            f"no {label} model"
        )
        return None
    traces = extract_traces(history)
    print(
        f"    Window {window}: {label} from {len(traces)} traces, "
        f"{len(history)} events, {first} to {last}",
        flush=True,
    )
    pm = mine_fodina(traces, *thresholds)
    if pm is None:
        print(
            f"    Warning: {label} model for window {window} failed: Fodina "
            "returned no model"
        )
        return None
    metadata = {
        "type": f"sliding_{family}",
        **window_meta,
        "method": "fodina_original",
        "history_start": first.strftime("%Y%m%d"),
        "history_end": last.strftime("%Y%m%d"),
        "traces": len(traces),
    }
    return fodina_to_json(pm, metadata), stats_from_fodina(pm)


def build_history_models(
    log_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    tD: float = 0.5,
    tL1L: float = 0.5,
    tL2L: float = 0.5,
    tPAT: float = 0.0,
    horizon: int = 7,
) -> dict[str, Any]:
    """Build the history models of every evaluation window of one event log.

    Writes the models, ``structural_metrics.csv`` and ``experiment_summary.json`` to
    ``output_dir``, as a pipeline run does. The windows are the pipeline's.

    Args:
        log_path:    Path to the XES event log, with ▶/■ markers.
        output_dir:  Directory for all outputs, ``<Name>_pmf``.
        train_ratio: Share of the timeline used for training.
        tD, tL1L, tL2L, tPAT: Fodina thresholds.
        horizon:     Window length in days, and the length of the Persistence
                     history.

    Returns:
        The run summary, as written to ``experiment_summary.json``.

    Raises:
        FileExistsError: ``output_dir`` is not empty, as for a pipeline run.
    """
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        raise FileExistsError(
            f"{output_dir} is not empty: move it aside or choose another output base"
        )
    enable_determinism()
    weekly_dir = Path(output_dir) / "weekly_models"
    weekly_dir.mkdir(parents=True, exist_ok=True)
    dataset = dataset_name_from_output_dir(output_dir)
    thresholds = (tD, tL1L, tL2L, tPAT)

    print("\nLoading and splitting the log...")
    log = load_and_split(log_path, train_ratio)
    windows = windows_of(log.events, train_ratio, horizon)

    print(f"\nBuilding the history models of {len(windows)} windows...")
    rows: list[dict[str, Any]] = []
    for w in windows.itertuples():
        window = int(w.window)
        window_meta = {
            "dataset": dataset,
            "window": window,
            "start_date": w.start_date,
            "end_date": w.end_date,
        }
        start = pd.to_datetime(w.start_date, format="%Y%m%d")
        day_before = (start - pd.Timedelta(days=1)).date()
        histories = {
            "persistence": ((start - pd.Timedelta(days=horizon)).date(), day_before),
            "expanding": (log.train_start, day_before),
        }
        for family, (first, last) in histories.items():
            built = _mine_history(
                log.events, first, last, family, window_meta, thresholds
            )
            if built is None:
                continue
            model_json, stats = built
            rows.append(
                structural_metrics_row(
                    window, family, stats, dataset=dataset, tPAT=tPAT
                )
            )
            save_model(weekly_dir, window, family, model_json)
        # Rewritten after every window, so a run that stops early keeps its rows.
        write_structural_metrics(output_dir, rows)

    metrics_path = write_structural_metrics(output_dir, rows)
    print(f"  Saved structural metrics: {metrics_path} ({len(rows)} rows)")

    # Every calendar day after the cutoff day, as the pipeline's daily series hold.
    test_days = pd.date_range(
        pd.Timestamp(log.train_end) + pd.Timedelta(days=1), pd.Timestamp(log.test_end)
    )
    summary = write_summary(
        output_dir,
        summary_periods(log, test_days),
        {"tD": tD, "tL1L": tL1L, "tL2L": tL2L, "tPAT": tPAT},
    )
    print(f"\nDone: {output_dir}/ ({len(rows)} structural-metrics rows)")
    return summary


# config_path is relative to this file: src/pmf_concurrency/ -> ../../conf
@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point; the pipeline's dataset, thresholds and root, from conf/."""
    build_history_models(
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
    )


if __name__ == "__main__":
    main()
