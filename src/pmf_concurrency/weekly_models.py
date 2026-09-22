"""The four model families of every evaluation window.

Each window gets up to four models:

  - forecast: rebuilt from the Chronos-2 counts summed over the window
  - re_mined: rebuilt the same way from the observed counts
  - direct_mined: Fodina mined on the window's events
  - static: Fodina mined once on the training log, reused in every window

Each model is saved as ``weekly_models/window_<n>_<family>.json`` and drawn as a PDF.
One ``structural_metrics.csv`` per run holds a row per window and family; the model
JSONs cannot be used to recount arity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from .causal_net import causal_graph_to_json, fodina_to_json, mine_fodina
from .data import extract_traces
from .families import model_json_path
from .paths import dataset_name_from_output_dir
from .reconstruction import build_causal_net_from_forecast, parse_ab_counts
from .rendering import render_petri_json_to_pdf
from .structural_metrics import (
    StructuralStats,
    stats_from_fodina,
    structural_metrics_row,
    write_structural_metrics,
)

#: The daily series every window is rebuilt from.
SERIES = ("A", "AB", "ABA", "IB", "OB")


class StaticModel(NamedTuple):
    """Fodina's net on the training log, the same in every window."""

    model_json: dict[str, Any]
    stats: StructuralStats


def _forecast_counts(
    forecasts: dict[str, dict[str, np.ndarray]], seq_idx: int
) -> dict[str, dict[str, float]]:
    """Forecast counts of one window: each series' forecast summed over the horizon."""
    counts: dict[str, dict[str, float]] = {}
    for name in SERIES:
        counts[name] = {}
        for key, predictions in sorted(forecasts[name].items()):
            if seq_idx < len(predictions):
                value = float(predictions[seq_idx].sum())
                if value > 0:
                    counts[name][key] = value
    return counts


def _observed_counts(
    test: dict[str, pd.DataFrame], seq_idx: int, horizon: int
) -> dict[str, dict[str, float]]:
    """Observed counts of one window: each series summed over the window's days."""
    counts: dict[str, dict[str, float]] = {}
    for name in SERIES:
        sums = test[name].iloc[seq_idx : seq_idx + horizon].sum()
        counts[name] = {}
        for key in sorted(test[name].columns):
            value = float(sums[key])
            if value > 0:
                counts[name][key] = value
    return counts


def _rebuilt_model(
    counts: dict[str, dict[str, float]],
    metadata: dict[str, Any],
    thresholds: tuple[float, float, float, float],
) -> tuple[dict[str, Any], StructuralStats]:
    """The Forecast or Re-mined model of one window, and its structural statistics."""
    graph, binding_nodes, binding_freqs, stats = build_causal_net_from_forecast(
        counts["A"],
        counts["AB"],
        counts["ABA"],
        counts["IB"],
        counts["OB"],
        *thresholds,
    )
    model_json = causal_graph_to_json(
        graph,
        binding_nodes,
        metadata,
        parse_ab_counts(counts["AB"]),
        binding_freqs,
        activity_frequencies=counts["A"],
    )
    return model_json, stats


def _save(weekly_dir: Path, window: int, family: str, model_json: dict) -> None:
    path = model_json_path(weekly_dir, window, family)
    with open(path, "w") as fh:
        json.dump(model_json, fh, indent=2)
    render_petri_json_to_pdf(model_json, str(path.with_suffix(".pdf")))


def build_weekly_models(
    forecasts: dict[str, dict[str, np.ndarray]],
    test: dict[str, pd.DataFrame],
    output_dir: str,
    tD: float,
    tL1L: float,
    tL2L: float,
    tPAT: float,
    horizon: int = 7,
    test_df: pd.DataFrame | None = None,
    static: StaticModel | None = None,
) -> None:
    """Build and save the models of every evaluation window.

    Args:
        forecasts:  Chronos-2 forecasts per series (see :data:`SERIES`), each of shape
                    (windows, horizon).
        test:       Observed daily counts of the test period per series.
        output_dir: The run's output directory, ``<Name>_pmf``.
        tD, tL1L, tL2L, tPAT: Fodina thresholds.
        horizon:    Window length in days.
        test_df:    Test-period events, for Direct-mined. None skips that family.
        static:     The Static model. None skips that family.

    A family whose model cannot be built for a window has no row for it; a warning
    says why.
    """
    weekly_dir = Path(output_dir) / "weekly_models"
    weekly_dir.mkdir(parents=True, exist_ok=True)
    dataset = dataset_name_from_output_dir(output_dir)
    thresholds = (tD, tL1L, tL2L, tPAT)
    rows: list[dict[str, Any]] = []

    days = test["A"].index
    # Windows are cut by row position, so every series must share one daily calendar.
    for name, frame in test.items():
        if not frame.index.equals(days):
            raise ValueError(f"{name} is not on the daily calendar of the other series")

    n_windows = len(days) - horizon + 1
    print(f"  Building models for {n_windows} windows of {horizon} days...")

    def record(window: int, family: str, stats: StructuralStats) -> None:
        rows.append(
            structural_metrics_row(window, family, stats, dataset=dataset, tPAT=tPAT)
        )

    for seq_idx in range(n_windows):
        window = seq_idx + 1
        span = days[seq_idx : seq_idx + horizon]
        window_meta = {
            "dataset": dataset,
            "window": window,
            "seq_start": seq_idx,
            "seq_end": seq_idx + horizon,
            "start_date": span[0].strftime("%Y%m%d"),
            "end_date": span[-1].strftime("%Y%m%d"),
        }

        for family, counts in (
            ("forecast", _forecast_counts(forecasts, seq_idx)),
            ("re_mined", _observed_counts(test, seq_idx, horizon)),
        ):
            metadata = {"type": f"sliding_{family}", **window_meta}
            model_json, stats = _rebuilt_model(counts, metadata, thresholds)
            record(window, family, stats)
            _save(weekly_dir, window, family, model_json)

        if test_df is not None:
            window_df = test_df[test_df["date"].isin({d.date() for d in span})]
            if window_df.empty:
                print(
                    f"    Warning: no events in window {window}, no Direct-mined model"
                )
            else:
                try:
                    traces = extract_traces(window_df)
                    pm = mine_fodina(traces, *thresholds)
                    if pm is None:
                        print(
                            f"    Warning: Direct-mined model for window {window} "
                            "failed: Fodina returned no model"
                        )
                    else:
                        # Recorded before the export, so a failed render cannot lose
                        # the row.
                        record(window, "direct_mined", stats_from_fodina(pm))
                        metadata = {
                            "type": "sliding_direct_mined",
                            **window_meta,
                            "method": "fodina_original",
                            "traces": len(traces),
                        }
                        _save(
                            weekly_dir,
                            window,
                            "direct_mined",
                            fodina_to_json(pm, metadata),
                        )
                except Exception as e:
                    print(
                        f"    Warning: Direct-mined model for window {window} failed: "
                        f"{e}"
                    )

        if static is not None:
            try:
                training = static.model_json.get("metadata", {})
                metadata = {
                    "type": "sliding_static",
                    **window_meta,
                    "method": "fodina_original",
                    "training_period": training.get("period", ""),
                    "training_traces": training.get("traces", 0),
                }
                model_json = {"metadata": metadata, "graph": static.model_json["graph"]}
                _save(weekly_dir, window, "static", model_json)
            except Exception as e:
                print(f"    Warning: Static model for window {window} failed: {e}")
            # The Static model does not depend on the window, so its row is recorded
            # even if this window's export failed.
            record(window, "static", static.stats)

        # Rewritten after every window, so a run that stops early keeps its rows.
        write_structural_metrics(output_dir, rows)

    metrics_path = write_structural_metrics(output_dir, rows)
    print(f"  Saved structural metrics: {metrics_path} ({len(rows)} rows)")
