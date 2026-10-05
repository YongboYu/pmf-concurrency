"""The 0.5 quantile level against the middle of the sorted quantiles.

The point forecast is Chronos-2's 0.5 level. The middle of the 21 sorted quantiles
differs from it only where the levels cross. A run with
``experiment.record_median_over_levels=true`` saves that middle under
``median_over_levels/``; this script compares the two for one pipeline run:

    n_values_differ            daily forecast values (series x window x day) that differ
    n_weekly_counts_differ     weekly sums (series x window) that differ, and the
    max_weekly_relative_change   largest relative change among those not 0
    n_weekly_counts_enter_or_leave  weekly sums that are 0 in one forecast only, so
                               the series is in one net and not the other
    n_nets_topology_differ     Forecast nets with other nodes or arcs
    n_nets_frequencies_differ  Forecast nets with other activity frequencies, which
                               ER uses as transition weights, even on one topology

It rebuilds every Forecast net from both forecasts with the run's thresholds, and
checks that the 0.5-level rebuild is the net the run saved (``n_nets_not_reproduced``
should be 0).

Usage, from the project root:
    uv run python scripts/median_choice.py outputs_tpat0/Sepsis_pmf outputs_tpatm1/Sepsis_pmf

Writes ``median_choice.json`` into each run directory and prints a table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pmf_concurrency.families import model_json_path
from pmf_concurrency.petri_net import activity_frequencies
from pmf_concurrency.pipeline import MEDIAN_OVER_LEVELS_DIR, _file_key
from pmf_concurrency.weekly_models import SERIES, _forecast_counts, _rebuilt_model


def _load(run_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    """Forecast arrays of a run directory, by series and key."""
    forecasts: dict[str, dict[str, np.ndarray]] = {}
    for name in SERIES:
        forecasts[name] = {}
        summary = run_dir / f"forecast_{name}_summary.csv"
        if not summary.exists():
            continue
        for key in pd.read_csv(summary)["series_key"].astype(str):
            path = run_dir / f"forecast_{name}" / f"{_file_key(key)}.npy"
            if path.exists():
                forecasts[name][key] = np.load(path)
    return forecasts


def _topology(model_json: dict) -> tuple:
    graph = model_json["graph"]
    nodes = sorted((n["id"], bool(n["data"].get("is_binding"))) for n in graph["nodes"])
    edges = sorted((e["source"], e["target"]) for e in graph["edges"])
    return nodes, edges


def median_choice_counts(run_dir: Path) -> dict:
    """The counts above for one pipeline run directory, ``<Name>_pmf``."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "experiment_summary.json").read_text())
    t = summary["thresholds"]
    thresholds = (t["tD"], t["tL1L"], t["tL2L"], t["tPAT"])
    point = _load(run_dir)
    middle = _load(run_dir / MEDIAN_OVER_LEVELS_DIR)

    n_values = n_values_differ = n_weekly = n_weekly_differ = n_enter_leave = 0
    max_relative = 0.0
    for name in SERIES:
        if set(point[name]) != set(middle[name]):
            raise ValueError(f"{run_dir}: {name} series differ between the two sets")
        for key, p in point[name].items():
            m = middle[name][key]
            n_values += p.size
            n_values_differ += int((p != m).sum())
            p_week, m_week = p.sum(axis=1), m.sum(axis=1)
            n_weekly += p_week.size
            differ = p_week != m_week
            n_weekly_differ += int(differ.sum())
            n_enter_leave += int(((p_week > 0) != (m_week > 0)).sum())
            nonzero = differ & (p_week != 0)
            if nonzero.any():
                rel = np.abs(m_week[nonzero] - p_week[nonzero]) / np.abs(
                    p_week[nonzero]
                )
                max_relative = max(max_relative, float(rel.max()))

    weekly_dir = run_dir / "weekly_models"
    windows = sorted(
        int(p.stem.split("_")[1]) for p in weekly_dir.glob("window_*_forecast.json")
    )
    topology_differ = frequencies_differ = not_reproduced = 0
    for window in windows:
        seq_idx = window - 1
        saved = json.loads(model_json_path(weekly_dir, window, "forecast").read_text())
        net_p, _ = _rebuilt_model(_forecast_counts(point, seq_idx), {}, thresholds)
        net_m, _ = _rebuilt_model(_forecast_counts(middle, seq_idx), {}, thresholds)
        not_reproduced += net_p["graph"] != saved["graph"]
        topology_differ += _topology(net_p) != _topology(net_m)
        frequencies_differ += activity_frequencies(net_p) != activity_frequencies(net_m)

    return {
        "run": str(run_dir),
        "tPAT": t["tPAT"],
        "n_values": n_values,
        "n_values_differ": n_values_differ,
        "n_weekly_counts": n_weekly,
        "n_weekly_counts_differ": n_weekly_differ,
        "max_weekly_relative_change": max_relative,
        "n_weekly_counts_enter_or_leave": n_enter_leave,
        "n_windows": len(windows),
        "n_nets_topology_differ": topology_differ,
        "n_nets_frequencies_differ": frequencies_differ,
        "n_nets_not_reproduced": not_reproduced,
    }


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    rows = []
    for arg in argv:
        counts = median_choice_counts(Path(arg))
        (Path(arg) / "median_choice.json").write_text(
            json.dumps(counts, indent=2) + "\n"
        )
        rows.append(counts)
    print(pd.DataFrame(rows).to_string(index=False))
    return 1 if any(r["n_nets_not_reproduced"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
