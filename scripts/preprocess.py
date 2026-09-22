"""Preprocess the raw event logs into the form the pipeline reads.

    raw log  ->  filter rare variants  ->  insert ▶/■  ->  trim both ends

The steps run in this order, matching the preprocessing of the pmf-benchmark repository.
Reads ``data/raw/<Filename>.xes`` (from ``scripts/fetch_data.py``) and writes
``data/<Filename>.xes``, the path in ``conf/dataset/``.

Usage:
    uv run python scripts/fetch_data.py         # 1. download the raw logs
    uv run python scripts/preprocess.py         # 2. preprocess all logs
    uv run python scripts/preprocess.py sepsis  #    or one log

Defaults: ``--filter-percentage 0.0001`` (minimum variant coverage) and ``--trim-
percentage 0.1`` (share of the timespan trimmed from each end). The start and end
markers are ▶ (U+25B6) and ■ (U+25A0).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pm4py

# Dataset -> filename mapping, shared with the fetch script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_data import DATA_DIR as RAW_DIR
from fetch_data import DATASETS

from pmf_concurrency.data import END_SYMBOL, START_SYMBOL

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data"


def _trace_count(df: pd.DataFrame) -> int:
    return int(df["case:concept:name"].nunique())


def _marker_counts(df: pd.DataFrame) -> tuple[int, int]:
    acts = df["concept:name"]
    return int((acts == START_SYMBOL).sum()), int((acts == END_SYMBOL).sum())


def _max_markers_per_trace(df: pd.DataFrame) -> tuple[int, int]:
    """Largest number of ▶ or ■ in any one trace (at most 1 is valid)."""
    per_case = df.groupby("case:concept:name")["concept:name"]
    max_start = int(per_case.apply(lambda s: (s == START_SYMBOL).sum()).max())
    max_end = int(per_case.apply(lambda s: (s == END_SYMBOL).sum()).max())
    return max_start, max_end


def preprocess(
    filename: str,
    *,
    filter_percentage: float,
    trim_percentage: float,
) -> bool:
    raw_path = RAW_DIR / filename
    out_path = OUT_DIR / filename
    if not raw_path.exists():
        print(
            f"[{filename}] raw log missing at {raw_path}\n"
            f"    run: uv run python scripts/fetch_data.py"
        )
        return False

    print(f"[{filename}] reading {raw_path}")
    df = pm4py.read_xes(str(raw_path))
    df = pm4py.convert_to_dataframe(df)
    print(f"  raw: {len(df)} events, {_trace_count(df)} cases")

    # 0. Strip any existing ▶/■ first. Some raw logs, such as Hospital Billing,
    #    already carry markers, and inserting again would double them.
    pre_start, pre_end = _marker_counts(df)
    if pre_start or pre_end:
        df = df[~df["concept:name"].isin([START_SYMBOL, END_SYMBOL])].copy()
        print(f"  stripped pre-existing markers (▶={pre_start}, ■={pre_end})")

    # 1. Filter infrequent variants.
    df = pm4py.filter_variants_by_coverage_percentage(
        df, min_coverage_percentage=filter_percentage
    )
    print(f"  after variant filter ({filter_percentage}): {_trace_count(df)} cases")

    # 2. Insert the ▶/■ markers, exactly once.
    df = pm4py.insert_artificial_start_end(df)

    # 3. Trim the timespan symmetrically.
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
    start_time = df["time:timestamp"].min()
    end_time = df["time:timestamp"].max()
    trim_days = int((end_time - start_time).days * trim_percentage)
    trim_start = start_time + pd.Timedelta(days=trim_days)
    trim_end = end_time - pd.Timedelta(days=trim_days)
    df = pm4py.filter_time_range(
        df,
        trim_start.strftime("%Y-%m-%d %H:%M:%S"),
        trim_end.strftime("%Y-%m-%d %H:%M:%S"),
        mode="events",
    )
    print(f"  after trim ({trim_percentage} each end): {_trace_count(df)} cases")

    # Each trace may hold at most one ▶ and one ■. Trimming can drop markers of traces
    # that cross a trim edge, so totals may be below the trace count.
    n_traces = _trace_count(df)
    n_start, n_end = _marker_counts(df)
    max_start, max_end = _max_markers_per_trace(df)
    print(f"  markers: ▶={n_start}, ■={n_end} over {n_traces} traces")
    ok = max_start <= 1 and max_end <= 1
    if not ok:
        print(
            f"  WARNING: a trace carries duplicate boundary markers "
            f"(max ▶/trace={max_start}, max ■/trace={max_end}); "
            f"insert_artificial_start_end appears to have run more than once."
        )
    else:
        print("  boundary markers OK: at most one ▶/■ per trace (no doubling)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pm4py.write_xes(df, str(out_path))
    print(f"  -> {out_path}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess raw event logs.")
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=[*DATASETS.keys()],
        help="datasets to preprocess (default: all)",
    )
    parser.add_argument("--filter-percentage", type=float, default=0.0001)
    parser.add_argument("--trim-percentage", type=float, default=0.1)
    args = parser.parse_args()

    keys = args.datasets or list(DATASETS)
    ok = True
    for key in keys:
        ok = (
            preprocess(
                DATASETS[key].filename,
                filter_percentage=args.filter_percentage,
                trim_percentage=args.trim_percentage,
            )
            and ok
        )
    if not ok:
        print("\nSome logs were missing or had marker anomalies; see above.")
        return 1
    print("\nAll requested logs preprocessed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
