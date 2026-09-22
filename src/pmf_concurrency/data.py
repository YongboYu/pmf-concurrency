"""Loading, splitting and windowing event logs."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date

import pandas as pd
import pm4py

#: Boundary markers every processed log starts and ends each case with.
START_SYMBOL = "▶"
END_SYMBOL = "■"


def assert_boundary_markers_present(
    df: pd.DataFrame, log_path: str, activity_key: str = "concept:name"
) -> None:
    """Raise if a log contains no ▶ or ■ activity.

    The pipeline requires logs with markers, which ``scripts/preprocess.py`` adds when
    it writes ``data/`` from ``data/raw/``. Relation extraction counts the markers as
    activities, and :func:`~pmf_concurrency.reconstruction.resolve_anchors` anchors the
    repair on them. A raw log would give a different model instead of an error.
    """
    activities = set(df[activity_key].unique())
    absent = [s for s in (START_SYMBOL, END_SYMBOL) if s not in activities]
    if absent:
        raise ValueError(
            f"{log_path} carries no {' / '.join(absent)} boundary marker(s). "
            "This pipeline requires marker-annotated input: run "
            "`python scripts/preprocess.py` to produce data/<Log>.xes from "
            "data/raw/<Log>.xes. Pass require_markers=False only for "
            "deliberate raw-log inspection."
        )


def load_event_log(
    log_path: str,
    activity_key: str = "concept:name",
    timestamp_key: str = "time:timestamp",
    case_key: str = "case:concept:name",
    require_markers: bool = True,
) -> pd.DataFrame:
    """Load a XES event log into a DataFrame sorted by case and timestamp.

    Adds a ``date`` column (the day of each event).

    Args:
        require_markers: Check for ▶ and ■ with
            :func:`assert_boundary_markers_present`. Set False only to inspect a
            raw log.
    """
    print(f"Loading event log from {log_path}...")
    log = pm4py.read_xes(log_path)
    df = pm4py.convert_to_dataframe(log)

    df[timestamp_key] = pd.to_datetime(df[timestamp_key])
    df = df.sort_values([case_key, timestamp_key])
    df["date"] = df[timestamp_key].dt.date

    if require_markers:
        assert_boundary_markers_present(df, log_path, activity_key)

    print(f"Loaded {len(df)} events, {df[case_key].nunique()} cases")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    return df


def split_train_test(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    timestamp_key: str = "time:timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, date, date, date]:
    """Split a DataFrame chronologically into train and test sets.

    Returns:
        (train_df, test_df, train_start, train_end, test_end)
    """
    min_ts = df[timestamp_key].min()
    max_ts = df[timestamp_key].max()
    training_end_ts = min_ts + (max_ts - min_ts) * train_ratio

    train_start = min_ts.date()
    train_end = training_end_ts.date()
    test_end = max_ts.date()

    train_df = df[df[timestamp_key] < training_end_ts].copy()
    test_df = df[df[timestamp_key] >= training_end_ts].copy()

    print(f"\nTrain period: {train_start} to {train_end}")
    print(f"Test period:  {train_end} to {test_end}")
    print(f"Train events: {len(train_df)}, Test events: {len(test_df)}")
    return train_df, test_df, train_start, train_end, test_end


def extract_traces(
    df: pd.DataFrame,
    activity_key: str = "concept:name",
    timestamp_key: str = "time:timestamp",
    case_key: str = "case:concept:name",
) -> list[list[str]]:
    """Extract ordered activity sequences (traces) from a DataFrame."""
    traces = []
    for _case_id, case_df in df.groupby(case_key):
        trace = case_df.sort_values(timestamp_key)[activity_key].tolist()
        if trace:
            traces.append(trace)
    return traces


@dataclass
class WindowedTrace:
    """The prefix of one case's trace up to and including a window's day."""

    case_id: str
    activities: list[str]
    timestamps: list[pd.Timestamp]


def slice_log_into_windows(
    df: pd.DataFrame,
    activity_key: str = "concept:name",
    timestamp_key: str = "time:timestamp",
    case_key: str = "case:concept:name",
) -> dict[date, list[WindowedTrace]]:
    """Cut an event log into daily windows, keeping the prefix of every active trace.

    For each day, every trace with at least one event on that day is included with all
    its events up to and including that day.

    Example: trace a-b-c-a-d-e-f over days 1 to 4
      - Day 1: [a-b]           (prefix, trace continues)
      - Day 2: [a-b-c]
      - Day 3: [a-b-c-a-d]
      - Day 4: [a-b-c-a-d-e-f]
      - Day 5: not included (no event that day)
    """
    all_dates = sorted(df["date"].unique())

    case_events: dict[str, list[tuple[str, pd.Timestamp, date]]] = defaultdict(list)
    for _, row in df.iterrows():
        case_events[row[case_key]].append(
            (row[activity_key], row[timestamp_key], row["date"])
        )
    for case_id in case_events:
        case_events[case_id].sort(key=lambda x: x[1])

    windows: dict[date, list[WindowedTrace]] = {}

    for window_date in all_dates:
        windowed_traces = []
        for case_id, events in case_events.items():
            if not any(ev_date == window_date for _, _, ev_date in events):
                continue

            prefix_events = [
                (act, ts) for act, ts, ev_date in events if ev_date <= window_date
            ]
            activities = [act for act, _ in prefix_events]
            timestamps = [ts for _, ts in prefix_events]
            windowed_traces.append(
                WindowedTrace(
                    case_id=case_id, activities=activities, timestamps=timestamps
                )
            )
        windows[window_date] = windowed_traces

    return windows
