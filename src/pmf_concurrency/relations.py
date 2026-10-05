"""Daily relation series from an event log.

Relation keys, without spaces around arrows:
  - activity:           "A"
  - direct succession:  "A→B"
  - length-2 loop:      "A→B→A"
"""

from collections.abc import Iterable
from datetime import date

import pandas as pd

from .data import sort_events


def daily_calendar(dates: Iterable[date]) -> pd.DatetimeIndex:
    """Every day from the first to the last of ``dates``.

    All daily series share this index, so a row is the same day in every family and a
    day without occurrences is a zero rather than a missing row.
    """
    dates = list(dates)
    return pd.date_range(min(dates), max(dates), freq="D")


def extract_relation_timeseries(
    df: pd.DataFrame,
    activity_key: str = "concept:name",
    timestamp_key: str = "time:timestamp",
    case_key: str = "case:concept:name",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Daily series of the relations A, AB and ABA.

    Each occurrence is dated by its last activity. Every series covers every day of
    the log (see :func:`daily_calendar`).

    Returns:
        (daily_A, daily_AB, daily_ABA), date-indexed count DataFrames.
    """
    relations_A: list[tuple[date, str]] = []
    relations_AB: list[tuple[date, str]] = []
    relations_ABA: list[tuple[date, str]] = []

    ordered = sort_events(df, case_key, timestamp_key)
    for _case_id, case_df in ordered.groupby(case_key):
        events = case_df[[activity_key, "date"]].values.tolist()

        for i, (activity, ev_date) in enumerate(events):
            # A: activity occurrence
            relations_A.append((ev_date, activity))

            # AB: direct succession (prev → current)
            if i > 0:
                relations_AB.append((ev_date, f"{events[i - 1][0]}→{activity}"))

            # ABA: length-2 loop (A → B → A, where A != B)
            if i >= 2:
                act_2back = events[i - 2][0]
                act_1back = events[i - 1][0]
                if act_2back == activity and act_1back != activity:
                    relations_ABA.append(
                        (ev_date, f"{activity}→{act_1back}→{activity}")
                    )

    calendar = daily_calendar(df["date"])
    daily_A = _aggregate_daily(relations_A, calendar)
    daily_AB = _aggregate_daily(relations_AB, calendar)
    daily_ABA = _aggregate_daily(relations_ABA, calendar)

    print("\nExtracted relation time series:")
    print(f"  A  (activities): {len(daily_A.columns)} unique, {len(daily_A)} days")
    print(f"  AB (direct):     {len(daily_AB.columns)} unique, {len(daily_AB)} days")
    print(f"  ABA (L2L):       {len(daily_ABA.columns)} unique, {len(daily_ABA)} days")

    return daily_A, daily_AB, daily_ABA


def _aggregate_daily(
    relations: list[tuple[date, str]], calendar: pd.DatetimeIndex
) -> pd.DataFrame:
    """Aggregate (date, key) pairs into a count DataFrame indexed by ``calendar``."""
    if not relations:
        return pd.DataFrame(index=calendar)
    rel_df = pd.DataFrame(relations, columns=["date", "key"])
    counts = rel_df.groupby(["date", "key"]).size().unstack(fill_value=0)
    counts.index = pd.to_datetime(counts.index)
    return counts.reindex(calendar, fill_value=0)
