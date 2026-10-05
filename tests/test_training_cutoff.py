"""Every model trains on the same period, which ends with the cutoff day.

The cutoff instant lies inside a day. The daily series keep that whole day for
training, so the Static model's training log keeps it too, and the test log starts on
the next day.
"""

from __future__ import annotations

import pandas as pd

from pmf_concurrency.data import split_train_test
from pmf_concurrency.pipeline import SplitLog, daily_series


def _log() -> pd.DataFrame:
    """One case a day for ten days, each with ▶, A, B and ■ at 08:00 to 20:00."""
    rows = []
    for day in range(10):
        start = pd.Timestamp("2021-03-01") + pd.Timedelta(days=day, hours=8)
        for i, activity in enumerate(("▶", "A", "B", "■")):
            rows.append((f"c{day}", activity, start + pd.Timedelta(hours=4 * i)))
    df = pd.DataFrame(
        rows, columns=["case:concept:name", "concept:name", "time:timestamp"]
    )
    df["date"] = df["time:timestamp"].dt.date
    return df


def test_the_training_log_keeps_the_whole_cutoff_day() -> None:
    df = _log()
    train, test, _start, train_end, _end = split_train_test(df, train_ratio=0.75)

    cutoff = (
        df["time:timestamp"].min()
        + (df["time:timestamp"].max() - df["time:timestamp"].min()) * 0.75
    )
    assert cutoff.date() == train_end
    assert cutoff.time() != pd.Timestamp(train_end).time()  # inside the day
    assert (df["time:timestamp"] > cutoff).loc[df["date"] == train_end].any()

    assert set(train["date"]) == {d for d in df["date"] if d <= train_end}
    assert set(test["date"]) == {d for d in df["date"] if d > train_end}
    assert len(train) + len(test) == len(df)


def test_static_and_the_daily_series_share_the_training_events() -> None:
    df = _log()
    log = SplitLog(df, *split_train_test(df, train_ratio=0.75))

    train, test = daily_series(log, tD=0.5, tL1L=0.5, tL2L=0.5)

    assert train["A"].to_numpy().sum() == len(log.train_events)
    assert test["A"].to_numpy().sum() == len(log.test_events)
