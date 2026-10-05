"""Events with equal timestamps keep the log's order in every step.

A case's events are ordered by timestamp, and ties keep the order in which the log lists
them. Preprocessing, the relation and binding series, replay and the ER traces must all
see the same sequence. Ties are many here, because numpy sorts short arrays stably even
with an unstable algorithm.
"""

from __future__ import annotations

from itertools import pairwise

import pandas as pd
import pytest

from pmf_concurrency import data
from pmf_concurrency.conformance import (
    extract_window_traces,
    filter_window_events,
    full_traces_of_active_cases,
)
from pmf_concurrency.data import (
    END_SYMBOL,
    START_SYMBOL,
    extract_traces,
    load_event_log,
    slice_log_into_windows,
)
from pmf_concurrency.relations import extract_relation_timeseries

DAY = pd.Timestamp("2021-03-01")
TIED = [f"t{i:02d}" for i in (7, 3, 31, 0, 12, 25, 18, 1, 39, 5)] + [
    f"u{i:02d}" for i in range(30, 0, -1)
]
EXPECTED = [START_SYMBOL, *TIED, END_SYMBOL]


def _log_as_listed() -> pd.DataFrame:
    """Case c1 has 40 events at one instant; c0's events are listed in between."""
    rows = [("c1", START_SYMBOL, DAY + pd.Timedelta(hours=8))]
    for i, activity in enumerate(TIED):
        rows.append(("c1", activity, DAY + pd.Timedelta(hours=9)))
        if i % 10 == 0:
            rows.append(("c0", f"x{i}", DAY + pd.Timedelta(hours=9, minutes=i)))
    rows.append(("c1", END_SYMBOL, DAY + pd.Timedelta(hours=10)))
    rows.insert(0, ("c0", START_SYMBOL, DAY + pd.Timedelta(hours=7)))
    rows.append(("c0", END_SYMBOL, DAY + pd.Timedelta(hours=11)))
    return pd.DataFrame(
        rows, columns=["case:concept:name", "concept:name", "time:timestamp"]
    )


@pytest.fixture
def events(monkeypatch) -> pd.DataFrame:
    listed = _log_as_listed()
    monkeypatch.setattr(data.pm4py, "read_xes", lambda _path: listed.copy())
    monkeypatch.setattr(data.pm4py, "convert_to_dataframe", lambda log: log)
    return load_event_log("tied.xes")


def _case(df: pd.DataFrame, case: str = "c1") -> list[str]:
    return df.loc[df["case:concept:name"] == case, "concept:name"].tolist()


def test_loading_keeps_tied_events_in_log_order(events) -> None:
    assert _case(events) == EXPECTED


def test_traces_keep_tied_events_in_log_order(events) -> None:
    assert EXPECTED in extract_traces(events)


def test_relation_series_see_the_log_order(events) -> None:
    _A, AB, _ABA = extract_relation_timeseries(events)
    expected = {f"{a}→{b}" for a, b in pairwise(EXPECTED)}
    assert {k for k in AB.columns if "x" not in k} == expected  # c0's labels hold x


def test_binding_input_sees_the_log_order(events) -> None:
    windows = slice_log_into_windows(events)
    (c1,) = [wt for wt in windows[DAY.date()] if wt.case_id == "c1"]
    assert c1.activities == EXPECTED


def test_replay_and_er_see_the_log_order(events) -> None:
    start, end = DAY, DAY + pd.Timedelta(days=1)
    assert _case(full_traces_of_active_cases(events, start, end)) == EXPECTED
    assert _case(filter_window_events(events, start, end)) == EXPECTED
    er_traces = [
        [e["concept:name"] for e in trace]
        for trace in extract_window_traces(events, start, end)
    ]
    assert EXPECTED in er_traces


def test_the_sort_breaks_ties_by_row_order_whatever_the_input_order() -> None:
    listed = _log_as_listed()
    shuffled = listed.sample(frac=1.0, random_state=3)
    # Row order is the tie-break, so a reshuffled frame gives a different but equally
    # deterministic order, and sorting twice changes nothing.
    once = data.sort_events(shuffled)
    assert _case(data.sort_events(once)) == _case(once)
    assert _case(data.sort_events(listed)) == EXPECTED
