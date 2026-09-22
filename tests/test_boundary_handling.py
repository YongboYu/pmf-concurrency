"""Boundary markers ▶ and ■.

``scripts/fetch_data.py`` downloads raw logs into ``data/raw/``, and
``scripts/preprocess.py`` adds the ▶/■ markers and writes to ``data/``, which is what
the pipeline reads. Three safeguards:

* :func:`load_event_log` rejects a log without ▶/■;
* :func:`extract_window_traces` strips existing markers before adding one
  boundary pair.
"""

import pandas as pd
import pytest

from pmf_concurrency.conformance import extract_window_traces
from pmf_concurrency.data import (
    END_SYMBOL,
    START_SYMBOL,
    assert_boundary_markers_present,
)


def _event_df(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case:concept:name": case,
                "concept:name": act,
                "time:timestamp": pd.Timestamp(ts),
            }
            for case, act, ts in rows
        ]
    )


def test_marker_precondition_rejects_a_raw_log() -> None:
    """A log without ▶/■ is rejected, and the message names the fix."""
    df = _event_df(
        [
            ("c1", "A", "2020-01-01T10:00:00"),
            ("c1", "B", "2020-01-01T11:00:00"),
        ]
    )
    with pytest.raises(ValueError, match="preprocess"):
        assert_boundary_markers_present(df, "data/raw/Sepsis.xes")


def test_marker_precondition_accepts_a_preprocessed_log() -> None:
    """A preprocessed log passes."""
    df = _event_df(
        [
            ("c1", START_SYMBOL, "2020-01-01T09:00:00"),
            ("c1", "A", "2020-01-01T10:00:00"),
            ("c1", END_SYMBOL, "2020-01-01T12:00:00"),
        ]
    )
    assert_boundary_markers_present(df, "data/Sepsis.xes")


def test_marker_precondition_names_only_the_absent_marker() -> None:
    """A half-annotated log reports which marker is missing, not both."""
    df = _event_df(
        [
            ("c1", START_SYMBOL, "2020-01-01T09:00:00"),
            ("c1", "A", "2020-01-01T10:00:00"),
        ]
    )
    with pytest.raises(ValueError, match=END_SYMBOL) as excinfo:
        assert_boundary_markers_present(df, "data/half.xes")
    assert START_SYMBOL not in str(excinfo.value).split("boundary marker")[0]


def test_extract_window_traces_strips_preexisting_boundaries() -> None:
    """Pre-injected ▶/■ are removed and exactly one boundary pair is re-added."""
    df = _event_df(
        [
            ("c1", START_SYMBOL, "2020-01-01T09:00:00"),
            ("c1", "A", "2020-01-01T10:00:00"),
            ("c1", "B", "2020-01-01T11:00:00"),
            ("c1", END_SYMBOL, "2020-01-01T12:00:00"),
        ]
    )
    traces = extract_window_traces(
        df, pd.Timestamp("2020-01-01T00:00:00"), pd.Timestamp("2020-01-01T23:59:59")
    )
    assert len(traces) == 1
    labels = [e["concept:name"] for e in traces[0]]
    # Exactly one boundary pair around the real activities, and no markers inside.
    assert labels == [START_SYMBOL, "A", "B", END_SYMBOL]
    assert labels.count(START_SYMBOL) == 1
    assert labels.count(END_SYMBOL) == 1


def test_extract_window_traces_wraps_raw_traces() -> None:
    """A raw trace (no markers) is wrapped with a single boundary pair."""
    df = _event_df(
        [
            ("c1", "A", "2020-01-01T10:00:00"),
            ("c1", "B", "2020-01-01T11:00:00"),
        ]
    )
    traces = extract_window_traces(
        df, pd.Timestamp("2020-01-01T00:00:00"), pd.Timestamp("2020-01-01T23:59:59")
    )
    assert [e["concept:name"] for e in traces[0]] == [
        START_SYMBOL,
        "A",
        "B",
        END_SYMBOL,
    ]
