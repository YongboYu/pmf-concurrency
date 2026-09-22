"""Every daily series shares one calendar, so a row means the same day in every family.

Windows are cut by row position across families. A family that skipped the days it
did not occur on would misdate its windows, as sparse loop series did before.
"""

import pandas as pd
import pytest

from pmf_concurrency import weekly_models
from pmf_concurrency.bindings import extract_binding_timeseries
from pmf_concurrency.data import slice_log_into_windows
from pmf_concurrency.relations import extract_relation_timeseries


def _log(events: list[tuple[str, str, str]]) -> pd.DataFrame:
    df = pd.DataFrame(events, columns=["case:concept:name", "concept:name", "ts"])
    df["time:timestamp"] = pd.to_datetime(df.pop("ts"))
    df["date"] = df["time:timestamp"].dt.date
    return df


# A loop on day 1 and day 4 only, and no events at all on day 2.
LOG = _log(
    [
        ("c1", "A", "2021-01-01T08:00"),
        ("c1", "B", "2021-01-01T09:00"),
        ("c1", "A", "2021-01-01T10:00"),
        ("c2", "A", "2021-01-03T08:00"),
        ("c2", "C", "2021-01-03T09:00"),
        ("c3", "A", "2021-01-04T08:00"),
        ("c3", "B", "2021-01-04T09:00"),
        ("c3", "A", "2021-01-04T10:00"),
    ]
)
CALENDAR = pd.date_range("2021-01-01", "2021-01-04", freq="D")


def test_relation_series_cover_every_day_of_the_log() -> None:
    daily_A, daily_AB, daily_ABA = extract_relation_timeseries(LOG)

    for frame in (daily_A, daily_AB, daily_ABA):
        assert list(frame.index) == list(CALENDAR)
    assert daily_A.loc["2021-01-02"].sum() == 0
    assert list(daily_ABA["A→B→A"]) == [1, 0, 0, 1]


def test_binding_series_cover_every_day_of_the_log() -> None:
    daily_IB, daily_OB = extract_binding_timeseries(slice_log_into_windows(LOG))

    for frame in (daily_IB, daily_OB):
        assert list(frame.index) == list(CALENDAR)
        assert frame.loc["2021-01-02"].sum() == 0


def test_weekly_models_reject_series_on_different_calendars(tmp_path) -> None:
    days = pd.date_range("2021-01-01", periods=7, freq="D")
    test_data = {
        "A": pd.DataFrame({"A": [1] * 7}, index=days),
        "AB": pd.DataFrame({"A→B": [1] * 7}, index=days),
        "ABA": pd.DataFrame({"A→B→A": [1] * 3}, index=days[[0, 3, 6]]),
        "IB": pd.DataFrame({"B:{A}": [1] * 7}, index=days),
        "OB": pd.DataFrame({"A:{B}": [1] * 7}, index=days),
    }
    forecasts = {series_type: {} for series_type in test_data}

    with pytest.raises(ValueError, match="ABA"):
        weekly_models.build_weekly_models(
            forecasts,
            test_data,
            str(tmp_path / "Toy_pmf"),
            tD=0.5,
            tL1L=0.5,
            tL2L=0.5,
            tPAT=0.0,
        )
