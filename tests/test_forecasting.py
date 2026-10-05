"""Chronos-2 forecasts: the pinned revision, the growing context and the 0.5 level.

Chronos-2 itself is replaced by a stub, so these tests check what the pipeline feeds it
and what it keeps from its output, not the forecasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from pmf_concurrency import forecasting

#: The commit ``main`` resolved to for every forecast in the paper.
PAPER_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"


def test_chronos_loads_the_papers_revision_once_per_device(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    def _from_pretrained(name, **kwargs):
        calls.append((name, kwargs))
        return object()

    monkeypatch.setattr(
        forecasting.Chronos2Pipeline, "from_pretrained", staticmethod(_from_pretrained)
    )
    forecasting.load_chronos.cache_clear()
    try:
        first = forecasting.load_chronos("cpu")
        assert forecasting.load_chronos("cpu") is first
    finally:
        forecasting.load_chronos.cache_clear()

    assert calls == [
        ("amazon/chronos-2", {"device_map": "cpu", "revision": PAPER_REVISION})
    ]


#: Chronos-2's quantile levels at the pinned revision.
LEVELS = [0.01, 0.05, *(round(0.1 + 0.05 * i, 2) for i in range(17)), 0.95, 0.99]

#: Values of the 21 quantile rows, before adding the context length. They are out of
#: order, so the levels cross: the 0.5 row holds 19, the middle of the sorted rows 10.
ROWS = torch.randperm(21, generator=torch.Generator().manual_seed(0))


class _StubChronos:
    """Returns Chronos-2's output shape, (1 variate, 21 quantiles, horizon).

    Row j holds ``ROWS[j] + len(context)``.
    """

    quantiles = LEVELS

    def __init__(self) -> None:
        self.contexts: list[np.ndarray] = []

    def predict(self, context: torch.Tensor, prediction_length: int):
        assert context.shape[:2] == (1, 1)
        self.contexts.append(context[0, 0].numpy().copy())
        levels = ROWS.to(torch.float32).reshape(1, 21, 1) + context.shape[-1]
        return [levels.expand(1, 21, prediction_length).clone()]


def _forecast(train_days: int, test_days: int, horizon: int):
    stub = _StubChronos()
    train = np.arange(train_days, dtype=np.float32)
    test = 100 + np.arange(test_days, dtype=np.float32)
    rows = forecasting._sliding_forecast(stub, train, test, horizon, seed=42)
    return stub, train, test, rows


def test_the_context_grows_by_one_observed_day_per_window() -> None:
    stub, train, test, rows = _forecast(train_days=12, test_days=5, horizon=2)

    assert rows.shape == (4, 2)
    for i, context in enumerate(stub.contexts):
        np.testing.assert_array_equal(context, np.concatenate([train, test[:i]]))


def test_each_window_keeps_the_0_5_quantile_level_even_when_levels_cross() -> None:
    _stub, _train, _test, rows = _forecast(train_days=12, test_days=5, horizon=2)

    assert LEVELS[10] == 0.5
    assert int(ROWS[10]) == 19  # the sorted middle would be 10
    expected = [[19 + 12 + i] * 2 for i in range(4)]
    np.testing.assert_array_equal(rows, expected)


def test_a_context_shorter_than_ten_days_forecasts_zeros() -> None:
    stub, _train, _test, rows = _forecast(train_days=8, test_days=5, horizon=2)

    np.testing.assert_array_equal(rows[:2], 0.0)
    np.testing.assert_array_equal(rows[2:], [[29, 29], [30, 30]])
    assert [len(c) for c in stub.contexts] == [10, 11]


def test_every_series_in_both_periods_is_forecast(monkeypatch) -> None:
    stub = _StubChronos()
    monkeypatch.setattr(forecasting, "load_chronos", lambda _device: stub)
    train = pd.DataFrame({"B": np.ones(12), "A": np.ones(12), "only_train": 0.0})
    test = pd.DataFrame({"A": np.ones(3), "B": np.ones(3), "only_test": 0.0})

    result = forecasting.chronos_sliding_forecast(train, test, horizon=2)

    assert list(result) == ["A", "B"]
    assert all(rows.shape == (2, 2) for rows in result.values())


def test_the_sorted_middle_is_recorded_on_request_without_changing_the_forecast() -> (
    None
):
    """Decision 1.6's counts compare the 0.5 level with the old median over levels."""
    stub = _StubChronos()
    train = np.arange(12, dtype=np.float32)
    test = 100 + np.arange(5, dtype=np.float32)
    middle: list[np.ndarray] = []

    rows = forecasting._sliding_forecast(
        stub, train, test, horizon=2, seed=42, median_over_levels=middle
    )

    np.testing.assert_array_equal(rows, [[19 + 12 + i] * 2 for i in range(4)])
    np.testing.assert_array_equal(
        np.array(middle), [[10 + 12 + i] * 2 for i in range(4)]
    )
