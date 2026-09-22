"""Chronos-2 forecasts: the pinned revision, the growing context and the median.

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


class _StubChronos:
    """Returns Chronos-2's output shape, (1 variate, 21 quantiles, horizon).

    The quantile rows are shuffled, and row q holds ``q + len(context)``, so the median
    over the rows is ``10 + len(context)``.
    """

    def __init__(self) -> None:
        self.contexts: list[np.ndarray] = []

    def predict(self, context: torch.Tensor, prediction_length: int):
        assert context.shape[:2] == (1, 1)
        self.contexts.append(context[0, 0].numpy().copy())
        rows = torch.randperm(21, generator=torch.Generator().manual_seed(0))
        levels = rows.to(torch.float32).reshape(1, 21, 1) + context.shape[-1]
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


def test_each_window_keeps_the_median_quantile() -> None:
    _stub, _train, _test, rows = _forecast(train_days=12, test_days=5, horizon=2)

    expected = [[10 + 12 + i] * 2 for i in range(4)]
    np.testing.assert_array_equal(rows, expected)


def test_a_context_shorter_than_ten_days_forecasts_zeros() -> None:
    stub, _train, _test, rows = _forecast(train_days=8, test_days=5, horizon=2)

    np.testing.assert_array_equal(rows[:2], 0.0)
    np.testing.assert_array_equal(rows[2:], [[20, 20], [21, 21]])
    assert [len(c) for c in stub.contexts] == [10, 11]


def test_every_series_in_both_periods_is_forecast(monkeypatch) -> None:
    stub = _StubChronos()
    monkeypatch.setattr(forecasting, "load_chronos", lambda _device: stub)
    train = pd.DataFrame({"B": np.ones(12), "A": np.ones(12), "only_train": 0.0})
    test = pd.DataFrame({"A": np.ones(3), "B": np.ones(3), "only_test": 0.0})

    result = forecasting.chronos_sliding_forecast(train, test, horizon=2)

    assert list(result) == ["A", "B"]
    assert all(rows.shape == (2, 2) for rows in result.values())
