"""Zero-shot Chronos-2 forecasts with a growing context."""

from functools import cache

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

CHRONOS_MODEL = "amazon/chronos-2"

#: Hub commit of ``amazon/chronos-2`` used for the paper. Its weights are the same in
#: every revision since the model's release; the later commits change documentation.
CHRONOS_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"


@cache
def load_chronos(device: str) -> Chronos2Pipeline:
    """The Chronos-2 pipeline, loaded once per device."""
    print(f"  Loading Chronos-2 ({CHRONOS_MODEL}) on {device}...")
    return Chronos2Pipeline.from_pretrained(
        CHRONOS_MODEL, device_map=device, revision=CHRONOS_REVISION
    )


def _sliding_forecast(
    pipeline: Chronos2Pipeline,
    train_values: np.ndarray,
    test_values: np.ndarray,
    horizon: int,
    seed: int,
) -> np.ndarray:
    """Forecasts of one series, one row per window of ``horizon`` test days.

    The context grows by one observed day per window, from the training length on.
    """
    n_sequences = len(test_values) - horizon + 1
    if n_sequences <= 0:
        return np.array([])

    # Re-seed per series, so the quantile forecasts do not depend on series order.
    torch.manual_seed(seed)

    forecasts = []
    for seq_idx in range(n_sequences):
        context = np.concatenate([train_values, test_values[:seq_idx]])
        if len(context) < 10:
            forecasts.append(np.zeros(horizon))
            continue

        context_tensor = (
            torch.tensor(context, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )
        forecast = pipeline.predict(context_tensor, prediction_length=horizon)
        if isinstance(forecast, list):
            forecast = forecast[0]
        # Chronos-2 returns quantiles, not samples: (n_quantiles, horizon) here. The
        # median over its 21 levels is the 0.5 quantile unless the levels cross.
        quantiles = forecast[0]
        forecasts.append(torch.median(quantiles, dim=0).values.cpu().numpy())

    return np.array(forecasts)


def chronos_sliding_forecast(
    train_series: pd.DataFrame,
    test_series: pd.DataFrame,
    horizon: int = 7,
    device: str = "cpu",
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Forecast every series present in both training and test data.

    Each series gets test_length - horizon + 1 forecasts of length ``horizon``, with
    context train + test[:i].

    Returns:
        Dict of series name -> array of shape (n_sequences, horizon)
    """
    if train_series.empty or test_series.empty:
        return {}

    # sorted() fixes the order of series.
    common_columns = sorted(set(train_series.columns) & set(test_series.columns))
    if not common_columns:
        return {}

    pipeline = load_chronos(device)
    results: dict[str, np.ndarray] = {}
    total = len(common_columns)
    for i, col in enumerate(common_columns):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"    Forecasting series {i + 1}/{total}...")
        results[col] = _sliding_forecast(
            pipeline,
            train_series[col].values.astype(np.float32),
            test_series[col].values.astype(np.float32),
            horizon,
            seed,
        )

    return results
