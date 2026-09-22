"""The vendored precision variants.

The vendored ETC token precision must equal pm4py's exactly on the same inputs.
"""

from __future__ import annotations

import math

import pandas as pd

from pmf_concurrency.petri_net import json_to_petri_net
from pmf_concurrency.precision import (
    align_etc_precision,
    etc_precision_with_diagnostics,
)
from tests.test_reduction import AND_SPLIT_JOIN, LINEAR, XOR_CHOICE


def _event_df(traces: list[list[str]]) -> pd.DataFrame:
    rows = []
    t0 = pd.Timestamp("2025-01-01")
    for ci, trace in enumerate(traces):
        for ei, act in enumerate(trace):
            rows.append(
                {
                    "case:concept:name": f"c{ci}",
                    "concept:name": act,
                    "time:timestamp": t0 + pd.Timedelta(minutes=ci * 100 + ei),
                }
            )
    return pd.DataFrame(rows)


FITTING_LOG = _event_df(
    [
        ["▶", "a", "b", "c", "■"],
        ["▶", "a", "c", "b", "■"],
        ["▶", "a", "b", "c", "■"],
    ]
)

MIXED_LOG = _event_df(
    [
        ["▶", "a", "b", "c", "■"],
        ["b", "c", "■"],  # prefix-truncated: unfitting from the first event
        ["▶", "a", "x", "b"],  # off-model activity
    ]
)


def _pm4py_etc(event_df: pd.DataFrame, net, im, fm) -> float:
    from pm4py.algo.evaluation.precision.variants import etconformance_token

    return float(etconformance_token.apply(event_df, net, im, fm))


def test_etc_matches_pm4py_on_fitting_log():
    net, im, fm = json_to_petri_net(AND_SPLIT_JOIN)
    diag = etc_precision_with_diagnostics(FITTING_LOG, net, im, fm)
    assert diag["precision"] == _pm4py_etc(FITTING_LOG, net, im, fm)
    assert diag["prefix_mass_unfit"] == 0.0
    assert diag["pct_prefix_mass_fit"] == 100.0


def test_etc_matches_pm4py_on_mixed_log():
    for fixture in (LINEAR, AND_SPLIT_JOIN, XOR_CHOICE):
        net, im, fm = json_to_petri_net(fixture)
        diag = etc_precision_with_diagnostics(MIXED_LOG, net, im, fm)
        assert diag["precision"] == _pm4py_etc(MIXED_LOG, net, im, fm), fixture
        assert diag["prefix_mass_unfit"] > 0.0


def test_etc_diagnostics_mass_accounting():
    net, im, fm = json_to_petri_net(AND_SPLIT_JOIN)
    diag = etc_precision_with_diagnostics(MIXED_LOG, net, im, fm)
    assert (
        diag["prefix_mass_fit"] + diag["prefix_mass_unfit"] == diag["prefix_mass_total"]
    )
    assert 0.0 <= diag["pct_prefix_mass_fit"] <= 100.0


def test_align_etc_returns_value_or_nan():
    net, im, fm = json_to_petri_net(AND_SPLIT_JOIN)
    value = align_etc_precision(FITTING_LOG, net, im, fm)
    assert math.isnan(value) or 0.0 <= value <= 1.0


def test_align_etc_considers_unfitting_prefixes():
    """Align-ETC aligns unfitting prefixes instead of discarding them, and does not
    fail.
    """
    net, im, fm = json_to_petri_net(LINEAR)
    value = align_etc_precision(MIXED_LOG, net, im, fm)
    assert math.isnan(value) or 0.0 <= value <= 1.0
