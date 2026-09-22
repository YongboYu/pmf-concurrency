"""A unit that runs out of memory is recorded as ``oom``, wherever it runs out.

The guard maps ``MemoryError`` to ``oom``. Precision, alignments and entropic relevance
turn other failures into NaN, so they must let ``MemoryError`` through.
"""

from __future__ import annotations

import math

import pandas as pd
import pm4py
import pytest
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator

from pmf_concurrency import entropic_relevance, precision
from pmf_concurrency.causal_net import fodina_to_json, mine_fodina
from pmf_concurrency.conformance import conformance_metrics
from pmf_concurrency.data import END_SYMBOL, START_SYMBOL
from pmf_concurrency.guard import run_guarded

TRACE = [START_SYMBOL, "A", "B", END_SYMBOL]


@pytest.fixture(autouse=True)
def _inline(monkeypatch):
    """Run units in-process, so the patches below apply."""
    monkeypatch.setenv("PMF_GUARD", "0")


@pytest.fixture
def model() -> dict:
    pm = mine_fodina([TRACE, TRACE])
    assert pm is not None
    return fodina_to_json(pm, {"type": "sliding_direct_mined"})


@pytest.fixture
def events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case:concept:name": ["c1"] * len(TRACE),
            "concept:name": TRACE,
            "time:timestamp": pd.date_range("2021-03-01 08:00", periods=4, freq="h"),
        }
    )


def _out_of_memory(*_args, **_kwargs):
    raise MemoryError


def _failure(*_args, **_kwargs):
    raise ValueError("not easy sound")


def test_etc_precision(monkeypatch, model, events) -> None:
    monkeypatch.setattr(precision, "etc_precision_with_diagnostics", _out_of_memory)
    assert run_guarded(conformance_metrics, model, events) == ("oom", None)


def test_align_etc_precision(monkeypatch, model, events) -> None:
    monkeypatch.setattr(precision_evaluator, "apply", _out_of_memory)
    status, _ = run_guarded(
        conformance_metrics, model, events, precision_variant="align_etc"
    )
    assert status == "oom"


@pytest.mark.parametrize("step", ["fitness_alignments", "precision_alignments"])
def test_alignments(monkeypatch, model, events, step) -> None:
    monkeypatch.setattr(pm4py, step, _out_of_memory)
    status, _ = run_guarded(conformance_metrics, model, events, method="alignment")
    assert status == "oom"


def test_entropic_relevance(monkeypatch, model) -> None:
    monkeypatch.setattr(entropic_relevance, "petri_net_to_transitions", _out_of_memory)
    traces = [[{"concept:name": a} for a in TRACE]]
    status, _ = run_guarded(entropic_relevance.compute_er, model, traces)
    assert status == "oom"


def test_other_failures_still_give_nan(monkeypatch, model, events) -> None:
    monkeypatch.setattr(pm4py, "fitness_alignments", _failure)
    status, metrics = run_guarded(
        conformance_metrics, model, events, method="alignment"
    )
    assert status == "ok"
    assert math.isnan(metrics["log_fitness"])
