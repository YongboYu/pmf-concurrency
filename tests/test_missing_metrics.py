"""A metric that could not be computed stays NaN, and nothing derived from it is zero.

F1 needs both fitness and precision, and ER's fitting ratio means nothing without the
ER value. The evaluation summary counts such units apart from the measured ones.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

from pmf_concurrency import conformance, entropic_relevance, precision
from pmf_concurrency.causal_net import fodina_to_json, mine_fodina
from pmf_concurrency.data import END_SYMBOL, START_SYMBOL

TRACE = [START_SYMBOL, "A", "B", END_SYMBOL]
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


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


def _failure(*_args, **_kwargs):
    raise ValueError("not easy sound")


def test_f1_is_missing_when_precision_is(monkeypatch, model, events) -> None:
    monkeypatch.setattr(precision, "etc_precision_with_diagnostics", _failure)
    metrics = conformance.conformance_metrics(model, events)
    assert metrics["log_fitness"] == 1.0
    assert math.isnan(metrics["precision"])
    assert math.isnan(metrics["f1"])


def test_f1_is_zero_only_for_a_measured_zero() -> None:
    assert conformance._f1(0.0, 0.8) == 0.0
    assert math.isnan(conformance._f1(float("nan"), 0.8))


def test_a_failed_er_automaton_has_no_fitting_ratio(monkeypatch, model) -> None:
    monkeypatch.setattr(entropic_relevance, "petri_net_to_transitions", _failure)
    traces = [[{"concept:name": a} for a in TRACE]]
    er, fitting_ratio, n_traces = entropic_relevance.compute_er(model, traces)
    assert math.isnan(er)
    assert math.isnan(fitting_ratio)
    assert n_traces == 1


def test_no_traces_give_no_fitting_ratio(model) -> None:
    er, fitting_ratio, n_traces = entropic_relevance.compute_er(model, [])
    assert math.isnan(er)
    assert math.isnan(fitting_ratio)
    assert n_traces == 0


def _load_evaluate():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("evaluate", SCRIPTS / "evaluate.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_summary_counts_measured_unavailable_and_intractable_units() -> None:
    evaluate = _load_evaluate()
    nan = float("nan")
    base = {
        "dataset": "sepsis",
        "model_type": "forecast",
        "mode": "case_complete",
        "reduce_level": "none",
    }
    rows = [
        {**base, "window": 1, "evaluator": "er", "guard_status": "ok", "er": 2.0,
         "fitting_ratio": 0.5},
        {**base, "window": 2, "evaluator": "er", "guard_status": "ok", "er": nan,
         "fitting_ratio": nan},
        {**base, "window": 3, "evaluator": "er", "guard_status": "timeout"},
        {**base, "window": 1, "evaluator": "token_etc", "guard_status": "ok",
         "log_fitness": 0.9, "precision": 0.8, "f1": 0.847},
        {**base, "window": 2, "evaluator": "token_etc", "guard_status": "ok",
         "log_fitness": 0.9, "precision": nan, "f1": nan},
    ]  # fmt: skip
    results = pd.DataFrame(rows).reindex(columns=evaluate.CSV_COLUMNS)

    summary = evaluate.summarise(results).set_index("evaluator")

    assert summary.loc["er", "n_units"] == 3
    assert summary.loc["er", "n_intractable"] == 1
    assert summary.loc["er", "n_unavailable"] == 1
    assert summary.loc["er", "er_count"] == 1
    assert summary.loc["er", "fitting_ratio_count"] == 1
    assert summary.loc["er", "fitting_ratio_mean"] == 0.5
    assert summary.loc["token_etc", "n_unavailable"] == 1
    assert summary.loc["token_etc", "f1_count"] == 1
    assert summary.loc["token_etc", "log_fitness_count"] == 2
