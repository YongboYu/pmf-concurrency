"""A model that accepts nothing, and a window without events.

A Forecast model is empty when no arc clears the dependency threshold. It can also keep
a few activities without an arc between them, so its final marking is out of reach. Both
accept no trace, and every evaluator scores them alike: fitness 0, no prefix replays,
precision and F1 undefined, ER as computed. The unit is ``ok`` with reason
``empty model``, and the summary counts it apart from the units without a result.

A window without events has nothing to score: every metric is NaN, and the unit is
``unavailable`` with category ``empty_window``.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

from pmf_concurrency import conformance
from pmf_concurrency.causal_net import fodina_to_json, mine_fodina
from pmf_concurrency.conformance import conformance_metrics, extract_window_traces
from pmf_concurrency.data import END_SYMBOL, START_SYMBOL
from pmf_concurrency.entropic_relevance import compute_er
from pmf_concurrency.petri_net import accepts_nothing

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
S, E = START_SYMBOL, END_SYMBOL


def _model(nodes: list[str], edges: list[tuple[str, str]]) -> dict:
    return {
        "metadata": {"type": "sliding_forecast"},
        "graph": {
            "directed": True,
            "nodes": [
                {"id": n, "data": {"label": n, "is_binding": False, "frequency": 2.0}}
                for n in nodes
            ],
            "edges": [{"source": a, "target": b, "data": {}} for a, b in edges],
        },
    }


#: No arc cleared tD, so no nodes at all, as in the sparse end of Sepsis.
EMPTY = _model([], [])
#: Two activities and no arc between them, as seen on Sepsis.
DISCONNECTED = _model(["Release A", E], [])
#: The same two, joined: an ordinary net.
JOINED = {
    "metadata": {"type": "sliding_forecast"},
    "graph": {
        "directed": True,
        "nodes": [
            {"id": "Release A", "data": {"label": "Release A", "frequency": 3.0}},
            {"id": "o", "data": {"label": "o", "is_binding": True, "frequency": 2.0}},
            {"id": E, "data": {"label": E, "frequency": 6.0}},
        ],
        "edges": [
            {"source": "Release A", "target": "o", "data": {"frequency": 2.0}},
            {"source": "o", "target": E, "data": {"frequency": 2.0}},
        ],
    },
}
ACCEPTS_NOTHING = [EMPTY, DISCONNECTED]

TRACES = [[S, "ER Registration", "Release A", E], [S, "Release A", E]]
START, END = pd.Timestamp("2015-03-01"), pd.Timestamp("2015-03-02")


@pytest.fixture(scope="module")
def ordinary() -> dict:
    pm = mine_fodina([TRACES[1]] * 2)
    assert pm is not None
    return fodina_to_json(pm, {"type": "sliding_direct_mined"})


def _events(traces: list[list[str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case:concept:name": f"c{ci}",
                "concept:name": activity,
                "time:timestamp": START + pd.Timedelta(hours=ci + 1, minutes=ei),
            }
            for ci, trace in enumerate(traces)
            for ei, activity in enumerate(trace)
        ]
    )


def test_a_net_without_a_path_to_its_final_place_accepts_nothing(ordinary) -> None:
    assert accepts_nothing(EMPTY)
    assert accepts_nothing(DISCONNECTED)
    assert not accepts_nothing(JOINED)
    assert not accepts_nothing(ordinary)


EVALUATORS = {
    "token_etc": {},
    "token_align_etc": {"precision_variant": "align_etc"},
    "alignment": {"method": "alignment"},
}


@pytest.mark.parametrize("reduce_level", ["none", "tau"])
@pytest.mark.parametrize("evaluator", list(EVALUATORS))
@pytest.mark.parametrize("model", ACCEPTS_NOTHING, ids=["empty", "disconnected"])
def test_a_model_that_accepts_nothing_fits_nothing(
    model, evaluator, reduce_level, monkeypatch
) -> None:
    import pm4py

    def _never(*_args, **_kwargs):
        raise AssertionError("pm4py is not asked to replay an empty model")

    monkeypatch.setattr(pm4py, "fitness_alignments", _never)
    monkeypatch.setattr(pm4py, "precision_alignments", _never)

    metrics = conformance_metrics(
        model, _events(TRACES), reduce_level=reduce_level, **EVALUATORS[evaluator]
    )

    assert metrics["log_fitness"] == 0.0
    assert metrics["avg_trace_fitness"] == 0.0
    assert metrics["perc_fit_traces"] == 0.0
    assert math.isnan(metrics["precision"])
    assert math.isnan(metrics["f1"])
    assert metrics["pct_events_unseen"] > 0.0
    if evaluator == "token_etc":
        assert metrics["pct_prefix_mass_fit"] == 0.0
        assert metrics["prefix_mass_fit"] == 0.0
        assert metrics["prefix_mass_total"] == 5.0  # 3 + 2 non-empty prefixes


@pytest.mark.parametrize("partial_trace", [True, False])
@pytest.mark.parametrize("model", ACCEPTS_NOTHING, ids=["empty", "disconnected"])
def test_er_scores_a_model_that_accepts_nothing(model, partial_trace) -> None:
    traces = extract_window_traces(_events(TRACES), START, END)
    er, fitting_ratio, n_traces = compute_er(model, traces, partial_trace=partial_trace)
    assert math.isfinite(er)
    assert n_traces == 2
    if model is EMPTY or not partial_trace:
        assert fitting_ratio == 0.0


@pytest.mark.parametrize("partial_trace", [True, False])
@pytest.mark.parametrize("evaluator", list(EVALUATORS))
@pytest.mark.parametrize("kind", ["ordinary", "empty", "disconnected"])
def test_a_window_without_events_has_no_metric(
    kind, evaluator, partial_trace, ordinary
) -> None:
    model = {"ordinary": ordinary, "empty": EMPTY, "disconnected": DISCONNECTED}[kind]
    metrics = conformance_metrics(model, _events([]).iloc[0:0], **EVALUATORS[evaluator])
    assert all(math.isnan(metrics[k]) for k in conformance.METRIC_KEYS)
    er, fitting_ratio, n_traces = compute_er(model, [], partial_trace=partial_trace)
    assert math.isnan(er) and math.isnan(fitting_ratio) and n_traces == 0


def test_an_unsound_net_with_behaviour_stays_without_a_result(
    ordinary, monkeypatch
) -> None:
    import pm4py

    def _unsound(*_args, **_kwargs):
        raise ValueError("not a easy sound net")

    monkeypatch.setattr(pm4py, "fitness_alignments", _unsound)
    metrics = conformance_metrics(ordinary, _events(TRACES), method="alignment")
    assert all(math.isnan(metrics[k]) for k in conformance.METRIC_KEYS)


def _load_evaluate():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("evaluate", SCRIPTS / "evaluate.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("evaluator", ["token_etc", "alignment", "er"])
def test_the_unit_is_ok_with_reason_empty_model(evaluator, monkeypatch) -> None:
    monkeypatch.setenv("PMF_GUARD", "0")
    evaluate = _load_evaluate()
    events = _events(TRACES)
    traces = extract_window_traces(events, START, END)

    status, metrics = evaluate._evaluate_cell(
        evaluator, DISCONNECTED, events, traces, True, "tau", 12.0, 20.0
    )

    assert status == "ok"
    assert metrics["guard_reason"] == "empty model"
    assert metrics["failure_category"] == "empty_model"


def test_an_empty_window_is_not_an_empty_model(monkeypatch) -> None:
    monkeypatch.setenv("PMF_GUARD", "0")
    evaluate = _load_evaluate()

    status, metrics = evaluate._evaluate_cell(
        "token_etc", EMPTY, _events([]).iloc[0:0], [], True, "tau", 12.0, 20.0
    )

    assert status == "unavailable"
    assert metrics["failure_category"] == "empty_window"


def test_the_summary_counts_empty_models_apart() -> None:
    evaluate = _load_evaluate()
    nan = float("nan")
    base = {
        "dataset": "sepsis",
        "model_type": "forecast",
        "mode": "case_complete",
        "reduce_level": "tau",
        "evaluator": "token_etc",
    }
    rows = [
        {**base, "window": 1, "guard_status": "ok", "log_fitness": 0.9,
         "precision": 0.8, "f1": 0.847},
        {**base, "window": 2, "guard_status": "ok", "guard_reason": "empty model",
         "log_fitness": 0.0, "precision": nan, "f1": nan},
        {**base, "window": 3, "guard_status": "unavailable", "log_fitness": 0.9,
         "precision": nan, "f1": nan},
        {**base, "window": 4, "guard_status": "timeout"},
    ]  # fmt: skip
    results = pd.DataFrame(rows).reindex(columns=evaluate.CSV_COLUMNS)

    (summary,) = evaluate.summarise(results).to_dict("records")

    assert summary["n_expected"] == 4
    assert summary["n_empty_model"] == 1
    assert summary["n_unavailable"] == 1
    assert summary["n_timeout"] == 1
    assert summary["n_log_fitness"] == 3
    assert summary["log_fitness_mean"] == pytest.approx(0.6)
    assert summary["n_f1"] == 1
