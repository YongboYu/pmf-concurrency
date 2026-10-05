"""Every unit says why it has no number, and every mean says over how many units.

A unit that finished without its main metric (F1, or ER) is ``unavailable``, with a
``failure_category`` from the evaluator's own check. A model that accepts nothing is
``ok``, scored, with precision and F1 undefined. The summary gives one count per status
and, next to each mean, the number of units with a finite value of that metric.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pandas as pd
import pm4py
import pytest

from pmf_concurrency import entropic_relevance, precision
from pmf_concurrency.conformance import extract_window_traces
from tests.test_empty_model import DISCONNECTED, END, START, TRACES, _events

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
NOT_EASY_SOUND = (
    "trying to apply alignments on a Petri net that is not a easy sound net!!"
)


def _load_evaluate():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("evaluate", SCRIPTS / "evaluate.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def evaluate():
    return _load_evaluate()


@pytest.fixture(scope="module")
def ordinary() -> dict:
    from pmf_concurrency.causal_net import fodina_to_json, mine_fodina

    pm = mine_fodina([TRACES[1]] * 2)
    assert pm is not None
    return fodina_to_json(pm, {"type": "sliding_direct_mined"})


def _unit(evaluate, evaluator: str, model: dict, traces=TRACES):
    events = _events(traces)
    window_traces = extract_window_traces(events, START, END)
    return evaluate._evaluate_cell(
        evaluator, model, events, window_traces, True, "tau", 12.0, 20.0
    )


def _raise(message: str):
    def _fail(*_args, **_kwargs):
        raise Exception(message)  # pm4py raises a plain Exception

    return _fail


def test_an_ordinary_unit_is_ok_and_carries_the_guard_measurements(
    evaluate, ordinary
) -> None:
    status, fields = _unit(evaluate, "token_etc", ordinary)

    assert status == "ok"
    assert fields["failure_category"] == ""
    assert fields["exit_code"] == 0
    assert fields["peak_rss_gb"] > 0
    assert math.isfinite(fields["f1"])


def test_an_alignment_on_a_net_that_is_not_easy_sound_is_unavailable(
    evaluate, ordinary, monkeypatch
) -> None:
    monkeypatch.setattr(pm4py, "fitness_alignments", _raise(NOT_EASY_SOUND))

    status, fields = _unit(evaluate, "alignment", ordinary)

    assert status == "unavailable"
    assert fields["failure_category"] == "not_easy_sound"
    assert "easy sound" in fields["guard_reason"]


def test_align_etc_on_a_net_that_is_not_easy_sound_keeps_its_fitness(
    evaluate, ordinary, monkeypatch
) -> None:
    from pm4py.algo.evaluation.precision import algorithm as precision_evaluator

    monkeypatch.setattr(precision_evaluator, "apply", _raise(NOT_EASY_SOUND))

    status, fields = _unit(evaluate, "token_align_etc", ordinary)

    assert status == "unavailable"
    assert fields["failure_category"] == "not_easy_sound"
    assert math.isfinite(fields["log_fitness"])


def test_a_failed_etc_precision_is_unavailable(evaluate, ordinary, monkeypatch) -> None:
    monkeypatch.setattr(
        precision, "etc_precision_with_diagnostics", _raise("no prefixes")
    )

    status, fields = _unit(evaluate, "token_etc", ordinary)

    assert status == "unavailable"
    assert fields["failure_category"] == "precision_failed"
    assert fields["guard_reason"] == "Exception('no prefixes')"


def test_a_failed_er_automaton_is_unavailable(evaluate, ordinary, monkeypatch) -> None:
    monkeypatch.setattr(
        entropic_relevance, "petri_net_to_transitions", _raise("bad marking")
    )

    status, fields = _unit(evaluate, "er", ordinary)

    assert status == "unavailable"
    assert fields["failure_category"] == "er_automaton_failed"
    assert "bad marking" in fields["guard_reason"]


@pytest.mark.parametrize("evaluator", ["token_etc", "alignment", "er"])
def test_an_empty_window_is_unavailable_and_not_evaluated(evaluate, evaluator) -> None:
    status, fields = _unit(evaluate, evaluator, DISCONNECTED, traces=[])

    assert status == "unavailable"
    assert fields["failure_category"] == "empty_window"


def test_a_model_that_accepts_nothing_is_ok(evaluate) -> None:
    status, fields = _unit(evaluate, "token_etc", DISCONNECTED)

    assert status == "ok"
    assert fields["failure_category"] == "empty_model"
    assert fields["guard_reason"] == "empty model"


def test_a_unit_over_its_time_budget_keeps_its_category(evaluate, ordinary) -> None:
    import time

    def _slow(*_args, **_kwargs):
        time.sleep(60)

    status, fields = evaluate._run_unit(_slow, (), {}, "token_etc", ordinary, 12, 0.02)

    assert status == "timeout"
    assert fields["failure_category"] == "time_budget"
    assert fields["peak_rss_gb"] > 0


def test_each_mean_counts_its_own_finite_values(evaluate) -> None:
    """Coverage is per metric: the status counts are diagnostics, not denominators."""
    nan = float("nan")
    base = {
        "dataset": "sepsis",
        "model_type": "forecast",
        "mode": "case_complete",
        "reduce_level": "tau",
        "evaluator": "alignment",
    }
    rows = [
        {**base, "window": 1, "guard_status": "ok", "log_fitness": 0.9,
         "precision": 0.8, "f1": 0.847},
        {**base, "window": 2, "guard_status": "ok", "guard_reason": "empty model",
         "failure_category": "empty_model", "log_fitness": 0.0, "precision": nan,
         "f1": nan},
        {**base, "window": 3, "guard_status": "unavailable",
         "failure_category": "not_easy_sound", "log_fitness": nan},
        {**base, "window": 4, "guard_status": "timeout",
         "failure_category": "time_budget"},
    ]  # fmt: skip
    results = pd.DataFrame(rows).reindex(columns=evaluate.CSV_COLUMNS)

    (summary,) = evaluate.summarise(results).to_dict("records")

    assert summary["n_expected"] == 4
    assert summary["n_log_fitness"] == 2
    assert summary["log_fitness_mean"] == pytest.approx(0.45)
    assert summary["n_f1"] == 1
    assert (summary["n_ok"], summary["n_unavailable"], summary["n_timeout"]) == (
        2,
        1,
        1,
    )
    assert summary["n_empty_model"] == 1
    assert summary["n_over_budget"] == 1
    assert summary["n_oom"] == summary["n_host_oom"] == summary["n_crashed"] == 0


def test_over_budget_is_timeout_and_oom_only(evaluate) -> None:
    base = {
        "dataset": "sepsis",
        "model_type": "re_mined",
        "mode": "case_complete",
        "reduce_level": "none",
        "evaluator": "token_etc",
    }
    statuses = ["timeout", "oom", "host_oom", "crashed", "error", "unavailable"]
    rows = [{**base, "window": i, "guard_status": s} for i, s in enumerate(statuses, 1)]
    results = pd.DataFrame(rows).reindex(columns=evaluate.CSV_COLUMNS)

    (summary,) = evaluate.summarise(results).to_dict("records")

    assert summary["n_over_budget"] == 2
    assert sum(summary[f"n_{s}"] for s in statuses) == 6


def test_failure_counts_say_why_units_have_no_number(evaluate) -> None:
    """The boundary analysis counts units per category, by log, family and evaluator."""
    base = {
        "dataset": "sepsis",
        "model_type": "re_mined",
        "mode": "case_complete",
        "reduce_level": "tau",
        "evaluator": "alignment",
    }
    rows = [
        {**base, "window": 1, "guard_status": "ok", "failure_category": ""},
        {**base, "window": 2, "guard_status": "unavailable",
         "failure_category": "not_easy_sound"},
        {**base, "window": 3, "guard_status": "unavailable",
         "failure_category": "not_easy_sound"},
        {**base, "window": 4, "guard_status": "crashed",
         "failure_category": "signal_SIGSEGV"},
    ]  # fmt: skip
    results = pd.DataFrame(rows).reindex(columns=evaluate.CSV_COLUMNS)

    counts = evaluate.failure_counts(results).set_index("failure_category")

    assert counts.loc["not_easy_sound", "n_units"] == 2
    assert counts.loc["signal_SIGSEGV", "guard_status"] == "crashed"
    assert "" not in counts.index


def test_the_summary_reports_memory_against_the_budget(evaluate) -> None:
    row = {
        "dataset": "sepsis",
        "model_type": "static",
        "mode": "case_complete",
        "reduce_level": "tau",
        "evaluator": "token_etc",
        "window": 1,
        "guard_status": "ok",
        "memory_peak_gb": 3.0,
    }
    results = pd.DataFrame([row]).reindex(columns=evaluate.CSV_COLUMNS)

    (summary,) = evaluate.summarise(results).to_dict("records")

    assert summary["memory_peak_gb_mean"] == 3.0
    assert summary["n_memory_peak_gb"] == 1
