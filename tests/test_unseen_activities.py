"""Events whose activity the net lacks.

pm4py's token replay skips such events: the net A B replays A U B with fitness 1. The
token-count fitness stays pm4py's value, and next to it the share of events the net
cannot label is reported. A trace or prefix holding such an event never counts as
fitting. pm4py's own flag for this shares its state across the log, so the evaluation
checks each trace itself.

A trace fits only if it has no missing tokens, no remaining tokens and no activity the
net lacks (the perfect-fit share).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from pmf_concurrency.causal_net import fodina_to_json, mine_fodina
from pmf_concurrency.conformance import conformance_metrics, extract_window_traces
from pmf_concurrency.data import END_SYMBOL, START_SYMBOL
from pmf_concurrency.entropic_relevance import compute_er
from pmf_concurrency.petri_net import json_to_petri_net
from pmf_concurrency.precision import (
    align_etc_precision,
    etc_precision_with_diagnostics,
)

S, E = START_SYMBOL, END_SYMBOL
SEEN = [S, "A", "B", E]
UNSEEN = [S, "A", "U", "B", E]


@pytest.fixture(scope="module")
def model() -> dict:
    pm = mine_fodina([SEEN, SEEN])
    assert pm is not None
    return fodina_to_json(pm, {"type": "sliding_direct_mined"})


def _events(traces: list[list[str]]) -> pd.DataFrame:
    t0 = pd.Timestamp("2021-03-01 08:00")
    return pd.DataFrame(
        [
            {
                "case:concept:name": f"c{ci}",
                "concept:name": activity,
                "time:timestamp": t0 + pd.Timedelta(minutes=100 * ci + ei),
            }
            for ci, trace in enumerate(traces)
            for ei, activity in enumerate(trace)
        ]
    )


def _pm4py_token_fitness(events: pd.DataFrame, model: dict) -> dict:
    from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
    from pm4py.algo.evaluation.replay_fitness.variants import token_replay as fit_eval

    net, im, fm = json_to_petri_net(model)
    replayed = token_replay.apply(
        events, net, im, fm, parameters={"show_progress_bar": False}
    )
    return fit_eval.evaluate(replayed)


MIXED_ORDERS = [[UNSEEN, UNSEEN, SEEN], [SEEN, UNSEEN, UNSEEN]]


@pytest.mark.parametrize("traces", MIXED_ORDERS)
def test_only_the_traces_with_an_unseen_activity_are_unfit(model, traces) -> None:
    events = _events(traces)
    metrics = conformance_metrics(model, events)

    assert metrics["perc_fit_traces"] == pytest.approx(100 / 3)
    pm4py_fit = _pm4py_token_fitness(events, model)
    assert pm4py_fit["log_fitness"] == 1.0  # pm4py ignores U
    assert metrics["log_fitness"] == pm4py_fit["log_fitness"]
    assert metrics["pct_events_unseen"] == pytest.approx(100 * 2 / 14)


@pytest.mark.parametrize("traces", MIXED_ORDERS)
def test_etc_counts_no_prefix_with_an_unseen_activity(model, traces) -> None:
    net, im, fm = json_to_petri_net(model)
    diag = etc_precision_with_diagnostics(_events(traces), net, im, fm)

    # Prefixes ▶, ▶A of all three traces and ▶AB fit; ▶AU and ▶AUB (twice each) do not.
    assert diag["prefix_mass_total"] == 11
    assert diag["prefix_mass_fit"] == 7
    assert diag["prefix_mass_unfit"] == 4


@pytest.fixture(scope="module")
def choice() -> tuple:
    """The net ▶ A (B | C) ■, where precision can fall below 1."""
    pm = mine_fodina([[S, "A", "B", E], [S, "A", "C", E]] * 2)
    assert pm is not None
    return json_to_petri_net(fodina_to_json(pm, {"type": "sliding_direct_mined"}))


def test_etc_ignores_what_follows_an_unseen_activity(choice) -> None:
    net, im, fm = choice
    others = [[S, "A", "B", E], [S, "A", "C", E]]
    then_b = etc_precision_with_diagnostics(
        _events([[S, "A", "U", "B", E], *others]), net, im, fm
    )
    then_end = etc_precision_with_diagnostics(
        _events([[S, "A", "U", E], *others]), net, im, fm
    )
    assert then_b["precision"] == then_end["precision"]
    assert then_b["prefix_mass_fit"] == then_end["prefix_mass_fit"]


def test_align_etc_aligns_an_unseen_activity_as_a_log_move(choice) -> None:
    """pm4py's Align-ETC: before U the log continues with an activity the net cannot
    do, so everything the net enables there escapes and precision drops.
    """
    net, im, fm = choice
    with_u = align_etc_precision(_events([[S, "A", "U", "B", E]]), net, im, fm)
    without = align_etc_precision(_events([[S, "A", "B", E]]), net, im, fm)
    assert math.isfinite(with_u)
    assert with_u < without


@pytest.mark.parametrize("traces", MIXED_ORDERS)
def test_alignments_count_an_unseen_activity_as_a_misfit(model, traces) -> None:
    metrics = conformance_metrics(model, _events(traces), method="alignment")

    assert metrics["perc_fit_traces"] == pytest.approx(100 / 3)
    assert metrics["log_fitness"] < 1.0
    assert metrics["pct_events_unseen"] == pytest.approx(100 * 2 / 14)


def test_er_scores_a_trace_with_an_unseen_activity_as_non_fitting(model) -> None:
    events = _events(MIXED_ORDERS[0])
    traces = extract_window_traces(
        events, pd.Timestamp("2021-03-01"), pd.Timestamp("2021-03-02")
    )
    er, fitting_ratio, n_traces = compute_er(model, traces)

    assert n_traces == 3
    assert fitting_ratio == pytest.approx(1 / 3)
    assert math.isfinite(er)


@pytest.mark.parametrize(
    ("trace", "why"),
    [
        ([S, "B", E], "missing token"),
        ([S, "A"], "remaining token"),
        (UNSEEN, "activity the net lacks"),
    ],
)
def test_each_condition_alone_makes_a_trace_imperfect(model, trace, why) -> None:
    metrics = conformance_metrics(model, _events([SEEN, trace]))
    assert metrics["perc_fit_traces"] == 50.0, why


def test_without_the_remaining_token_check_only_that_condition_is_dropped(
    model,
) -> None:
    events = _events([SEEN, [S, "A"], UNSEEN])
    metrics = conformance_metrics(model, events, consider_remaining=False)
    assert metrics["perc_fit_traces"] == pytest.approx(200 / 3)
