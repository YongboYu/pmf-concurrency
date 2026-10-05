"""Conformance checking of the weekly models against the test log.

Fitness and precision by token replay or alignments, and entropic relevance. Three
replay modes handle traces cut by the window boundaries:

* ``window_raw``: events inside the window, pm4py defaults. Truncated traces
  are penalized at both ends.
* ``window_no_remain``: events inside the window, and a trace with remaining
  tokens can still count as fitting. pm4py's fitness values still include the
  remaining tokens.
* ``case_complete``: the full trace of every case active in the window,
  taken from the reference log. The paper reports this mode.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Literal

import pandas as pd

from .data import END_SYMBOL, START_SYMBOL, sort_events
from .petri_net import accepts_nothing, json_to_petri_net, visible_labels
from .reduction import ReductionLevel, reduce_net

logger = logging.getLogger(__name__)

EvaluationMode = Literal["window_raw", "window_no_remain", "case_complete"]
PrecisionVariant = Literal["etc", "align_etc", "none"]

#: Fitness and precision measures of one unit, in CSV order.
METRIC_KEYS = ("log_fitness", "avg_trace_fitness", "perc_fit_traces", "precision", "f1")


def _nan_metrics() -> dict[str, float]:
    return dict.fromkeys(METRIC_KEYS, float("nan"))


def _f1(fitness: float, precision: float) -> float:
    """Harmonic mean of fitness and precision; NaN if either is missing."""
    if math.isnan(fitness) or math.isnan(precision):
        return float("nan")
    if fitness > 0 and precision > 0:
        return 2 * fitness * precision / (fitness + precision)
    return 0.0


def _empty_model_metrics(
    event_df: pd.DataFrame, precision_variant: PrecisionVariant | None
) -> dict[str, float]:
    """Scores of a model that accepts nothing: no trace and no prefix fits, and
    precision is undefined.
    """
    from .precision import prefix_mass

    metrics = {
        "log_fitness": 0.0,
        "avg_trace_fitness": 0.0,
        "perc_fit_traces": 0.0,
        "precision": float("nan"),
        "f1": float("nan"),
    }
    if precision_variant == "etc":
        mass = prefix_mass(event_df)
        metrics |= {
            **mass,
            "prefix_mass_fit": 0.0,
            "prefix_mass_unfit": mass["prefix_mass_total"],
            "pct_prefix_mass_fit": 0.0 if mass["prefix_mass_total"] else float("nan"),
        }
    return metrics


def _run_token_replay(
    event_df: pd.DataFrame,
    net: Any,
    im: Any,
    fm: Any,
    consider_remaining: bool,
    precision_variant: PrecisionVariant = "etc",
) -> dict[str, float]:
    """Token replay and precision, with an explicit remaining-token flag.

    Uses pm4py's low-level API, since the high-level wrapper always sets
    ``CONSIDER_REMAINING_IN_FITNESS`` to True. The fitness values are pm4py's, which
    skip events the net cannot label. A trace counts as fitting only if pm4py finds it
    fitting and the net labels all its activities.

    ``precision_variant``:
      * ``etc``: ETC token precision (equal to pm4py's), plus the share of
        prefix mass it discards as non-fitting;
      * ``align_etc``: Align-ETC, which aligns non-fitting prefixes;
      * ``none``: no precision.
    """
    from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
    from pm4py.algo.conformance.tokenreplay.variants.token_replay import (
        Parameters as TRParams,
    )
    from pm4py.algo.evaluation.replay_fitness.variants import token_replay as fit_eval
    from pm4py.objects.conversion.log import converter as log_converter

    from .precision import align_etc_precision, etc_precision_with_diagnostics

    t0 = time.perf_counter()
    tr_params = {
        TRParams.CONSIDER_REMAINING_IN_FITNESS: consider_remaining,
        TRParams.SHOW_PROGRESS_BAR: False,
    }
    log = log_converter.apply(event_df, variant=log_converter.Variants.TO_EVENT_LOG)
    aligned_traces = token_replay.apply(log, net, im, fm, parameters=tr_params)
    fit_dict = fit_eval.evaluate(aligned_traces)
    fitness_wall_s = time.perf_counter() - t0

    log_fitness = float(fit_dict.get("log_fitness", float("nan")))
    avg_trace_fitness = float(fit_dict.get("average_trace_fitness", float("nan")))
    labels = visible_labels(net)
    fitting = [
        replayed["trace_is_fit"]
        and all(event["concept:name"] in labels for event in trace)
        for trace, replayed in zip(log, aligned_traces, strict=True)
    ]
    perc_fit = 100.0 * sum(fitting) / len(fitting) if fitting else float("nan")

    extras: dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        if precision_variant == "etc":
            diag = etc_precision_with_diagnostics(event_df, net, im, fm)
            precision = diag.pop("precision")
            extras = diag
        elif precision_variant == "align_etc":
            precision = align_etc_precision(event_df, net, im, fm)
        elif precision_variant == "none":
            precision = float("nan")
        else:
            raise ValueError(f"Unknown precision variant: {precision_variant!r}")
    except MemoryError:
        raise  # the budget guard records it as oom
    except Exception as exc:
        logger.warning("Precision computation failed: %s", exc)
        precision = float("nan")
    precision_wall_s = time.perf_counter() - t0

    return {
        "log_fitness": log_fitness,
        "avg_trace_fitness": avg_trace_fitness,
        "perc_fit_traces": perc_fit,
        "precision": precision,
        "f1": _f1(log_fitness, precision),
        "fitness_wall_s": fitness_wall_s,
        "precision_wall_s": precision_wall_s,
        **extras,
    }


def _run_alignments(
    event_df: pd.DataFrame,
    net: Any,
    im: Any,
    fm: Any,
) -> dict[str, float]:
    """Alignment-based fitness and precision through pm4py's high-level API.

    Alignments need an easy-sound net. If the net fails that check, the result is NaN
    and a warning is logged. An activity the net lacks is a log move, so its trace
    never fits.
    """
    import pm4py

    log = pm4py.convert_to_event_log(event_df)
    t0 = time.perf_counter()
    try:
        fit_dict = pm4py.fitness_alignments(log, net, im, fm)
    except MemoryError:
        raise  # the budget guard records it as oom
    except Exception as exc:
        logger.warning("Alignment fitness failed (likely non-easy-sound net): %s", exc)
        return _nan_metrics()
    fitness_wall_s = time.perf_counter() - t0

    log_fitness = float(fit_dict.get("log_fitness", float("nan")))
    avg_trace_fitness = float(fit_dict.get("average_trace_fitness", float("nan")))
    perc_fit = float(fit_dict.get("percentage_of_fitting_traces", float("nan")))

    t0 = time.perf_counter()
    try:
        precision = float(pm4py.precision_alignments(log, net, im, fm))
    except MemoryError:
        raise  # the budget guard records it as oom
    except Exception as exc:
        logger.warning("Alignment precision failed: %s", exc)
        precision = float("nan")
    precision_wall_s = time.perf_counter() - t0

    return {
        "log_fitness": log_fitness,
        "avg_trace_fitness": avg_trace_fitness,
        "perc_fit_traces": perc_fit,
        "precision": precision,
        "f1": _f1(log_fitness, precision),
        "fitness_wall_s": fitness_wall_s,
        "precision_wall_s": precision_wall_s,
    }


def conformance_metrics(
    model_json: dict[str, Any],
    event_df: pd.DataFrame,
    consider_remaining: bool = True,
    method: Literal["token", "alignment"] = "token",
    reduce_level: ReductionLevel = "none",
    precision_variant: PrecisionVariant = "etc",
) -> dict[str, float]:
    """Fitness and precision of one model against one event log.

    Args:
        model_json:          The model JSON, converted to a pm4py Petri net.
        event_df:            Events to replay.
        consider_remaining:  False disables the remaining-token penalty (token
                             replay only).
        method:              ``token`` (default) or ``alignment``.
        reduce_level:        ``none`` (the WF-net as built), ``tau``
                             (silent series fusion), or ``tau_implicit`` (fusion
                             plus implicit-place removal). See :mod:`.reduction`.
        precision_variant:   ``etc`` (default), ``align_etc`` or ``none``, for
                             token replay. Alignments always use Align-ETC.

    A model that accepts nothing (:func:`.petri_net.accepts_nothing`) is scored
    without replay: fitness and the fitting shares are 0, precision and F1 NaN. A log
    without events gives NaN everywhere.

    Returns:
        Dict with ``log_fitness``, ``avg_trace_fitness``, ``perc_fit_traces``,
        ``precision``, ``f1``, ``pct_events_unseen`` (the share of events whose
        activity the net lacks), wall times per phase and, for reduced nets, the net
        size before and after reduction.
    """
    if event_df.empty:
        return _nan_metrics()

    net, im, fm = json_to_petri_net(model_json)
    unseen = ~event_df["concept:name"].isin(visible_labels(net))
    pct_events_unseen = 100.0 * float(unseen.mean())

    extras: dict[str, float] = {}
    if reduce_level != "none":
        t0 = time.perf_counter()
        net, im, fm, red_stats = reduce_net(net, im, fm, level=reduce_level)
        red_stats["reduce_wall_s"] = time.perf_counter() - t0
        red_stats.pop("reduction_level", None)
        extras = red_stats

    if accepts_nothing(model_json):
        metrics = _empty_model_metrics(
            event_df, None if method == "alignment" else precision_variant
        )
    elif method == "alignment":
        metrics = _run_alignments(event_df, net, im, fm)
    else:
        metrics = _run_token_replay(
            event_df, net, im, fm, consider_remaining, precision_variant
        )
    return {**metrics, "pct_events_unseen": pct_events_unseen, **extras}


def parse_window_date(date_str: str) -> pd.Timestamp:
    """Parse a YYYYMMDD metadata date string to a Timestamp."""
    return pd.Timestamp(date_str[:4] + "-" + date_str[4:6] + "-" + date_str[6:8])


def normalise_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with ``time:timestamp`` cast to tz-naive (if necessary)."""
    ts = df["time:timestamp"]
    if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
        df = df.copy()
        df["time:timestamp"] = ts.dt.tz_localize(None)
    return df


def filter_window_events(
    event_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Keep only events whose timestamp falls within [start, end]."""
    mask = (event_df["time:timestamp"] >= start) & (event_df["time:timestamp"] <= end)
    return event_df.loc[mask].copy()


def extract_window_traces(
    event_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[list[dict[str, str]]]:
    """Traces restricted to a window, in the format :func:`compute_er` expects.

    Each active case yields the activities inside the window, wrapped in ▶ and ■.
    Existing ▶/■ events are removed first so a trace is never wrapped twice. A truncated
    trace starts or ends mid-process; the wildcard ▶/■ of the automaton absorbs that.
    """
    if event_df.empty:
        return []

    sub = sort_events(
        filter_window_events(event_df, start, end)[
            ["case:concept:name", "concept:name", "time:timestamp"]
        ]
    )
    if sub.empty:
        return []

    traces: list[list[dict[str, str]]] = []
    for _case_id, group in sub.groupby("case:concept:name", sort=False):
        labels = [
            str(a)
            for a in group["concept:name"].tolist()
            if str(a) not in (START_SYMBOL, END_SYMBOL)
        ]
        if not labels:
            continue
        trace = (
            [{"concept:name": START_SYMBOL}]
            + [{"concept:name": a} for a in labels]
            + [{"concept:name": END_SYMBOL}]
        )
        traces.append(trace)
    return traces


def full_traces_of_active_cases(
    event_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Events of the cases active in ``[start, end]``, across the reference log."""
    ts = event_df["time:timestamp"]
    in_window = (ts >= start) & (ts <= end)
    active_cases = event_df.loc[in_window, "case:concept:name"].unique()
    if len(active_cases) == 0:
        return event_df.iloc[0:0].copy()
    keep = event_df["case:concept:name"].isin(active_cases)
    return sort_events(event_df.loc[keep])
