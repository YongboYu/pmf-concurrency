"""Precision variants that report how many prefixes they discard.

pm4py's ETC token precision replays every prefix and ignores those that do not fit
(except for the empty prefix). On windowed logs, where many traces are truncated,
precision is then computed on a small subset of the log, and the result does not show
how small.

* :func:`etc_precision_with_diagnostics`: ETC token precision, equal to
  pm4py's, plus the share of prefix mass it discards.
* :func:`align_etc_precision`: Align-ETC, which aligns every prefix to the
  model. Needs an easy-sound net and returns NaN otherwise.

Both take the event DataFrame used in :mod:`.conformance`.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def etc_precision_with_diagnostics(
    event_df: pd.DataFrame,
    net: Any,
    im: Any,
    fm: Any,
) -> dict[str, float]:
    """ETC token precision with prefix-coverage diagnostics.

    Uses the same parameters as pm4py's ``etconformance_token`` and only records which
    prefixes it discards.

    Returns:
        Dict with:
          * ``precision``: equal to pm4py's ETC value;
          * ``prefix_variants``: distinct non-empty prefixes;
          * ``prefix_mass_total``: prefix occurrences in the log;
          * ``prefix_mass_fit``: occurrences that replay without error, the only
            ones ETC uses;
          * ``prefix_mass_unfit``: discarded occurrences;
          * ``pct_prefix_mass_fit``: 100 × fit / total (NaN if empty).
    """
    from pm4py.algo.conformance.tokenreplay import algorithm as executor
    from pm4py.algo.conformance.tokenreplay.variants import token_replay
    from pm4py.algo.evaluation.precision import utils as precision_utils
    from pm4py.objects.conversion.log import converter as log_converter
    from pm4py.objects.log.util import xes as xes_util
    from pm4py.objects.petri_net.utils.align_utils import (
        get_visible_transitions_eventually_enabled_by_marking,
    )
    from pm4py.statistics.start_activities.log.get import get_start_activities

    activity_key = xes_util.DEFAULT_NAME_KEY

    log = log_converter.apply(event_df, variant=log_converter.Variants.TO_EVENT_LOG)

    prefixes, prefix_count = precision_utils.get_log_prefixes(
        log, activity_key=activity_key
    )
    prefixes_keys = list(prefixes.keys())
    fake_log = precision_utils.form_fake_log(prefixes_keys, activity_key=activity_key)

    parameters_tr = {
        token_replay.Parameters.SHOW_PROGRESS_BAR: False,
        token_replay.Parameters.CONSIDER_REMAINING_IN_FITNESS: False,
        token_replay.Parameters.TRY_TO_REACH_FINAL_MARKING_THROUGH_HIDDEN: False,
        token_replay.Parameters.STOP_IMMEDIATELY_UNFIT: True,
        token_replay.Parameters.WALK_THROUGH_HIDDEN_TRANS: True,
        token_replay.Parameters.CLEANING_TOKEN_FLOOD: False,
        token_replay.Parameters.ACTIVITY_KEY: activity_key,
    }
    aligned_traces = executor.apply(
        fake_log,
        net,
        im,
        fm,
        variant=executor.Variants.TOKEN_REPLAY,
        parameters=parameters_tr,
    )

    precision = 1.0
    sum_ee = 0.0
    sum_at = 0.0

    # Empty-prefix correction, as in pm4py: counted for every trace, fitting or not.
    start_activities = set(get_start_activities(log))
    trans_en_ini_marking = {
        x.label for x in get_visible_transitions_eventually_enabled_by_marking(net, im)
    }
    diff = trans_en_ini_marking.difference(start_activities)
    n_traces = len(log)
    sum_at += n_traces * len(trans_en_ini_marking)
    sum_ee += n_traces * len(diff)

    mass_total = 0
    mass_fit = 0
    for i in range(len(aligned_traces)):
        occurrences = prefix_count[prefixes_keys[i]]
        mass_total += occurrences
        if aligned_traces[i]["trace_is_fit"]:
            mass_fit += occurrences
            log_transitions = set(prefixes[prefixes_keys[i]])
            activated_transitions_labels = {
                x.label
                for x in aligned_traces[i]["enabled_transitions_in_marking"]
                if x.label is not None
            }
            sum_at += len(activated_transitions_labels) * occurrences
            escaping_edges = activated_transitions_labels.difference(log_transitions)
            sum_ee += len(escaping_edges) * occurrences

    if sum_at > 0:
        precision = 1 - float(sum_ee) / float(sum_at)

    return {
        "precision": float(precision),
        "prefix_variants": float(len(prefixes_keys)),
        "prefix_mass_total": float(mass_total),
        "prefix_mass_fit": float(mass_fit),
        "prefix_mass_unfit": float(mass_total - mass_fit),
        "pct_prefix_mass_fit": (
            100.0 * mass_fit / mass_total if mass_total else float("nan")
        ),
    }


def align_etc_precision(
    event_df: pd.DataFrame,
    net: Any,
    im: Any,
    fm: Any,
) -> float:
    """Align-ETC precision, where every prefix counts through its alignment.

    Returns NaN if the net is not easy-sound or the computation fails.
    """
    from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
    from pm4py.objects.conversion.log import converter as log_converter

    log = log_converter.apply(event_df, variant=log_converter.Variants.TO_EVENT_LOG)
    try:
        return float(
            precision_evaluator.apply(
                log,
                net,
                im,
                fm,
                variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE,
                parameters={"show_progress_bar": False},
            )
        )
    except MemoryError:
        raise  # the budget guard records it as oom
    except Exception as exc:
        logger.warning("Align-ETC precision failed: %s", exc)
        return float("nan")
