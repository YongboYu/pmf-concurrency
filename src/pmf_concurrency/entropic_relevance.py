"""Entropic relevance (ER) of Petri nets, with support for partial traces.

Ports the DFG-based ER of pmf-tsfm to Petri nets, following Polyvyanyy et al. (2020).
The ER core (``BackgroundModel``, ``_run_source``) is the same as the DFG version; only
the conversion of a model into a stochastic automaton (SA) differs.

* The SA's alphabet is the set of activities. Transitions labelled τ, ▶ or ■
  are silent and absorbed into τ-closures.
* SA states are the reachable markings grouped by forward τ-closure.
* With ``partial_trace=True`` (the default) the SA gets two extra states, ▶
  and ■, with wildcard arcs: ▶ to every activity and every state to ■, each
  with probability 1 (■ arcs are left out of the normalisation). Every partial
  trace can then be replayed, and only mismatches inside it are penalized.
* Arc probabilities are frequencies normalised per source state over visible,
  non-■ transitions.

Public functions:
    petri_net_to_transitions(model_json, partial_trace=True) -> (trans_table, sources)
    compute_er(model_json, traces, partial_trace=True) -> (er, fitting_ratio, n_traces)
    er_metrics(model_json, traces, partial_trace=True) -> dict, with the failure reason
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Hashable
from typing import Any

from pm4py.objects.petri_net.obj import Marking, PetriNet

from .data import END_SYMBOL, START_SYMBOL
from .petri_net import activity_frequencies, json_to_petri_net

logger = logging.getLogger(__name__)

# An SA state is a frozenset of marking keys (a τ-closure) or one of the extra boundary
# states "__START__" and "__END__".
_SA_State = Hashable
_TransTable = dict[tuple[_SA_State, str], tuple[_SA_State, float]]

_START_STATE: _SA_State = "__START__"
_END_STATE: _SA_State = "__END__"


def _canonical_state_key(state: _SA_State) -> tuple:
    """Sort key that orders SA states totally.

    An SA state is a frozenset of marking keys. ``sorted()`` on frozensets uses the
    subset order, which leaves incomparable states in hash order. Sorting each marking
    key as a tuple gives an order that does not depend on the hash seed.
    """
    if isinstance(state, frozenset):
        return tuple(sorted(tuple(sorted(mk)) for mk in state))
    return (str(state),)


class BackgroundModel:
    """Trace statistics for ER, copied from the DFG implementation so both give the same
    numbers.
    """

    def __init__(self) -> None:
        self.number_of_events: int = 0
        self.number_of_traces: int = 0
        self.total_number_non_fitting_traces: int = 0
        self.labels: set[str] = set()
        self.trace_frequency: dict[str, int] = {}

        self._trace_str: str = ""
        self._lprob: float = 0.0
        self._trace_size: dict[str, int] = {}
        self._log2_of_model_probability: dict[str, float] = {}

    def open_trace(self) -> None:
        self._lprob = 0.0
        self._trace_str = ""

    def process_event(self, label: str, log2_prob: float) -> None:
        self._trace_str += label
        self.number_of_events += 1
        self.labels.add(label)
        self._lprob += log2_prob

    def close_trace(self, trace_length: int, fitting: bool) -> None:
        key = self._trace_str
        self._trace_size[key] = trace_length
        self.number_of_traces += 1
        if fitting:
            self._log2_of_model_probability[key] = self._lprob
        else:
            self.total_number_non_fitting_traces += 1
        self.trace_frequency[key] = self.trace_frequency.get(key, 0) + 1

    def _h0(self, rho: int) -> float:
        n = self.number_of_traces
        if rho == 0 or rho == n:
            return 0.0
        p = rho / n
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    def compute_relevance(self) -> float:
        rho = 0
        cost_bits = 0.0
        for key, freq in self.trace_frequency.items():
            if key in self._log2_of_model_probability:
                bits = -self._log2_of_model_probability[key]
                rho += freq
            else:
                bits = (1 + self._trace_size[key]) * math.log2(1 + len(self.labels))
            cost_bits += (bits * freq) / self.number_of_traces
        return self._h0(rho) + cost_bits


def _marking_key(marking: Marking) -> frozenset[tuple[str, int]]:
    """Hashable form of a marking: ``frozenset((place_name, count))``."""
    return frozenset(
        (place.name, int(count)) for place, count in marking.items() if count > 0
    )


def _is_silent(transition: PetriNet.Transition) -> bool:
    """True if a transition is invisible or labelled ▶ or ■.

    In the expanded WF-nets ``act::▶`` and ``act::■`` are visible transitions, but they
    only mark the start and end of a trace, so the SA treats them as silent.
    """
    return (
        transition.label is None
        or transition.label == START_SYMBOL
        or transition.label == END_SYMBOL
    )


def _enabled(net: PetriNet, marking: Marking) -> list[PetriNet.Transition]:
    """Transitions enabled in ``marking`` (every input place holds a token)."""
    enabled: list[PetriNet.Transition] = []
    for transition in net.transitions:
        ok = True
        for arc in transition.in_arcs:
            if marking.get(arc.source, 0) < 1:
                ok = False
                break
        if ok:
            enabled.append(transition)
    return enabled


def _fire(marking: Marking, transition: PetriNet.Transition) -> Marking:
    """Marking after firing ``transition``; ``marking`` itself is not changed."""
    new = Marking()
    for place, count in marking.items():
        new[place] = count
    for arc in transition.in_arcs:
        new[arc.source] -= 1
        if new[arc.source] == 0:
            del new[arc.source]
    for arc in transition.out_arcs:
        new[arc.target] = new.get(arc.target, 0) + 1
    return new


def _tau_closure(
    net: PetriNet, start: Marking, marking_lookup: dict[frozenset, Marking]
) -> frozenset[tuple[str, int]]:
    """Markings reachable from ``start`` through silent transitions, as a frozenset of
    keys.

    ``marking_lookup`` is updated in place with a ``Marking`` for every key seen, so
    callers can fire visible transitions from any member of the closure.
    """
    start_key = _marking_key(start)
    if start_key not in marking_lookup:
        marking_lookup[start_key] = start
    closure: set[frozenset[tuple[str, int]]] = {start_key}
    stack: list[Marking] = [start]
    while stack:
        m = stack.pop()
        for t in _enabled(net, m):
            if not _is_silent(t):
                continue
            successor = _fire(m, t)
            key = _marking_key(successor)
            if key in closure:
                continue
            closure.add(key)
            marking_lookup.setdefault(key, successor)
            stack.append(successor)
    return frozenset(closure)


def _label_freq(
    transition: PetriNet.Transition, activity_freq: dict[str, float]
) -> float:
    """Frequency of a visible transition, looked up by its label.

    Transitions that share a label share its frequency. Labels missing from the
    frequency table get a small positive value, so they stay in the SA.
    """
    label = transition.label
    if label is None:
        return 0.0
    f = activity_freq.get(label)
    return float(f) if f is not None and f > 0 else 1.0


def petri_net_to_transitions(
    model_json: dict[str, Any],
    partial_trace: bool = True,
    reduce_level: str = "none",
) -> tuple[_TransTable, list[_SA_State]]:
    """Convert a model JSON into the stochastic automaton ER is computed on.

    Args:
        model_json:    A ``window_N_<family>.json`` dict from the pipeline.
        partial_trace: If True (default), add wildcard ▶/■ states so every
            partial trace can be replayed. If False, the source is τ*(IM) and ■
            is reachable only from closures that contain the final marking.
        reduce_level:  Optional language-preserving reduction before the
            reachability search (``none``, ``tau`` or ``tau_implicit``; see
            :mod:`.reduction`). It shrinks the state space. The automaton keeps one
            target per state and label, so ER can differ between reduction levels.

    Returns:
        ``(trans_table, sources)`` for ``_run_source``. ``trans_table`` maps
        ``(source_state, label)`` to ``(target_state, log2 probability)``;
        ``sources`` holds the start state.
    """
    net, im, fm = json_to_petri_net(model_json)
    if reduce_level != "none":
        from .reduction import reduce_net

        net, im, fm, _ = reduce_net(net, im, fm, level=reduce_level)
    activity_freq = activity_frequencies(model_json)

    # A representative Marking for every marking key, for firing visible transitions out
    # of a closure.
    marking_lookup: dict[frozenset[tuple[str, int]], Marking] = {}

    # 1. Reachability: explore the full reachable marking graph.
    initial_closure = _tau_closure(net, im, marking_lookup)

    # SA state = τ-equivalence class (frozenset of marking keys).
    visited: set[frozenset[tuple[str, int]]] = {initial_closure}
    queue: list[frozenset[tuple[str, int]]] = [initial_closure]

    # Per source closure and visible label: candidate target closures and their summed
    # frequency. The target with the highest frequency is kept.
    raw: dict[
        frozenset[tuple[str, int]],
        dict[str, dict[frozenset[tuple[str, int]], float]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    fm_key = _marking_key(fm)

    while queue:
        closure = queue.pop()
        for member_key in closure:
            member = marking_lookup[member_key]
            for t in _enabled(net, member):
                if _is_silent(t):
                    continue
                successor = _fire(member, t)
                target_closure = _tau_closure(net, successor, marking_lookup)
                freq = _label_freq(t, activity_freq)
                raw[closure][t.label][target_closure] += freq
                if target_closure not in visited:
                    visited.add(target_closure)
                    queue.append(target_closure)

    # 2. For each (closure, label) pick the dominant target closure.
    aggregated: dict[
        tuple[frozenset[tuple[str, int]], str],
        tuple[frozenset[tuple[str, int]], float],
    ] = {}
    for src_closure, by_label in sorted(
        raw.items(), key=lambda kv: _canonical_state_key(kv[0])
    ):
        for label, targets in sorted(by_label.items()):
            # Break ties by the canonical key of the target, so the choice does not
            # depend on hash order (see ``_canonical_state_key``).
            target, freq = max(
                targets.items(), key=lambda kv: (kv[1], _canonical_state_key(kv[0]))
            )
            aggregated[(src_closure, label)] = (target, freq)

    # 3. Normalise per source closure, over visible non-■ frequencies.
    per_source_total: dict[frozenset[tuple[str, int]], float] = defaultdict(float)
    for (src, label), (_target, freq) in sorted(
        aggregated.items(), key=lambda kv: (_canonical_state_key(kv[0][0]), kv[0][1])
    ):
        if label == END_SYMBOL:  # ■ is silent and should not appear here
            continue
        per_source_total[src] += freq

    trans_table: _TransTable = {}
    for (src, label), (target, freq) in sorted(
        aggregated.items(), key=lambda kv: (_canonical_state_key(kv[0][0]), kv[0][1])
    ):
        denom = per_source_total[src]
        if denom <= 0 or freq <= 0:
            continue
        prob = freq / denom
        trans_table[(src, label)] = (target, math.log2(prob))

    # 4. The extra ▶ and ■ states.
    if partial_trace:
        # ▶ leads to every visible activity with probability 1. The labels come from
        # both ``aggregated`` and the net, so activities reachable only from other
        # markings are included.
        all_visible_labels: set[str] = set()
        for t in net.transitions:
            if not _is_silent(t):
                all_visible_labels.add(t.label)

        # For each label, the most frequent target closure in the whole graph. Labels
        # that never fire are skipped.
        per_label_targets: dict[str, dict[frozenset[tuple[str, int]], float]] = (
            defaultdict(lambda: defaultdict(float))
        )
        for (_src, label), (target, freq) in aggregated.items():
            per_label_targets[label][target] += freq

        for label in all_visible_labels:
            if label not in per_label_targets:
                continue
            target, _freq = max(
                per_label_targets[label].items(),
                key=lambda kv: (kv[1], _canonical_state_key(kv[0])),
            )
            trans_table[(_START_STATE, label)] = (target, 0.0)  # log2(1) = 0

        # Every reachable state leads to ■ with probability 1, outside the
        # normalisation.
        for closure in visited:
            trans_table[(closure, END_SYMBOL)] = (_END_STATE, 0.0)

        sources: list[_SA_State] = [_START_STATE]
    else:
        # Strict mode: the source is τ*(IM), and ■ is reachable only from closures
        # holding the final marking.
        for closure in visited:
            if fm_key in closure:
                trans_table[(closure, END_SYMBOL)] = (_END_STATE, 0.0)
        sources = [initial_closure]

    return trans_table, sources


def _run_source(
    source: _SA_State,
    trans_table: _TransTable,
    traces: list[list[dict[str, str]]],
) -> tuple[float, BackgroundModel]:
    """Replay the traces on the SA, trace by trace, exactly as the DFG version does."""
    model = BackgroundModel()

    for trace in traces:
        curr = source
        non_fitting = False
        model.open_trace()
        trace_len = 0
        last_label = ""

        for event in trace:
            label = event["concept:name"]
            if label == START_SYMBOL:
                continue
            trace_len += 1
            log2_prob = 0.0
            if not non_fitting and (curr, label) in trans_table:
                curr, log2_prob = trans_table[(curr, label)]
            else:
                non_fitting = True
            model.process_event(label, log2_prob)
            last_label = label

        model.close_trace(
            trace_len,
            fitting=not non_fitting and last_label == END_SYMBOL,
        )

    return model.compute_relevance(), model


def er_metrics(
    model_json: dict[str, Any],
    traces: list[list[dict[str, str]]],
    partial_trace: bool = True,
    reduce_level: str = "none",
) -> dict[str, Any]:
    """Entropic relevance of a model JSON against a list of traces.

    Args:
        model_json:     A model JSON from the pipeline.
        traces:         Traces as lists of ``{"concept:name": label}`` dicts,
                        wrapped in ▶ and ■. :func:`conformance.extract_window_traces`
                        produces them.
        partial_trace:  True (default) for wildcard ▶/■; False for strict start
                        and end states.
        reduce_level:   Optional reduction before building the SA (see
                        :func:`petri_net_to_transitions`).

    Returns:
        ``er``, ``fitting_ratio`` and ``n_traces``. ER and the fitting ratio are NaN if
        ER cannot be computed; then ``failure_category`` is ``er_automaton_failed``
        and ``guard_reason`` says why.
    """
    nan = float("nan")
    if not traces:
        return {"er": nan, "fitting_ratio": nan, "n_traces": 0}

    failed = {"er": nan, "fitting_ratio": nan, "n_traces": len(traces)}
    try:
        trans_table, sources = petri_net_to_transitions(
            model_json, partial_trace=partial_trace, reduce_level=reduce_level
        )
    except MemoryError:
        raise  # the budget guard records it as oom
    except Exception as exc:
        logger.warning("ER automaton could not be built: %s", exc)
        return {
            **failed,
            "failure_category": "er_automaton_failed",
            "guard_reason": repr(exc),
        }

    if not trans_table or not sources:
        return {
            **failed,
            "failure_category": "er_automaton_failed",
            "guard_reason": "the automaton has no transitions or no source state",
        }

    results = [_run_source(s, trans_table, traces) for s in sources]
    best_er, best_model = min(results, key=lambda x: x[0])

    total = best_model.number_of_traces
    non_fit = best_model.total_number_non_fitting_traces
    fitting_ratio = 1.0 - non_fit / total if total > 0 else nan

    return {"er": best_er, "fitting_ratio": fitting_ratio, "n_traces": total}


def compute_er(
    model_json: dict[str, Any],
    traces: list[list[dict[str, str]]],
    partial_trace: bool = True,
    reduce_level: str = "none",
) -> tuple[float, float, int]:
    """``(er, fitting_ratio, n_traces)`` of :func:`er_metrics`."""
    m = er_metrics(model_json, traces, partial_trace, reduce_level)
    return m["er"], m["fitting_ratio"], m["n_traces"]
