"""Language-preserving Petri-net reduction, to make evaluation tractable.

The WF-net expansion of a causal net adds one silent transition per binding and dedicated
pI/pO/pD places. On concurrent nets these make token replay, alignments and the ER
automaton expensive.

This module removes silent transitions whose only role is to pass a token through a
dedicated place, with two series-fusion rules (from Murata 1989, restricted to silent
transitions). Both preserve the visible language and the reachability of the final
marking, so values stay comparable; only replay heuristics and runtime can change. τ
transitions of AND and XOR splits and joins share their places and are left alone.

Rules (τ is a transition with ``label is None``; all arcs have weight 1):

  merge-out   τ t with one output place q, where t is q's only producer and q
              holds no initial or final token: connect every input place of
              t to every consumer of q, then remove t and q.

  merge-in    τ t with one input place p, where t is p's only consumer and p
              holds no initial or final token: connect every producer of p to
              every output place of t, then remove t and p.

A rule is skipped rather than create a duplicate arc, which would change the arc weight.

An optional second stage removes implicit places with pm4py's ILP check
(``pm4py.objects.petri_net.utils.murata``), which helps on AND-heavy nets.

Public function:
    reduce_net(net, im, fm, level="tau") -> (net, im, fm, stats)
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils.petri_utils import (
    add_arc_from_to,
    remove_place,
    remove_transition,
)

logger = logging.getLogger(__name__)

ReductionLevel = Literal["none", "tau", "tau_implicit"]

#: Recognised values for the ``level`` argument of :func:`reduce_net`.
REDUCTION_LEVELS: tuple[ReductionLevel, ...] = ("none", "tau", "tau_implicit")


def _in_places(t: PetriNet.Transition) -> list[PetriNet.Place]:
    return [arc.source for arc in t.in_arcs]


def _out_places(t: PetriNet.Transition) -> list[PetriNet.Place]:
    return [arc.target for arc in t.out_arcs]


def _all_weight_one(arcs: Any) -> bool:
    return all(getattr(arc, "weight", 1) == 1 for arc in arcs)


def _arc_exists(src: Any, tgt: Any) -> bool:
    return any(arc.target is tgt for arc in src.out_arcs)


def net_stats(net: PetriNet) -> dict[str, int]:
    """Structural counts used to report the effect of a reduction."""
    n_silent = sum(1 for t in net.transitions if t.label is None)
    return {
        "places": len(net.places),
        "transitions": len(net.transitions),
        "silent_transitions": n_silent,
        "visible_transitions": len(net.transitions) - n_silent,
        "arcs": len(net.arcs),
    }


def _try_merge_out(
    net: PetriNet,
    t: PetriNet.Transition,
    marked: set[PetriNet.Place],
) -> bool:
    """merge-out: absorb τ ``t`` and its dedicated single output place."""
    if t.label is not None or len(t.out_arcs) != 1:
        return False
    q = next(iter(t.out_arcs)).target
    if (
        q in marked
        or len(q.in_arcs) != 1  # q's only producer must be t
        or len(q.out_arcs) < 1  # keep dead-end places out of scope
        or q in _in_places(t)  # no self-loop through q
        or not _all_weight_one(t.in_arcs)
        or not _all_weight_one(t.out_arcs)
        or not _all_weight_one(q.out_arcs)
    ):
        return False

    sources = _in_places(t)
    consumers = [arc.target for arc in q.out_arcs]
    # An existing p→u arc would be doubled by the rewiring, so skip.
    for p in sources:
        for u in consumers:
            if _arc_exists(p, u):
                return False

    for p in sources:
        for u in consumers:
            add_arc_from_to(p, u, net)
    remove_transition(net, t)
    remove_place(net, q)
    return True


def _try_merge_in(
    net: PetriNet,
    t: PetriNet.Transition,
    marked: set[PetriNet.Place],
) -> bool:
    """merge-in: absorb τ ``t`` and its dedicated single input place."""
    if t.label is not None or len(t.in_arcs) != 1:
        return False
    p = next(iter(t.in_arcs)).source
    if (
        p in marked
        or len(p.out_arcs) != 1  # p's only consumer must be t
        or len(p.in_arcs) < 1
        or p in _out_places(t)  # no self-loop through p
        or not _all_weight_one(t.in_arcs)
        or not _all_weight_one(t.out_arcs)
        or not _all_weight_one(p.in_arcs)
    ):
        return False

    producers = [arc.source for arc in p.in_arcs]
    targets = _out_places(t)
    for w in producers:
        for q in targets:
            if _arc_exists(w, q):
                return False

    for w in producers:
        for q in targets:
            add_arc_from_to(w, q, net)
    remove_transition(net, t)
    remove_place(net, p)
    return True


def reduce_tau_series(net: PetriNet, im: Marking, fm: Marking) -> PetriNet:
    """Apply merge-in and merge-out until nothing changes, in name order.

    Changes ``net`` in place and returns it. ``im`` and ``fm`` are untouched, since
    neither rule removes a marked place.
    """
    marked: set[PetriNet.Place] = set(im.keys()) | set(fm.keys())
    changed = True
    while changed:
        changed = False
        for t in sorted(net.transitions, key=lambda x: x.name):
            if t.label is not None:
                continue
            if _try_merge_out(net, t, marked) or _try_merge_in(net, t, marked):
                changed = True
                break  # the net changed, so restart the scan
    return net


def reduce_net(
    net: PetriNet,
    im: Marking,
    fm: Marking,
    level: ReductionLevel = "tau",
) -> tuple[PetriNet, Marking, Marking, dict[str, Any]]:
    """Reduce ``net`` at the given level and report its size before and after.

    Args:
        net, im, fm: The net to reduce (changed in place for ``tau``; the
                     implicit-place stage may return new objects).
        level:       ``none``: no change, statistics only;
                     ``tau``: silent series fusion (:func:`reduce_tau_series`);
                     ``tau_implicit``: fusion, implicit-place removal, then
                     fusion again.

    Returns:
        ``(net, im, fm, stats)``, where stats holds ``*_before``/``*_after``
        counts and the ``reduction_level``.
    """
    before = net_stats(net)

    if level == "tau":
        net = reduce_tau_series(net, im, fm)
    elif level == "tau_implicit":
        net = reduce_tau_series(net, im, fm)
        try:
            from pm4py.objects.petri_net.utils import murata

            net, im, fm = murata.apply_reduction(net, im, fm)
            net = reduce_tau_series(net, im, fm)
        except Exception as exc:
            logger.warning(
                "Implicit-place reduction failed (%s); keeping τ-reduced net.", exc
            )
    elif level != "none":
        raise ValueError(f"Unknown reduction level: {level!r}")

    after = net_stats(net)
    stats: dict[str, Any] = {"reduction_level": level}
    stats.update({f"{k}_before": v for k, v in before.items()})
    stats.update({f"{k}_after": v for k, v in after.items()})
    return net, im, fm, stats
