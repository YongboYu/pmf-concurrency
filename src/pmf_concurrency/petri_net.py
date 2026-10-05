"""Causal nets expanded into WF-nets.

The expansion follows the mapping of causal nets onto Petri nets in W. M. P. van der
Aalst, A. Adriansyah and B. F. van Dongen, "Causal Nets: A Modeling Language Tailored
towards Process Discovery", CONCUR 2011, LNCS 6901, pp. 28-42.

Every model family is stored as a causal-net JSON and expanded into a WF-net for
conformance checking and rendering. Bindings become τ transitions:

  XOR split: an activity puts its token in one of N ≥ 2 output places. An
             intermediate place and one τ per branch are inserted.
  AND split: one place feeds N ≥ 2 places, which a Petri net cannot express
             directly. A τ_AND transition with N output arcs is inserted.
"""

from collections import defaultdict
from functools import cache
from typing import Any

from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils import petri_utils

from .data import END_SYMBOL, START_SYMBOL


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _edge_binding_frequency(edge_data: dict[str, Any], side: str) -> float:
    """Extract a binding frequency from an edge, preferring side-specific counts."""
    primary = "ib_frequency" if side == "input" else "ob_frequency"
    return _as_float(edge_data.get(primary) or edge_data.get("frequency"))


def _ab_frequency(data: dict[str, Any]) -> float:
    """The AB count of an edge or activity node, else its plain count."""
    return _as_float(data.get("ab_frequency") or data.get("frequency"))


def _format_place_id(prefix: str, *parts: str) -> str:
    return "::".join((prefix, *parts))


def causal_net_to_wf_net(model: dict[str, Any]) -> dict[str, Any]:
    """Expand a causal-net JSON into the places, transitions and arcs of its WF-net."""
    graph_data = model["graph"]
    node_meta: dict[str, dict[str, Any]] = {
        node["id"]: node.get("data", {}) for node in graph_data["nodes"]
    }
    binding_ids = {
        node_id for node_id, data in node_meta.items() if data.get("is_binding", False)
    }
    activity_ids = set(node_meta) - binding_ids

    out_adj: dict[str, list[str]] = defaultdict(list)
    in_adj: dict[str, list[str]] = defaultdict(list)
    edge_meta: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in graph_data["edges"]:
        src, tgt = edge["source"], edge["target"]
        out_adj[src].append(tgt)
        in_adj[tgt].append(src)
        edge_meta[(src, tgt)] = edge.get("data", {})

    @cache
    def forward_visible_targets(binding_id: str) -> tuple[str, ...]:
        seen: set[str] = set()
        targets: set[str] = set()
        stack = [binding_id]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            for nxt in out_adj.get(current, []):
                if nxt in binding_ids:
                    stack.append(nxt)
                elif nxt in activity_ids:
                    targets.add(nxt)
        return tuple(sorted(targets))

    @cache
    def backward_visible_sources(binding_id: str) -> tuple[str, ...]:
        seen: set[str] = set()
        sources: set[str] = set()
        stack = [binding_id]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            for prev in in_adj.get(current, []):
                if prev in binding_ids:
                    stack.append(prev)
                elif prev in activity_ids:
                    sources.add(prev)
        return tuple(sorted(sources))

    def p_in(activity: str) -> str:
        return _format_place_id("pI", activity)

    def p_out(activity: str) -> str:
        return _format_place_id("pO", activity)

    def p_dep(src: str, tgt: str) -> str:
        return _format_place_id("pD", src, tgt)

    output_bindings: dict[str, list[dict[str, Any]]] = defaultdict(list)
    input_bindings: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for activity in sorted(activity_ids):
        for binding_id in sorted(out_adj.get(activity, [])):
            if binding_id not in binding_ids:
                continue
            output_bindings[activity].append(
                {
                    "binding_id": binding_id,
                    "members": forward_visible_targets(binding_id),
                    "frequency": _edge_binding_frequency(
                        edge_meta.get((activity, binding_id), {}), "output"
                    ),
                }
            )

        for binding_id in sorted(in_adj.get(activity, [])):
            if binding_id not in binding_ids:
                continue
            input_bindings[activity].append(
                {
                    "binding_id": binding_id,
                    "members": backward_visible_sources(binding_id),
                    "frequency": _edge_binding_frequency(
                        edge_meta.get((binding_id, activity), {}), "input"
                    ),
                }
            )

    dependency_pairs: set[tuple[str, str]] = set()
    for src_activity, bindings in output_bindings.items():
        for binding in bindings:
            dependency_pairs.update((src_activity, tgt) for tgt in binding["members"])
    for tgt_activity, bindings in input_bindings.items():
        for binding in bindings:
            dependency_pairs.update((src, tgt_activity) for src in binding["members"])

    dependency_frequency: dict[tuple[str, str], float] = defaultdict(float)
    for (src, tgt), data in edge_meta.items():
        weight = _ab_frequency(data)
        if weight <= 0:
            continue
        src_tasks = (
            (src,) if src in activity_ids else backward_visible_sources(src)  # type: ignore[arg-type]
        )
        tgt_tasks = (
            (tgt,) if tgt in activity_ids else forward_visible_targets(tgt)  # type: ignore[arg-type]
        )
        for src_task in src_tasks:
            for tgt_task in tgt_tasks:
                dependency_frequency[(src_task, tgt_task)] = max(
                    dependency_frequency[(src_task, tgt_task)], weight
                )

    if START_SYMBOL in activity_ids:
        source_activity = START_SYMBOL
    else:
        source_candidates = sorted(
            activity for activity in activity_ids if not input_bindings.get(activity)
        )
        source_activity = source_candidates[0] if source_candidates else None

    if END_SYMBOL in activity_ids:
        sink_activity = END_SYMBOL
    else:
        sink_candidates = sorted(
            activity for activity in activity_ids if not output_bindings.get(activity)
        )
        sink_activity = sink_candidates[0] if sink_candidates else None

    places: dict[str, dict[str, Any]] = {}
    transitions: dict[str, dict[str, Any]] = {}
    arcs: list[dict[str, Any]] = []

    for activity in sorted(activity_ids):
        places[p_in(activity)] = {
            "kind": "activity_input",
            "is_source": activity == source_activity,
            "is_sink": False,
            "frequency": 0.0,
        }
        places[p_out(activity)] = {
            "kind": "activity_output",
            "is_source": False,
            "is_sink": activity == sink_activity,
            "frequency": 0.0,
        }

        transition_id = _format_place_id("act", activity)
        transitions[transition_id] = {
            "kind": "activity",
            # The start and end activities stay explicit; the source and sink places
            # only hold the markings around them.
            "label": node_meta[activity].get("label", activity),
            "invisible": False,
            "frequency": _ab_frequency(node_meta[activity]),
        }
        arcs.append(
            {"source": p_in(activity), "target": transition_id, "frequency": 0.0}
        )
        arcs.append(
            {"source": transition_id, "target": p_out(activity), "frequency": 0.0}
        )

    for src_activity, tgt_activity in sorted(dependency_pairs):
        places[p_dep(src_activity, tgt_activity)] = {
            "kind": "dependency",
            "is_source": False,
            "is_sink": False,
            "frequency": dependency_frequency.get((src_activity, tgt_activity), 0.0),
        }

    for activity, bindings in sorted(output_bindings.items()):
        for idx, binding in enumerate(bindings):
            tau_id = _format_place_id("tO", activity, str(idx), binding["binding_id"])
            transitions[tau_id] = {
                "kind": "tau_output",
                "label": None,
                "invisible": True,
                "frequency": binding["frequency"],
            }
            arcs.append({"source": p_out(activity), "target": tau_id, "frequency": 0.0})
            for tgt_activity in binding["members"]:
                arcs.append(
                    {
                        "source": tau_id,
                        "target": p_dep(activity, tgt_activity),
                        "frequency": dependency_frequency.get(
                            (activity, tgt_activity), 0.0
                        ),
                    }
                )

    for activity, bindings in sorted(input_bindings.items()):
        for idx, binding in enumerate(bindings):
            tau_id = _format_place_id("tI", activity, str(idx), binding["binding_id"])
            transitions[tau_id] = {
                "kind": "tau_input",
                "label": None,
                "invisible": True,
                "frequency": binding["frequency"],
            }
            for src_activity in binding["members"]:
                arcs.append(
                    {
                        "source": p_dep(src_activity, activity),
                        "target": tau_id,
                        "frequency": dependency_frequency.get(
                            (src_activity, activity), 0.0
                        ),
                    }
                )
            arcs.append({"source": tau_id, "target": p_in(activity), "frequency": 0.0})

    return {
        "places": places,
        "transitions": transitions,
        "arcs": arcs,
        "initial_place": p_in(source_activity) if source_activity else None,
        "final_place": p_out(sink_activity) if sink_activity else None,
    }


def json_to_petri_net(model: dict) -> tuple[PetriNet, Marking, Marking]:
    """Expand a causal-net JSON into a pm4py PetriNet."""
    explicit = causal_net_to_wf_net(model)
    net = PetriNet(model["metadata"].get("type", "model"))
    places: dict[str, PetriNet.Place] = {}
    transitions: dict[str, PetriNet.Transition] = {}

    for place_id in explicit["places"]:
        place = PetriNet.Place(place_id)
        net.places.add(place)
        places[place_id] = place

    for transition_id, meta in explicit["transitions"].items():
        transition = PetriNet.Transition(
            transition_id, None if meta.get("invisible", False) else meta.get("label")
        )
        net.transitions.add(transition)
        transitions[transition_id] = transition

    for arc in explicit["arcs"]:
        src_obj = transitions.get(arc["source"]) or places.get(arc["source"])
        tgt_obj = transitions.get(arc["target"]) or places.get(arc["target"])
        if src_obj is not None and tgt_obj is not None:
            petri_utils.add_arc_from_to(src_obj, tgt_obj, net)

    initial_marking = Marking()
    final_marking = Marking()
    if explicit["initial_place"] and explicit["initial_place"] in places:
        initial_marking[places[explicit["initial_place"]]] = 1
    if explicit["final_place"] and explicit["final_place"] in places:
        final_marking[places[explicit["final_place"]]] = 1

    return net, initial_marking, final_marking


def accepts_nothing(model: dict[str, Any]) -> bool:
    """True if the model's WF-net can never reach its final marking.

    That holds when the net has no transitions, lacks an initial or final place, or has
    no directed path from the initial to the final place: tokens only move along arcs.
    A Forecast model is like this when no arc, or no arc on the way to ■, clears the
    dependency threshold.
    """
    explicit = causal_net_to_wf_net(model)
    source, sink = explicit["initial_place"], explicit["final_place"]
    if not explicit["transitions"] or source is None or sink is None:
        return True
    successors: dict[str, list[str]] = defaultdict(list)
    for arc in explicit["arcs"]:
        successors[arc["source"]].append(arc["target"])
    seen = {source}
    stack = [source]
    while stack:
        for node in successors[stack.pop()]:
            if node not in seen:
                seen.add(node)
                stack.append(node)
    return sink not in seen


def visible_labels(net: PetriNet) -> set[str]:
    """Labels of the net's visible transitions, ▶ and ■ included."""
    return {t.label for t in net.transitions if t.label is not None}


def activity_frequencies(data: dict[str, Any]) -> dict[str, float]:
    """``{activity label: frequency}`` for the activities of a causal-net JSON."""
    freq: dict[str, float] = {}
    graph = data.get("graph")
    if isinstance(graph, dict):
        for node in graph.get("nodes", []):
            d = node.get("data", {}) if isinstance(node, dict) else {}
            if d.get("is_binding"):
                continue
            label = d.get("label") or node.get("id")
            f = d.get("frequency")
            if label is not None and isinstance(f, (int, float)):
                freq[str(label)] = float(f)
    return freq
