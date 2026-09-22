"""Causal nets with binding nodes: built from bindings, mined by Fodina, and JSON.

Every model family is stored in the same JSON format (:func:`causal_graph_to_json`).
"""

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from .bindings import BindingCounter, mine_dependency_graph
from .fodina import PatternMiner


def mine_fodina(
    traces: list[list[str]],
    tD: float = 0.5,
    tL1L: float = 0.5,
    tL2L: float = 0.5,
    tPAT: float = 0.0,
) -> PatternMiner | None:
    """Mine a causal net with Fodina, or None without traces.

    The miner carries its tasks (``.tm``) and dependency graph (``.dg``).
    """
    if not traces:
        return None
    pm = PatternMiner(dg=mine_dependency_graph(traces, tD, tL1L, tL2L), tPAT=tPAT)
    pm.mine()
    return pm


def _numbered(
    side: str, task: str, bindings: set[frozenset[str]]
) -> list[tuple[str, frozenset[str]]]:
    """Node ids of a task's input ("i") or output ("o") bindings, in a fixed order."""
    return [
        (f"{side}_{task}_{i}", binding)
        for i, binding in enumerate(sorted(bindings, key=sorted))
    ]


def bindings_to_causal_graph(
    tasks: set[str],
    IB: dict[str, set[frozenset[str]]],
    OB: dict[str, set[frozenset[str]]],
) -> tuple[dict[str, set[str]], set[str], dict[str, set[str]]]:
    """Convert bindings into a causal graph with binding nodes.

    Steps:
    1. Create output binding nodes (o_T_i) and input binding nodes (i_T_i).
    2. Connect o_T → i_X when T's output binding contains X and X's input
       binding contains T.
    3. Merge consecutive o_X → i_Y pairs into one binding node.
    4. Remove binding nodes with one in-edge and one out-edge that cross a
       task/binding boundary.

    Returns:
        (graph, binding_nodes, binding_lineage), where binding_lineage maps each
        remaining binding node to the i_/o_ nodes merged into it.
    """
    graph: dict[str, set[str]] = {t: set() for t in tasks}
    binding_nodes: set[str] = set()
    binding_lineage: dict[str, set[str]] = {}

    for t in tasks:
        for node_id, _ in _numbered("o", t, OB.get(t, set())):
            graph[node_id] = set()
            binding_nodes.add(node_id)
            binding_lineage[node_id] = {node_id}
            graph[t].add(node_id)

    for t in tasks:
        for node_id, _ in _numbered("i", t, IB.get(t, set())):
            graph[node_id] = set()
            binding_nodes.add(node_id)
            binding_lineage[node_id] = {node_id}
            graph[node_id].add(t)

    for t in tasks:
        for node_id, binding in _numbered("o", t, OB.get(t, set())):
            for x in binding:
                for in_id, in_binding in _numbered("i", x, IB.get(x, set())):
                    if t in in_binding:
                        graph[node_id].add(in_id)

    _merge_oi_bindings(graph, binding_nodes, binding_lineage)
    _simplify_binding_nodes(graph, binding_nodes, binding_lineage)

    return graph, binding_nodes, binding_lineage


def binding_frequencies(
    tasks: set[str],
    IB: dict[str, set[frozenset[str]]],
    OB: dict[str, set[frozenset[str]]],
    binding_lineage: dict[str, set[str]],
    count: Callable[[str, str, frozenset[str]], float],
) -> dict[str, dict[str, float]]:
    """IB and OB counts of each binding node, summed over the bindings merged into it.

    ``count(side, task, binding)`` is the count of one binding, with side "i" or "o".
    """
    raw_ib: dict[str, float] = {}
    raw_ob: dict[str, float] = {}
    for task in tasks:
        for node_id, binding in _numbered("i", task, IB.get(task, set())):
            raw_ib[node_id] = count("i", task, binding)
        for node_id, binding in _numbered("o", task, OB.get(task, set())):
            raw_ob[node_id] = count("o", task, binding)
    return {
        node_id: {
            "ib_frequency": sum(raw_ib.get(orig, 0.0) for orig in lineage),
            "ob_frequency": sum(raw_ob.get(orig, 0.0) for orig in lineage),
        }
        for node_id, lineage in binding_lineage.items()
    }


def _merge_oi_bindings(
    graph: dict[str, set[str]],
    binding_nodes: set[str],
    binding_lineage: dict[str, set[str]],
) -> None:
    """Merge consecutive o_X → i_Y binding nodes into one node.

    Merges only when each is the other's sole neighbour. The merged node keeps the o_X
    name.
    """
    doing = True
    while doing:
        doing = False
        for a_node in sorted(binding_nodes):
            if not a_node.startswith("o_") or a_node not in graph:
                continue
            for i_node in sorted(graph[a_node]):
                if not i_node.startswith("i_") or i_node not in binding_nodes:
                    continue
                in_edges = {x for x in graph if i_node in graph[x]}
                if len(in_edges) == 1 and next(iter(in_edges)) == a_node:
                    i_outputs = graph.get(i_node, set()).copy()
                    graph[a_node].discard(i_node)
                    graph[a_node].update(i_outputs)
                    binding_lineage[a_node].update(
                        binding_lineage.get(i_node, {i_node})
                    )
                    binding_nodes.discard(i_node)
                    graph.pop(i_node, None)
                    binding_lineage.pop(i_node, None)
                    doing = True
                    break
            if doing:
                break


def _simplify_binding_nodes(
    graph: dict[str, set[str]],
    binding_nodes: set[str],
    binding_lineage: dict[str, set[str]],
) -> None:
    """Bypass binding nodes with exactly one in-edge and one out-edge.

    Only when the bypass crosses a task/binding boundary, so nodes between two tasks or
    two binding nodes stay.
    """
    doing = True
    while doing:
        doing = False
        for a_node in sorted(binding_nodes):
            if a_node not in graph:
                continue
            in_edges = {x for x in graph if a_node in graph[x]}
            out_edges = graph[a_node]
            if len(in_edges) != 1 or len(out_edges) != 1:
                continue
            in_s = next(iter(in_edges))
            out_t = next(iter(out_edges))
            if (in_s in binding_nodes) ^ (out_t in binding_nodes):
                survivor = in_s if in_s in binding_nodes else out_t
                binding_lineage.setdefault(survivor, set()).update(
                    binding_lineage.get(a_node, {a_node})
                )
                graph[in_s].add(out_t)
                for k in graph:
                    graph[k].discard(a_node)
                binding_nodes.discard(a_node)
                del graph[a_node]
                binding_lineage.pop(a_node, None)
                doing = True
                break


def causal_graph_to_json(
    graph: dict[str, set[str]],
    binding_nodes: set[str],
    metadata: dict[str, Any],
    edge_frequencies: dict[tuple[str, str], float] | None = None,
    binding_frequencies: dict[str, dict[str, float]] | None = None,
    activity_frequencies: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Serialize a causal graph to a JSON-compatible dict, in sorted order.

    Binding nodes and the edges next to them carry ``ib_frequency``, ``ob_frequency``
    and ``frequency`` (the IB count, else the OB count). Binding nodes and edges also
    carry ``ab_frequency``, the AB counts between the activities they connect. Activity
    nodes carry their A count as ``frequency``.
    """
    task_nodes = set(graph) - binding_nodes
    reverse_graph: dict[str, set[str]] = defaultdict(set)
    for src, targets in graph.items():
        for tgt in targets:
            reverse_graph[tgt].add(src)

    def tasks_from(node: str, adjacency: dict[str, set[str]]) -> list[str]:
        """Activities reached from ``node`` through binding nodes only, sorted."""
        found: set[str] = set()
        seen = {node}
        stack = [node]
        while stack:
            current = stack.pop()
            if current in task_nodes:
                found.add(current)
                continue
            for nxt in adjacency.get(current, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return sorted(found)

    ab_counts = edge_frequencies or {}

    def ab_frequency(sources: list[str], targets: list[str]) -> float:
        return sum(ab_counts.get((s, t), 0) for s in sources for t in targets)

    def binding_data(node_id: str) -> dict[str, float]:
        if not binding_frequencies or node_id not in binding_frequencies:
            return {}
        counts = binding_frequencies[node_id]
        result: dict[str, float] = {}
        ib_freq = float(counts.get("ib_frequency", 0.0))
        ob_freq = float(counts.get("ob_frequency", 0.0))
        if ib_freq > 0:
            result["ib_frequency"] = ib_freq
        if ob_freq > 0:
            result["ob_frequency"] = ob_freq
        if result:
            result["frequency"] = result.get("ib_frequency", ob_freq)
        return result

    nodes = []
    edges_list = []

    for node_id, targets in sorted(graph.items()):
        is_binding = node_id in binding_nodes
        node_data: dict[str, Any] = {
            "id": node_id,
            "data": {"label": node_id, "is_binding": is_binding},
        }

        if not is_binding and activity_frequencies:
            act_freq = activity_frequencies.get(node_id, 0.0)
            if act_freq > 0:
                node_data["data"]["frequency"] = act_freq

        if is_binding:
            node_data["data"].update(binding_data(node_id))
            if edge_frequencies:
                ab_sum = ab_frequency(
                    tasks_from(node_id, reverse_graph), tasks_from(node_id, graph)
                )
                if ab_sum > 0:
                    node_data["data"]["ab_frequency"] = ab_sum

        nodes.append(node_data)

        for target in sorted(targets):
            edge_data: dict[str, Any] = {}

            if is_binding:
                edge_data.update(binding_data(node_id))
            elif target in binding_nodes:
                edge_data.update(binding_data(target))

            if edge_frequencies:
                ab_sum = ab_frequency(
                    tasks_from(node_id, reverse_graph), tasks_from(target, graph)
                )
                if ab_sum > 0:
                    edge_data["ab_frequency"] = ab_sum

            edges_list.append({"source": node_id, "target": target, "data": edge_data})

    return {
        "metadata": metadata,
        "graph": {"directed": True, "nodes": nodes, "edges": edges_list},
    }


def _fodina_named_bindings(
    pm: PatternMiner,
) -> tuple[set[str], dict[str, set[frozenset[str]]], dict[str, set[frozenset[str]]]]:
    """Convert a Fodina PatternMiner's integer-coded bindings to task-name bindings."""
    tasks: set[str] = set()
    IB: dict[str, set[frozenset[str]]] = {}
    OB: dict[str, set[frozenset[str]]] = {}

    for task_id in pm.tm.TC:
        task_name = pm.tm.tau(task_id)
        if not task_name:
            continue
        tasks.add(task_name)

        ib_bindings: set[frozenset[str]] = set()
        for binding in pm.IB.get(task_id, set()):
            names = frozenset(
                name for pred in binding if (name := pm.tm.tau(pred)) is not None
            )
            if names:
                ib_bindings.add(names)
        IB[task_name] = ib_bindings

        ob_bindings: set[frozenset[str]] = set()
        for binding in pm.OB.get(task_id, set()):
            names = frozenset(
                name for succ in binding if (name := pm.tm.tau(succ)) is not None
            )
            if names:
                ob_bindings.add(names)
        OB[task_name] = ob_bindings

    return tasks, IB, OB


def fodina_to_json(pm: PatternMiner, metadata: dict[str, Any]) -> dict[str, Any]:
    """A Fodina net in the JSON format, with its IB/OB, AB and activity counts."""
    tasks, IB, OB = _fodina_named_bindings(pm)
    graph, binding_nodes, binding_lineage = bindings_to_causal_graph(tasks, IB, OB)

    counter = BindingCounter(dg=pm.dg)
    counter.mine(event_log=pm.tm.event_log, ob_skip=set())
    ib_named, ob_named = counter.get_all_counts()
    binding_freqs = binding_frequencies(
        tasks,
        IB,
        OB,
        binding_lineage,
        lambda side, task, binding: float(
            (ib_named if side == "i" else ob_named).get((task, binding), 0.0)
        ),
    )

    edge_freqs: dict[tuple[str, str], float] = {}
    for (a, b), count in pm.dg.measureAB.items():
        if count > 0:
            src, tgt = pm.tm.tau(a), pm.tm.tau(b)
            if src and tgt:
                edge_freqs[(src, tgt)] = count

    act_freqs: dict[str, float] = {}
    for task_id, count in pm.dg.measureA.items():
        name = pm.tm.tau(task_id)
        if name and count > 0:
            act_freqs[name] = float(count)
    return causal_graph_to_json(
        graph,
        binding_nodes,
        metadata,
        edge_freqs,
        binding_freqs,
        activity_frequencies=act_freqs,
    )
