"""Causal nets rebuilt from relation and binding counts.

The Forecast and Re-mined families are rebuilt from counts summed over an evaluation
window: Chronos-2 forecasts for Forecast, observed counts for Re-mined.
:func:`build_causal_net_from_forecast` follows Fodina's path on those counts, so the
rebuilt nets carry the same guarantees as the mined ones: no dead-end place, and no
unreachable activity whenever the connectivity repair succeeds. Relation keys are
written "A→B", without spaces.
"""

from collections import defaultdict
from dataclasses import dataclass

from .causal_net import binding_frequencies, bindings_to_causal_graph
from .data import END_SYMBOL, START_SYMBOL
from .structural_metrics import StructuralStats, stats_from_bindings


def parse_ab_counts(count_AB: dict[str, float]) -> dict[tuple[str, str], float]:
    """Parse "A→B" direct-succession keys into (source, target) pairs."""
    ab_counts: dict[tuple[str, str], float] = {}
    for key, count in count_AB.items():
        if "→" in key:
            parts = key.split("→")
            if len(parts) == 2:
                ab_counts[(parts[0], parts[1])] = count
    return ab_counts


def dependency_measure(
    ab_counts: dict[tuple[str, str], float], a: str, b: str
) -> float:
    """Fodina's dependency measure ``ab / (ab + ba + 1)`` over forecast counts.

    Same as :meth:`fodina.DependencyGraph.dep_measure`. Both the dependency graph and
    the step-3 repair use this one definition.
    """
    ab = ab_counts.get((a, b), 0.0)
    ba = ab_counts.get((b, a), 0.0)
    return ab / (ab + ba + 1)


def build_dependency_graph_from_forecast(
    count_A: dict[str, float],
    count_AB: dict[str, float],
    count_ABA: dict[str, float],
    tD: float = 0.5,
    tL1L: float = 0.5,
    tL2L: float = 0.5,
) -> set[tuple[str, str]]:
    """Build a dependency graph from forecasted relation counts.

    count_AB keys use the format "A→B" (no spaces).

    Returns:
        Set of (source, target) edge pairs.
    """
    ab_counts = parse_ab_counts(count_AB)

    aba_counts: dict[tuple[str, str], float] = {}
    for key, count in count_ABA.items():
        if "→" in key:
            parts = key.split("→")
            if len(parts) == 3 and parts[0] == parts[2]:
                a, b = parts[0], parts[1]
                aba_counts[(a, b)] = aba_counts.get((a, b), 0) + count

    activities = set(count_A.keys())

    def l1l_measure(a: str) -> float:
        aa = ab_counts.get((a, a), 0)
        return aa / (aa + 1)

    def l2l_measure(a: str, b: str) -> float:
        aba = aba_counts.get((a, b), 0)
        bab = aba_counts.get((b, a), 0)
        return (aba + bab) / (aba + bab + 1)

    edges: set[tuple[str, str]] = set()

    for a in activities:
        if count_A.get(a, 0) == 0:
            continue
        if l1l_measure(a) >= tL1L:
            edges.add((a, a))
        for b in activities:
            if count_A.get(b, 0) == 0:
                continue
            if dependency_measure(ab_counts, a, b) >= tD:
                edges.add((a, b))

    # A length-2 loop stays a pair of arcs; Fodina may replace it by an AND.
    for a in activities:
        for b in activities:
            if a == b or ((a, a) in edges and (b, b) in edges):
                continue
            if l2l_measure(a, b) >= tL2L:
                edges.add((a, b))
                edges.add((b, a))

    return edges


def parse_binding_counts(
    counts_by_key: dict[str, float],
) -> dict[str, dict[frozenset[str], float]]:
    """Parse binding counts from "task:{a,b,c}" keys.

    Returns:
        Dict of task -> {frozenset(activities): count}
    """
    result: dict[str, dict[frozenset[str], float]] = defaultdict(dict)
    for key, count in counts_by_key.items():
        if count <= 0 or ":{" not in key:
            continue
        task, rest = key.split(":{", 1)
        activities_str = rest.rstrip("}")
        binding = (
            frozenset(activities_str.split(",")) if activities_str else frozenset()
        )
        result[task][binding] = count
    return dict(result)


def resolve_anchors(
    count_A: dict[str, float],
    ab_counts: dict[tuple[str, str], float],
    tasks: set[str],
) -> tuple[str, str]:
    """Pick the start and end activities the repair works from.

    Unlike Fodina, which uses the most frequent first and last activities, this anchors
    on the ▶ and ■ markers. Preprocessing trims both ends of each log, so the empirical
    end activity can drift into the middle of the process, and decoupling it would
    delete real arcs.

    A window only contains ▶ if a trace starts inside it, so the markers can be missing
    on sparse logs. The fallback then ranks activities by how many traces they start
    (``count_A[a] - Σ_b count_AB[b→a]``) or end, breaking ties by count, so the two
    anchors stay distinct. Markers are looked up in ``tasks``, since an anchor outside
    the arc set could not reach anything.
    """
    if not tasks:
        raise ValueError("cannot resolve anchors for an empty task set")

    def fallback(from_start: bool) -> str:
        def score(a: str) -> tuple[float, float]:
            attached = sum(
                count
                for (src, tgt), count in ab_counts.items()
                if (tgt if from_start else src) == a
            )
            return (count_A.get(a, 0.0) - attached, count_A.get(a, 0.0))

        return max(sorted(tasks), key=score)

    start = START_SYMBOL if START_SYMBOL in tasks else fallback(True)
    end = END_SYMBOL if END_SYMBOL in tasks else fallback(False)
    return start, end


def decouple_start_end(
    DG: set[tuple[str, str]], tasks: set[str], start: str, end: str
) -> None:
    """Port of :meth:`fodina.DependencyGraph._decouple_start_end` (step 2).

    Removes every arc into the start anchor and every arc out of the end anchor, in
    place.
    """
    for a in tasks:
        DG.discard((a, start))
        DG.discard((end, a))


def _unconnected_tasks(
    DG: set[tuple[str, str]], tasks: set[str], anchor: str, from_start: bool
) -> set[str]:
    """Port of :meth:`fodina.DependencyGraph.get_unconnected_tasks`."""
    todo = {anchor}
    visited = {anchor}
    while todo:
        t = todo.pop()
        next_tasks = (
            {b for a, b in DG if a == t} if from_start else {a for a, b in DG if b == t}
        )
        next_tasks -= visited
        visited |= next_tasks
        todo |= next_tasks
    return tasks - visited


def assure_connected(
    DG: set[tuple[str, str]],
    tasks: set[str],
    ab_counts: dict[tuple[str, str], float],
    start: str,
    end: str,
) -> bool:
    """Port of :meth:`fodina.DependencyGraph._assure_connected` (step 3).

    Repeatedly adds the admissible arc with the highest dependency measure that connects
    a task not reachable from ``start`` or not reaching ``end``. Candidates are visited
    in sorted order, so ties resolve deterministically.

    Returns:
        True if every task ends up connected in both directions. False if no
        admissible arc is left, which happens on degenerate windows; the
        remaining tasks then stay unconnected.
    """
    ordered = sorted(tasks)
    while True:
        unc_start = _unconnected_tasks(DG, tasks, start, from_start=True)
        unc_end = _unconnected_tasks(DG, tasks, end, from_start=False)
        if not unc_start and not unc_end:
            return True

        best_dep = -1.0
        best_edge: tuple[str, str] | None = None
        for c in ordered:
            for d in sorted(unc_start):
                if (c, d) not in DG and d != start and c != end:
                    measure = dependency_measure(ab_counts, c, d)
                    if measure > best_dep:
                        best_dep = measure
                        best_edge = (c, d)
            for d in sorted(unc_end):
                if (d, c) not in DG and c != start and d != end:
                    measure = dependency_measure(ab_counts, d, c)
                    if measure > best_dep:
                        best_dep = measure
                        best_edge = (d, c)

        if best_edge is None:
            return False
        DG.add(best_edge)


def admit_bindings(
    bindings: dict[frozenset[str], float], neighbours: set[str]
) -> dict[frozenset[str], float]:
    """Intersect each forecast binding with the task's neighbours (step 5).

    Bindings that intersect to the same set have their counts summed; step 6
    thresholds the sum.
    """
    admitted: dict[frozenset[str], float] = defaultdict(float)
    for binding, count in bindings.items():
        filtered = frozenset(binding & neighbours)
        if filtered and count > 0:
            admitted[filtered] += count
    return dict(admitted)


@dataclass(frozen=True)
class FilterOutcome:
    """What step 6 kept, and what its threshold and re-add each did.

    A binding dropped by the threshold and restored as a singleton appears in both
    halves, so the counts are taken inside the filter.

    The re-add is split by reason, because only one reason is a filter effect:

    ``readded_orphaned``
        the threshold dropped every binding that named this neighbour. This is
        the conversion of AND bindings into choices.
    ``readded_uncovered``
        no admitted binding named the neighbour, for example an arc the step-3
        repair added. Not a filter effect.

    ``threshold_skipped`` is True when a side had bindings but no positive ``count_A``,
    so the threshold could not run. It keeps "kept everything" apart from "never ran".
    """

    kept: dict[frozenset[str], float]
    removed: int
    readded_orphaned: int
    readded_uncovered: int
    threshold_skipped: bool

    @property
    def readded(self) -> int:
        """Every neighbour restored as a singleton, whatever the reason."""
        return self.readded_orphaned + self.readded_uncovered


def apply_binding_filter(
    counts: dict[frozenset[str], float],
    neighbours: set[str],
    activity_count: float,
    tPAT: float,
) -> FilterOutcome:
    """Port of Fodina's binding filter (step 6): threshold, then singleton re-add.

    The threshold is over ``pat_measure(t, x) = x / count_A[t]``. The re-add then
    restores every neighbour the threshold orphaned, which is what turns AND bindings
    into choices, so both halves are needed.

    Fodina's full threshold expression is kept. At ``tPAT = 0.0`` it keeps bindings at
    or above the side's mean, and at ``tPAT = -1.0`` the threshold is zero.
    """
    chosen: dict[frozenset[str], float] = {}

    if counts and activity_count > 0:
        measures = {p: c / activity_count for p, c in counts.items()}
        threshold_base = sum(measures.values()) / len(counts)
        threshold_add = (
            tPAT * threshold_base if tPAT <= 0 else tPAT * (1 - threshold_base)
        )
        threshold = threshold_base + threshold_add
        for binding, count in counts.items():
            if measures[binding] >= threshold:
                chosen[binding] = count
    else:
        # Without a positive activity count pat_measure is undefined, so every admitted
        # binding is kept.
        chosen = dict(counts)

    removed = len(counts) - len(chosen)

    # Neighbours the threshold could orphan: those some admitted binding named. Taken
    # before the re-add changes ``chosen``.
    admitted_neighbours = {ct for binding in counts for ct in binding}

    readded_orphaned = 0
    readded_uncovered = 0
    for ct in sorted(neighbours):
        if not any(ct in binding for binding in chosen):
            singleton = frozenset([ct])
            chosen[singleton] = counts.get(singleton, 0.0)
            if ct in admitted_neighbours:
                readded_orphaned += 1
            else:
                readded_uncovered += 1

    return FilterOutcome(
        kept=chosen,
        removed=removed,
        readded_orphaned=readded_orphaned,
        readded_uncovered=readded_uncovered,
        threshold_skipped=bool(counts) and activity_count <= 0,
    )


def _reachable(graph: dict[str, set[str]], origin: str, forward: bool) -> set[str]:
    """Nodes reachable from ``origin``, following arcs forward or backward."""
    todo = [origin]
    seen = {origin}
    while todo:
        node = todo.pop()
        nxt = (
            graph.get(node, set())
            if forward
            else {src for src, targets in graph.items() if node in targets}
        )
        for n in nxt - seen:
            seen.add(n)
            todo.append(n)
    return seen


def assert_reconstruction_sound(
    graph: dict[str, set[str]],
    binding_nodes: set[str],
    tasks: set[str],
    start: str,
    end: str,
    fully_connected: bool,
) -> None:
    """Check the reconstructed net (step 8) and raise on a violation.

    Raises explicitly rather than with ``assert``, which ``python -O`` removes.

    Checks:

    1. No dead-end place: every binding node has an incoming and an outgoing
       arc. Step 6's re-add guarantees this.
    2. No unreachable activity, checked only when step 3 connected every task.
    """
    for node in sorted(binding_nodes):
        if not graph.get(node):
            raise AssertionError(
                f"dead-end place: binding node {node!r} has no outgoing arc"
            )
        if not any(node in targets for targets in graph.values()):
            raise AssertionError(
                f"unsourced place: binding node {node!r} has no incoming arc"
            )

    if not fully_connected:
        return

    reachable = _reachable(graph, start, forward=True)
    co_reachable = _reachable(graph, end, forward=False)
    for task in sorted(tasks):
        if task not in reachable:
            raise AssertionError(
                f"unreachable activity: {task!r} is not reachable from {start!r}"
            )
        if task not in co_reachable:
            raise AssertionError(f"dead-end activity: {task!r} cannot reach {end!r}")


def build_causal_net_from_forecast(
    count_A: dict[str, float],
    count_AB: dict[str, float],
    count_ABA: dict[str, float],
    forecasted_IB: dict[str, float],
    forecasted_OB: dict[str, float],
    tD: float = 0.5,
    tL1L: float = 0.5,
    tL2L: float = 0.5,
    tPAT: float = 0.0,
) -> tuple[dict[str, set[str]], set[str], dict[str, dict[str, float]], StructuralStats]:
    """Build a causal net from forecast relation counts.

    Runs Fodina's path in eight steps, so every family uses the same tPAT filter and
    Forecast and Re-mined nets have no dead-end place. They also have no unreachable
    activity when step 3 connects every task; on a degenerate window where it cannot,
    the remaining tasks stay unconnected and step 8 skips that check. The steps:

    1. build the dependency graph from the counts
    2. anchor ▶ and ■ and decouple them
    3. repair connectedness
    4. derive predecessors and successors
    5. admit bindings (binding ∩ neighbours)
    6. tPAT filter and singleton re-add
    7. singleton fallback
    8. check the result

    Two rules differ from Fodina's miner. The tasks are the endpoints of the arcs of
    step 1, so an activity with no admitted arc is left out, where Fodina keeps every
    task and connects it in step 3. A length-2 loop stays a pair of arcs, where Fodina
    replaces it by an AND when it can (``prefer_and_to_l2l``).

    Structural statistics are measured on the binding sets of step 6, before step 8
    turns them into graph nodes (see :mod:`structural_metrics`). Arcs added by step 3
    and bindings removed by step 6 are counted separately.

    Returns:
        (graph, binding_nodes, binding_freqs, stats)
    """
    # 1. Dependency graph from the forecast counts.
    DG = build_dependency_graph_from_forecast(
        count_A, count_AB, count_ABA, tD, tL1L, tL2L
    )
    if not DG:
        # No arc cleared tD, so steps 3 to 7 had nothing to do. Their counters are 0,
        # not missing.
        empty_stats = stats_from_bindings(
            tasks=0,
            arcs=0,
            IB={},
            OB={},
            repair_arcs_added=0,
            filter_bindings_removed=0,
            singleton_readds_orphaned=0,
            singleton_readds_uncovered=0,
            singleton_fallback_fires=0,
        )
        return {}, set(), {}, empty_stats

    # Only activities on an admitted arc become tasks (see the docstring).
    tasks = {a for a, b in DG} | {b for a, b in DG}
    ab_counts = parse_ab_counts(count_AB)

    # 2. Anchor on the declared boundary markers, then decouple them.
    start_anchor, end_anchor = resolve_anchors(count_A, ab_counts, tasks)
    decouple_start_end(DG, tasks, start_anchor, end_anchor)

    # 3. Repair connectedness. Adds arcs, so it runs before bindings are derived.
    arcs_before_repair = len(DG)
    fully_connected = assure_connected(DG, tasks, ab_counts, start_anchor, end_anchor)
    repair_arcs_added = len(DG) - arcs_before_repair

    # 4. Derive neighbour sets from the repaired arc set.
    predecessors: dict[str, set[str]] = defaultdict(set)
    successors: dict[str, set[str]] = defaultdict(set)
    for a, b in DG:
        if a != b:
            successors[a].add(b)
            predecessors[b].add(a)

    ib_bindings = parse_binding_counts(forecasted_IB)
    ob_bindings = parse_binding_counts(forecasted_OB)

    # 5. Admit bindings, then 6. filter them with tPAT + singleton re-add.
    ib_counts: dict[str, dict[frozenset[str], float]] = {}
    ob_counts: dict[str, dict[frozenset[str], float]] = {}
    filter_bindings_removed = 0
    singleton_readds_orphaned = 0
    singleton_readds_uncovered = 0
    for task in tasks:
        activity_count = count_A.get(task, 0.0)
        ib_outcome = apply_binding_filter(
            admit_bindings(ib_bindings.get(task, {}), predecessors[task]),
            predecessors[task],
            activity_count,
            tPAT,
        )
        ob_outcome = apply_binding_filter(
            admit_bindings(ob_bindings.get(task, {}), successors[task]),
            successors[task],
            activity_count,
            tPAT,
        )
        ib_counts[task] = ib_outcome.kept
        ob_counts[task] = ob_outcome.kept
        for outcome in (ib_outcome, ob_outcome):
            filter_bindings_removed += outcome.removed
            singleton_readds_orphaned += outcome.readded_orphaned
            singleton_readds_uncovered += outcome.readded_uncovered
            # Step 1 only admits arcs between activities with a positive count, so
            # every task has one and filter_bindings_removed == 0 means the threshold
            # ran and kept everything.
            if outcome.threshold_skipped:
                raise AssertionError(
                    f"task {task!r} reached the dependency graph with no positive "
                    "activity count: the tPAT threshold could not be applied to "
                    "its bindings, so the filter counters are not attributable"
                )

    # 7. Singleton fallback for any side left empty, counted per side. Step 6
    #    already covers every neighbour, so this should not fire.
    singleton_fallback_fires = 0
    for t in tasks:
        if predecessors[t] and not ib_counts[t]:
            singleton_fallback_fires += 1
            for pred in predecessors[t]:
                ib_counts[t][frozenset([pred])] = 0.0
        if successors[t] and not ob_counts[t]:
            singleton_fallback_fires += 1
            for succ in successors[t]:
                ob_counts[t][frozenset([succ])] = 0.0

    IB: dict[str, set[frozenset[str]]] = {t: set(ib_counts[t]) for t in tasks}
    OB: dict[str, set[frozenset[str]]] = {t: set(ob_counts[t]) for t in tasks}

    # Measured on the binding sets, before they become graph nodes.
    stats = stats_from_bindings(
        tasks=len(tasks),
        arcs=len(DG),
        IB=IB,
        OB=OB,
        repair_arcs_added=repair_arcs_added,
        filter_bindings_removed=filter_bindings_removed,
        singleton_readds_orphaned=singleton_readds_orphaned,
        singleton_readds_uncovered=singleton_readds_uncovered,
        singleton_fallback_fires=singleton_fallback_fires,
    )

    graph, binding_nodes, binding_lineage = bindings_to_causal_graph(tasks, IB, OB)
    binding_freqs = binding_frequencies(
        tasks,
        IB,
        OB,
        binding_lineage,
        lambda side, task, binding: (ib_counts if side == "i" else ob_counts)[task][
            binding
        ],
    )

    # 8. Fail loudly if the repair did not hold.
    assert_reconstruction_sound(
        graph, binding_nodes, tasks, start_anchor, end_anchor, fully_connected
    )

    return graph, binding_nodes, binding_freqs, stats
