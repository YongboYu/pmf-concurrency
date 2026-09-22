"""Fodina (vanden Broucke and De Weerdt, Decision Support Systems 100, 2017).

Task mining, the dependency graph and the input and output bindings of each task.
"""


class TaskMiner:
    def __init__(
        self,
        event_log: list[list[str]],
        collapse_repeats: bool = True,
        count_collapsed: bool = True,
        backward_context: int = 1,
        forward_context: int = 1,
        tDUP: float = 0.90,
    ):
        self.event_log = event_log
        self.collapse_repeats = collapse_repeats
        self.count_collapsed = count_collapsed
        self.backward_context = backward_context
        self.forward_context = forward_context
        self.tDUP = tDUP

        self._reset()

    def _reset(self) -> None:
        self.MAPPING: dict[tuple[str, tuple[str, ...], tuple[str, ...]], int] = {}
        self.TC: set[int] = set()
        self.counts: dict[tuple[str, tuple[str, ...], tuple[str, ...]], int] = {}
        self.next_task = 0
        self.idx_to_task: dict[int, str] = {}
        self.most_started = -1
        self.most_ended = -1

    def _fragment(
        self, sigma: list[str], i: int
    ) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        b = tuple(sigma[max(0, i - self.backward_context) : i])
        f = tuple(sigma[i + 1 : min(len(sigma), i + 1 + self.forward_context)])
        return (sigma[i], b, f)

    def _collapse_fragments(self) -> None:
        done = False
        while not done:
            done = True
            for frag1 in self.MAPPING:
                for frag2 in self.MAPPING:
                    if (
                        self.MAPPING[frag1] == self.MAPPING[frag2]
                        or frag1[0] != frag2[0]
                    ):
                        continue
                    if frag1[1] == frag2[1] or frag1[2] == frag2[2]:
                        done = False
                        for frag3 in self.MAPPING:
                            if self.MAPPING[frag3] == self.MAPPING[frag2]:
                                self.MAPPING[frag3] = self.MAPPING[frag1]

    def _filter_tDUP(self) -> None:
        task_log = set(x for y in self.event_log for x in y)
        for act in task_log:
            total_freq = sum(
                self.counts[fragment] for fragment in self.MAPPING if fragment[0] == act
            )
            fragments = [fragment for fragment in self.MAPPING if fragment[0] == act]
            if not fragments:
                continue
            max_fragment = max(fragments, key=lambda x: self.counts[x] / total_freq)
            highest_task = self.MAPPING[max_fragment]
            for fragment in fragments:
                freq = self.counts[fragment] / total_freq
                if freq < self.tDUP:
                    for fragment2 in self.MAPPING:
                        if self.MAPPING[fragment2] == self.MAPPING[fragment]:
                            self.MAPPING[fragment2] = highest_task

    def mine(self) -> None:
        self._reset()

        # Map fragments and collapse duplicates
        for sigma in self.event_log:
            prev = None
            prev_task_idx = -1
            for i in range(len(sigma)):
                fragment = self._fragment(sigma, i)
                if self.collapse_repeats and prev == sigma[i]:
                    self.MAPPING[fragment] = prev_task_idx
                    if self.count_collapsed:
                        self.counts[fragment] = self.counts.get(fragment, 0) + 1
                else:
                    if fragment not in self.MAPPING:
                        self.MAPPING[fragment] = self.next_task
                        self.counts[fragment] = 0
                        self.next_task += 1
                    self.counts[fragment] += 1
                    prev_task_idx = self.MAPPING[fragment]
                prev = sigma[i]

        # Collapse mappings
        self._collapse_fragments()

        # Filter by tDUP
        self._filter_tDUP()
        self.TC = set(self.MAPPING.values())
        self.idx_to_task = {v: k[0] for k, v in self.MAPPING.items()}

        # Most started and ended
        self.most_started = max(
            self.TC,
            key=lambda x: sum(self.mu(sigma, 0) == x for sigma in self.event_log),
        )
        self.most_ended = max(
            self.TC,
            key=lambda x: sum(
                self.mu(sigma, len(sigma) - 1) == x for sigma in self.event_log
            ),
        )

    def mu(self, sigma: list[str], i: int) -> int:
        return self.MAPPING[self._fragment(sigma, i)]

    def tau(self, idx: int) -> str | None:
        return self.idx_to_task.get(idx)


class DependencyGraph:
    def __init__(
        self,
        tm: "TaskMiner",
        tD: float = 0.5,
        tL1L: float = 0.5,
        tL2L: float = 0.5,
        tLD: float = 0.9,
        prevent_l2l_with_l1l: bool = True,
        prefer_and_to_l2l: bool = True,
        all_connected: bool = True,
        long_dependencies: bool = False,
    ):
        self.tm = tm
        self.TC = tm.TC
        self.tD = tD
        self.tL1L = tL1L
        self.tL2L = tL2L
        self.tLD = tLD
        self.prevent_l2l_with_l1l = prevent_l2l_with_l1l
        self.prefer_and_to_l2l = prefer_and_to_l2l
        self.all_connected = all_connected
        self.long_dependencies = long_dependencies

        self._reset()

    def _reset(self) -> None:
        self.measureA = dict.fromkeys(self.TC, 0)
        self.measureAB = {(a, b): 0 for a in self.TC for b in self.TC}
        self.measureABA = {(a, b): 0 for a in self.TC for b in self.TC}
        self.measureAXB = {(a, b): 0 for a in self.TC for b in self.TC}

        self.DG: set[tuple[int, int]] = set()
        self.LONGD: set[tuple[int, int]] = set()

    def dep_measure(self, a: int, b: int) -> float:
        ab = self.measureAB[(a, b)]
        ba = self.measureAB[(b, a)]
        return ab / (ab + ba + 1)

    def l1l_measure(self, a: int) -> float:
        return self.measureAB[(a, a)] / (self.measureAB[(a, a)] + 1)

    def l2l_measure(self, a: int, b: int) -> float:
        aba = self.measureABA[(a, b)]
        bab = self.measureABA[(b, a)]
        return (aba + bab) / (aba + bab + 1)

    def ldi_measure(self, a: int, b: int) -> float:
        axb = self.measureAXB[(a, b)]
        aa = self.measureA[a]
        bb = self.measureA[b]
        return (2.0 * axb) / (aa + bb + 1) - (2.0 * abs(aa - bb)) / (aa + bb + 1)

    def pat_measure(self, t: int, x: float) -> float:
        return x / self.measureA[t]

    def path_exists_without_visiting(self, start: int, end: int, visiting: int) -> bool:
        todo = set([start])
        visited = set([start])
        while len(todo):
            t = todo.pop()
            if t == visiting:
                continue
            visited |= {t}
            outputs = set([ab[1] for ab in self.DG if ab[0] == t])
            outputs -= visited
            todo |= outputs
            if end in visited:
                return True
        return False

    def get_unconnected_tasks(self, from_start: bool = True) -> set[int]:
        start = self.tm.most_started if from_start else self.tm.most_ended
        todo = {start}
        visited = {start}
        while todo:
            t = todo.pop()
            outputs = {b for a, b in self.DG if a == t}
            inputs = {a for a, b in self.DG if b == t}
            next_tasks = outputs if from_start else inputs
            next_tasks -= visited
            visited |= next_tasks
            todo |= next_tasks
        return set(self.TC) - visited

    def _extract_relations(self, event_log: list[list[str]]) -> None:
        for sigma in event_log:
            for i in range(len(sigma)):
                a = self.tm.mu(sigma, i)
                self.measureA[a] += 1
            for i in range(len(sigma) - 1):
                a, b = self.tm.mu(sigma, i), self.tm.mu(sigma, i + 1)
                self.measureAB[(a, b)] += 1
            for i in range(len(sigma) - 2):
                a, b, c = (
                    self.tm.mu(sigma, i),
                    self.tm.mu(sigma, i + 1),
                    self.tm.mu(sigma, i + 2),
                )
                if a == c:
                    self.measureABA[(a, b)] += 1
            for i in range(len(sigma)):
                for j in range(i + 1, len(sigma)):
                    skip = False
                    for k in range(i + 1, j):
                        if self.tm.mu(sigma, j) == self.tm.mu(sigma, k):
                            skip = True
                            break
                    if not skip:
                        a, b = self.tm.mu(sigma, i), self.tm.mu(sigma, j)
                        self.measureAXB[(a, b)] += 1
                    if self.tm.mu(sigma, i) == self.tm.mu(sigma, j):
                        break

    def _build_basic_graph(self) -> None:
        # Basic dependencies and L1L
        for a in self.TC:
            if self.l1l_measure(a) >= self.tL1L:
                self.DG.add((a, a))
            for b in self.TC:
                if self.dep_measure(a, b) >= self.tD:
                    self.DG.add((a, b))

        # Add L2L
        for a in self.TC:
            for b in self.TC:
                if a == b:
                    continue
                if (
                    self.prevent_l2l_with_l1l
                    and (a, a) in self.DG
                    and (b, b) in self.DG
                ):
                    continue
                if self.l2l_measure(a, b) >= self.tL2L:
                    self.DG.add((a, b))
                    self.DG.add((b, a))

    def _replace_l2l_by_and(self) -> None:
        to_remove = set()
        to_add = set()
        for a in self.TC:
            for b in self.TC:
                if a == b:
                    continue  # skip self-loops!
                if (a, b) not in self.DG or (b, a) not in self.DG:
                    continue
                if self.measureABA[(a, b)] > 0:
                    to_add.add((a, a))
                if self.measureABA[(b, a)] > 0:
                    to_add.add((b, b))
                for c in self.TC:
                    if c == a or c == b:
                        continue
                    if (c, a) in self.DG or (c, b) in self.DG:
                        to_add.add((c, a))
                        to_add.add((c, b))
                    if (a, c) in self.DG or (b, c) in self.DG:
                        to_add.add((a, c))
                        to_add.add((b, c))
                to_remove.add((a, b))
                to_remove.add((b, a))
        self.DG -= to_remove
        self.DG |= to_add

    def _decouple_start_end(self) -> None:
        for a in list(self.TC):
            self.DG.discard((a, self.tm.most_started))
            self.DG.discard((self.tm.most_ended, a))

    def _assure_connected(self) -> None:
        while True:
            unc_start = self.get_unconnected_tasks(from_start=True)
            unc_end = self.get_unconnected_tasks(from_start=False)
            if not unc_start and not unc_end:
                break
            best_dep: float = -1
            best_edge = None
            for c in self.TC:
                for d in unc_start:
                    measure = self.dep_measure(c, d)
                    if (
                        (c, d) not in self.DG
                        and d != self.tm.most_started
                        and c != self.tm.most_ended
                    ):
                        if measure > best_dep:
                            best_dep = measure
                            best_edge = (c, d)
                for d in unc_end:
                    measure = self.dep_measure(d, c)
                    if (
                        (d, c) not in self.DG
                        and c != self.tm.most_started
                        and d != self.tm.most_ended
                    ):
                        if measure > best_dep:
                            best_dep = measure
                            best_edge = (d, c)
            if best_edge is None:
                # No admissible edge can connect the remaining tasks (e.g. a
                # degenerate low-volume day of length-1 traces with no
                # successions, where most_started == most_ended). Without this
                # guard the loop spins forever, since unc_start/unc_end never
                # empty. Leave the residual tasks unconnected and stop.
                break
            self.DG.add(best_edge)

    def _long_dependencies(self) -> None:
        for a in self.TC:
            for b in self.TC:
                if (
                    self.ldi_measure(a, b) >= self.tLD
                    and self.path_exists_without_visiting(
                        self.tm.most_started, self.tm.most_ended, a
                    )
                    and self.path_exists_without_visiting(
                        self.tm.most_started, self.tm.most_ended, b
                    )
                    and self.path_exists_without_visiting(a, self.tm.most_ended, b)
                ):
                    self.DG |= {(a, b)}
                    self.LONGD |= {(a, b)}

    def mine(self, event_log: list[list[str]] | None = None) -> None:
        self._reset()

        event_log = event_log if event_log is not None else self.tm.event_log

        # Extract log relations
        self._extract_relations(event_log)

        # Build basic dependency graph
        self._build_basic_graph()

        # Optionally relink L2L as AND-splits
        if self.prefer_and_to_l2l:
            self._replace_l2l_by_and()

        # Set start and end tasks
        self._decouple_start_end()

        # Optionally make all tasks connected
        if self.all_connected:
            self._assure_connected()

        # Optionally mine long dependencies
        if self.long_dependencies:
            self._long_dependencies()


class PatternMiner:
    def __init__(self, dg: "DependencyGraph", tPAT: float = 0.00):
        self.dg = dg
        self.tm = dg.tm
        self.TC = dg.TC
        self.tPAT = tPAT

        self._reset()

    def _reset(self) -> None:
        self.IB: dict[int, set[frozenset[int]]] = {k: set() for k in self.tm.TC}
        self.OB: dict[int, set[frozenset[int]]] = {k: set() for k in self.tm.TC}

    def _find_all_patterns(
        self,
        forTask: int,
        connectedTasks: set[int],
        forInput: bool,
        event_log: list[list[str]],
    ) -> set[frozenset[int]]:
        patternsWithCounts = {}
        chosenpatterns = set()
        for sigma in event_log:
            for i in range(len(sigma)):
                if self.tm.mu(sigma, i) != forTask:
                    continue
                set_pattern: set[int] = set()
                for ct in connectedTasks:
                    inputsForConnected = {ab[0] for ab in self.dg.DG if ab[1] == ct}
                    outputsForConnected = {ab[1] for ab in self.dg.DG if ab[0] == ct}
                    crange = (
                        range(i - 1, -1, -1) if forInput else range(i + 1, len(sigma))
                    )
                    for pos in crange:
                        if self.tm.mu(sigma, pos) == forTask:
                            break
                        if self.tm.mu(sigma, pos) != ct:
                            continue
                        checks = True
                        mrange = (
                            range(pos + 1, i) if forInput else range(pos - 1, i, -1)
                        )
                        for mid in mrange:
                            ai = self.tm.mu(sigma, mid)
                            if (ai, ct) not in self.dg.LONGD and (
                                (ai in inputsForConnected and not forInput)
                                or (ai in outputsForConnected and forInput)
                            ):
                                checks = False
                                break
                        if checks:
                            set_pattern.add(ct)
                        break
                if not set_pattern:
                    continue
                pattern = frozenset(set_pattern)
                if pattern not in patternsWithCounts:
                    patternsWithCounts[pattern] = 0
                patternsWithCounts[pattern] += 1
        # Select the most relevant patterns
        if patternsWithCounts:
            frequencyTotal = sum(
                [
                    self.dg.pat_measure(forTask, patternsWithCounts[pattern])
                    for pattern in patternsWithCounts
                ]
            )
            thresholdBase = frequencyTotal / len(patternsWithCounts)
            thresholdAdd = (
                self.tPAT * thresholdBase
                if self.tPAT <= 0
                else self.tPAT * (1 - thresholdBase)
            )
            threshold = thresholdBase + thresholdAdd
            for pattern in patternsWithCounts:
                if (
                    self.dg.pat_measure(forTask, patternsWithCounts[pattern])
                    >= threshold
                ):
                    chosenpatterns.add(pattern)
        # Ensure all connected tasks appear
        for ct in connectedTasks:
            found = False
            for pattern in chosenpatterns:
                if ct in pattern:
                    found = True
                    break
            if not found:
                chosenpatterns.add(frozenset([ct]))
        return chosenpatterns

    def mine(self, event_log: list[list[str]] | None = None) -> None:
        self._reset()

        event_log = event_log if event_log is not None else self.dg.tm.event_log

        for t in self.tm.TC:
            inputs = {ab[0] for ab in self.dg.DG if ab[1] == t}
            outputs = {ab[1] for ab in self.dg.DG if ab[0] == t}
            if inputs:
                inputPatterns = self._find_all_patterns(t, inputs, True, event_log)
                self.IB[t] = inputPatterns
            if outputs:
                outputPatterns = self._find_all_patterns(t, outputs, False, event_log)
                self.OB[t] = outputPatterns
