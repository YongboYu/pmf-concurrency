"""Daily input and output binding counts (IB/OB) from windowed traces.

Uses an adaptation of Fodina's PatternMiner that counts bindings instead of selecting
them.

IB counting: on day D an input binding is counted only if its task fires on day D, so no
occurrence is counted twice.

OB counting: an output binding is only visible after its task fires, often on a later
day. Each task occurrence, keyed by (case, position), is counted once, on the first day
a non-empty output binding is visible in its trace prefix. Only data up to day D is
used, so there is no look-ahead.
"""

from collections import defaultdict
from datetime import date

import pandas as pd

from .data import WindowedTrace
from .fodina import DependencyGraph, TaskMiner
from .relations import daily_calendar


class BindingCounter:
    """Fodina's PatternMiner, returning raw binding counts instead of a filtered set.

    Uses the same detection as ``PatternMiner._find_all_patterns`` but returns the
    counts before the tPAT filter. After ``mine()``, ``newly_counted_ob`` holds the
    (sigma_idx, position) pairs whose OB was counted for the first time.
    """

    def __init__(self, dg: DependencyGraph):
        self.dg = dg
        self.tm = dg.tm
        self.IB_counts: dict[int, dict[frozenset[int], int]] = {}
        self.OB_counts: dict[int, dict[frozenset[int], int]] = {}
        self.newly_counted_ob: set[tuple[int, int]] = set()

    def mine(
        self,
        event_log: list[list[str]] | None = None,
        trace_dates: list[list[date]] | None = None,
        window_date: date | None = None,
        ob_skip: set[tuple[int, int]] | None = None,
    ) -> None:
        """Count bindings in the event log.

        Args:
            event_log:    Traces to mine. Defaults to the TaskMiner's log.
            trace_dates:  Per-event dates, parallel to event_log; needed for IB.
            window_date:  Count an IB only if its task fires on this date.
            ob_skip:      (sigma_idx, position) pairs whose OB was counted in an
                          earlier window and is skipped. Empty or None at first.
        """
        self.IB_counts = {}
        self.OB_counts = {}
        self.newly_counted_ob = set()

        event_log = event_log if event_log is not None else self.dg.tm.event_log

        for t in self.tm.TC:
            inputs = {ab[0] for ab in self.dg.DG if ab[1] == t}
            outputs = {ab[1] for ab in self.dg.DG if ab[0] == t}

            if inputs:
                counts = self._find_bindings_with_counts(
                    t,
                    inputs,
                    True,
                    event_log,
                    trace_dates,
                    window_date=window_date,
                    ob_skip=None,
                    ob_newly_counted=None,
                )
                if counts:
                    self.IB_counts[t] = counts

            if outputs:
                counts = self._find_bindings_with_counts(
                    t,
                    outputs,
                    False,
                    event_log,
                    trace_dates,
                    window_date=None,
                    ob_skip=ob_skip,
                    ob_newly_counted=self.newly_counted_ob,
                )
                if counts:
                    self.OB_counts[t] = counts

    def _find_bindings_with_counts(
        self,
        forTask: int,
        connectedTasks: set[int],
        forInput: bool,
        event_log: list[list[str]],
        trace_dates: list[list[date]] | None = None,
        window_date: date | None = None,
        ob_skip: set[tuple[int, int]] | None = None,
        ob_newly_counted: set[tuple[int, int]] | None = None,
    ) -> dict[frozenset[int], int]:
        """Find all bindings with their counts, without the tPAT filter.

        IB (forInput=True): counted only when the task fires on window_date. OB
        (forInput=False): counted unless (sigma_idx, i) is in ob_skip; newly counted
        occurrences go into ob_newly_counted.
        """
        patternsWithCounts: dict[frozenset[int], int] = {}

        for sigma_idx, sigma in enumerate(event_log):
            sigma_dates = (
                trace_dates[sigma_idx]
                if trace_dates is not None and sigma_idx < len(trace_dates)
                else None
            )

            for i in range(len(sigma)):
                if self.tm.mu(sigma, i) != forTask:
                    continue

                if forInput:
                    # IB: count only when task fires on window_date
                    if (
                        window_date is not None
                        and sigma_dates is not None
                        and i < len(sigma_dates)
                        and sigma_dates[i] != window_date
                    ):
                        continue
                else:
                    # OB: skip occurrences already counted in a prior window
                    if ob_skip is not None and (sigma_idx, i) in ob_skip:
                        continue

                set_pattern: set[int] = set()

                for ct in connectedTasks:
                    inputs_for_ct = {ab[0] for ab in self.dg.DG if ab[1] == ct}
                    outputs_for_ct = {ab[1] for ab in self.dg.DG if ab[0] == ct}
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
                                (ai in inputs_for_ct and not forInput)
                                or (ai in outputs_for_ct and forInput)
                            ):
                                checks = False
                                break

                        if checks:
                            set_pattern.add(ct)
                        break

                if not set_pattern:
                    continue

                pattern = frozenset(set_pattern)
                patternsWithCounts[pattern] = patternsWithCounts.get(pattern, 0) + 1

                # Record it so the caller skips it in later windows.
                if not forInput and ob_newly_counted is not None:
                    ob_newly_counted.add((sigma_idx, i))

        return patternsWithCounts

    def get_all_counts(
        self,
    ) -> tuple[
        dict[tuple[str, frozenset[str]], int], dict[tuple[str, frozenset[str]], int]
    ]:
        """Binding counts keyed by task names instead of indices.

        Returns:
            (ib_counts, ob_counts), each mapping (task name, binding) to a count.
        """
        ib_named: dict[tuple[str, frozenset[str]], int] = {}
        ob_named: dict[tuple[str, frozenset[str]], int] = {}

        for t, patterns in self.IB_counts.items():
            task_name = self.tm.tau(t)
            if not task_name:
                continue
            for pattern, count in patterns.items():
                names: frozenset[str] = frozenset(
                    n for x in pattern if (n := self.tm.tau(x)) is not None
                )
                if names:
                    ib_named[(task_name, names)] = count

        for t, patterns in self.OB_counts.items():
            task_name = self.tm.tau(t)
            if not task_name:
                continue
            for pattern, count in patterns.items():
                names = frozenset(
                    n for x in pattern if (n := self.tm.tau(x)) is not None
                )
                if names:
                    ob_named[(task_name, names)] = count

        return ib_named, ob_named


def mine_dependency_graph(
    traces: list[list[str]], tD: float, tL1L: float, tL2L: float
) -> DependencyGraph:
    """Fodina's tasks and dependency graph, with the settings every model uses.

    The binding series and the Direct-mined and Static nets share these settings, so
    their bindings are counted on the same kind of dependency graph.
    """
    tm = TaskMiner(
        event_log=traces,
        collapse_repeats=True,
        count_collapsed=True,
        backward_context=1,
        forward_context=1,
        tDUP=0.90,
    )
    tm.mine()
    dg = DependencyGraph(
        tm=tm,
        tD=tD,
        tL1L=tL1L,
        tL2L=tL2L,
        prevent_l2l_with_l1l=True,
        prefer_and_to_l2l=True,
        all_connected=True,
        long_dependencies=False,
    )
    dg.mine()
    return dg


def count_window_bindings(
    windowed_traces: list[WindowedTrace],
    tD: float = 0.5,
    tL1L: float = 0.5,
    tL2L: float = 0.5,
    window_date: date | None = None,
    case_id_to_ob_positions: dict[str, set[int]] | None = None,
) -> tuple[
    dict[tuple[str, frozenset[str]], int],
    dict[tuple[str, frozenset[str]], int],
    set[tuple[str, int]],
]:
    """IB and OB binding counts of one day's window.

    The counts are taken before any binding filter, so they do not depend on tPAT.

    Args:
        windowed_traces:         Trace prefixes visible in this window.
        tD, tL1L, tL2L:          Fodina's dependency thresholds.
        window_date:             Date an IB's task must fire on.
        case_id_to_ob_positions: case id -> trace positions whose OB is already
                                 counted. Shared across windows; the caller
                                 updates it after this call.

    Returns:
        (ib_counts, ob_counts, newly_counted_ob): counts per (task, binding), and
        the (case id, position) pairs counted for the first time.
    """
    if not windowed_traces:
        return {}, {}, set()

    # Keep non-empty traces and their case ids in the same order, so the miner's
    # sigma_idx maps back to the case id.
    active = [(wt.case_id, wt) for wt in windowed_traces if wt.activities]
    if not active:
        return {}, {}, set()

    case_ids = [cid for cid, _ in active]
    traces = [wt.activities for _, wt in active]
    trace_dates = [[ts.date() for ts in wt.timestamps] for _, wt in active]

    # Global (case id, position) pairs to the miner's (sigma_idx, position).
    ob_skip: set[tuple[int, int]] = set()
    if case_id_to_ob_positions is not None:
        for sigma_idx, case_id in enumerate(case_ids):
            for pos in case_id_to_ob_positions.get(case_id, ()):
                ob_skip.add((sigma_idx, pos))

    pm = BindingCounter(dg=mine_dependency_graph(traces, tD, tL1L, tL2L))
    pm.mine(traces, trace_dates=trace_dates, window_date=window_date, ob_skip=ob_skip)
    ib_counts, ob_counts = pm.get_all_counts()

    # Translate miner-local (sigma_idx, pos) back to global (case_id, pos).
    newly_counted: set[tuple[str, int]] = {
        (case_ids[sigma_idx], pos)
        for sigma_idx, pos in pm.newly_counted_ob
        if sigma_idx < len(case_ids)
    }

    return ib_counts, ob_counts, newly_counted


def extract_binding_timeseries(
    windows: dict[date, list[WindowedTrace]],
    tD: float = 0.5,
    tL1L: float = 0.5,
    tL2L: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily IB and OB series, one column per binding.

    IB columns count a binding on the days its task fires. OB columns count a binding on
    the day it first becomes visible, once per task occurrence (see the module
    docstring). Columns are named "task:{a,b,c}" with sorted activities.

    Returns:
        (daily_IB, daily_OB) DataFrames of counts for every day from the first to the
        last window.
    """
    ib_series: dict[str, dict[date, int]] = defaultdict(lambda: defaultdict(int))
    ob_series: dict[str, dict[date, int]] = defaultdict(lambda: defaultdict(int))

    # case id -> trace positions whose OB is already counted, across windows.
    case_id_to_ob_positions: dict[str, set[int]] = defaultdict(set)

    total = len(windows)
    print(f"  Counting bindings in {total} daily windows...")

    for i, (window_date, windowed_traces) in enumerate(sorted(windows.items())):
        if (i + 1) % 50 == 0:
            print(f"  Processing window {i + 1}/{total}...")

        ib_counts, ob_counts, newly_counted = count_window_bindings(
            windowed_traces,
            tD,
            tL1L,
            tL2L,
            window_date=window_date,
            case_id_to_ob_positions=case_id_to_ob_positions,
        )

        # Remember newly counted OB positions for later windows.
        for case_id, pos in newly_counted:
            case_id_to_ob_positions[case_id].add(pos)

        for (task, pattern), count in ib_counts.items():
            key = f"{task}:{{{','.join(sorted(pattern))}}}"
            ib_series[key][window_date] += count

        for (task, pattern), count in ob_counts.items():
            key = f"{task}:{{{','.join(sorted(pattern))}}}"
            ob_series[key][window_date] += count

    calendar = daily_calendar(windows)

    def _series_to_df(series: dict[str, dict[date, int]]) -> pd.DataFrame:
        if not series:
            return pd.DataFrame(index=calendar)
        all_dates = sorted({d for counts in series.values() for d in counts})
        data = {
            pattern: [counts.get(d, 0) for d in all_dates]
            for pattern, counts in sorted(series.items())
        }
        frame = pd.DataFrame(data, index=pd.to_datetime(all_dates))
        # Every day of the log, so these rows line up with the relation series.
        return frame.reindex(calendar, fill_value=0)

    daily_IB = _series_to_df(dict(ib_series))
    daily_OB = _series_to_df(dict(ob_series))

    print(f"  IB bindings: {len(daily_IB.columns)} unique")
    print(f"  OB bindings: {len(daily_OB.columns)} unique")
    if not daily_IB.empty:
        print(f"  Total IB occurrences: {int(daily_IB.sum().sum())}")
    if not daily_OB.empty:
        print(f"  Total OB occurrences: {int(daily_OB.sum().sum())}")

    return daily_IB, daily_OB
