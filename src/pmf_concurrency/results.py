"""The paper's numbers, read from one results directory.

A results directory holds the files that the paper's tables, figure and prose numbers
are computed from, and ``MANIFEST.yaml``, which names their sources and the SHA-256 of
every file. ``scripts/paper.py collect`` writes it from the campaign and pipeline
directories; ``scripts/paper.py build`` reads nothing else.

    sha256_file, names_and_bytes_sha256 fingerprints, the second as campaigns record
    read_manifest                       the manifest, every file checked against it
    read_campaign                       a campaign's units, once its gate holds
    describe_log, window_composition    Table 1, from a log and its windows
    pool                                the mean of one metric over finished units
    over_budget_share                   the share of units stopped by the budget
    reduction_effect                    unreduced against reduced nets (Section 5.2)
    replay_mode_ranges                  the BPI2017 replay modes (Section 5.2)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]

from .campaign import UNIT_KEY, read_expected, reconcile
from .conformance import normalise_timestamp, parse_window_date
from .guard import OVER_BUDGET

#: The manifest's file name inside a results directory.
MANIFEST = "MANIFEST.yaml"

CASE_KEY = "case:concept:name"
ACTIVITY_KEY = "concept:name"
TIMESTAMP_KEY = "time:timestamp"

#: How a window-active case overlaps its window, in Table 1's order.
CASE_CLASSES = ("complete", "prefix_truncated", "suffix_truncated", "middle_crossing")


class ResultsError(ValueError):
    """A results directory, or a campaign in it, that the paper must not be built from."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def names_and_bytes_sha256(paths: list[Path]) -> str:
    """One SHA-256 over each file's name and bytes, in order.

    ``scripts/evaluate.py`` fingerprints a log (``log_sha256``) and a run's
    ``window_*.json`` in name order (``models_sha256``) this way.
    """
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _listed_files(results_dir: Path) -> set[str]:
    return {
        p.relative_to(results_dir).as_posix()
        for p in results_dir.rglob("*")
        if p.is_file() and p.name != MANIFEST and not p.name.startswith(".")
    }


def read_manifest(results_dir: Path) -> dict:
    """``MANIFEST.yaml``, once every file it lists is present and unchanged.

    A file on disk that the manifest does not list is refused as well, so nothing can
    enter the paper without a recorded hash.
    """
    path = results_dir / MANIFEST
    if not path.exists():
        raise ResultsError(f"{results_dir}: no {MANIFEST}")
    manifest: dict = yaml.safe_load(path.read_text())
    files: dict[str, str] = manifest.get("files", {})
    on_disk = _listed_files(results_dir)
    problems = [f"missing: {f}" for f in sorted(set(files) - on_disk)]
    problems += [f"not in the manifest: {f}" for f in sorted(on_disk - set(files))]
    problems += [
        f"changed: {f}"
        for f in sorted(set(files) & on_disk)
        if sha256_file(results_dir / f) != files[f]
    ]
    if problems:
        raise ResultsError(f"{path}:\n  " + "\n  ".join(problems))
    return manifest


def read_campaign(campaign_dir: Path) -> pd.DataFrame:
    """A campaign's recorded units, if its gate passed and they equal its grid."""
    gate = json.loads((campaign_dir / "gate.json").read_text())
    if not gate.get("passed"):
        raise ResultsError(f"{campaign_dir}: gate.json does not record a passed gate")
    expected = read_expected(campaign_dir / "expected_units.csv")
    recorded = pd.read_csv(campaign_dir / "evaluation.csv")
    rec = reconcile(expected, recorded)
    if not rec.passed:
        raise ResultsError(
            f"{campaign_dir}: recorded units differ from expected_units.csv "
            f"({rec.n_missing} missing, {rec.n_duplicated} duplicated, "
            f"{rec.n_extra} extra)"
        )
    return recorded


# Table 1


def describe_log(events: pd.DataFrame) -> dict[str, float]:
    """Cases, events, activities, trace variants and median case duration in days.

    ``events`` is a log as :func:`~pmf_concurrency.data.load_event_log` returns it,
    ordered by case and timestamp with ties in file order, so a variant is the
    activity sequence of a case in that order.
    """
    by_case = events.groupby(CASE_KEY, sort=False)
    span = by_case[TIMESTAMP_KEY].max() - by_case[TIMESTAMP_KEY].min()
    return {
        "cases": int(events[CASE_KEY].nunique()),
        "events": len(events),
        "activities": int(events[ACTIVITY_KEY].nunique()),
        "variants": int(by_case[ACTIVITY_KEY].agg(tuple).nunique()),
        "median_case_days": float((span.dt.total_seconds() / 86400.0).median()),
    }


def window_composition(events: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Per window, its active cases and how each overlaps the window.

    A case is active if it has an event in the window, whose first and last instants
    are inclusive as in the evaluation. Its first and last events in the whole log
    decide its class: ``complete`` (both inside), ``prefix_truncated`` (starts before),
    ``suffix_truncated`` (ends after) or ``middle_crossing`` (both outside).
    """
    log = normalise_timestamp(events[[CASE_KEY, TIMESTAMP_KEY]])
    ts = log[TIMESTAMP_KEY]
    first = ts.groupby(log[CASE_KEY]).min()
    last = ts.groupby(log[CASE_KEY]).max()

    rows = []
    for window, start_date, end_date in windows[
        ["window", "start_date", "end_date"]
    ].itertuples(index=False):
        start = parse_window_date(str(start_date))
        end = (
            parse_window_date(str(end_date))
            + pd.Timedelta(days=1)
            - pd.Timedelta(nanoseconds=1)
        )
        active = log.loc[(ts >= start) & (ts <= end), CASE_KEY].unique()
        before = first.loc[active] < start
        after = last.loc[active] > end
        rows.append(
            {
                "window": int(window),
                "n_cases": len(active),
                "n_complete": int((~before & ~after).sum()),
                "n_prefix_truncated": int((before & ~after).sum()),
                "n_suffix_truncated": int((~before & after).sum()),
                "n_middle_crossing": int((before & after).sum()),
            }
        )
    return pd.DataFrame(rows)


# Table 2 and Section 5


@dataclass(frozen=True)
class Pooled:
    """One metric over the units of a cell that finished with a finite value."""

    n_units: int
    n_values: int
    mean: float
    sd: float


def pool(units: pd.DataFrame, metric: str) -> Pooled:
    """Mean and sample SD of ``metric`` over the ``ok`` units with a finite value.

    ``n_values`` is that metric's coverage out of ``n_units``. An empty model's
    fitness is a measured zero and counts; its precision is NaN and does not.
    """
    ok = units[units["guard_status"] == "ok"]
    values = pd.to_numeric(ok[metric], errors="coerce").dropna()
    return Pooled(
        n_units=len(units),
        n_values=len(values),
        mean=float(values.mean()) if len(values) else float("nan"),
        sd=float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
    )


def over_budget_share(units: pd.DataFrame) -> float:
    """Share of the units that exceeded their time or memory budget."""
    return float(units["guard_status"].isin(OVER_BUDGET).mean())


@dataclass(frozen=True)
class Invariance:
    """One measure on the units that finished at both levels."""

    n: int
    max_abs_diff: float
    share_equal: float


@dataclass(frozen=True)
class ReductionEffect:
    """What τ-reduction changed, unit by unit, over one grid."""

    n_units: int
    n_over_budget_unreduced: int
    n_rescued: int
    n_lost: int
    time_saved_all: float
    time_saved_both_ok: float
    invariance: dict[tuple[str, str], Invariance]

    @property
    def share_rescued(self) -> float:
        return self.n_rescued / self.n_over_budget_unreduced


#: (evaluator, column) compared between the levels.
INVARIANCE_MEASURES = (
    ("alignment", "log_fitness"),
    ("alignment", "precision"),
    ("er", "er"),
)


def reduction_effect(unreduced: pd.DataFrame, reduced: pd.DataFrame) -> ReductionEffect:
    """Compare every unit of a grid on the unreduced and the τ-reduced nets.

    The two frames must hold the same units apart from ``reduce_level``. A unit is
    rescued when it exceeded the budget unreduced and finished ``ok`` reduced, and lost
    in the opposite case. Time saved is one minus the ratio of summed wall times, over
    all units (a stopped unit counts the time it ran) and over the units ``ok`` at both
    levels.
    """
    key = [k for k in UNIT_KEY if k != "reduce_level"]
    merged = unreduced.merge(
        reduced, on=key, suffixes=("_none", "_tau"), validate="one_to_one"
    )
    if not len(merged) == len(unreduced) == len(reduced):
        raise ResultsError(
            f"the levels hold different units: {len(unreduced)} unreduced, "
            f"{len(reduced)} reduced, {len(merged)} in both"
        )
    status_none, status_tau = merged["guard_status_none"], merged["guard_status_tau"]
    over = status_none.isin(OVER_BUDGET)
    both_ok = (status_none == "ok") & (status_tau == "ok")

    def saved(rows: pd.Series) -> float:
        return float(
            1
            - merged.loc[rows, "wall_s_tau"].sum()
            / merged.loc[rows, "wall_s_none"].sum()
        )

    invariance = {}
    for evaluator, column in INVARIANCE_MEASURES:
        pair = merged.loc[
            both_ok & (merged["evaluator"] == evaluator),
            [f"{column}_none", f"{column}_tau"],
        ].dropna()
        diff = (pair[f"{column}_none"] - pair[f"{column}_tau"]).abs()
        invariance[(evaluator, column)] = Invariance(
            n=len(pair),
            max_abs_diff=float(diff.max()) if len(diff) else float("nan"),
            share_equal=float((diff == 0).mean()) if len(diff) else float("nan"),
        )
    return ReductionEffect(
        n_units=len(merged),
        n_over_budget_unreduced=int(over.sum()),
        n_rescued=int((over & (status_tau == "ok")).sum()),
        n_lost=int(((status_none == "ok") & (status_tau != "ok")).sum()),
        time_saved_all=saved(pd.Series(True, index=merged.index)),
        time_saved_both_ok=saved(both_ok),
        invariance=invariance,
    )


def replay_mode_ranges(units: pd.DataFrame) -> pd.DataFrame:
    """Per (evaluator, mode), the range over arms of the arms' mean fitness and
    perfect-fit share.

    An arm is one (tPAT, family); its means pool its windows. ``n_not_ok`` counts the
    units that did not finish, which the means leave out.
    """
    rows = []
    for (evaluator, mode), cell in units.groupby(["evaluator", "mode"], sort=True):
        arms = cell.groupby(["tpat", "model_type"])
        fit = arms.apply(lambda g: pool(g, "log_fitness").mean, include_groups=False)
        perf = arms.apply(
            lambda g: pool(g, "perc_fit_traces").mean, include_groups=False
        )
        rows.append(
            {
                "evaluator": evaluator,
                "mode": mode,
                "n_arms": arms.ngroups,
                "n_not_ok": int((cell["guard_status"] != "ok").sum()),
                "fit_min": float(np.min(fit)),
                "fit_max": float(np.max(fit)),
                "perf_min": float(np.min(perf)),
                "perf_max": float(np.max(perf)),
            }
        )
    return pd.DataFrame(rows)
