"""Structural statistics per model family, and the CSV that stores them.

The key rule: the arity of a binding is the size of its set, never a node degree. In the
persisted graph an input binding node is linked both to its source activity and to the
``o_`` node pointing at it, so a singleton has in-degree 2 and a degree count would
inflate the AND share about 3×. The measurement therefore takes binding sets, never a
graph.

:func:`stats_from_bindings` is the one definition. The reconstruction passes its
filtered IB/OB before they become graph nodes; the Fodina families pass Fodina's IB/OB
through :func:`stats_from_fodina`.

Two values are deliberately missing rather than zero:

* ``and_share_pct`` of a net with no bindings is NaN, since 0% would claim a
  pure XOR net.
* The process counters of the Fodina families are ``None``: Fodina's own
  repair and filter are not instrumented.

``singleton_fallback_fires`` is 0 by construction, because the binding filter's
singleton re-add leaves no side empty for the fallback. It is recorded as a check that
the re-add still covers every neighbour.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .families import MODEL_FAMILIES, check_families
from .fodina import PatternMiner

#: One file per campaign output directory.
STRUCTURAL_METRICS_FILENAME = "structural_metrics.csv"

#: Identifies a row. ``dataset`` and ``tPAT`` are constant within a file but kept, since
#: the tPAT setting is not part of the directory name.
STRUCTURAL_METRICS_KEY = ("dataset", "tPAT", "window", "model_family")

#: Column order of :data:`STRUCTURAL_METRICS_FILENAME`.
STRUCTURAL_METRICS_COLUMNS = (
    *STRUCTURAL_METRICS_KEY,
    "tasks",
    "arcs",
    "bindings",
    "and_bindings",
    "and_share_pct",
    "input_bindings",
    "input_and_bindings",
    "output_bindings",
    "output_and_bindings",
    "repair_arcs_added",
    "filter_bindings_removed",
    "singleton_readds",
    "singleton_readds_orphaned",
    "singleton_readds_uncovered",
    "singleton_fallback_fires",
)

#: Nullable integers, so one missing value does not turn a column into floats.
_COUNT_COLUMNS = tuple(
    c
    for c in STRUCTURAL_METRICS_COLUMNS
    if c not in ("dataset", "model_family", "tPAT", "and_share_pct")
)


@dataclass(frozen=True)
class StructuralStats:
    """Structural measurements of one causal net.

    A binding with arity ≥ 2 is an AND binding. The four process counters exist only on
    the reconstruction path; ``None`` means not measured, not zero.
    """

    tasks: int
    arcs: int
    input_bindings: int
    input_and_bindings: int
    output_bindings: int
    output_and_bindings: int
    repair_arcs_added: int | None = None
    filter_bindings_removed: int | None = None
    singleton_readds_orphaned: int | None = None
    singleton_readds_uncovered: int | None = None
    singleton_fallback_fires: int | None = None

    @property
    def singleton_readds(self) -> int | None:
        """Every neighbour restored as a singleton, for either reason.

        Only ``singleton_readds_orphaned`` is a filter effect (see
        :class:`reconstruction.FilterOutcome`).
        """
        if (
            self.singleton_readds_orphaned is None
            and self.singleton_readds_uncovered is None
        ):
            return None
        return (self.singleton_readds_orphaned or 0) + (
            self.singleton_readds_uncovered or 0
        )

    @property
    def bindings(self) -> int:
        return self.input_bindings + self.output_bindings

    @property
    def and_bindings(self) -> int:
        """Number of AND bindings, reported next to the share."""
        return self.input_and_bindings + self.output_and_bindings

    @property
    def and_share_pct(self) -> float:
        """``100 × |{b : arity(b) ≥ 2}| / |bindings|``, or NaN without bindings."""
        if self.bindings == 0:
            return math.nan
        return 100.0 * self.and_bindings / self.bindings


def _count(bindings: Mapping[Any, Iterable[frozenset[Any]]]) -> tuple[int, int]:
    """``(total, arity ≥ 2)`` over the bindings of an IB or OB mapping.

    Works on activity names and on Fodina's integer task indices alike.
    """
    total = 0
    and_count = 0
    for task_bindings in bindings.values():
        for binding in task_bindings:
            total += 1
            if len(binding) >= 2:
                and_count += 1
    return total, and_count


def stats_from_bindings(
    *,
    tasks: int,
    arcs: int,
    IB: Mapping[Any, Iterable[frozenset[Any]]],
    OB: Mapping[Any, Iterable[frozenset[Any]]],
    repair_arcs_added: int | None = None,
    filter_bindings_removed: int | None = None,
    singleton_readds_orphaned: int | None = None,
    singleton_readds_uncovered: int | None = None,
    singleton_fallback_fires: int | None = None,
) -> StructuralStats:
    """Measure a causal net from its bindings.

    ``IB``/``OB`` map each task to a set of frozensets, or to a ``{frozenset: count}``
    dict whose keys are used.
    """
    input_total, input_and = _count(IB)
    output_total, output_and = _count(OB)
    return StructuralStats(
        tasks=tasks,
        arcs=arcs,
        input_bindings=input_total,
        input_and_bindings=input_and,
        output_bindings=output_total,
        output_and_bindings=output_and,
        repair_arcs_added=repair_arcs_added,
        filter_bindings_removed=filter_bindings_removed,
        singleton_readds_orphaned=singleton_readds_orphaned,
        singleton_readds_uncovered=singleton_readds_uncovered,
        singleton_fallback_fires=singleton_fallback_fires,
    )


def stats_from_fodina(pm: PatternMiner) -> StructuralStats:
    """Measure a Direct-mined or Static net from Fodina's own IB/OB.

    The process counters stay unmeasured. ``tasks`` counts the endpoints of dependency-
    graph arcs, as the reconstruction does, so task counts compare across families.
    """
    dg = pm.dg
    return stats_from_bindings(
        tasks=len({a for a, _ in dg.DG} | {b for _, b in dg.DG}),
        arcs=len(dg.DG),
        IB=pm.IB,
        OB=pm.OB,
    )


def structural_metrics_row(
    window: int,
    model_family: str,
    stats: StructuralStats,
    *,
    dataset: str,
    tPAT: float,
) -> dict[str, Any]:
    """One record in the :data:`STRUCTURAL_METRICS_COLUMNS` layout.

    ``dataset`` and ``tPAT`` are required, so each row says which campaign it comes
    from.
    """
    return {
        "dataset": dataset,
        "tPAT": tPAT,
        "window": window,
        "model_family": model_family,
        **{
            column: getattr(stats, column)
            for column in STRUCTURAL_METRICS_COLUMNS[len(STRUCTURAL_METRICS_KEY) :]
        },
    }


def _with_declared_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.reindex(columns=list(STRUCTURAL_METRICS_COLUMNS))
    for column in _COUNT_COLUMNS:
        out[column] = out[column].astype("Int64")
    out["and_share_pct"] = out["and_share_pct"].astype("Float64")
    out["tPAT"] = out["tPAT"].astype("Float64")
    out["dataset"] = out["dataset"].astype("string")
    # Ordered categorical, so families sort in paper order, not alphabetically.
    out["model_family"] = pd.Categorical(
        out["model_family"], categories=list(MODEL_FAMILIES), ordered=True
    )
    return out


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.sort_values(
        list(STRUCTURAL_METRICS_KEY), kind="stable", ignore_index=True
    )


def structural_metrics_frame(
    rows: Iterable[dict[str, Any]] | Sequence[dict[str, Any]],
) -> pd.DataFrame:
    """Build the campaign table from :func:`structural_metrics_row` records.

    Raises if two rows share a :data:`STRUCTURAL_METRICS_KEY`.
    """
    records = list(rows)
    if not records:
        return _with_declared_dtypes(
            pd.DataFrame(columns=list(STRUCTURAL_METRICS_COLUMNS))
        )
    frame = _sorted(_with_declared_dtypes(pd.DataFrame(records)))
    duplicated = frame.duplicated(subset=list(STRUCTURAL_METRICS_KEY), keep=False)
    if duplicated.any():
        clashes = (
            frame.loc[duplicated, list(STRUCTURAL_METRICS_KEY)]
            .astype(str)
            .agg(" / ".join, axis=1)
            .unique()
        )
        raise ValueError(
            "structural metrics must be unique per "
            f"{' / '.join(STRUCTURAL_METRICS_KEY)}; duplicated: "
            f"{', '.join(clashes)}"
        )
    return frame


def write_structural_metrics(
    output_dir: str,
    rows: Iterable[dict[str, Any]] | Sequence[dict[str, Any]],
) -> str:
    """Write the campaign's :data:`STRUCTURAL_METRICS_FILENAME` and return its path.

    A campaign without models still writes the header.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, STRUCTURAL_METRICS_FILENAME)
    structural_metrics_frame(rows).to_csv(path, index=False)
    return path


def read_structural_metrics(path: str) -> pd.DataFrame:
    """Read a structural-metrics CSV and check its columns.

    Restores the nullable integer dtypes, so unmeasured counters stay missing, and
    raises on family names other than the four.
    """
    raw = pd.read_csv(path)
    missing = [c for c in STRUCTURAL_METRICS_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(
            f"{path} is not a structural-metrics file: missing columns "
            f"{', '.join(missing)}"
        )
    check_families(raw["model_family"], path)
    return _sorted(_with_declared_dtypes(raw))
