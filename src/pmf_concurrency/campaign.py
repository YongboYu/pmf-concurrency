"""The units an evaluation campaign must record, and the checks of its gate.

The expected units come from the protocol: every log's calendar windows, the Table 2
arms (all four families at tPAT 0, Forecast and Re-mined at tPAT -1) or the history
arms (Persistence and Expanding at tPAT 0), and the Table 2 evaluators. They never come
from the files on disk, so a model the pipeline failed to write, or a unit the
evaluation did not record, shows up as missing.

    windows_of       the windows of one log, from its timestamps
    expected_units   the grid of campaign 1 (reduced nets), 2 (unreduced, replay modes),
                     3 (history families, reduced nets) or 4 (history families,
                     unreduced nets)
    check_models     every model a unit needs exists, parses and has the grid's dates
    reconcile        recorded units against the grid: missing, duplicated, extra
    near_budget      finished units above 75 % of the time or memory budget
    oom_attribution  every OOM kill on the machine matched by a unit's own
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data import split_train_test
from .families import HISTORY_FAMILIES, MODEL_FAMILIES, model_json_path

#: Families evaluated at each tPAT: the mined families do not use the forecasts, so
#: Table 2 reports them at Fodina's default only.
TABLE2_ARMS: dict[float, tuple[str, ...]] = {
    0.0: MODEL_FAMILIES,
    -1.0: ("forecast", "re_mined"),
}

#: Families of campaigns 3 and 4: the history families are mined, so at Fodina's
#: default only.
HISTORY_ARMS: dict[float, tuple[str, ...]] = {0.0: HISTORY_FAMILIES}

#: Evaluators of Table 2.
TABLE2_EVALUATORS = ("token_etc", "alignment", "er")

#: Replay mode of the Table 2 token and alignment units.
TABLE2_MODE = "case_complete"

#: ``mode`` of an ER unit, which always scores the window's fragments.
ER_MODE = "window"

#: Campaign 2's extra BPI2017 rows: evaluator and replay mode, at both levels.
REPLAY_MODE_UNITS = (
    ("token_etc", "window_raw"),
    ("token_etc", "window_no_remain"),
    ("alignment", "window_raw"),
)
REPLAY_MODE_DATASET = "bpi2017"

#: Identifies one unit, as ``scripts/evaluate.py`` keys its rows.
UNIT_KEY = (
    "dataset",
    "tpat",
    "window",
    "model_type",
    "mode",
    "evaluator",
    "reduce_level",
)

#: Columns of ``expected_units.csv``. ``cohort`` keeps the replay-mode and history rows
#: out of the pooled rates of the base grid.
EXPECTED_COLUMNS = (*UNIT_KEY, "start_date", "end_date", "cohort")

#: Share of a budget above which a finished unit counts as near it.
NEAR_BUDGET = 0.75

#: Statuses of a unit that ran to the end.
FINISHED = ("ok", "unavailable")


def windows_of(
    events: pd.DataFrame,
    train_ratio: float = 0.8,
    horizon: int = 7,
    timestamp_key: str = "time:timestamp",
) -> pd.DataFrame:
    """The evaluation windows of a log: ``window``, ``start_date``, ``end_date``.

    The test period runs from the day after the cutoff day to the last event's day, on
    the calendar, with or without events. Windows are ``horizon`` days long and start
    on every test day that leaves room for a whole window, as the pipeline builds them.
    Dates are ``YYYYMMDD``, as in the models' metadata.
    """
    _, _, _, train_end, test_end = split_train_test(events, train_ratio, timestamp_key)
    days = pd.date_range(
        pd.Timestamp(train_end) + pd.Timedelta(days=1), pd.Timestamp(test_end)
    )
    starts = days[: max(len(days) - horizon + 1, 0)]
    return pd.DataFrame(
        {
            "window": range(1, len(starts) + 1),
            "start_date": starts.strftime("%Y%m%d"),
            "end_date": (starts + pd.Timedelta(days=horizon - 1)).strftime("%Y%m%d"),
        }
    )


def _rows(
    dataset: str,
    windows: pd.DataFrame,
    arms: dict[float, tuple[str, ...]],
    units: list[tuple[str, str]],
    levels: tuple[str, ...],
    cohort: str,
) -> list[dict]:
    return [
        {
            "dataset": dataset,
            "tpat": tpat,
            "window": int(w.window),
            "model_type": family,
            "mode": mode,
            "evaluator": evaluator,
            "reduce_level": level,
            "start_date": w.start_date,
            "end_date": w.end_date,
            "cohort": cohort,
        }
        for tpat, families in arms.items()
        for family in families
        for w in windows.itertuples()
        for evaluator, mode in units
        for level in levels
    ]


def expected_units(windows: dict[str, pd.DataFrame], campaign: int) -> pd.DataFrame:
    """The units a campaign must record, one row each, :data:`EXPECTED_COLUMNS`.

    ``windows`` maps each log's dataset key to :func:`windows_of`. Campaign 1 is the
    Table 2 grid on the τ-reduced nets. Campaign 2 is the same grid on the unreduced
    nets (cohort ``table2``) plus the BPI2017 replay-mode comparison (cohort
    ``replay_modes``) at both levels; its ``case_complete`` rows come from the two
    base grids and are not repeated. Campaign 3 is the Table 2 evaluators on the
    history families' τ-reduced nets (cohort ``history``), and campaign 4 the same
    units on their unreduced nets.
    """
    table2 = [(e, ER_MODE if e == "er" else TABLE2_MODE) for e in TABLE2_EVALUATORS]
    rows: list[dict] = []
    if campaign == 1:
        for dataset, w in windows.items():
            rows += _rows(dataset, w, TABLE2_ARMS, table2, ("tau",), "table2")
    elif campaign == 2:
        for dataset, w in windows.items():
            rows += _rows(dataset, w, TABLE2_ARMS, table2, ("none",), "table2")
        if REPLAY_MODE_DATASET in windows:
            rows += _rows(
                REPLAY_MODE_DATASET,
                windows[REPLAY_MODE_DATASET],
                TABLE2_ARMS,
                list(REPLAY_MODE_UNITS),
                ("tau", "none"),
                "replay_modes",
            )
    elif campaign in (3, 4):
        level = "tau" if campaign == 3 else "none"
        for dataset, w in windows.items():
            rows += _rows(dataset, w, HISTORY_ARMS, table2, (level,), "history")
    else:
        raise ValueError(f"campaign must be 1, 2, 3 or 4, not {campaign}")
    return pd.DataFrame(rows, columns=list(EXPECTED_COLUMNS))


def read_expected(path: str | Path) -> pd.DataFrame:
    """Read an ``expected_units.csv``, dates as text."""
    return pd.read_csv(path, dtype={"start_date": str, "end_date": str})


def run_tpat(weekly_dir: Path) -> float | None:
    """tPAT of the pipeline run that wrote ``weekly_dir``, or None if unrecorded."""
    try:
        summary = json.loads(
            (weekly_dir.parent / "experiment_summary.json").read_text()
        )
        return float(summary["thresholds"]["tPAT"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def check_models(
    expected: pd.DataFrame, weekly_dirs: dict[tuple[str, float], Path]
) -> list[str]:
    """Problems with the models the expected units need; empty if there are none.

    ``weekly_dirs`` maps (dataset, tPAT) to that run's ``weekly_models``. Every model
    must exist, parse as JSON and carry the grid's window dates, and the run's
    ``experiment_summary.json`` must record the same tPAT.
    """
    problems: list[str] = []
    models = expected[
        ["dataset", "tpat", "window", "model_type", "start_date", "end_date"]
    ].drop_duplicates()
    for (dataset, tpat), group in models.groupby(["dataset", "tpat"], sort=True):
        weekly = weekly_dirs.get((dataset, float(tpat)))
        if weekly is None:
            problems.append(f"{dataset} tPAT {float(tpat)}: no model directory given")
            continue
        recorded = run_tpat(weekly)
        if recorded != float(tpat):
            problems.append(
                f"{weekly}: the run records tPAT {recorded}, the grid expects "
                f"{float(tpat)}"
            )
        for m in group.itertuples():
            path = model_json_path(weekly, int(m.window), m.model_type)
            try:
                meta = json.loads(path.read_text())["metadata"]
            except FileNotFoundError:
                problems.append(f"{path}: missing")
                continue
            except (ValueError, KeyError, TypeError) as exc:
                problems.append(f"{path}: does not parse ({exc!r})")
                continue
            dates = (str(meta.get("start_date")), str(meta.get("end_date")))
            if dates != (m.start_date, m.end_date):
                problems.append(
                    f"{path}: window {dates[0]}-{dates[1]}, the grid expects "
                    f"{m.start_date}-{m.end_date}"
                )
    return problems


def _keys(frame: pd.DataFrame) -> pd.DataFrame:
    keys = frame[list(UNIT_KEY)].copy()
    keys["tpat"] = keys["tpat"].astype(float)
    keys["window"] = keys["window"].astype(int)
    return keys


@dataclass(frozen=True)
class Reconciliation:
    """Recorded units against the expected ones."""

    missing: pd.DataFrame
    duplicated: pd.DataFrame
    extra: pd.DataFrame

    @property
    def n_missing(self) -> int:
        return len(self.missing)

    @property
    def n_duplicated(self) -> int:
        return len(self.duplicated)

    @property
    def n_extra(self) -> int:
        return len(self.extra)

    @property
    def passed(self) -> bool:
        return self.n_missing == self.n_duplicated == self.n_extra == 0


def reconcile(expected: pd.DataFrame, recorded: pd.DataFrame) -> Reconciliation:
    """Expected units not recorded, units recorded more than once (one row per extra
    copy), and recorded units the grid does not expect.
    """
    want = _keys(expected).drop_duplicates()
    got = _keys(recorded)
    merged = got.drop_duplicates().merge(want, how="outer", indicator=True)
    return Reconciliation(
        missing=merged[merged["_merge"] == "right_only"].drop(columns="_merge"),
        duplicated=got[got.duplicated()],
        extra=merged[merged["_merge"] == "left_only"].drop(columns="_merge"),
    )


def near_budget(
    recorded: pd.DataFrame,
    mem_gb: float,
    time_min: float,
    share: float = NEAR_BUDGET,
    by: tuple[str, ...] = ("dataset",),
) -> pd.DataFrame:
    """Finished units above ``share`` of the time budget or of the memory cap.

    Memory is the unit cgroup's ``memory.peak``, the quantity the cap limits.
    """
    finished = recorded[recorded["guard_status"].isin(FINISHED)]
    over_time = (
        pd.to_numeric(finished["wall_s"], errors="coerce") > share * time_min * 60
    )
    peak = pd.to_numeric(finished["memory_peak_gb"], errors="coerce")
    over_memory = peak > share * mem_gb
    flags = pd.DataFrame(
        {
            "n_finished": 1,
            "n_over_75pct_time": over_time,
            "n_over_75pct_memory": over_memory,
            "n_near_budget": over_time | over_memory,
        },
        index=finished.index,
    ).astype(int)
    return flags.groupby([finished[k] for k in by]).sum().reset_index()


@dataclass(frozen=True)
class OomAttribution:
    """The machine's OOM kills against the units' own."""

    global_rise: int
    unit_kills: int
    n_host_oom: int

    @property
    def unattributed(self) -> int:
        return max(self.global_rise - self.unit_kills, 0)

    @property
    def passed(self) -> bool:
        return self.n_host_oom == 0 and self.unattributed == 0


def oom_attribution(global_rise: int, recorded: pd.DataFrame) -> OomAttribution:
    """Match the rise of ``oom_kill`` in ``/proc/vmstat`` during the campaign against
    the kills in the units' own cgroups. A rise left over is a host OOM: something on
    the machine was killed that no unit's limit explains.
    """
    kills = pd.to_numeric(recorded["memcg_oom_kill"], errors="coerce").fillna(0)
    return OomAttribution(
        global_rise=int(global_rise),
        unit_kills=int(kills.sum()),
        n_host_oom=int((recorded["guard_status"] == "host_oom").sum()),
    )
