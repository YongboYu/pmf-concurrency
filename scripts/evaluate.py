"""Evaluate the weekly models against the test log, one unit at a time.

A unit is one (window, family, reduce_level, evaluator):

    token_etc        token replay fitness, ETC precision, prefix coverage
    token_align_etc  token replay fitness, Align-ETC precision
    alignment        alignment fitness, Align-ETC precision
    er               entropic relevance with partial-trace wildcards

ETC precision discards non-fitting prefixes and Align-ETC aligns them, so the share of
prefix mass ETC discards is recorded too. ``tau`` evaluates the net after
language-preserving τ-reduction, and reduced units record the net size before and after.
ER always scores the events inside the window, so its rows have mode ``window`` and
count the fragments it scored, whatever ``--mode`` says.

Every unit runs under the budget guard (12 GiB and 20 min by default, see
:mod:`pmf_concurrency.guard`) and is appended to the CSV as soon as it finishes, so a
stopped run resumes where it left off. Each row records the unit's status, why it has no
number if it has none (``failure_category``, ``guard_reason``), and what it used. A run
needs the per-unit cgroup (start it with ``scripts/local/delegated_scope.sh``);
``--development`` allows the weaker fallbacks and the manifest records which was used.

Units are keyed by tPAT too, read from the pipeline run's ``experiment_summary.json``.
A campaign runs exactly the units of its ``expected_units.csv`` (``--expected``, see
``scripts/evaluation_campaign.py``) and stops before the first unit if a model they
need is missing, does not parse or has other window dates.

A resume must use the same log, models, grid, budgets, code, environment, machine,
concurrency and budget enforcement, which ``evaluation_manifest.json`` records. OMP and MKL
default to 8 threads, because the thread count can change floating-point results.

Usage, from the project root:
    bash scripts/local/delegated_scope.sh uv run python scripts/evaluate.py \
        --dataset sepsis --expected expected_units.csv --windows 6 7 8 --concurrency 3
    uv run python scripts/evaluate.py --development --dataset bpi2017 \
        --reduce-levels tau --evaluators token_etc alignment er

Outputs, under $PMF_OUTPUT_BASE/<Dataset>_pmf/ unless --out-dir:
    evaluation.csv            one row per unit
    evaluation_summary.csv    by (mode, evaluator, level, family): each metric's mean,
                              std and number of finite values (n_<metric>) out of
                              n_expected, and one count per status
    evaluation_failures.csv   units per failure_category
    evaluation_manifest.json  the inputs and settings every unit shares
"""

from __future__ import annotations

import os

# Before numpy is imported, which fixes its thread pools.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import argparse
import hashlib
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import get_args

import numpy as np
import pandas as pd
from _dataset import DATASETS, resolve_dataset

from pmf_concurrency.campaign import (
    ER_MODE,
    UNIT_KEY,
    check_models,
    read_expected,
    run_tpat,
)
from pmf_concurrency.conformance import (
    METRIC_KEYS,
    EvaluationMode,
    conformance_metrics,
    extract_window_traces,
    filter_window_events,
    full_traces_of_active_cases,
    normalise_timestamp,
    parse_window_date,
)
from pmf_concurrency.data import load_event_log
from pmf_concurrency.entropic_relevance import er_metrics
from pmf_concurrency.families import MODEL_FAMILIES, check_families, model_json_path
from pmf_concurrency.guard import (
    GUARD_STATUSES,
    OUTCOME_COLUMNS,
    OVER_BUDGET,
    GuardSetupError,
    enforcement,
    resolve_budgets,
    run_guarded,
)
from pmf_concurrency.petri_net import accepts_nothing
from pmf_concurrency.provenance import code_fingerprint, file_sha256, machine, versions
from pmf_concurrency.reduction import REDUCTION_LEVELS
from pmf_concurrency.reproducibility import enable_determinism

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVALUATORS = ("token_etc", "token_align_etc", "alignment", "er")

MANIFEST_FILENAME = "evaluation_manifest.json"

#: ``guard_reason`` of an ``ok`` unit whose model accepts no trace.
EMPTY_MODEL = "empty model"

# Evaluators return different metrics, so every row is reindexed to this schema before
# it is appended.
CSV_COLUMNS = [
    "dataset",
    "tpat",
    "window",
    "model_type",
    "mode",
    "evaluator",
    "reduce_level",
    "n_events",
    "n_cases",
    *OUTCOME_COLUMNS,
    "wall_s",
    *METRIC_KEYS,
    "pct_events_unseen",
    "fitness_wall_s",
    "precision_wall_s",
    "prefix_variants",
    "prefix_mass_total",
    "prefix_mass_fit",
    "prefix_mass_unfit",
    "pct_prefix_mass_fit",
    "er",
    "fitting_ratio",
    "reduce_wall_s",
    "places_before",
    "places_after",
    "transitions_before",
    "transitions_after",
    "silent_transitions_before",
    "silent_transitions_after",
    "visible_transitions_before",
    "visible_transitions_after",
    "arcs_before",
    "arcs_after",
]


def read_units(path: str | Path) -> pd.DataFrame:
    """Read an ``evaluation.csv``.

    Raises if its columns are not :data:`CSV_COLUMNS`, since new rows would not line
    up with them, on family names other than the four, and if two rows describe the
    same unit, which pooling would count twice.
    """
    df = pd.read_csv(path)
    if list(df.columns) != CSV_COLUMNS:
        changed = sorted(set(df.columns) ^ set(CSV_COLUMNS)) or ["the column order"]
        raise ValueError(
            f"{path}: columns differ from this script's ({', '.join(changed)}). "
            "Evaluate into a new --out-dir."
        )
    check_families(df["model_type"], str(path))
    duplicated = df.duplicated(subset=list(UNIT_KEY), keep=False)
    if duplicated.any():
        units = (
            df.loc[duplicated, list(UNIT_KEY)]
            .astype(str)
            .agg(" / ".join, axis=1)
            .unique()
        )
        raise ValueError(
            f"{path}: duplicate units: {', '.join(units)}. Keep one row per unit."
        )
    return df


def _sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def evaluation_manifest(
    log_path: str,
    weekly_dir: Path,
    mem_gb: float | None,
    time_min: float | None,
    concurrency: int = 1,
    tpat: float | None = None,
    expected: Path | None = None,
) -> dict:
    """What every unit in one output directory must share.

    The inputs (log, models, their tPAT, the expected grid if any) and budgets; the
    code (commit, and a hash of the uncommitted changes),
    ``uv.lock``, Python and pm4py; the machine; how many units of the campaign run at
    once; and the guard: ``PMF_GUARD``, the enforcement mechanism and whether it is in
    force. The mode, windows, families, levels and evaluators are part of each unit's
    key, so one directory can collect several such runs.
    """
    mem_gb, time_min = resolve_budgets(mem_gb, time_min)
    enforced = enforcement()
    return {
        "log_sha256": _sha256([Path(log_path)]),
        "models_sha256": _sha256(sorted(weekly_dir.glob("window_*.json"))),
        "tpat": tpat,
        "expected_sha256": _sha256([expected]) if expected is not None else None,
        "mem_gb": mem_gb,
        "time_min": time_min,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        **code_fingerprint(PROJECT_ROOT),
        "uv_lock_sha256": file_sha256(PROJECT_ROOT / "uv.lock"),
        **versions(),
        **machine(),
        "concurrency": concurrency,
        "pmf_guard": os.environ.get("PMF_GUARD", "1"),
        "enforcement": enforced.mechanism,
        "enforcement_in_force": enforced.in_force,
    }


def check_manifest(out_dir: Path, manifest: dict) -> None:
    """Write the manifest on a fresh run; on a resume, stop if it has changed."""
    path = out_dir / MANIFEST_FILENAME
    if path.exists():
        recorded = json.loads(path.read_text())
        changed = sorted(
            k
            for k in manifest.keys() | recorded.keys()
            if manifest.get(k) != recorded.get(k)
        )
        if changed:
            raise SystemExit(
                f"{path}: {', '.join(changed)} changed since this run started. "
                "Evaluate into a new --out-dir."
            )
    elif (out_dir / "evaluation.csv").exists():
        raise SystemExit(
            f"{out_dir / 'evaluation.csv'} has no {MANIFEST_FILENAME}, so a resume "
            "cannot check its inputs. Evaluate into a new --out-dir."
        )
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")


def window_bounds(metadata: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last instant of a model's window, both inclusive."""
    start = parse_window_date(str(metadata["start_date"]))
    last_day = parse_window_date(str(metadata["end_date"]))
    return start, last_day + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)


def _run_unit(
    fn: Callable[..., dict],
    args: tuple,
    kwargs: dict,
    evaluator: str,
    model_json: dict,
    mem_gb: float | None,
    time_min: float | None,
) -> tuple[str, dict]:
    """Run ``fn`` under the guard; return (status, row fields).

    A finished unit is ``ok``, with ``failure_category`` ``empty_model`` if its model
    accepts nothing, or ``unavailable`` if its main metric (F1, or ER) is missing, with
    the evaluator's reason.
    """
    outcome = run_guarded(fn, *args, mem_gb=mem_gb, time_min=time_min, **kwargs)
    fields = outcome.row()
    status = fields.pop("guard_status")
    if status != "ok":
        return status, fields

    metrics = dict(outcome.result)
    category = metrics.pop("failure_category", None)
    reason = metrics.pop("guard_reason", "")
    if accepts_nothing(model_json):
        category, reason = "empty_model", EMPTY_MODEL
    elif math.isnan(metrics.get("er" if evaluator == "er" else "f1", math.nan)):
        status = "unavailable"
        category = category or (
            "er_automaton_failed" if evaluator == "er" else "precision_failed"
        )
    else:
        category, reason = "", ""
    return status, {
        **fields,
        **metrics,
        "failure_category": category,
        "guard_reason": reason,
    }


def _evaluate_cell(
    evaluator: str,
    model_json: dict,
    event_subset: pd.DataFrame,
    traces: list,
    consider_remaining: bool,
    reduce_level: str,
    mem_gb: float | None,
    time_min: float | None,
) -> tuple[str, dict]:
    """Evaluate one unit; return (status, row fields).

    A window without events is ``unavailable`` (``empty_window``) without running.
    """
    has_events = bool(traces) if evaluator == "er" else not event_subset.empty
    if not has_events:
        return "unavailable", {
            "failure_category": "empty_window",
            "guard_reason": "no events in the window",
        }
    if evaluator == "er":
        return _run_unit(
            er_metrics,
            (model_json, traces),
            {"partial_trace": True, "reduce_level": reduce_level},
            evaluator,
            model_json,
            mem_gb,
            time_min,
        )
    method = "alignment" if evaluator == "alignment" else "token"
    precision_variant = "align_etc" if evaluator == "token_align_etc" else "etc"
    return _run_unit(
        conformance_metrics,
        (model_json, event_subset),
        {
            "consider_remaining": consider_remaining,
            "method": method,
            "reduce_level": reduce_level,
            "precision_variant": precision_variant,
        },
        evaluator,
        model_json,
        mem_gb,
        time_min,
    )


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Coverage, mean and std of each metric by (mode, evaluator, level, family).

    ``n_<metric>`` counts the units with a finite value of that metric, out of
    ``n_expected`` units; it is the denominator of that metric's mean. The status
    counts (``n_ok``, ``n_unavailable``, ``n_timeout``, ...), ``n_over_budget``
    (``timeout`` and ``oom``) and ``n_empty_model`` are diagnostics.
    """
    keys = ["mode", "evaluator", "reduce_level", "model_type"]
    metric_cols = [
        *METRIC_KEYS,
        "er",
        "fitting_ratio",
        "wall_s",
        "pct_prefix_mass_fit",
        "pct_events_unseen",
        "memory_peak_gb",
        "peak_rss_gb",
    ]
    values = results[metric_cols].apply(pd.to_numeric, errors="coerce")
    by = [results[k] for k in keys]
    stats = values.groupby(by).agg(["mean", "std"])
    stats.columns = ["_".join(c) for c in stats.columns]
    coverage = np.isfinite(values).astype(int).groupby(by).sum().add_prefix("n_")

    status = results["guard_status"]
    flags = {
        "n_expected": pd.Series(1, index=results.index),
        **{f"n_{s}": status == s for s in GUARD_STATUSES},
        "n_over_budget": status.isin(OVER_BUDGET),
        "n_empty_model": (status == "ok") & (results["guard_reason"] == EMPTY_MODEL),
    }
    counts = pd.DataFrame(flags).astype(int).groupby(by).sum()
    ordered = [c for m in metric_cols for c in (f"{m}_mean", f"{m}_std", f"n_{m}")]
    return counts.join(stats.join(coverage)[ordered]).reset_index()


def failure_counts(results: pd.DataFrame) -> pd.DataFrame:
    """Units per ``failure_category``, by log, mode, evaluator, level and family.

    Every unit without a number has a category, and so does an ``ok`` unit whose model
    accepts nothing (``empty_model``).
    """
    keys = ["dataset", "mode", "evaluator", "reduce_level", "model_type"]
    flagged = results[results["failure_category"].fillna("") != ""]
    return (
        flagged.groupby([*keys, "guard_status", "failure_category"])
        .size()
        .rename("n_units")
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the weekly models, one guarded unit at a time."
    )
    parser.add_argument("--dataset", choices=DATASETS, default="bpi2017")
    parser.add_argument(
        "--mode",
        choices=get_args(EvaluationMode),
        default="case_complete",
        help="partial-trace evaluation mode for the token/alignment cells",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=None,
        help="restrict to these window numbers (default: all)",
    )
    parser.add_argument(
        "--model-types", nargs="+", default=list(MODEL_FAMILIES), choices=MODEL_FAMILIES
    )
    parser.add_argument(
        "--evaluators", nargs="+", default=list(EVALUATORS), choices=EVALUATORS
    )
    parser.add_argument(
        "--reduce-levels",
        nargs="+",
        default=["none", "tau"],
        choices=REDUCTION_LEVELS,
    )
    parser.add_argument(
        "--weekly-dir",
        type=Path,
        default=None,
        help="read the weekly models from this directory instead",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="override the output directory"
    )
    parser.add_argument(
        "--mem-gb", type=float, default=None, help="guard memory budget"
    )
    parser.add_argument(
        "--time-min", type=float, default=None, help="guard time budget"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="units of this campaign running at once on this machine (recorded)",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=None,
        help="expected_units.csv of the campaign: run exactly its units for this log "
        "and tPAT (narrowed by the other options), after checking every model they "
        "need; required unless --development",
    )
    parser.add_argument(
        "--development",
        action="store_true",
        help="allow running without the per-unit cgroup (RLIMIT_AS or time only)",
    )
    args = parser.parse_args()
    enable_determinism()

    LOG_PATH, OUTPUT_DIR, WEEKLY_DIR = resolve_dataset(args.dataset)
    weekly_dir = args.weekly_dir if args.weekly_dir is not None else WEEKLY_DIR
    out_dir = args.out_dir if args.out_dir is not None else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    res_csv = out_dir / "evaluation.csv"
    summary_csv = out_dir / "evaluation_summary.csv"
    failures_csv = out_dir / "evaluation_failures.csv"

    print("=" * 72)
    print(
        f"EVALUATION: {args.dataset} mode={args.mode} "
        f"levels={args.reduce_levels} evaluators={args.evaluators}"
    )
    print(f"weekly models: {weekly_dir}")
    print("=" * 72)

    # Every window lies after the training period, so the whole log serves all modes.
    df = normalise_timestamp(load_event_log(LOG_PATH))

    tpat = run_tpat(weekly_dir)
    if tpat is None:
        raise SystemExit(
            f"evaluate.py: {weekly_dir.parent / 'experiment_summary.json'} records no "
            "tPAT, so the units cannot be keyed by it."
        )

    # The units to run: the expected grid's, or else every model on disk.
    wanted: set[tuple] | None = None
    if args.expected is not None:
        grid = read_expected(args.expected)
        grid = grid[
            (grid["dataset"] == args.dataset)
            & (grid["tpat"].astype(float) == tpat)
            & grid["model_type"].isin(args.model_types)
            & grid["evaluator"].isin(args.evaluators)
            & grid["reduce_level"].isin(args.reduce_levels)
            & ((grid["evaluator"] == "er") | (grid["mode"] == args.mode))
        ]
        if args.windows:
            grid = grid[grid["window"].isin(args.windows)]
        problems = check_models(grid, {(args.dataset, tpat): weekly_dir})
        if problems:
            raise SystemExit(
                "evaluate.py: the models do not match the expected grid:\n  "
                + "\n  ".join(problems)
            )
        wanted = {
            tuple(r) for r in grid[list(UNIT_KEY)].itertuples(index=False, name=None)
        }
        window_numbers = sorted({int(w) for w in grid["window"]})
    else:
        window_numbers = sorted(
            {int(p.stem.split("_")[1]) for p in weekly_dir.glob("window_*.json")}
        )
        if args.windows:
            window_numbers = [w for w in window_numbers if w in set(args.windows)]
    if not window_numbers:
        print(f"No units to evaluate for {weekly_dir}")
        return
    try:
        manifest = evaluation_manifest(
            LOG_PATH,
            weekly_dir,
            args.mem_gb,
            args.time_min,
            args.concurrency,
            tpat,
            args.expected,
        )
    except GuardSetupError as exc:
        raise SystemExit(f"evaluate.py: {exc}") from exc
    if manifest["enforcement"] != "cgroup" and not args.development:
        raise SystemExit(
            f"evaluate.py: the memory budget would be enforced by "
            f"{manifest['enforcement']}, not by a per-unit cgroup. Start the run with "
            "scripts/local/delegated_scope.sh, or pass --development for a run that "
            "is not part of a campaign."
        )
    if args.expected is None and not args.development:
        raise SystemExit(
            "evaluate.py: a campaign runs the units of its expected grid: pass "
            "--expected expected_units.csv, or --development for a run that is not "
            "part of a campaign."
        )
    check_manifest(out_dir, manifest)

    # Remaining tokens make a trace imperfect, except in window_no_remain.
    consider_remaining = args.mode != "window_no_remain"
    n_cells = 0
    n_skipped = 0

    # Units already in the CSV are skipped, so a resubmitted job resumes.
    done: set[tuple] = set()
    if res_csv.exists():
        done = {
            (
                r.dataset,
                float(r.tpat),
                int(r.window),
                r.model_type,
                r.mode,
                r.evaluator,
                r.reduce_level,
            )
            for r in read_units(res_csv).itertuples()
        }
        print(f"Resuming: {len(done)} cells already in {res_csv}")

    for window_num in window_numbers:
        # Window dates from the first available model JSON.
        model_jsons: dict[str, dict] = {}
        for mt in args.model_types:
            p = model_json_path(weekly_dir, window_num, mt)
            if p.exists():
                model_jsons[mt] = json.loads(p.read_text())
        if not model_jsons:
            continue
        meta = next(iter(model_jsons.values()))["metadata"]
        start, end = window_bounds(meta)

        if args.mode == "case_complete":
            event_subset = full_traces_of_active_cases(df, start, end)
        else:
            event_subset = filter_window_events(df, start, end)
        traces = (
            extract_window_traces(df, start, end) if "er" in args.evaluators else []
        )
        # What each evaluator scores: ER the window's fragments (without ▶ and ■),
        # the others the replay mode's events.
        scored = {
            "er": (len(traces), sum(len(t) - 2 for t in traces)),
            "replay": (event_subset["case:concept:name"].nunique(), len(event_subset)),
        }

        print(
            f"  window {window_num}/{window_numbers[-1]}: "
            f"{event_subset['case:concept:name'].nunique()} cases, "
            f"{len(event_subset)} events",
            flush=True,
        )

        for model_type, model_json in model_jsons.items():
            for reduce_level in args.reduce_levels:
                for evaluator in args.evaluators:
                    mode = ER_MODE if evaluator == "er" else args.mode
                    n_cases, n_events = scored["er" if evaluator == "er" else "replay"]
                    key = (
                        args.dataset,
                        tpat,
                        window_num,
                        model_type,
                        mode,
                        evaluator,
                        reduce_level,
                    )
                    if wanted is not None and key not in wanted:
                        continue
                    if key in done:
                        n_skipped += 1
                        continue
                    t0 = time.perf_counter()
                    status, metrics = _evaluate_cell(
                        evaluator,
                        model_json,
                        event_subset,
                        traces,
                        consider_remaining,
                        reduce_level,
                        args.mem_gb,
                        args.time_min,
                    )
                    wall_s = time.perf_counter() - t0
                    row = {
                        "dataset": args.dataset,
                        "tpat": tpat,
                        "window": window_num,
                        "model_type": model_type,
                        "mode": mode,
                        "evaluator": evaluator,
                        "reduce_level": reduce_level,
                        "n_events": n_events,
                        "n_cases": n_cases,
                        **metrics,
                        "guard_status": status,
                        "wall_s": wall_s,
                    }
                    pd.DataFrame([row]).reindex(columns=CSV_COLUMNS).to_csv(
                        res_csv, mode="a", header=not res_csv.exists(), index=False
                    )
                    # A repeated command-line value names the same unit again.
                    done.add(key)
                    n_cells += 1
                    brief = (
                        f"er={metrics.get('er', float('nan')):.3f}"
                        if evaluator == "er" and status == "ok"
                        else f"fit={metrics.get('log_fitness', float('nan')):.3f} "
                        f"prec={metrics.get('precision', float('nan')):.3f}"
                        if status == "ok"
                        else metrics.get("failure_category", "")
                    )
                    print(
                        f"    {model_type:<10} {reduce_level:<12} {evaluator:<16} "
                        f"-> {status:<8} {wall_s:7.1f}s  {brief}",
                        flush=True,
                    )

    if not res_csv.exists():
        print("No units evaluated.")
        return

    # Summary over every row in the CSV, including earlier runs.
    results = read_units(res_csv)
    this_log = results[results["dataset"] == args.dataset]
    summarise(this_log).to_csv(summary_csv, index=False)
    failure_counts(this_log).to_csv(failures_csv, index=False)
    print(f"\n{n_cells} cells evaluated this run ({n_skipped} skipped as done).")
    print(f"Per-cell results: {res_csv}")
    print(f"Summary:          {summary_csv}")
    print(f"Failures:         {failures_csv}")


if __name__ == "__main__":
    main()
