"""Evaluate the weekly models against the test log, one unit at a time.

A unit is one (window, family, reduce_level, evaluator):

    token_etc        token replay fitness, ETC precision, prefix coverage
    token_align_etc  token replay fitness, Align-ETC precision
    alignment        alignment fitness, Align-ETC precision
    er               entropic relevance with partial-trace wildcards

ETC precision discards non-fitting prefixes and Align-ETC aligns them, so the share of
prefix mass ETC discards is recorded too. ``tau`` evaluates the net after
language-preserving τ-reduction, and reduced units record the net size before and after.

Every unit runs under the budget guard (12 GB and 20 min by default) and is appended to
the CSV as soon as it finishes, so a stopped run resumes where it left off. A resume must
use the same log, models, budgets and thread counts, which ``evaluation_manifest.json``
records. OMP and MKL default to 8 threads, because the thread count can change
floating-point results.

Usage, from the project root:
    uv run python scripts/evaluate.py --dataset sepsis --windows 6 7 8
    uv run python scripts/evaluate.py --dataset bpi2017 --reduce-levels tau \
        --evaluators token_etc alignment er

Outputs, under $PMF_OUTPUT_BASE/<Dataset>_pmf/ unless --out-dir:
    evaluation.csv            one row per unit
    evaluation_summary.csv    mean, std and count by (evaluator, level, family), and
                              the number of units without a result
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
import time
from pathlib import Path
from typing import get_args

import pandas as pd
from _dataset import DATASETS, resolve_dataset

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
from pmf_concurrency.entropic_relevance import compute_er
from pmf_concurrency.families import MODEL_FAMILIES, check_families, model_json_path
from pmf_concurrency.guard import resolve_budgets, run_guarded
from pmf_concurrency.reduction import REDUCTION_LEVELS
from pmf_concurrency.reproducibility import enable_determinism

EVALUATORS = ("token_etc", "token_align_etc", "alignment", "er")

MANIFEST_FILENAME = "evaluation_manifest.json"

#: Identifies one evaluation unit.
UNIT_KEY = ("dataset", "window", "model_type", "mode", "evaluator", "reduce_level")

# Evaluators return different metrics, so every row is reindexed to this schema before
# it is appended.
CSV_COLUMNS = [
    "dataset",
    "window",
    "model_type",
    "mode",
    "evaluator",
    "reduce_level",
    "n_events",
    "n_cases",
    "guard_status",
    "wall_s",
    *METRIC_KEYS,
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

    Raises on family names other than the four, and if two rows describe the same unit,
    which pooling would count twice.
    """
    df = pd.read_csv(path)
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
    log_path: str, weekly_dir: Path, mem_gb: float | None, time_min: float | None
) -> dict:
    """What every unit in one output directory must share.

    The mode, windows, families, levels and evaluators are part of each unit's key, so
    one directory can collect several such runs.
    """
    mem_gb, time_min = resolve_budgets(mem_gb, time_min)
    return {
        "log_sha256": _sha256([Path(log_path)]),
        "models_sha256": _sha256(sorted(weekly_dir.glob("window_*.json"))),
        "mem_gb": mem_gb,
        "time_min": time_min,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
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


def _evaluate_cell(
    evaluator: str,
    model_json: dict,
    event_subset: pd.DataFrame,
    traces: list,
    consider_remaining: bool,
    reduce_level: str,
    mem_gb: float,
    time_min: float,
) -> tuple[str, dict]:
    """Run one guarded unit; return (guard_status, metrics)."""
    if evaluator == "er":
        status, payload = run_guarded(
            compute_er,
            model_json,
            traces,
            partial_trace=True,
            reduce_level=reduce_level,
            mem_gb=mem_gb,
            time_min=time_min,
        )
        if status == "ok":
            er, fitting_ratio, _n_traces = payload
            return status, {"er": er, "fitting_ratio": fitting_ratio}
        return status, {}

    method = "alignment" if evaluator == "alignment" else "token"
    precision_variant = "align_etc" if evaluator == "token_align_etc" else "etc"
    status, payload = run_guarded(
        conformance_metrics,
        model_json,
        event_subset,
        consider_remaining=consider_remaining,
        method=method,
        reduce_level=reduce_level,
        precision_variant=precision_variant,
        mem_gb=mem_gb,
        time_min=time_min,
    )
    if status == "ok":
        return status, payload
    return status, {}


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean, std and count of each metric by (mode, evaluator, level, family).

    A metric's count covers only the units that measured it. ``n_units`` counts every
    unit, ``n_intractable`` those the guard stopped, and ``n_unavailable`` those that
    finished without their main metric (ER, or F1, which needs fitness and precision).
    """
    keys = ["mode", "evaluator", "reduce_level", "model_type"]
    metric_cols = [*METRIC_KEYS, "er", "fitting_ratio", "wall_s", "pct_prefix_mass_fit"]
    summary = results.groupby(keys)[metric_cols].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]

    finished = results["guard_status"] == "ok"
    main_metric = results["f1"].where(results["evaluator"] != "er", results["er"])
    counts = (
        results.assign(
            n_units=1,
            n_intractable=~finished,
            n_unavailable=finished & main_metric.isna(),
        )
        .groupby(keys)[["n_units", "n_intractable", "n_unavailable"]]
        .sum()
    )
    return summary.join(counts).reset_index()


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
    args = parser.parse_args()
    enable_determinism()

    LOG_PATH, OUTPUT_DIR, WEEKLY_DIR = resolve_dataset(args.dataset)
    weekly_dir = args.weekly_dir if args.weekly_dir is not None else WEEKLY_DIR
    out_dir = args.out_dir if args.out_dir is not None else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    res_csv = out_dir / "evaluation.csv"
    summary_csv = out_dir / "evaluation_summary.csv"

    print("=" * 72)
    print(
        f"EVALUATION: {args.dataset} mode={args.mode} "
        f"levels={args.reduce_levels} evaluators={args.evaluators}"
    )
    print(f"weekly models: {weekly_dir}")
    print("=" * 72)

    # Every window lies after the training period, so the whole log serves all modes.
    df = normalise_timestamp(load_event_log(LOG_PATH))

    window_numbers = sorted(
        {int(p.stem.split("_")[1]) for p in weekly_dir.glob("window_*.json")}
    )
    if args.windows:
        window_numbers = [w for w in window_numbers if w in set(args.windows)]
    if not window_numbers:
        print(f"No weekly model JSONs found in {weekly_dir}")
        return
    check_manifest(
        out_dir, evaluation_manifest(LOG_PATH, weekly_dir, args.mem_gb, args.time_min)
    )

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

        print(
            f"  window {window_num}/{window_numbers[-1]}: "
            f"{event_subset['case:concept:name'].nunique()} cases, "
            f"{len(event_subset)} events",
            flush=True,
        )

        for model_type, model_json in model_jsons.items():
            for reduce_level in args.reduce_levels:
                for evaluator in args.evaluators:
                    key = (
                        args.dataset,
                        window_num,
                        model_type,
                        args.mode,
                        evaluator,
                        reduce_level,
                    )
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
                        "window": window_num,
                        "model_type": model_type,
                        "mode": args.mode,
                        "evaluator": evaluator,
                        "reduce_level": reduce_level,
                        "n_events": len(event_subset),
                        "n_cases": event_subset["case:concept:name"].nunique(),
                        "guard_status": status,
                        "wall_s": wall_s,
                        **metrics,
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
                        else ""
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
    summarise(results[results["dataset"] == args.dataset]).to_csv(
        summary_csv, index=False
    )
    print(f"\n{n_cells} cells evaluated this run ({n_skipped} skipped as done).")
    print(f"Per-cell results: {res_csv}")
    print(f"Summary:          {summary_csv}")


if __name__ == "__main__":
    main()
