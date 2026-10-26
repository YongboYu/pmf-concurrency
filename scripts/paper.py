"""Build the paper's tables, figure and prose numbers from one results directory.

    collect  copy the gated campaigns and their pipeline runs into a results directory,
             with the Table 1 statistics, the net audit and MANIFEST.yaml
    build    write the manuscript's generated files from a results directory
    check    rebuild into a temporary directory and compare with a manuscript

``collect`` refuses a campaign whose gate did not pass, a campaign that evaluated
other nets than the pipeline runs it is given, and a log other than the one the
campaigns evaluated. The history campaign and root, given together, add Persistence
and Expanding to Table 2. ``build`` reads only the results directory and refuses it
unless every file matches its hash in ``MANIFEST.yaml``.

Usage, from the project root:
    uv run python scripts/paper.py collect --out results \\
        --reduced-campaign campaigns/c1 --unreduced-campaign campaigns/c2 \\
        --pipeline-roots outputs_tpat0 outputs_tpatm1 --data-dir data \\
        --history-campaign campaigns/c3 --history-root outputs_history_tpat0
    uv run python scripts/paper.py build --results results --out manuscript/paper \\
        --highlight-against manuscript/submitted
    uv run python scripts/paper.py check --results results --manuscript manuscript/paper \\
        --highlight-against manuscript/submitted

``--highlight-against`` names the submitted manuscript's directory; every table value
that differs from it is wrapped in ``\\rev``. Leave it out for the camera-ready.

Files ``build`` writes:
    numbers.tex                      one macro per number in the prose (\\pmf...)
    tab-logs-rows.tex                Table 1 body rows
    tab-quality-rows-structure.tex   Table 2 body rows
    fig-w1-expanded.{dot,pdf}        Figure 2 (a)
    fig-w1-reduced.{dot,pdf}         Figure 2 (b)
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from net_pair_figure import draw
from omegaconf import OmegaConf

from pmf_concurrency.campaign import (
    HISTORY_ARMS,
    TABLE2_ARMS,
    TABLE2_EVALUATORS,
    windows_of,
)
from pmf_concurrency.data import load_event_log
from pmf_concurrency.families import (
    FAMILY_LABELS,
    HISTORY_FAMILIES,
    MODEL_FAMILIES,
    model_json_path,
)
from pmf_concurrency.petri_net import unfinishable_places
from pmf_concurrency.results import (
    CASE_CLASSES,
    MANIFEST,
    ResultsError,
    describe_log,
    names_and_bytes_sha256,
    over_budget_share,
    pool,
    read_campaign,
    read_manifest,
    reduction_effect,
    replay_mode_ranges,
    sha256_file,
    window_composition,
)

ROOT = Path(__file__).resolve().parents[1]

#: Dataset keys in the paper's order, with their names in the tables.
LOGS = (
    ("bpi2017", "BPI2017"),
    ("bpi2019", "BPI2019"),
    ("sepsis", "Sepsis"),
    ("hospital_billing", "Hosp.\\ Bill."),
)

#: Relation families of Table 1, in its column order.
SERIES_FAMILIES = ("A", "AB", "ABA", "IB", "OB")

#: Files copied from a campaign directory.
CAMPAIGN_FILES = (
    "evaluation.csv",
    "evaluation_summary.csv",
    "evaluation_failures.csv",
    "expected_units.csv",
    "gate.json",
    "campaign_manifest.json",
    "environment.json",
    "near_budget.csv",
    "batches.csv",
)

#: Files copied from each log's pipeline output directory.
RUN_FILES = (
    "experiment_summary.json",
    "structural_metrics.csv",
    "median_choice.json",
    *(f"forecast_{f}_summary.csv" for f in SERIES_FAMILIES),
)

#: Files copied from each log's directory in the history root.
HISTORY_FILES = ("experiment_summary.json", "structural_metrics.csv")

#: The net of Figure 2: (dataset, tPAT, window, family).
FIGURE_NET = ("bpi2017", -1.0, 1, "forecast")

#: Fewer finished units than this print a dagger instead of a mean.
MIN_VALUES = 2

DAGGER = "$^{\\dagger}$"

#: The submitted cell of a row the submitted table lacks: every value differs from it.
ABSENT = ""

#: Value cells of a Table 2 row, after the model and tPAT cells.
TABLE2_CELLS = 11

WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight")


def _dataset(key: str) -> tuple[str, str]:
    """(output directory name, log file name) of a dataset key."""
    cfg = OmegaConf.load(ROOT / "conf" / "dataset" / f"{key}.yaml")
    return f"{cfg.name}_pmf", Path(str(cfg.log_path)).name


def _tag(tpat: float) -> str:
    return f"{tpat:+.1f}"


# collect


def _pipeline_runs(roots: list[Path]) -> dict[float, Path]:
    """tPAT -> pipeline root, from each root's experiment summaries."""
    runs: dict[float, Path] = {}
    for root in roots:
        tpats = {
            float(json.loads(p.read_text())["thresholds"]["tPAT"])
            for p in root.glob("*_pmf/experiment_summary.json")
        }
        if len(tpats) != 1:
            raise ResultsError(
                f"{root}: expected one tPAT across its logs, got {tpats}"
            )
        tpat = tpats.pop()
        if tpat in runs:
            raise ResultsError(f"{runs[tpat]} and {root} both hold tPAT {tpat}")
        manifest = json.loads((root / "campaign_manifest.json").read_text())
        failed = [r["dataset"] for r in manifest["runs"] if r["exit_code"] != 0]
        if failed or manifest.get("git_dirty"):
            raise ResultsError(
                f"{root}: failed runs {failed}, dirty {manifest['git_dirty']}"
            )
        runs[tpat] = root
    return runs


def _history_root(root: Path) -> dict[float, Path]:
    """tPAT -> the history root, once every log holds both history families."""
    summaries = [
        json.loads(p.read_text()) for p in root.glob("*_pmf/experiment_summary.json")
    ]
    tpats = {float(s["thresholds"]["tPAT"]) for s in summaries}
    families = {tuple(s["structural_metrics"]["families"]) for s in summaries}
    if (
        len(summaries) != len(LOGS)
        or tpats != set(HISTORY_ARMS)
        or families != {HISTORY_FAMILIES}
    ):
        raise ResultsError(
            f"{root}: expected {', '.join(HISTORY_FAMILIES)} for {len(LOGS)} logs at "
            f"tPAT {sorted(HISTORY_ARMS)}, got {len(summaries)} logs, tPAT "
            f"{sorted(tpats)}, families {sorted(families)}"
        )
    return {tpats.pop(): root}


def _check_models(campaign: dict, runs: dict[float, Path], source: Path) -> None:
    """Every net a campaign evaluated is the net in the given pipeline run."""
    for label, entry in campaign["models"].items():
        key, _, tpat = label.split(" ")
        weekly = runs[float(tpat)] / _dataset(key)[0] / "weekly_models"
        got = names_and_bytes_sha256(sorted(weekly.glob("window_*.json")))
        if got != entry["models_sha256"]:
            raise ResultsError(f"{source}: {label} evaluated other nets than {weekly}")


def _log_hashes(campaign_dir: Path) -> dict[str, str]:
    """dataset -> the one ``log_sha256`` its batches recorded."""
    hashes: dict[str, set[str]] = {}
    for path in campaign_dir.glob("eval/*/evaluation_manifest.json"):
        manifest = json.loads(path.read_text())
        dataset = path.parent.name.split("_tpat")[0]
        hashes.setdefault(dataset, set()).add(manifest["log_sha256"])
    if any(len(h) != 1 for h in hashes.values()):
        raise ResultsError(f"{campaign_dir}: batches of one log read different logs")
    return {d: h.pop() for d, h in hashes.items()}


def _audit(
    arms: list[tuple[Path, float, tuple[str, ...]]], grids: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Unfinishable nets per (log, tPAT, family) over Table 2's arms and windows.

    ``arms`` holds one (pipeline root, tPAT, families) per group of arms.
    """
    rows = []
    for key, _ in LOGS:
        for root, tpat, families in arms:
            weekly = root / _dataset(key)[0] / "weekly_models"
            for family in families:
                counts = [
                    unfinishable_places(
                        json.loads(model_json_path(weekly, int(w), family).read_text())
                    )
                    for w in grids[key]["window"]
                ]
                rows.append(
                    {
                        "dataset": key,
                        "tpat": tpat,
                        "model_type": family,
                        "nets": len(counts),
                        "unfinishable": sum(c > 0 for c in counts),
                    }
                )
    return pd.DataFrame(rows)


def _campaign_entry(directory: Path, name: str) -> dict:
    manifest = json.loads((directory / "campaign_manifest.json").read_text())
    gate = json.loads((directory / "gate.json").read_text())
    keep = (
        "git_commit",
        "host",
        "cpu",
        "ram_gb",
        "os",
        "python",
        "pm4py",
        "concurrency",
        "mem_gb",
        "time_min",
        "omp_num_threads",
        "mkl_num_threads",
    )
    return {
        "dir": name,
        **{k: manifest[k] for k in keep},
        "gate_passed": bool(gate["passed"]),
        "n_units": int(gate["n_expected"]),
        "status_counts": gate["status_counts"],
    }


def collect(args: argparse.Namespace) -> int:
    out: Path = args.out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty: collect into a new directory")
    if (args.history_campaign is None) != (args.history_root is None):
        raise SystemExit("give --history-campaign and --history-root together")
    campaigns = {"reduced": args.reduced_campaign, "unreduced": args.unreduced_campaign}
    if args.history_campaign is not None:
        campaigns["history"] = args.history_campaign
    for d in campaigns.values():
        read_campaign(d)  # the gate passed and the units equal the grid
    runs = _pipeline_runs(args.pipeline_roots)
    arms = [(runs[tpat], tpat, families) for tpat, families in TABLE2_ARMS.items()]
    history_runs = {}
    if args.history_root is not None:
        history_runs = _history_root(args.history_root)
        arms += [(history_runs[t], t, f) for t, f in HISTORY_ARMS.items()]
    for role, d in campaigns.items():
        _check_models(
            json.loads((d / "campaign_manifest.json").read_text()),
            history_runs if role == "history" else runs,
            d,
        )

    hashes = [_log_hashes(d) for d in campaigns.values()]
    if any(h != hashes[0] for h in hashes[1:]):
        raise ResultsError("the campaigns evaluated different logs")
    log_entries, described, compositions, grids = {}, [], [], {}
    for key, _ in LOGS:
        log_file = args.data_dir / _dataset(key)[1]
        if names_and_bytes_sha256([log_file]) != hashes[0][key]:
            raise ResultsError(f"{log_file} is not the log the campaigns evaluated")
        log_entries[key] = {"file": log_file.name, "log_sha256": hashes[0][key]}
        events = load_event_log(str(log_file))
        grids[key] = windows_of(events)
        described.append({"dataset": key, **describe_log(events)})
        compositions.append(window_composition(events, grids[key]).assign(dataset=key))

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(described).to_csv(out / "logs.csv", index=False)
    composition = pd.concat(compositions, ignore_index=True)
    composition[
        ["dataset", *[c for c in composition.columns if c != "dataset"]]
    ].to_csv(out / "window_composition.csv", index=False)
    _audit(arms, grids).to_csv(out / "unfinishable_nets.csv", index=False)

    entries = {}
    for role, d in campaigns.items():
        (out / d.name).mkdir()
        for name in CAMPAIGN_FILES:
            shutil.copy2(d / name, out / d.name / name)
        entries[role] = _campaign_entry(d, d.name)

    pipeline = {}
    for tpat, root in sorted(runs.items()):
        target = out / "pipeline" / root.name
        target.mkdir(parents=True)
        shutil.copy2(root / "campaign_manifest.json", target / "run_manifest.json")
        models = {}
        for key, _ in LOGS:
            name = _dataset(key)[0]
            (target / name).mkdir()
            for f in RUN_FILES:
                if (root / name / f).exists():
                    shutil.copy2(root / name / f, target / name / f)
            models[key] = names_and_bytes_sha256(
                sorted((root / name / "weekly_models").glob("window_*.json"))
            )
        run = json.loads((root / "campaign_manifest.json").read_text())
        pipeline[_tag(tpat)] = {
            "dir": f"pipeline/{root.name}",
            "tpat": tpat,
            **{k: run[k] for k in ("git_commit", "host", "device", "gpu")},
            "models_sha256": models,
        }

    history = {}
    for tpat, root in history_runs.items():
        models = {}
        for key, _ in LOGS:
            name = _dataset(key)[0]
            target = out / "history" / root.name / name
            target.mkdir(parents=True)
            for f in HISTORY_FILES:
                shutil.copy2(root / name / f, target / f)
            models[key] = names_and_bytes_sha256(
                sorted((root / name / "weekly_models").glob("window_*.json"))
            )
        history = {
            "dir": f"history/{root.name}",
            "tpat": tpat,
            "families": list(HISTORY_FAMILIES),
            "models_sha256": models,
        }

    key, tpat, window, family = FIGURE_NET
    net = model_json_path(
        runs[tpat] / _dataset(key)[0] / "weekly_models", window, family
    )
    (out / "figure2").mkdir()
    shutil.copy2(net, out / "figure2" / net.name)

    files = {
        p.relative_to(out).as_posix(): sha256_file(p)
        for p in sorted(out.rglob("*"))
        if p.is_file()
    }
    manifest = {
        "note": "Written by scripts/paper.py collect. Do not edit: collect again.",
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "collected_by_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT
        ).stdout.strip(),
        "campaigns": entries,
        "pipeline": pipeline,
        **({"history": history} if history else {}),
        "logs": log_entries,
        "figure2": {
            "file": f"figure2/{net.name}",
            "dataset": key,
            "tpat": tpat,
            "window": window,
            "model_type": family,
        },
        "files": files,
    }
    (out / MANIFEST).write_text(yaml.safe_dump(manifest, sort_keys=False, width=88))
    print(f"{len(files)} files -> {out}/{MANIFEST}")
    return 0


# build: formatting


def _int(n: float) -> str:
    return f"{round(n):,}".replace(",", "{,}")


def _fixed(x: float, decimals: int) -> str:
    return f"{x:.{decimals}f}"


def _range(low: float, high: float, decimals: int) -> str:
    """``low--high`` as printed, or one value when both ends print alike."""
    a, b = _fixed(low, decimals), _fixed(high, decimals)
    return a if a == b else f"{a}--{b}"


def _cell(
    p, decimals: int, width: int, sd_decimals: int, sd_width: int = 0
) -> tuple[str, str | None]:
    """(mean, SD), each padded to a fixed width; a dagger without two values."""
    if p.n_values < MIN_VALUES or p.mean != p.mean:
        return DAGGER, None
    body = _fixed(p.mean, decimals)
    lead = "\\phantom{" + "0" * (width - len(body)) + "}" if len(body) < width else ""
    if p.sd != p.sd:
        return lead + body, None
    text = _fixed(p.sd, sd_decimals)
    fill = (
        "\\phantom{" + "0" * (sd_width - len(text)) + "}"
        if len(text) < sd_width
        else ""
    )
    return lead + body, text + fill


def _coverage(p) -> tuple[str, None]:
    text = f"{p.n_values}/{p.n_units}"
    width = len(str(p.n_units)) * 2 + 1
    lead = "\\phantom{" + "0" * (width - len(text)) + "}" if len(text) < width else ""
    return lead + text, None


def _plain(text: str) -> str:
    """A printed value without padding, highlighting, math shifts or braces."""
    text = re.sub(r"\\phantom\{0*\}", "", text)
    text = re.sub(r"\\(?:chg|rev|pend)\{", "", text)
    return text.replace("{,}", ",").translate(str.maketrans("", "", "${}")).strip()


def _split(cell: str) -> tuple[str, str | None]:
    """(mean, SD) of a printed cell, plain."""
    mean, _, sd = _plain(cell).partition("_\\pm")
    return mean, sd or None


def _rev(text: str) -> str:
    return "\\rev{" + text + "}"


def _math(parts: tuple[str, str | None], old: str | None) -> str:
    """A Table 2 cell; with the submitted cell, what changed is wrapped in \\rev."""
    mean, sd = parts
    if mean == DAGGER:
        return (
            _rev(DAGGER)
            if old is not None and _plain(old) != _plain(DAGGER)
            else DAGGER
        )
    old_mean, old_sd = _split(old) if old is not None else (None, None)
    if old is not None and _plain(mean) != old_mean:
        mean = _rev(mean)
    if sd is not None and old is not None and _plain(sd) != old_sd:
        sd = _rev(sd)
    return "$" + mean + ("_{\\pm" + sd + "}" if sd is not None else "") + "$"


def _text(value: str, old: str | None) -> str:
    """A Table 1 cell; with the submitted cell, wrapped in \\rev if it changed."""
    return _rev(value) if old is not None and _plain(value) != _plain(old) else value


def submitted_rows(manuscript: Path) -> tuple[dict[str, list[str]], list[list[str]]]:
    """The submitted manuscript's Table 1 cells by log label and Table 2 cells by row."""
    main = (manuscript / "main.tex").read_text()
    block = main[main.index("\\label{tab:logs}") :]
    block = block[: block.index("\\end{tabular}")]
    labels = {label for _, label in LOGS}
    table1 = {}
    for line in block.splitlines():
        cells = [c.strip() for c in line.strip().removesuffix("\\\\").split("&")]
        if cells[0] in labels:
            table1[cells[0]] = cells[1:]
    table2 = [
        [c.strip() for c in line.strip().removesuffix("\\\\").split("&")][3:]
        for line in (manuscript / "tab-quality-rows-structure.tex")
        .read_text()
        .splitlines()
        if "&" in line
    ]
    if set(table1) != labels or len(table2) != 24:
        raise ResultsError(
            f"{manuscript}: cannot find the submitted Table 1 and 2 rows"
        )
    return table1, table2


# build: Table 1


def _series_counts(results: Path, manifest: dict) -> pd.DataFrame:
    """Series per relation family and log; the same at both tPAT or an error."""
    frames = []
    for run in manifest["pipeline"].values():
        rows = []
        for key, _ in LOGS:
            directory = results / run["dir"] / _dataset(key)[0]
            counts = {}
            for family in SERIES_FAMILIES:
                path = directory / f"forecast_{family}_summary.csv"
                counts[family] = len(pd.read_csv(path)) if path.exists() else 0
            rows.append({"dataset": key, **counts})
        frames.append(pd.DataFrame(rows).set_index("dataset"))
    if any(not f.equals(frames[0]) for f in frames[1:]):
        raise ResultsError("the pipeline runs extracted different series")
    return frames[0]


def table1(
    results: Path, manifest: dict, submitted: dict[str, list[str]] | None = None
) -> tuple[str, dict[str, str]]:
    logs = pd.read_csv(results / "logs.csv").set_index("dataset")
    composition = pd.read_csv(results / "window_composition.csv")
    series = _series_counts(results, manifest)
    lines, numbers = [], {}
    ratios, binding_shares = [], []
    for key, label in LOGS:
        d = logs.loc[key]
        s = series.loc[key]
        c = composition[composition["dataset"] == key]
        total = c["n_cases"].sum()
        shares = [100 * c[f"n_{k}"].sum() / total for k in CASE_CLASSES]
        cells = [
            _int(d["cases"]),
            _int(d["events"]),
            _int(d["activities"]),
            _int(d["variants"]),
            _fixed(d["median_case_days"], 1),
            *(_int(s[f]) for f in SERIES_FAMILIES),
            _int(len(c)),
            _int(c["n_cases"].mean()),
            *(_fixed(x, 1) for x in shares),
        ]
        old = submitted[label] if submitted else [None] * len(cells)
        cells = [_text(c, o) for c, o in zip(cells, old, strict=True)]
        lines.append(f"    {label:<12} & " + " & ".join(cells) + " \\\\")
        n_series = s.sum()
        ratios.append(n_series / s["AB"])
        binding_shares.append(100 * (s["IB"] + s["OB"]) / n_series)
    medians = logs["median_case_days"]
    numbers.update(
        {
            "SeriesRatioMin": _fixed(min(ratios), 1),
            "SeriesRatioMax": _fixed(max(ratios), 1),
            "BindingSeriesShareMin": _fixed(min(binding_shares), 0),
            "BindingSeriesShareMax": _fixed(max(binding_shares), 0),
            "LogsMedianOverWeek": WORDS[int((medians > 7).sum())],
            "SepsisMedianDays": _fixed(medians["sepsis"], 1),
        }
    )
    # \bottomrule must not follow an \input inside a tabular, so the file carries it.
    lines.append("    \\bottomrule")
    return "\n".join(lines) + "\n", numbers


# build: Table 2


def _structure(results: Path, run: dict, key: str, family: str):
    path = results / run["dir"] / _dataset(key)[0] / "structural_metrics.csv"
    frame = pd.read_csv(path)
    share = frame.loc[frame["model_family"] == family, "and_share_pct"]
    units = pd.DataFrame(
        {"guard_status": "ok", "and_share_pct": pd.to_numeric(share, errors="coerce")}
    )
    return pool(units, "and_share_pct")


def table2(
    results: Path,
    manifest: dict,
    reduced: pd.DataFrame,
    submitted: list[list[str]] | None = None,
    history: pd.DataFrame | None = None,
) -> str:
    """Body rows: τ-reduced nets, case-complete replay, ER on window fragments.

    With ``history``, the history campaign's units, Persistence and Expanding follow
    Static. The submitted table lacks their rows, so all of each is marked.
    """
    groups: list[
        tuple[tuple[str, ...], dict[float, tuple[str, ...]], pd.DataFrame, dict | None]
    ] = [(MODEL_FAMILIES, TABLE2_ARMS, reduced, None)]
    if history is not None:
        groups.append((HISTORY_FAMILIES, HISTORY_ARMS, history, manifest["history"]))
    n_rows = sum(len(arms[t]) for _, arms, _, _ in groups for t in arms)
    lines, row = [], 0
    for index, (key, label) in enumerate(LOGS):
        lines.append(f"    \\multirow{{{n_rows}}}{{*}}{{{label}}}")
        for families, arms, campaign, run in groups:
            for family in families:
                settings = [t for t in (-1.0, 0.0) if family in arms.get(t, ())]
                for i, tpat in enumerate(settings):
                    units = campaign[
                        (campaign["dataset"] == key)
                        & (campaign["tpat"] == tpat)
                        & (campaign["model_type"] == family)
                    ]
                    lines.append(
                        _row(
                            results,
                            run or manifest["pipeline"][_tag(tpat)],
                            key,
                            family,
                            tpat,
                            units,
                            _name(family, i, len(settings)),
                            _submitted(submitted, row, run is not None),
                        )
                    )
                    if run is None:
                        row += 1
                if family == "re_mined":
                    lines.append("      \\cmidrule(l){2-14}")
        lines.append("    \\midrule" if index < len(LOGS) - 1 else "    \\bottomrule")
    return "\n".join(lines) + "\n"


def _name(family: str, i: int, n_settings: int) -> str:
    """The model cell of a family's ``i``-th row."""
    if i:
        return ""
    if n_settings == 1:
        return FAMILY_LABELS[family]
    return f"\\multirow{{{n_settings}}}{{*}}{{{FAMILY_LABELS[family]}}}"


def _submitted(
    submitted: list[list[str]] | None, row: int, new: bool
) -> list[str] | None:
    """The submitted cells a row is compared with, or None without highlighting."""
    if submitted is None:
        return None
    return [ABSENT] * TABLE2_CELLS if new else submitted[row]


def _row(
    results: Path,
    run: dict,
    key: str,
    family: str,
    tpat: float,
    units: pd.DataFrame,
    name: str,
    old: list[str] | None,
) -> str:
    """One Table 2 row of a family at one tPAT."""
    by = {ev: units[units["evaluator"] == ev] for ev in TABLE2_EVALUATORS}
    tok, aln, er = by["token_etc"], by["alignment"], by["er"]
    cells = [
        _cell(_structure(results, run, key, family), 1, 4, 1, 4),
        _cell(pool(tok, "log_fitness"), 3, 5, 3),
        _cell(pool(tok, "precision"), 3, 5, 3),
        _cell(pool(tok, "perc_fit_traces"), 1, 5, 1, 4),
        _cell(pool(tok, "pct_prefix_mass_fit"), 1, 5, 1, 4),
        _cell(pool(aln, "log_fitness"), 3, 5, 3),
        _cell(pool(aln, "precision"), 3, 5, 3),
        _cell(pool(aln, "perc_fit_traces"), 1, 5, 1, 4),
        _cell(pool(er, "er"), 3, 6, 2, 5),
        _cell(pool(er, "fitting_ratio"), 4, 6, 3),
        _coverage(pool(er, "er")),
    ]
    marks = old or [None] * len(cells)
    cells = [_math(c, o) for c, o in zip(cells, marks, strict=True)]
    tpat_cell = f"${tpat:.1f}$"
    if old == [ABSENT] * TABLE2_CELLS:
        name, tpat_cell = _text(name, ABSENT), _text(tpat_cell, ABSENT)
    return f"      & {name:<24} & {tpat_cell:<12} & " + " & ".join(cells) + " \\\\"


# build: prose numbers


def _same(entries: list[dict], key: str):
    values = {str(e[key]) for e in entries}
    if len(values) != 1:
        raise ResultsError(f"the campaigns differ in {key}: {sorted(values)}")
    return entries[0][key]


def setup_numbers(results: Path, manifest: dict) -> dict[str, str]:
    """Pipeline time, memory and GPU; the evaluation budget, CPU and concurrency."""
    minutes, peaks, gpus = [], [], set()
    for run in manifest["pipeline"].values():
        record = json.loads((results / run["dir"] / "run_manifest.json").read_text())
        minutes += [r["wall_clock_sec"] / 60 for r in record["runs"]]
        peaks += [r["peak_rss_gb"] for r in record["runs"]]
        gpus.add(record["gpu"])
    if len(gpus) != 1:
        raise ResultsError(f"the pipeline runs used different GPUs: {sorted(gpus)}")
    campaigns = list(manifest["campaigns"].values())
    cpu = str(_same(campaigns, "cpu"))
    cpu = cpu.replace("(R)", "").split(" CPU")[0].replace("  ", " ")
    return {
        "PipelineMinutesMin": _fixed(min(minutes), 0),
        "PipelineMinutesMax": _fixed(max(minutes), 0),
        "PipelinePeakGB": _fixed(max(peaks), 1),
        "PipelineGPU": gpus.pop(),
        "EvalCPU": cpu,
        "EvalThreads": WORDS[int(_same(campaigns, "omp_num_threads"))],
        "EvalConcurrency": WORDS[int(_same(campaigns, "concurrency"))],
        "BudgetMinutes": _fixed(float(_same(campaigns, "time_min")), 0),
        "BudgetGB": _fixed(float(_same(campaigns, "mem_gb")), 0),
    }


def _empty_models(reduced: pd.DataFrame) -> dict[str, str]:
    """The Sepsis Forecast nets that accept nothing, and Sepsis without them.

    Empty models stay pooled in Table 2 and are reported apart here: the windows, and
    Forecast against Re-mined fitness on the other windows at tPAT 0.
    """
    empty = reduced[
        (reduced["evaluator"] == "alignment")
        & (reduced["guard_reason"] == "empty model")
    ]
    if set(empty["dataset"]) - {"sepsis"} or set(empty["model_type"]) - {"forecast"}:
        raise ResultsError("an empty model outside Sepsis Forecast: revise the text")
    windows = {t: sorted(empty.loc[empty["tpat"] == t, "window"]) for t in (0.0, -1.0)}
    if windows[0.0] != windows[-1.0] or windows[0.0] != list(
        range(windows[0.0][0], windows[0.0][-1] + 1)
    ):
        raise ResultsError("the empty Sepsis windows differ by tPAT or have gaps")
    sepsis = reduced[(reduced["dataset"] == "sepsis") & (reduced["tpat"] == 0.0)]
    kept = sepsis[~sepsis["window"].isin(windows[0.0])]
    numbers = {
        "SepsisEmptyForecast": _int(len(windows[0.0])),
        "SepsisEmptyFirst": str(windows[0.0][0]),
        "SepsisEmptyLast": str(windows[0.0][-1]),
        "SepsisNonEmptyWindows": _int(kept["window"].nunique()),
    }
    for family, name in (("forecast", "Forecast"), ("re_mined", "ReMined")):
        for evaluator, metric in (("token_etc", "Token"), ("alignment", "Align")):
            units = kept[
                (kept["model_type"] == family) & (kept["evaluator"] == evaluator)
            ]
            numbers[f"SepsisNonEmpty{name}{metric}Fit"] = _fixed(
                pool(units, "log_fitness").mean, 2
            )
    return numbers


def _er_coverage(reduced: pd.DataFrame) -> dict[str, str]:
    """ER coverage of Forecast where the unfiltered nets cover under half the windows."""
    er = reduced[(reduced["evaluator"] == "er") & (reduced["model_type"] == "forecast")]
    poor, missing, windows = 0, 0, 0
    for key, _ in LOGS:
        at = {t: er[(er["dataset"] == key) & (er["tpat"] == t)] for t in (0.0, -1.0)}
        unfiltered = pool(at[-1.0], "er")
        if unfiltered.n_values < unfiltered.n_units / 2:
            filtered = pool(at[0.0], "er")
            poor += 1
            missing += filtered.n_units - filtered.n_values
            windows += filtered.n_units
    return {
        "ERPoorLogs": WORDS[poor],
        "ERPoorWindows": _int(windows),
        "ERMissingForecastTpatZero": WORDS[missing],
    }


def section5_numbers(
    results: Path, manifest: dict, reduced: pd.DataFrame, unreduced: pd.DataFrame
) -> dict[str, str]:
    numbers = {}
    grid = unreduced[unreduced["cohort"] == "table2"]
    numbers["OverBudgetUnreduced"] = _fixed(100 * over_budget_share(grid), 1)
    numbers["OverBudgetReduced"] = _fixed(100 * over_budget_share(reduced), 1)
    for tpat, name in ((0.0, "TpatZero"), (-1.0, "TpatMinusOne")):
        numbers[f"OverBudgetUnreduced{name}"] = _fixed(
            100 * over_budget_share(grid[grid["tpat"] == tpat]), 1
        )
        numbers[f"OverBudgetReduced{name}"] = _fixed(
            100 * over_budget_share(reduced[reduced["tpat"] == tpat]), 1
        )
    numbers["OomShareOfOverBudgetReduced"] = _fixed(
        100
        * (reduced["guard_status"] == "oom").sum()
        / reduced["guard_status"].isin(("oom", "timeout")).sum(),
        0,
    )

    effect = reduction_effect(grid.drop(columns="cohort"), reduced)
    if effect.n_lost:
        raise ResultsError(
            f"τ-reduction lost {effect.n_lost} units that finished unreduced"
        )
    fitness = effect.invariance[("alignment", "log_fitness")]
    exponent = int(np.floor(np.log10(fitness.max_abs_diff)))
    numbers.update(
        {
            "OverBudgetUnreducedUnits": _int(effect.n_over_budget_unreduced),
            "RescuedUnits": _int(effect.n_rescued),
            "RescuedShare": _fixed(100 * effect.share_rescued, 0),
            "TimeSavedAll": _fixed(100 * effect.time_saved_all, 0),
            "TimeSavedBothOk": _fixed(100 * effect.time_saved_both_ok, 0),
            "AlignFitnessMaxDiff": (
                f"${fitness.max_abs_diff / 10**exponent:.1f}\\times10^{{{exponent}}}$"
            ),
            "AlignPrecisionEqualShare": _fixed(
                100 * effect.invariance[("alignment", "precision")].share_equal, 1
            ),
            "EREqualShare": _fixed(
                100 * effect.invariance[("er", "er")].share_equal, 1
            ),
        }
    )

    replay = pd.concat(
        [
            unreduced[unreduced["cohort"] == "replay_modes"],
            reduced[
                (reduced["dataset"] == "bpi2017")
                & reduced["evaluator"].isin(("token_etc", "alignment"))
            ],
        ]
    )
    replay = replay[replay["reduce_level"] == "tau"]
    for tpat, suffix in ((0.0, "TpatZero"), (-1.0, "TpatMinusOne")):
        ranges = replay_mode_ranges(replay[replay["tpat"] == tpat]).set_index(
            ["evaluator", "mode"]
        )
        if ranges["n_not_ok"].any():
            raise ResultsError("a BPI2017 replay-mode unit did not finish")
        for (evaluator, mode), name in (
            (("token_etc", "window_raw"), "RawToken"),
            (("token_etc", "window_no_remain"), "NoRemainToken"),
            (("token_etc", "case_complete"), "CaseToken"),
            (("alignment", "window_raw"), "RawAlign"),
            (("alignment", "case_complete"), "CaseAlign"),
        ):
            r = ranges.loc[(evaluator, mode)]
            numbers[f"Replay{name}Fit{suffix}"] = _range(r["fit_min"], r["fit_max"], 2)
            numbers[f"Replay{name}Perf{suffix}"] = _range(
                r["perf_min"], r["perf_max"], 1
            )

    audit = pd.read_csv(results / "unfinishable_nets.csv")
    numbers["AuditedNets"] = _int(audit["nets"].sum())
    numbers["UnfinishableNets"] = _int(audit["unfinishable"].sum())

    numbers.update(_empty_models(reduced))
    numbers.update(_er_coverage(reduced))

    choice = {"topology": {}, "frequencies": 0, "nets": 0}
    for tag, run in manifest["pipeline"].items():
        topology = 0
        for key, _ in LOGS:
            path = results / run["dir"] / _dataset(key)[0] / "median_choice.json"
            record = json.loads(path.read_text())
            if record["n_nets_not_reproduced"]:
                raise ResultsError(f"{path}: saved nets do not reproduce")
            topology += record["n_nets_topology_differ"]
            if tag == "+0.0":
                choice["frequencies"] += record["n_nets_frequencies_differ"]
                choice["nets"] += record["n_windows"]
        choice["topology"][tag] = topology
    numbers.update(
        {
            "MedianChoiceTopologyTpatZero": _int(choice["topology"]["+0.0"]),
            "MedianChoiceTopologyTpatMinusOne": _int(choice["topology"]["-1.0"]),
            "MedianChoiceFrequencies": _int(choice["frequencies"]),
            "ForecastNetsPerTpat": _int(choice["nets"]),
        }
    )
    return numbers


def history_numbers(reduced: pd.DataFrame, history: pd.DataFrame) -> dict[str, str]:
    """Persistence and Expanding against the families they derive from and Forecast.

    Precision is Align-ETC precision pooled per log. The F1 counts are over the
    windows where Forecast and the history family both have an alignment F1 at tPAT 0.
    """
    if (history["guard_reason"] == "empty model").any():
        raise ResultsError("a history net is empty: revise the text")
    units = pd.concat([reduced[reduced["tpat"] == 0.0], history])
    aln = units[units["evaluator"] == "alignment"]

    def precision(key: str, family: str) -> float:
        cell = aln[(aln["dataset"] == key) & (aln["model_type"] == family)]
        return pool(cell, "precision").mean

    keys = [key for key, _ in LOGS]
    gap = max(abs(precision(k, "expanding") - precision(k, "static")) for k in keys)
    less = sum(precision(k, "persistence") < precision(k, "direct_mined") for k in keys)
    numbers = {
        "ExpandingStaticPrecisionMaxDiff": _fixed(gap, 3),
        "PersistenceLessPreciseLogs": WORDS[less],
    }
    ok = aln[aln["guard_status"] == "ok"]
    f1 = ok.pivot_table(index=["dataset", "window"], columns="model_type", values="f1")
    for family in HISTORY_FAMILIES:
        both = f1[["forecast", family]].dropna()
        name = FAMILY_LABELS[family]
        numbers[f"ForecastOver{name}Wins"] = _int(
            (both["forecast"] > both[family]).sum()
        )
        numbers[f"ForecastOver{name}Windows"] = _int(len(both))
    return numbers


def _numbers_tex(numbers: dict[str, str]) -> str:
    lines = [
        "% Generated by scripts/paper.py build from the results directory. Do not edit.",
        "% Each macro is one number of the prose; the tables and Figure 2 are",
        "% generated files of their own.",
    ]
    for name, value in numbers.items():
        if not name.isalpha():
            raise ValueError(f"macro name {name!r} must be letters only")
        lines.append(f"\\newcommand{{\\pmf{name}}}{{{value}}}")
    return "\n".join(lines) + "\n"


def build_into(
    results: Path, out: Path, highlight_against: Path | None = None
) -> dict[str, str]:
    """Write every generated file into ``out``; return the prose numbers.

    With ``highlight_against``, the submitted manuscript's directory, every table value
    that differs from the submitted one is wrapped in ``\\rev``.
    """
    manifest = read_manifest(results)
    old1, old2 = (
        submitted_rows(highlight_against) if highlight_against else (None, None)
    )
    reduced = read_campaign(results / manifest["campaigns"]["reduced"]["dir"])
    unreduced = read_campaign(results / manifest["campaigns"]["unreduced"]["dir"])
    history = None
    if "history" in manifest["campaigns"]:
        history = read_campaign(results / manifest["campaigns"]["history"]["dir"])
    for campaign in (reduced, history):
        if campaign is not None and set(campaign["reduce_level"]) != {"tau"}:
            raise ResultsError("a campaign of Table 2 holds unreduced units")
    out.mkdir(parents=True, exist_ok=True)

    rows1, numbers = table1(results, manifest, old1)
    (out / "tab-logs-rows.tex").write_text(rows1)
    (out / "tab-quality-rows-structure.tex").write_text(
        table2(results, manifest, reduced, old2, history)
    )

    figure = manifest["figure2"]
    counts = draw(json.loads((results / figure["file"]).read_text()), out, "fig-w1")
    numbers.update(
        {
            "FigExpandedPlaces": str(counts["expanded_places"]),
            "FigExpandedTransitions": str(counts["expanded_transitions"]),
            "FigExpandedSilent": str(counts["expanded_silent"]),
            "FigExpandedArcs": str(counts["expanded_arcs"]),
            "FigReducedPlaces": str(counts["reduced_places"]),
            "FigReducedTransitions": str(counts["reduced_transitions"]),
            "FigReducedSilent": str(counts["reduced_silent"]),
            "FigReducedArcs": str(counts["reduced_arcs"]),
        }
    )
    numbers.update(setup_numbers(results, manifest))
    numbers.update(section5_numbers(results, manifest, reduced, unreduced))
    if history is not None:
        numbers.update(history_numbers(reduced, history))
    (out / "numbers.tex").write_text(_numbers_tex(numbers))
    return numbers


GENERATED_TEXT = (
    "numbers.tex",
    "tab-logs-rows.tex",
    "tab-quality-rows-structure.tex",
    "fig-w1-expanded.dot",
    "fig-w1-reduced.dot",
)


def build(args: argparse.Namespace) -> int:
    numbers = build_into(args.results, args.out, args.highlight_against)
    width = max(map(len, numbers))
    for name, value in numbers.items():
        print(f"  \\pmf{name:<{width}}  {value}")
    print(f"wrote {', '.join(GENERATED_TEXT)} and the two PDFs -> {args.out}")
    return 0


def check(args: argparse.Namespace) -> int:
    """Rebuild and compare every generated text file with the manuscript's copy."""
    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp)
        numbers = build_into(args.results, fresh, args.highlight_against)
        for name in GENERATED_TEXT:
            current = args.manuscript / name
            if not current.exists():
                print(f"MISSING {name}")
                failed = True
                continue
            new, old = (fresh / name).read_text(), current.read_text()
            if new != old:
                failed = True
                print(f"DIFFERS {name}")
                sys.stdout.writelines(
                    difflib.unified_diff(
                        old.splitlines(True), new.splitlines(True), name, "rebuilt", n=0
                    )
                )
            else:
                print(f"ok      {name}")
    main_tex = (args.manuscript / "main.tex").read_text()
    for needed in ("\\input{numbers}", "tab-logs-rows", "tab-quality-rows-structure"):
        if needed not in main_tex:
            print(f"main.tex does not use {needed}")
            failed = True
    unused = [n for n in numbers if f"\\pmf{n}" not in main_tex]
    if unused:
        print(f"note: {len(unused)} macros not used in main.tex: {', '.join(unused)}")
    print("check FAILED" if failed else "check passed")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("collect", help="write a results directory and its manifest")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reduced-campaign", type=Path, required=True)
    p.add_argument("--unreduced-campaign", type=Path, required=True)
    p.add_argument(
        "--pipeline-roots",
        type=Path,
        nargs="+",
        required=True,
        help="pipeline campaign roots (PMF_OUTPUT_BASE), one per tPAT",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="directory holding the processed logs the campaigns evaluated",
    )
    p.add_argument(
        "--history-campaign",
        type=Path,
        default=None,
        help="gated campaign of Persistence and Expanding (with --history-root)",
    )
    p.add_argument(
        "--history-root",
        type=Path,
        default=None,
        help="root the history models were built into (with --history-campaign)",
    )

    highlight = {
        "type": Path,
        "default": None,
        "help": "submitted manuscript directory: mark table values that changed",
    }
    p = sub.add_parser("build", help="write the manuscript's generated files")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--highlight-against", **highlight)

    p = sub.add_parser("check", help="rebuild and compare with a manuscript")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--manuscript", type=Path, required=True)
    p.add_argument("--highlight-against", **highlight)

    args = parser.parse_args()
    return {"collect": collect, "build": build, "check": check}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
