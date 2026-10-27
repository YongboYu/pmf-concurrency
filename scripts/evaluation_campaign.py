"""Run one evaluation campaign on one machine, and check its gate.

A campaign is one directory. Its units are fixed before anything runs, in
``expected_units.csv``, from the protocol (see :mod:`pmf_concurrency.campaign`):

    grid  write expected_units.csv from the logs' calendar windows
    run   check the code and the models, then evaluate every unit, N at a time
    gate  check the recorded units, manifests and OOM kills against the grid

``run`` refuses uncommitted code (the campaign names its code by commit) and archives
that commit, ``uv.lock`` and the environment in the campaign directory. It checks that
every model the grid needs exists, parses and has the grid's window dates, and stops if
not. It then runs one ``scripts/evaluate.py`` per (log, tPAT, family, evaluator, mode,
level), each in its own delegated scope so every unit gets its own cgroup, at most
``--concurrency`` at once, heaviest first. While it runs it samples ``oom_kill`` from
``/proc/vmstat`` and the free memory every 10 s. A stopped run resumes: every batch
resumes its own directory, and ``campaign_manifest.json`` must not have changed.

Usage, from the project root of the frozen checkout, inside tmux:
    uv run python scripts/evaluation_campaign.py grid --campaign 1 \\
        --out campaigns/c1/expected_units.csv
    uv run python scripts/evaluation_campaign.py run --campaign-dir campaigns/c1 \\
        --pipeline-roots outputs_tpat0 outputs_tpatm1 --concurrency 3
    uv run python scripts/evaluation_campaign.py gate --campaign-dir campaigns/c1

Campaigns 3 and 4 score the history families on the reduced and the unreduced nets, so
their only pipeline root is the one ``pmf_concurrency.history_models`` wrote
(``--pipeline-roots outputs_history_tpat0``).

Outputs in the campaign directory:
    expected_units.csv       the units to record
    campaign_manifest.json   code, environment, machine, budgets, grid and models
    source.tar.gz, uv.lock, environment.json
    eval/<batch>/              each batch's evaluate.py output directory
    logs/<batch>.log, batches.csv the batches' output and exit codes per session
    oom_kill_samples.csv     session, time, MemAvailable, global oom_kill
    kernel_oom_<session>.txt the kernel's OOM reports during that session, if readable
    evaluation.csv           every recorded unit (written by gate)
    evaluation_summary.csv   per log and tPAT, as evaluate.py summarises
    near_budget.csv          finished units above 75 % of the time or memory budget
    gate.json                the gate's verdict and counts
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from _dataset import resolve_dataset
from evaluate import (
    CSV_COLUMNS,
    _sha256,
    failure_counts,
    read_units,
    summarise,
)
from omegaconf import OmegaConf

from pmf_concurrency.campaign import (
    UNIT_KEY,
    check_models,
    expected_units,
    near_budget,
    oom_attribution,
    read_expected,
    reconcile,
    run_tpat,
    windows_of,
)
from pmf_concurrency.data import load_event_log
from pmf_concurrency.guard import GUARD_STATUSES, resolve_budgets
from pmf_concurrency.provenance import code_fingerprint, file_sha256, machine, versions

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPECTED = "expected_units.csv"
MANIFEST = "campaign_manifest.json"
SAMPLES = "oom_kill_samples.csv"

#: Fields every batch's ``evaluation_manifest.json`` shares with the campaign's.
SHARED_KEYS = (
    "git_commit",
    "git_diff_sha256",
    "uv_lock_sha256",
    "python",
    "pm4py",
    "host",
    "cpu",
    "ram_gb",
    "os",
    "concurrency",
    "mem_gb",
    "time_min",
    "omp_num_threads",
    "mkl_num_threads",
    "pmf_guard",
    "expected_sha256",
)

#: Environment variables archived with the campaign.
ENV_PREFIXES = ("PMF_", "OMP_", "MKL_", "HF_", "CUDA_", "PYTHON", "UV_")
ENV_NAMES = ("PATH", "LANG", "LC_ALL", "TZ", "USER", "HOSTNAME")

#: Relative cost, for ordering: the recorded wall times put BPI2019 and Hospital
#: Billing ER far ahead.
DATASET_WEIGHT = {"bpi2019": 4, "hospital_billing": 3, "sepsis": 2, "bpi2017": 1}
EVALUATOR_WEIGHT = {"er": 3, "alignment": 2, "token_align_etc": 1, "token_etc": 1}

#: Resident memory of an ``evaluate.py`` parent holding each log, in GiB: the largest
#: ``base_rss_gb`` of campaign 1 on the office box.
PARENT_RSS_GB = {
    "bpi2019": 4.53,
    "hospital_billing": 2.31,
    "bpi2017": 1.15,
    "sepsis": 0.79,
}

#: Memory kept free for the system and the driver, in GiB.
SYSTEM_RESERVE_GB = 2.0

SAMPLE_EVERY_S = 10.0


def _tag(tpat: float) -> str:
    return f"{tpat:g}".replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class Batch:
    """One ``evaluate.py`` run: the units of one log, tPAT, family, evaluator, mode
    and level."""

    dataset: str
    tpat: float
    model_type: str
    evaluator: str
    mode: str
    reduce_level: str
    units: pd.DataFrame

    @property
    def name(self) -> str:
        return "_".join(
            (
                self.dataset,
                f"tpat{_tag(self.tpat)}",
                self.model_type,
                self.evaluator,
                self.mode,
                self.reduce_level,
            )
        )

    @property
    def n_units(self) -> int:
        return len(self.units)

    def out_dir(self, campaign_dir: Path) -> Path:
        return campaign_dir / "eval" / self.name


def plan_batches(grid: pd.DataFrame) -> list[Batch]:
    """The batches of a grid, heaviest first."""
    keys = ["dataset", "tpat", "model_type", "evaluator", "mode", "reduce_level"]
    batches = [
        Batch(ds, float(tp), mt, ev, mode, level, units.reset_index(drop=True))
        for (ds, tp, mt, ev, mode, level), units in grid.groupby(keys, sort=True)
    ]
    return sorted(
        batches,
        key=lambda j: (
            -DATASET_WEIGHT.get(j.dataset, 0) * EVALUATOR_WEIGHT.get(j.evaluator, 1)
        ),
    )


def batch_command(
    batch: Batch,
    campaign_dir: Path,
    weekly_dir: Path,
    concurrency: int,
    development: bool,
) -> list[str]:
    """The command of a batch: ``evaluate.py`` on its units, in a delegated scope
    (without one under ``development``, where systemd may be missing)."""
    # ER ignores the replay mode; its rows are keyed "window" whatever --mode says.
    mode = "case_complete" if batch.evaluator == "er" else batch.mode
    scope = [] if development else ["bash", "scripts/local/delegated_scope.sh"]
    argv = [
        *scope,
        "uv",
        "run",
        "python",
        "scripts/evaluate.py",
        "--dataset",
        batch.dataset,
        "--weekly-dir",
        str(weekly_dir),
        "--out-dir",
        str(batch.out_dir(campaign_dir)),
        "--expected",
        str(campaign_dir / EXPECTED),
        "--model-types",
        batch.model_type,
        "--evaluators",
        batch.evaluator,
        "--reduce-levels",
        batch.reduce_level,
        "--mode",
        mode,
        "--concurrency",
        str(concurrency),
    ]
    return [*argv, "--development"] if development else argv


def oom_kill_rise(samples: pd.DataFrame) -> int:
    """Rise of the global ``oom_kill`` while the campaign ran, summed over sessions."""
    if samples.empty:
        return 0
    by = samples.groupby("session")["oom_kill"]
    return int((by.last() - by.first()).sum())


def check_campaign_manifest(campaign_dir: Path, manifest: dict) -> None:
    """Write the campaign manifest on the first run; on a resume, stop if it changed."""
    path = campaign_dir / MANIFEST
    if not path.exists():
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        return
    recorded = json.loads(path.read_text())
    changed = sorted(
        k
        for k in manifest.keys() | recorded.keys()
        if manifest.get(k) != recorded.get(k)
    )
    if changed:
        raise SystemExit(
            f"{path}: {', '.join(changed)} changed since the campaign started. "
            "Resume with the same code, machine and settings, or start a new campaign."
        )


# grid


def _experiment() -> Any:
    return OmegaConf.load(PROJECT_ROOT / "conf" / "config.yaml").experiment


def write_grid(
    campaign: int, datasets: list[str], out: Path, windows: list[int] | None = None
) -> pd.DataFrame:
    """Write the campaign's grid; ``windows`` keeps only those, for a pilot batch."""
    exp = _experiment()
    per_log = {}
    for key in datasets:
        log_path, _, _ = resolve_dataset(key)
        per_log[key] = windows_of(
            load_event_log(log_path), float(exp.train_ratio), int(exp.horizon)
        )
        print(f"{key}: {len(per_log[key])} windows")
    grid = expected_units(per_log, campaign)
    if windows:
        grid = grid[grid["window"].isin(windows)]
    if out.exists():
        raise SystemExit(f"{out} exists: a campaign's grid is written once.")
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out, index=False)
    print(f"{len(grid)} units -> {out}")
    print(grid.groupby(["dataset", "cohort"]).size().to_string())
    return grid


# run


def _weekly_dirs(
    grid: pd.DataFrame, roots: list[Path]
) -> dict[tuple[str, float], Path]:
    """(log, tPAT) -> weekly_models, from pipeline roots that each hold one tPAT."""
    dirs: dict[tuple[str, float], Path] = {}
    for key in sorted(set(grid["dataset"])):
        _, out_dir, _ = resolve_dataset(key)
        for root in roots:
            weekly = root / out_dir.name / "weekly_models"
            tpat = run_tpat(weekly)
            if tpat is None:
                continue
            if (key, tpat) in dirs:
                raise SystemExit(f"two pipeline roots hold {key} at tPAT {tpat}")
            dirs[(key, tpat)] = weekly.resolve()
    return dirs


def check_concurrency(
    datasets: list[str], concurrency: int, mem_gb: float, ram_gb: float
) -> None:
    """Stop if ``concurrency`` lanes of the heaviest log, each parent holding the log
    and its unit at the cap, do not fit in RAM with :data:`SYSTEM_RESERVE_GB` left."""
    largest = max(PARENT_RSS_GB.values())
    parent, heaviest = max((PARENT_RSS_GB.get(d, largest), d) for d in datasets)
    need = concurrency * (parent + mem_gb) + SYSTEM_RESERVE_GB
    if need > ram_gb:
        raise SystemExit(
            f"evaluation_campaign.py: {concurrency} lanes of {heaviest} need "
            f"{concurrency} x ({parent} GiB parent + {mem_gb} GiB unit) + "
            f"{SYSTEM_RESERVE_GB} GiB = {need:.1f} GiB, more than {ram_gb} GiB."
        )


def _dirty_code() -> str:
    return subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "src",
            "scripts",
            "conf",
            "uv.lock",
            "pyproject.toml",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def campaign_manifest(
    campaign_dir: Path,
    weekly_dirs: dict[tuple[str, float], Path],
    concurrency: int,
) -> dict:
    mem_gb, time_min = resolve_budgets()
    grid_path = campaign_dir / EXPECTED
    return {
        "expected_sha256": _sha256([grid_path]),
        "models": {
            f"{ds} tPAT {tp}": {
                "weekly_dir": str(w),
                "models_sha256": _sha256(sorted(w.glob("window_*.json"))),
            }
            for (ds, tp), w in sorted(weekly_dirs.items())
        },
        **code_fingerprint(PROJECT_ROOT),
        "uv_lock_sha256": file_sha256(PROJECT_ROOT / "uv.lock"),
        **versions(),
        **machine(),
        "concurrency": concurrency,
        "mem_gb": mem_gb,
        "time_min": time_min,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "pmf_guard": os.environ.get("PMF_GUARD", "1"),
    }


def archive(campaign_dir: Path, commit: str) -> None:
    """The commit's source, ``uv.lock`` and the environment, once per campaign."""
    tarball = campaign_dir / "source.tar.gz"
    if tarball.exists():
        return
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(tarball), commit],
        cwd=PROJECT_ROOT,
        check=True,
    )
    shutil.copy2(PROJECT_ROOT / "uv.lock", campaign_dir / "uv.lock")
    env = {
        k: v
        for k, v in sorted(os.environ.items())
        if k.startswith(ENV_PREFIXES) or k in ENV_NAMES
    }
    (campaign_dir / "environment.json").write_text(json.dumps(env, indent=2) + "\n")


def _vmstat_oom_kill() -> int | None:
    try:
        for line in Path("/proc/vmstat").read_text().splitlines():
            if line.startswith("oom_kill "):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def _mem_available_kb() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


class Sampler:
    """Appends (session, time, MemAvailable, oom_kill) every few seconds."""

    def __init__(self, path: Path, session: int) -> None:
        self.path, self.session = path, session
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def sample(self) -> None:
        oom_kill = _vmstat_oom_kill()
        if oom_kill is None:
            return
        new = not self.path.exists()
        with self.path.open("a") as fh:
            if new:
                fh.write("session,time,mem_available_kb,oom_kill\n")
            fh.write(
                f"{self.session},{time.time():.0f},{_mem_available_kb()},{oom_kill}\n"
            )

    def _loop(self) -> None:
        while not self._stop.wait(SAMPLE_EVERY_S):
            self.sample()

    def __enter__(self) -> Sampler:
        self.sample()
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join()
        self.sample()


def _next_session(campaign_dir: Path) -> int:
    batches = campaign_dir / "batches.csv"
    if not batches.exists():
        return 1
    return int(pd.read_csv(batches)["session"].max()) + 1


def _finished(batch: Batch, campaign_dir: Path) -> bool:
    """Whether every unit of the batch is already recorded, so it need not start."""
    csv = batch.out_dir(campaign_dir) / "evaluation.csv"
    if not csv.exists():
        return False
    try:
        recorded = read_units(csv)
    except ValueError:
        return False
    return bool(reconcile(batch.units, recorded).n_missing == 0)


def run(args: argparse.Namespace) -> int:
    campaign_dir: Path = args.campaign_dir.resolve()
    grid = read_expected(campaign_dir / EXPECTED)
    os.chdir(PROJECT_ROOT)

    dirty = _dirty_code()
    if dirty and not args.development:
        raise SystemExit(
            "evaluation_campaign.py: the code has uncommitted changes, so no commit "
            f"names it:\n{dirty}\nCommit (locally) and start from a clean checkout."
        )
    check_concurrency(
        sorted(set(grid["dataset"])),
        args.concurrency,
        resolve_budgets()[0],
        float(machine()["ram_gb"]),
    )

    weekly_dirs = _weekly_dirs(grid, [Path(r).resolve() for r in args.pipeline_roots])
    problems = check_models(grid, weekly_dirs)
    if problems:
        raise SystemExit(
            "evaluation_campaign.py: the models do not match the expected grid:\n  "
            + "\n  ".join(problems)
        )

    manifest = campaign_manifest(campaign_dir, weekly_dirs, args.concurrency)
    check_campaign_manifest(campaign_dir, manifest)
    if not args.development:
        archive(campaign_dir, str(manifest["git_commit"]))

    batches = [j for j in plan_batches(grid) if not _finished(j, campaign_dir)]
    session = _next_session(campaign_dir)
    logs = campaign_dir / "logs"
    logs.mkdir(exist_ok=True)
    print(
        f"campaign {campaign_dir.name}, session {session}: {len(batches)} batches to run, "
        f"{args.concurrency} at once"
    )

    lock = threading.Lock()

    def launch(batch: Batch) -> None:
        argv = batch_command(
            batch,
            campaign_dir,
            weekly_dirs[(batch.dataset, batch.tpat)],
            args.concurrency,
            args.development,
        )
        t0 = time.time()
        with (logs / f"{batch.name}.log").open("a") as log:
            rc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT).returncode
        with lock:
            new = not (campaign_dir / "batches.csv").exists()
            with (campaign_dir / "batches.csv").open("a") as fh:
                if new:
                    fh.write("session,batch,exit_code,started,finished\n")
                fh.write(f"{session},{batch.name},{rc},{t0:.0f},{time.time():.0f}\n")
            print(f"  {time.strftime('%H:%M:%S')} {batch.name}: exit {rc}", flush=True)

    started = time.time()
    try:
        with (
            Sampler(campaign_dir / SAMPLES, session),
            ThreadPoolExecutor(max_workers=args.concurrency) as pool,
        ):
            list(pool.map(launch, batches))
    finally:
        _kernel_oom_log(campaign_dir / f"kernel_oom_{session}.txt", started)

    report = gate(campaign_dir)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def _kernel_oom_log(path: Path, since: float) -> None:
    """The kernel's OOM reports since ``since``, where the journal is readable."""
    try:
        out = subprocess.run(
            ["journalctl", "-k", "--no-pager", "--since", f"@{since:.0f}"],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return
    lines = [ln for ln in out.splitlines() if "oom" in ln.lower()]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


# gate


def _with_cohort(recorded: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    """``recorded`` with each unit's cohort from the grid (``extra`` if not in it)."""
    keys = list(UNIT_KEY)
    cohorts = expected[[*keys, "cohort"]].drop_duplicates(subset=keys).copy()
    for frame in (cohorts, recorded):
        frame["tpat"] = frame["tpat"].astype(float)
        frame["window"] = frame["window"].astype(int)
    merged = recorded.merge(cohorts, on=keys, how="left")
    merged["cohort"] = merged["cohort"].fillna("extra")
    return merged


def campaign_summary(recorded: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    """``evaluate.summarise`` per cohort, log and tPAT, with ``n_expected`` from the
    grid, so a missing unit lowers every metric's coverage."""
    runs = ["cohort", "dataset", "tpat"]
    cells = ["mode", "evaluator", "reduce_level", "model_type"]
    want = expected.assign(tpat=expected["tpat"].astype(float))
    want = want.groupby([*runs, *cells]).size().rename("n_grid").reset_index()
    parts = [
        summarise(g).assign(cohort=c, dataset=ds, tpat=tp)
        for (c, ds, tp), g in recorded.groupby(runs, sort=True)
    ]
    summary = pd.concat(parts, ignore_index=True).merge(want, on=[*runs, *cells])
    summary["n_expected"] = summary.pop("n_grid")
    return summary[[*runs, *[c for c in summary.columns if c not in runs]]]


def gate(campaign_dir: Path) -> dict:
    """Check a campaign against its grid and write the pooled results.

    Passes when the recorded units equal ``expected_units.csv`` exactly, every batch's
    manifest matches the campaign's and records the per-unit cgroup in force, no unit
    is ``host_oom`` and every global OOM kill is matched by a unit's own.
    """
    expected = read_expected(campaign_dir / EXPECTED)
    campaign = json.loads((campaign_dir / MANIFEST).read_text())

    frames, problems = [], []
    for batch in plan_batches(expected):
        out = batch.out_dir(campaign_dir)
        csv = out / "evaluation.csv"
        if csv.exists():
            try:
                frames.append(read_units(csv))
            except ValueError as exc:
                # A unit recorded twice is counted below; other columns are not rows.
                problems.append(f"{batch.name}: {exc}")
                raw = pd.read_csv(csv)
                if list(raw.columns) == CSV_COLUMNS:
                    frames.append(raw)
        manifest_path = out / "evaluation_manifest.json"
        if not manifest_path.exists():
            problems.append(f"{batch.name}: no evaluation_manifest.json")
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("enforcement") != "cgroup" or not manifest.get(
            "enforcement_in_force"
        ):
            problems.append(
                f"{batch.name}: enforcement {manifest.get('enforcement')}, in force "
                f"{manifest.get('enforcement_in_force')}, not the per-unit cgroup"
            )
        for key in SHARED_KEYS:
            if key in campaign and manifest.get(key) != campaign[key]:
                problems.append(
                    f"{batch.name}: {key} {manifest.get(key)!r}, the campaign's "
                    f"{campaign[key]!r}"
                )
    recorded = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=CSV_COLUMNS)
    )
    recorded = _with_cohort(recorded, expected)

    rec = reconcile(expected, recorded)
    for name, frame in (
        ("missing", rec.missing),
        ("duplicated", rec.duplicated),
        ("extra", rec.extra),
    ):
        path = campaign_dir / f"units_{name}.csv"
        if len(frame):
            frame.to_csv(path, index=False)
        elif path.exists():
            path.unlink()

    samples_path = campaign_dir / SAMPLES
    if samples_path.exists():
        rise = oom_kill_rise(pd.read_csv(samples_path))
    else:
        rise = 0
        problems.append(f"no {SAMPLES}: the machine's OOM kills were not sampled")
    oom = oom_attribution(rise, recorded)

    if frames:
        recorded.to_csv(campaign_dir / "evaluation.csv", index=False)
        campaign_summary(recorded, expected).to_csv(
            campaign_dir / "evaluation_summary.csv", index=False
        )
        failure_counts(recorded).to_csv(
            campaign_dir / "evaluation_failures.csv", index=False
        )
    mem_gb = float(campaign.get("mem_gb", resolve_budgets()[0]))
    time_min = float(campaign.get("time_min", resolve_budgets()[1]))
    near = near_budget(
        recorded,
        mem_gb,
        time_min,
        by=("cohort", "dataset", "tpat", "model_type", "evaluator", "reduce_level"),
    )
    near.to_csv(campaign_dir / "near_budget.csv", index=False)

    counts = recorded["guard_status"].value_counts()
    report = {
        "passed": rec.passed and oom.passed and not problems,
        "n_expected": len(expected),
        "n_recorded": len(recorded),
        "units": {
            "missing": rec.n_missing,
            "duplicated": rec.n_duplicated,
            "extra": rec.n_extra,
        },
        "status_counts": {s: int(counts[s]) for s in GUARD_STATUSES if s in counts},
        "manifest_problems": problems,
        "oom": {
            "global_rise": oom.global_rise,
            "unit_kills": oom.unit_kills,
            "unattributed": oom.unattributed,
            "host_oom_units": oom.n_host_oom,
        },
        "near_budget": {
            c: int(near[c].sum())
            for c in (
                "n_finished",
                "n_over_75pct_time",
                "n_over_75pct_memory",
                "n_near_budget",
            )
        },
    }
    (campaign_dir / "gate.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("grid", help="write expected_units.csv")
    p.add_argument("--campaign", type=int, choices=(1, 2, 3, 4), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["bpi2017", "bpi2019", "hospital_billing", "sepsis"],
    )
    p.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=None,
        help="keep only these windows: a pilot grid to time a batch, not a campaign",
    )

    p = sub.add_parser("run", help="evaluate every unit of the grid")
    p.add_argument("--campaign-dir", type=Path, required=True)
    p.add_argument(
        "--pipeline-roots",
        type=Path,
        nargs="+",
        required=True,
        help="pipeline campaign roots (PMF_OUTPUT_BASE), one per tPAT",
    )
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument(
        "--development",
        action="store_true",
        help="allow uncommitted code and the guard's fallbacks: not a campaign",
    )

    p = sub.add_parser("gate", help="check a campaign against its grid")
    p.add_argument("--campaign-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "grid":
        write_grid(args.campaign, args.datasets, args.out, args.windows)
        return 0
    if args.command == "run":
        return run(args)
    report = gate(args.campaign_dir.resolve())
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
