"""Provoke each guard status once and check that the guard reports it.

Runs six test units through the same guard ``scripts/evaluate.py`` uses: one that
returns, one that raises, one that allocates past the memory budget, one that sends
itself SIGSEGV, one that exits without a result and one that outlives the time budget.
Prints one row per unit, writes them to ``--out`` if given, and exits 1 if a status is
not the expected one, or if a child process or a unit cgroup is left behind.

Usage, from the project root, on the machine and in the scope of the campaign:
    bash scripts/local/delegated_scope.sh uv run python scripts/guard_probe.py \
        --out guard_probe.csv
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import pandas as pd

from pmf_concurrency.guard import OUTCOME_COLUMNS, enforcement, run_guarded


def _returns() -> int:
    return 42


def _raises() -> None:
    raise ValueError("probe")


def _allocates() -> None:
    hoard = []
    while True:
        hoard.append(bytearray(64 * 2**20))


def _segfaults() -> None:
    os.kill(os.getpid(), signal.SIGSEGV)


def _exits() -> None:
    os._exit(3)


def _sleeps() -> None:
    time.sleep(3600)


#: (unit, expected status, expected failure_category)
PROBES = [
    (_returns, "ok", ""),
    (_raises, "error", "ValueError"),
    (_allocates, "oom", "memory_budget"),
    (_segfaults, "crashed", "signal_SIGSEGV"),
    (_exits, "crashed", "no_result"),
    (_sleeps, "timeout", "time_budget"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mem-gb", type=float, default=1.0)
    parser.add_argument("--time-min", type=float, default=0.1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    enforced = enforcement()
    print(f"enforcement: {enforced.mechanism} (in force: {enforced.in_force})")
    rows = []
    for fn, status, category in PROBES:
        outcome = run_guarded(fn, mem_gb=args.mem_gb, time_min=args.time_min)
        rows.append(
            {
                "probe": fn.__name__.lstrip("_"),
                "expected": f"{status}/{category}",
                **outcome.row(),
                "as_expected": (outcome.status, outcome.failure_category)
                == (status, category),
            }
        )

    table = pd.DataFrame(
        rows, columns=["probe", "expected", *OUTCOME_COLUMNS, "as_expected"]
    )
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(table.to_string(index=False))
    if args.out is not None:
        table.to_csv(args.out, index=False)

    problems = [
        f"{r['probe']}: got {r['guard_status']}" for r in rows if not r["as_expected"]
    ]
    try:
        os.waitpid(-1, os.WNOHANG)
        problems.append("a child process is left")
    except ChildProcessError:
        pass
    if enforced.cgroups is not None:
        left = sorted(p.name for p in enforced.cgroups.root.glob("unit_*"))
        if left:
            problems.append(f"unit cgroups left: {', '.join(left)}")
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("every probe as expected; no child process or unit cgroup left")


if __name__ == "__main__":
    main()
