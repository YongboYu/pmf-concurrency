"""Budget guard for single evaluation units.

Token replay can explode on concurrent nets, and one such unit could exhaust a whole
job's memory. :func:`run_guarded` runs each unit in a forked child with an address-space
limit and a wall-clock budget, so the unit ends as a recorded ``oom`` or ``timeout``
instead of killing the job.

Budgets come from ``PMF_GUARD_MEM_GB`` and ``PMF_GUARD_TIME_MIN`` (12 GB, 20 min, the
paper's budget). A budget of 0 disables that limit, and ``PMF_GUARD=0`` runs units
inline, still mapping exceptions to statuses. The memory budget is added to the parent's
current usage.

The child is a fork with the parent's thread settings, and units are pure functions, so
results match an unguarded run bit for bit.

``RLIMIT_AS`` is enforced on Linux only; on macOS only the time budget applies.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

#: Possible ``guard_status`` values of an evaluation unit.
GUARD_STATUSES = ("ok", "oom", "timeout", "crashed", "error")

#: Budget per unit when neither an argument nor the environment sets one.
DEFAULT_MEM_GB = 12.0
DEFAULT_TIME_MIN = 20.0


def _self_vsz_bytes() -> int:
    """Current virtual size of this process (0 where /proc is unavailable)."""
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def _run_inline(fn: Callable[..., Any], args: tuple, kwargs: dict) -> tuple[str, Any]:
    try:
        return "ok", fn(*args, **kwargs)
    except MemoryError:
        return "oom", None
    except Exception as exc:
        return "error", repr(exc)


def _child(
    conn: Any,
    fn: Callable[..., Any],
    args: tuple,
    kwargs: dict,
    mem_limit_bytes: int,
) -> None:
    if mem_limit_bytes:
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes))
        except (ImportError, ValueError, OSError) as exc:
            logger.warning("Guard child could not set RLIMIT_AS: %s", exc)
    try:
        conn.send(_run_inline(fn, args, kwargs))
    finally:
        conn.close()


def resolve_budgets(
    mem_gb: float | None = None, time_min: float | None = None
) -> tuple[float, float]:
    """Budgets of a unit: the arguments, else the environment, else the defaults."""
    if mem_gb is None:
        mem_gb = float(os.environ.get("PMF_GUARD_MEM_GB", DEFAULT_MEM_GB))
    if time_min is None:
        time_min = float(os.environ.get("PMF_GUARD_TIME_MIN", DEFAULT_TIME_MIN))
    return float(mem_gb), float(time_min)


def run_guarded(
    fn: Callable[..., Any],
    *args: Any,
    mem_gb: float | None = None,
    time_min: float | None = None,
    **kwargs: Any,
) -> tuple[str, Any]:
    """Run ``fn(*args, **kwargs)`` in a forked child under memory and time budgets.

    Args:
        fn:       The unit to run; its return value must be picklable.
        mem_gb:   Memory the child may add to the parent's usage. ``None`` reads
                  ``$PMF_GUARD_MEM_GB`` (default 12); 0 disables the limit.
        time_min: Wall-clock budget. ``None`` reads ``$PMF_GUARD_TIME_MIN``
                  (default 20); 0 disables it.

    Returns:
        ``(status, result)``, with ``status`` in :data:`GUARD_STATUSES`.
        ``result`` is the return value for ``ok``, the exception's ``repr`` for
        ``error``, and ``None`` otherwise. ``crashed`` means the child died
        without reporting, for example a native crash.
    """
    mem_gb, time_min = resolve_budgets(mem_gb, time_min)

    if (
        os.environ.get("PMF_GUARD", "1") == "0"
        or "fork" not in mp.get_all_start_methods()
    ):
        return _run_inline(fn, args, kwargs)

    # At fork the child has the parent's address space, so the limit is the parent's
    # size plus the budget.
    mem_limit = int(_self_vsz_bytes() + mem_gb * 2**30) if mem_gb else 0

    # fork, not spawn: arguments are shared copy-on-write without pickling, and the
    # thread settings carry over.
    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(child_conn, fn, args, kwargs, mem_limit))
    proc.start()
    child_conn.close()

    status: str = "crashed"
    result: Any = None
    # poll() also returns on EOF, so a crashed child is noticed at once.
    if parent_conn.poll(time_min * 60 if time_min else None):
        try:
            status, result = parent_conn.recv()
        except EOFError:
            status, result = "crashed", None
    else:
        status = "timeout"
        proc.kill()
    proc.join()
    parent_conn.close()
    return status, result
