"""Budget guard for single evaluation units.

Token replay can explode on concurrent nets, and one such unit could exhaust a whole
job's memory. :func:`run_guarded` runs each unit in a forked child under a memory and a
wall-clock budget, and returns a :class:`UnitOutcome`: the status, its reason, and what
the unit used.

Budgets come from ``PMF_GUARD_MEM_GB`` and ``PMF_GUARD_TIME_MIN`` (12 GiB and 20 min,
the paper's budget). Memory figures are in GiB (2**30 bytes), the budget included. A budget
of 0 disables that limit, and ``PMF_GUARD=0`` runs units inline, still mapping
exceptions to statuses.

How the memory budget is enforced (:func:`enforcement`):

* ``cgroup``: each unit runs in its own cgroup with ``memory.max`` set to the budget
  and swap off. The cap is on what the unit's processes fault in; pages the parent
  already had stay charged to the parent. Requested with ``PMF_GUARD_ENFORCE=cgroup``
  inside a delegated systemd scope (``scripts/local/delegated_scope.sh``). If the
  cgroup cannot be set up, the guard stops instead of falling back.
* ``rlimit_as``: on Linux otherwise. ``RLIMIT_AS`` caps the child's address space at
  the parent's virtual size plus the budget. For development only.
* ``time_only``: elsewhere (macOS). Only the time budget applies.

Statuses:

* ``ok``: the unit returned. ``unavailable`` (finished without its main metric) is
  assigned by the evaluator, not here.
* ``timeout``: the time budget ran out and the unit was killed.
* ``oom``: the unit hit its own memory cap: a caught ``MemoryError``, or a kill with
  both ``oom`` and ``oom_kill`` in its cgroup's ``memory.events``.
* ``host_oom``: a kill with ``oom_kill`` but no ``oom``: the global OOM killer, so the
  box ran out of memory. Never counted as over budget.
* ``error``: the unit raised; ``guard_reason`` is the exception's ``repr``.
* ``crashed``: the child died without a result: a signal, at any memory use, or an
  exit code. ``suspected_cause`` may say "near memory limit", but that is not proof.

The parent alone reaps the child, with ``os.wait4``, so the exit status and the child's
peak RSS are known for every unit, including one killed at its deadline. The child is a
fork with the parent's thread settings, and units are pure functions, so results match
an unguarded run bit for bit.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing.connection import Pipe
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Possible ``guard_status`` values of an evaluation unit.
GUARD_STATUSES = ("ok", "unavailable", "timeout", "oom", "host_oom", "error", "crashed")

#: The statuses that mean the unit exceeded its budget.
OVER_BUDGET = ("timeout", "oom")

#: Budget per unit when neither an argument nor the environment sets one.
DEFAULT_MEM_GB = 12.0
DEFAULT_TIME_MIN = 20.0

GIB = 2**30

#: Where the cgroup v2 hierarchy is mounted.
CGROUP_MOUNT = Path("/sys/fs/cgroup")

#: A crash at or above this share of the memory cap is suspected to be memory.
NEAR_LIMIT = 0.9

#: The fields of :meth:`UnitOutcome.row`, in CSV order.
OUTCOME_COLUMNS = (
    "guard_status",
    "guard_reason",
    "failure_category",
    "suspected_cause",
    "exit_code",
    "peak_rss_gb",
    "base_rss_gb",
    "base_vsz_gb",
    "memory_peak_gb",
    "memory_max_gb",
    "memcg_max",
    "memcg_oom",
    "memcg_oom_kill",
    "host_oom_kill_delta",
)

NAN = float("nan")


class GuardSetupError(RuntimeError):
    """The requested enforcement cannot be put in force."""


@dataclass(frozen=True)
class UnitOutcome:
    """What happened to one unit.

    ``exit_code`` is the child's exit code, or minus the signal that ended it, and
    ``None`` for an inline run. ``peak_rss_gb`` is the child's own peak RSS, which
    counts the parent's pages it touched, so it is no measure of the unit's own use;
    ``base_rss_gb`` and ``base_vsz_gb`` are the parent's RSS and virtual size at fork.
    The ``memory_*`` and ``memcg_*`` fields come from the unit's cgroup, and
    ``host_oom_kill_delta`` is the rise of the global ``oom_kill`` counter in
    ``/proc/vmstat`` while the unit ran.
    """

    status: str
    result: Any = None
    reason: str = ""
    failure_category: str = ""
    suspected_cause: str = ""
    exit_code: int | None = None
    peak_rss_gb: float = NAN
    base_rss_gb: float = NAN
    base_vsz_gb: float = NAN
    memory_peak_gb: float = NAN
    memory_max_gb: float = NAN
    memcg_max: int | None = None
    memcg_oom: int | None = None
    memcg_oom_kill: int | None = None
    host_oom_kill_delta: int | None = None

    def row(self) -> dict[str, Any]:
        """The outcome as CSV fields, :data:`OUTCOME_COLUMNS`."""
        fields = {
            "guard_status": self.status,
            "guard_reason": self.reason,
            **{k: getattr(self, k) for k in OUTCOME_COLUMNS[2:]},
        }
        return fields


def resolve_budgets(
    mem_gb: float | None = None, time_min: float | None = None
) -> tuple[float, float]:
    """Budgets of a unit: the arguments, else the environment, else the defaults."""
    if mem_gb is None:
        mem_gb = float(os.environ.get("PMF_GUARD_MEM_GB", DEFAULT_MEM_GB))
    if time_min is None:
        time_min = float(os.environ.get("PMF_GUARD_TIME_MIN", DEFAULT_TIME_MIN))
    return float(mem_gb), float(time_min)


# Enforcement


class UnitCgroups:
    """One cgroup per unit, below the delegated cgroup this process was started in.

    cgroup v2 lets a cgroup with processes of its own have no children with
    controllers, so the processes of the delegated root move into a leaf ``parent/``
    first, and every unit gets a sibling ``unit_<pid>_<n>/``.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.owner_pid = os.getpid()
        self._seq = 0

    @classmethod
    def claim(cls, mount: Path) -> UnitCgroups:
        """Set up the tree below this process's cgroup, or raise GuardSetupError."""
        try:
            line = Path("/proc/self/cgroup").read_text().strip()
            own = mount / line.split("::", 1)[1].lstrip("/")
            root = own.parent if own.name == "parent" else own
            leaf = root / "parent"
            leaf.mkdir(exist_ok=True)
            for pid in (root / "cgroup.procs").read_text().split():
                with contextlib.suppress(ProcessLookupError):  # exited meanwhile
                    (leaf / "cgroup.procs").write_text(pid)
            (root / "cgroup.subtree_control").write_text("+memory")
            enabled = (root / "cgroup.subtree_control").read_text().split()
        except (OSError, IndexError) as exc:
            raise GuardSetupError(
                f"cannot set up per-unit cgroups ({exc}). Start the run in a delegated "
                "scope (scripts/local/delegated_scope.sh) or unset PMF_GUARD_ENFORCE."
            ) from exc
        if "memory" not in enabled:
            raise GuardSetupError(f"{root}: the memory controller is not delegated")
        return cls(root)

    def create(self, mem_bytes: int) -> Path:
        self._seq += 1
        unit = self.root / f"unit_{os.getpid()}_{self._seq}"
        unit.mkdir()
        cap = str(mem_bytes) if mem_bytes else "max"
        (unit / "memory.max").write_text(cap)
        (unit / "memory.swap.max").write_text("0")
        (unit / "memory.oom.group").write_text("1")
        # The kernel rounds the cap down to whole pages.
        in_force = (unit / "memory.max").read_text().strip()
        if cap != "max" and abs(int(in_force) - mem_bytes) >= os.sysconf(
            "SC_PAGE_SIZE"
        ):
            self.remove(unit)
            raise GuardSetupError(f"{unit}: memory.max is {in_force}, not {cap}")
        return unit

    @staticmethod
    def attach(unit: Path, pid: int) -> None:
        (unit / "cgroup.procs").write_text(str(pid))

    @staticmethod
    def kill(unit: Path, pid: int) -> None:
        """Kill every process of the unit, grandchildren too."""
        try:
            (unit / "cgroup.kill").write_text("1")
        except OSError:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)  # kernels before 5.14

    @staticmethod
    def read(unit: Path) -> dict[str, Any]:
        events = dict.fromkeys(("max", "oom", "oom_kill"), 0)
        for line in (unit / "memory.events").read_text().splitlines():
            key, value = line.split()
            if key in events:
                events[key] = int(value)
        cap = (unit / "memory.max").read_text().strip()
        try:
            peak = int((unit / "memory.peak").read_text()) / GIB
        except OSError:
            peak = NAN  # kernels before 5.19
        return {
            "memory_peak_gb": peak,
            "memory_max_gb": math.inf if cap == "max" else int(cap) / GIB,
            "memcg_max": events["max"],
            "memcg_oom": events["oom"],
            "memcg_oom_kill": events["oom_kill"],
        }

    @staticmethod
    def remove(unit: Path) -> None:
        # The cgroup empties a moment after its last process is reaped.
        for _ in range(100):
            try:
                unit.rmdir()
                return
            except OSError:
                time.sleep(0.01)
        logger.warning("Could not remove %s", unit)


@dataclass(frozen=True)
class Enforcement:
    """How the memory budget is enforced, and whether it is.

    ``mechanism`` is ``cgroup``, ``rlimit_as``, ``time_only`` or ``inline``.
    """

    mechanism: str
    in_force: bool
    cgroups: UnitCgroups | None = field(default=None, compare=False, repr=False)


_cgroups: UnitCgroups | None = None


def enforcement() -> Enforcement:
    """The enforcement of this process's units, set up on first use."""
    global _cgroups
    if os.environ.get("PMF_GUARD", "1") == "0" or not hasattr(os, "fork"):
        return Enforcement("inline", in_force=False)
    requested = os.environ.get("PMF_GUARD_ENFORCE", "auto")
    if requested == "cgroup":
        if _cgroups is None or _cgroups.owner_pid != os.getpid():
            _cgroups = UnitCgroups.claim(CGROUP_MOUNT)
        return Enforcement("cgroup", in_force=True, cgroups=_cgroups)
    if requested not in ("auto", "rlimit"):
        raise GuardSetupError(f"PMF_GUARD_ENFORCE={requested!r}: use cgroup or rlimit")
    if sys.platform == "linux":
        return Enforcement("rlimit_as", in_force=True)
    return Enforcement("time_only", in_force=False)


# Measurement


def _proc_status_gb(key: str) -> float:
    """A ``kB`` field of /proc/self/status in GiB (NaN where /proc is unavailable)."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(key + ":"):
                return int(line.split()[1]) * 1024 / GIB
    except OSError:
        pass
    return NAN


def _host_oom_kills() -> int | None:
    try:
        for line in Path("/proc/vmstat").read_text().splitlines():
            if line.startswith("oom_kill "):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def _maxrss_gb(ru_maxrss: int) -> float:
    # kB on Linux, bytes on macOS.
    return ru_maxrss * (1 if sys.platform == "darwin" else 1024) / GIB


# Classification

#: A child's report: (status, result, reason, failure_category).
Message = tuple[str, Any, str, str]


def _run_inline(fn: Callable[..., Any], args: tuple, kwargs: dict) -> Message:
    try:
        return "ok", fn(*args, **kwargs), "", ""
    except MemoryError as exc:
        return "oom", None, repr(exc), "memory_budget"
    except Exception as exc:
        return "error", None, repr(exc), type(exc).__name__


def _exit_text(exit_code: int | None) -> str:
    if exit_code is not None and exit_code < 0:
        return f"signal {signal.Signals(-exit_code).name}"
    return f"exit code {exit_code}"


def classify(
    message: Message | None,
    timed_out: bool,
    exit_code: int | None,
    oom: int | None,
    oom_kill: int | None,
) -> tuple[str, str, str]:
    """``(status, failure_category, reason)`` of a finished child."""
    if timed_out:
        return "timeout", "time_budget", "over the time budget"
    if message is not None:
        status, _result, reason, category = message
        return status, category, reason
    if oom_kill:
        events = f"{_exit_text(exit_code)}, memory.events oom={oom} oom_kill={oom_kill}"
        if oom:
            return "oom", "memory_budget", events
        return "host_oom", "host_memory", events
    if exit_code is not None and exit_code < 0:
        name = signal.Signals(-exit_code).name
        return "crashed", f"signal_{name}", f"signal {name}"
    return "crashed", "no_result", f"exit code {exit_code} without a result"


def _outcome(
    message: Message | None,
    timed_out: bool,
    exit_code: int | None,
    memory: dict[str, Any],
    mem_gb: float = NAN,
) -> UnitOutcome:
    status, category, reason = classify(
        message,
        timed_out,
        exit_code,
        memory.get("memcg_oom"),
        memory.get("memcg_oom_kill"),
    )
    suspected = ""
    if status == "crashed" and exit_code is not None and exit_code < 0:
        peak, cap = memory.get("memory_peak_gb", NAN), memory.get("memory_max_gb", NAN)
        if math.isnan(peak) or math.isnan(cap):
            # Without the cgroup the child's RSS includes the parent's pages.
            peak = memory.get("peak_rss_gb", NAN)
            cap = mem_gb + memory.get("base_rss_gb", 0.0)
        if peak >= NEAR_LIMIT * cap:
            suspected = "near memory limit"
        elif memory.get("host_oom_kill_delta"):
            suspected = "host oom_kill rose"
    return UnitOutcome(
        status=status,
        result=message[1] if message is not None and status == "ok" else None,
        reason=reason,
        failure_category=category,
        suspected_cause=suspected,
        exit_code=exit_code,
        **memory,
    )


# The guard


def run_guarded(
    fn: Callable[..., Any],
    *args: Any,
    mem_gb: float | None = None,
    time_min: float | None = None,
    **kwargs: Any,
) -> UnitOutcome:
    """Run ``fn(*args, **kwargs)`` in a forked child under memory and time budgets.

    Args:
        fn:       The unit to run; its return value must be picklable.
        mem_gb:   Memory budget of the child, in GiB. ``None`` reads
                  ``$PMF_GUARD_MEM_GB`` (default 12); 0 disables the limit.
        time_min: Wall-clock budget. ``None`` reads ``$PMF_GUARD_TIME_MIN``
                  (default 20); 0 disables it.

    Returns:
        The :class:`UnitOutcome`; ``result`` holds the return value of an ``ok`` unit.

    Raises:
        GuardSetupError: the requested enforcement could not be put in force.
    """
    mem_gb, time_min = resolve_budgets(mem_gb, time_min)
    enforced = enforcement()
    if enforced.mechanism == "inline":
        return _outcome(_run_inline(fn, args, kwargs), False, None, {})

    cgroups = enforced.cgroups
    memory: dict[str, Any] = {
        "base_rss_gb": _proc_status_gb("VmRSS"),
        "base_vsz_gb": _proc_status_gb("VmSize"),
    }
    # At fork the child has the parent's address space, so the limit is the parent's
    # size plus the budget.
    rlimit = 0
    if enforced.mechanism == "rlimit_as" and mem_gb:
        rlimit = int(memory["base_vsz_gb"] * GIB + mem_gb * GIB)
    unit = cgroups.create(int(mem_gb * GIB)) if cgroups is not None else None
    oom_kills_before = _host_oom_kills()

    reader, writer = Pipe(duplex=False)
    go_read, go_write = os.pipe()
    try:
        pid = os.fork()
    except OSError:
        for end in (reader, writer):
            end.close()
        os.close(go_read)
        os.close(go_write)
        if unit is not None:
            UnitCgroups.remove(unit)
        raise
    if pid == 0:
        _child(fn, args, kwargs, reader, writer, go_read, go_write, rlimit)

    writer.close()
    os.close(go_read)
    message: Message | None = None
    timed_out = False
    reaped = False
    try:
        # The child waits until it is inside its cgroup.
        try:
            if unit is not None and cgroups is not None:
                cgroups.attach(unit, pid)
            os.write(go_write, b"1")
        except OSError as exc:
            raise GuardSetupError(f"cannot move unit {pid} into {unit}: {exc}") from exc
        finally:
            os.close(go_write)

        # poll() also returns on EOF, so a crashed child is noticed at once.
        if reader.poll(time_min * 60 if time_min else None):
            try:
                message = reader.recv()
            except EOFError:
                message = None
        else:
            timed_out = True
            if unit is not None:
                UnitCgroups.kill(unit, pid)
            else:
                os.kill(pid, signal.SIGKILL)
        _, wait_status, usage = os.wait4(pid, 0)
        reaped = True
    finally:
        reader.close()
        if not reaped:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        if unit is not None:
            # Whatever the child left running dies with the unit.
            try:
                UnitCgroups.kill(unit, pid)
                memory |= UnitCgroups.read(unit)
            finally:
                UnitCgroups.remove(unit)

    oom_kills_after = _host_oom_kills()
    if oom_kills_before is not None and oom_kills_after is not None:
        memory["host_oom_kill_delta"] = oom_kills_after - oom_kills_before
    memory["peak_rss_gb"] = _maxrss_gb(usage.ru_maxrss)
    exit_code = os.waitstatus_to_exitcode(wait_status)
    return _outcome(message, timed_out, exit_code, memory, mem_gb)


def _child(
    fn: Callable[..., Any],
    args: tuple,
    kwargs: dict,
    reader: Any,
    writer: Any,
    go_read: int,
    go_write: int,
    rlimit: int,
) -> None:
    """Run the unit in the forked child and send its report. Never returns."""
    try:
        reader.close()
        os.close(go_write)
        if os.read(go_read, 1) != b"1":
            os._exit(70)  # the parent could not place this child
        os.close(go_read)
        if rlimit:
            import resource

            resource.setrlimit(resource.RLIMIT_AS, (rlimit, rlimit))
        message = _run_inline(fn, args, kwargs)
        try:
            writer.send(message)
        except Exception as exc:  # an unpicklable result
            writer.send(("error", None, repr(exc), type(exc).__name__))
    finally:
        os._exit(0)
