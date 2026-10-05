"""What produced a set of results: the code, the environment and the machine.

:func:`code_fingerprint` names the code even when it is not committed: the commit, plus a
hash of the diff against it and of the untracked files under ``src/`` and ``scripts/``.
Outside a git checkout, for example in a source tarball, it hashes the files themselves.
"""

from __future__ import annotations

import hashlib
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

#: The directories whose files are the code.
CODE_DIRS = ("src", "scripts")


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout


def _code_files(root: Path) -> list[Path]:
    return sorted(
        p
        for d in CODE_DIRS
        for p in (root / d).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    )


def code_fingerprint(root: Path) -> dict[str, str | None]:
    """``git_commit`` and ``git_diff_sha256`` of the code under ``root``.

    The diff hash covers the tracked changes against the commit and the untracked,
    not ignored, files under :data:`CODE_DIRS`. Outside git, ``git_commit`` is None and
    the hash covers every file under those directories.
    """
    digest = hashlib.sha256()
    try:
        commit: str | None = _git(root, "rev-parse", "HEAD").decode().strip()
        digest.update(_git(root, "diff", "HEAD", "--binary", "--", *CODE_DIRS))
        untracked = _git(
            root, "ls-files", "--others", "--exclude-standard", "-z", "--", *CODE_DIRS
        )
        files = [root / name for name in sorted(untracked.decode().split("\0")) if name]
    except (OSError, subprocess.CalledProcessError):
        commit, files = None, _code_files(root)
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return {"git_commit": commit, "git_diff_sha256": digest.hexdigest()}


def file_sha256(path: Path) -> str | None:
    """SHA-256 of a file, or None if it does not exist."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def _cpu() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            return (
                subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                or platform.machine()
            )
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


def machine() -> dict[str, str | float]:
    """Host name, CPU model, RAM in GiB and operating system."""
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    return {
        "host": socket.gethostname(),
        "cpu": _cpu(),
        "ram_gb": round(ram, 1),
        "os": platform.platform(),
    }


def versions() -> dict[str, str]:
    """Python and pm4py versions."""
    from importlib.metadata import version

    return {"python": platform.python_version(), "pm4py": version("pm4py")}
