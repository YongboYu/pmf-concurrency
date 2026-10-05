#!/usr/bin/env bash
# Run a command in its own systemd scope with the memory controller delegated, so the
# budget guard gives every evaluation unit its own cgroup (PMF_GUARD_ENFORCE=cgroup).
#
# Usage, from the project root (inside tmux for long runs):
#   bash scripts/local/delegated_scope.sh uv run python scripts/evaluate.py --dataset sepsis
#   bash scripts/local/delegated_scope.sh uv run pytest tests/test_guard.py
#
# Needs cgroup v2 and a systemd user manager with the memory controller delegated to
# the user (check: cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/cgroup.controllers).
# Swap is off per unit, so the cap is RAM.

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "usage: bash scripts/local/delegated_scope.sh <command> [args...]" >&2
    exit 2
fi
command -v systemd-run >/dev/null || {
    echo "delegated_scope.sh: systemd-run not found; per-unit cgroups need systemd" >&2
    exit 2
}

export PMF_GUARD_ENFORCE=cgroup
exec systemd-run --user --scope --quiet -p Delegate=yes "$@"
