#!/usr/bin/env bash
# Run the pipeline for every log on one machine, without a scheduler.
#
# Usage, from the project root (inside tmux, see "Detached runs"):
#   PMF_OUTPUT_BASE=outputs_tpatm1 PMF_HYDRA_OVERRIDES="experiment.tPAT=-1.0" \
#     bash scripts/local/run_campaign.sh                          # all four logs
#   ... bash scripts/local/run_campaign.sh sepsis bpi2017         # a subset, in order
#   ... DRY_RUN=1 bash scripts/local/run_campaign.sh              # print, run nothing
#   ... DEVICE=cpu bash scripts/local/run_campaign.sh             # override the torch device
#
# PMF_OUTPUT_BASE names the campaign root and PMF_HYDRA_OVERRIDES is passed on to
# Hydra. The nets are evaluated afterwards, one guarded unit at a time, with
# scripts/evaluate.py.
#
# Cold runs only:
# Every campaign recomputes relations, bindings and forecasts from the log. An existing
# output directory is refused, since writing into it would look like a cold run.
#
# Detached runs:
# On machines where systemd-logind ends a user's processes when the SSH session closes,
# run `loginctl enable-linger` once and launch inside tmux. The script checks both and
# warns.
#
# Determinism:
# OMP and MKL use 8 threads because the thread count changes the floating-point
# reduction order. Results are byte-reproducible on the same device; the
# manifest records the device.

set -uo pipefail

DEVICE="${DEVICE:-cuda}"
DRY_RUN="${DRY_RUN:-0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PMF_ROOT sets the tree this runs against. data/, the campaign root and logs/ are
# relative to it, so tests can use a throwaway tree.
ROOT="${PMF_ROOT:-$(cd "$HERE/../.." && pwd)}"
cd "$ROOT" || { echo "run_campaign.sh: cannot cd to '$ROOT'" >&2; exit 2; }

die() { echo "run_campaign.sh: $*" >&2; exit 2; }

# The campaign root is never defaulted.
if [ -z "${PMF_OUTPUT_BASE:-}" ]; then
    cat >&2 <<'MSG'
run_campaign.sh: PMF_OUTPUT_BASE is unset: refusing to run.

It names the output tree this campaign owns, and defaulting it here is the one
mistake that cannot be caught later: the run would land in `outputs/`, on top of
whatever another campaign is still reading, and every affected row would be
silently mislabelled rather than missing. Name the root:

    PMF_OUTPUT_BASE=outputs_tpatm1 PMF_HYDRA_OVERRIDES="experiment.tPAT=-1.0" \
      bash scripts/local/run_campaign.sh

Add DRY_RUN=1 to see the whole run without starting it.
MSG
    exit 2
fi
case "$PMF_OUTPUT_BASE" in
    *[!A-Za-z0-9._/-]*)
        die "PMF_OUTPUT_BASE='$PMF_OUTPUT_BASE': use only letters, digits, . _ - /" ;;
esac
export PMF_OUTPUT_BASE

# What the pipeline runs are told.
overrides="${PMF_HYDRA_OVERRIDES:-}"
read -r -a HYDRA_ARGS <<< "$overrides"

if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    # Smallest log first, so a machine that cannot cope fails early.
    DATASETS=(sepsis bpi2017 hospital_billing bpi2019)
fi

# A dataset's log path and output name both come from its Hydra config. They differ for
# bpi2019, which reads data/BPI2019_1.xes but writes BPI2019_pmf.
dataset_field() {
    local conf="conf/dataset/$1.yaml"
    [ -f "$conf" ] || die "unknown dataset '$1': no $conf"
    local value
    value="$(sed -n "s/^$2:[[:space:]]*//p" "$conf" | head -1 \
             | tr -d '\r' | sed 's/[[:space:]]*$//')"
    [ -n "$value" ] || die "$conf has no '$2:' field"
    printf '%s\n' "$value"
}

# The processed log the pipeline reads (`dataset.log_path`).
dataset_log() { dataset_field "$1" log_path; }

# The output directory of this dataset's run, <root>/<dataset.name>_pmf, as in
# src/pmf_concurrency/paths.py.
dataset_dir() { dataset_field "$1" name; }

for ds in "${DATASETS[@]}"; do dataset_dir "$ds" >/dev/null; dataset_log "$ds" >/dev/null; done

mkdir -p logs

DRY_EXIT=""

# Fixed thread count (see "Determinism"); do not derive it from nproc.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Run offline when the pinned Chronos-2 revision is already cached.
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
revision="$(sed -n 's/^CHRONOS_REVISION = "\(.*\)"$/\1/p' src/pmf_concurrency/forecasting.py 2>/dev/null)"
if [ -n "$revision" ] \
   && [ -d "$hf_home/hub/models--amazon--chronos-2/snapshots/$revision" ]; then
    export HF_HUB_OFFLINE=1
fi

warn=""
if [ -z "${TMUX:-}" ] && [ "$DRY_RUN" = 0 ]; then
    warn="not running inside tmux: an SSH disconnect may kill this run"
fi
if command -v loginctl >/dev/null 2>&1 \
   && [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" = "no" ]; then
    warn="${warn:+$warn; }systemd lingering is off: run \`loginctl enable-linger\`"
fi

# The manifest.
# One JSON per campaign root: what ran, where, from which commit, and how long each
# dataset took. Written as the run goes, so a killed run keeps it.
git_commit="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
git_dirty=false
git diff --quiet HEAD 2>/dev/null || git_dirty=true
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "campaign root: PMF_OUTPUT_BASE=$PMF_OUTPUT_BASE"
echo "overrides:     ${overrides:-none}"
echo "device:        $DEVICE${gpu_name:+ ($gpu_name)}"
echo "host:          $(hostname)"
echo "commit:        $git_commit$([ "$git_dirty" = true ] && echo ' (dirty)')"
echo "threads:       OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS"
echo "datasets:      ${DATASETS[*]}"
[ -n "$warn" ] && echo "WARNING:       $warn"
echo

if [ "$DRY_RUN" != 0 ]; then
    echo "DRY RUN: nothing runs. Per dataset:"
    for ds in "${DATASETS[@]}"; do
        echo "  uv run python -m pmf_concurrency.pipeline dataset=$ds" \
             "experiment.device=$DEVICE${overrides:+ $overrides}"
    done
    echo
    echo "Re-run without DRY_RUN=1 to start."
    DRY_EXIT=0
fi

# Preconditions.
command -v uv >/dev/null || die "uv is not on PATH"

for ds in "${DATASETS[@]}"; do
    log="$(dataset_log "$ds")"
    [ -f "$log" ] || die "$log is missing: run scripts/preprocess.py first"
    out="$PMF_OUTPUT_BASE/$(dataset_dir "$ds")_pmf"
    if [ -d "$out" ]; then
        die "$out already exists: refusing to run.

This campaign runs cold, so a pre-existing output directory is ambiguous: the
pipeline would write into it and the result would look identical to a cold run.
Move it aside or choose another PMF_OUTPUT_BASE."
    fi
done

[ -n "$DRY_EXIT" ] && exit "$DRY_EXIT"

mkdir -p "$PMF_OUTPUT_BASE"
manifest="$PMF_OUTPUT_BASE/campaign_manifest.json"

# python rather than jq, which is not always installed.
manifest_write() {
    MANIFEST="$manifest" python3 - "$@" <<'PY'
import json, os, sys
path = os.environ["MANIFEST"]
patch = json.loads(sys.argv[1])
try:
    with open(path) as fh:
        doc = json.load(fh)
except (FileNotFoundError, json.JSONDecodeError):
    doc = {}
runs = doc.pop("runs", [])
doc.update(patch.pop("_top", {}))
runs.extend(patch.pop("runs", []))
doc["runs"] = runs
with open(path, "w") as fh:
    json.dump(doc, fh, indent=2)
PY
}

manifest_write "$(cat <<JSON
{"_top": {
  "campaign_root": "$PMF_OUTPUT_BASE",
  "hydra_overrides": "$overrides",
  "launcher": "scripts/local/run_campaign.sh",
  "host": "$(hostname)",
  "device": "$DEVICE",
  "gpu": "${gpu_name:-none}",
  "git_commit": "$git_commit",
  "git_dirty": $git_dirty,
  "omp_num_threads": $OMP_NUM_THREADS,
  "datasets_requested": [$(printf '"%s",' "${DATASETS[@]}" | sed 's/,$//')],
  "started_utc": "$started"
}}
JSON
)"

status=0
for ds in "${DATASETS[@]}"; do
    dir="$(dataset_dir "$ds")"
    log="logs/local-pipe-$ds-$(date -u +%Y%m%dT%H%M%SZ).out"
    t0=$(date +%s)
    echo "=== $(date -u +%H:%M:%SZ) $ds -> $PMF_OUTPUT_BASE/${dir}_pmf  (log: $log)"

    # /usr/bin/time -v reports peak memory. Without GNU time the run goes on unmeasured.
    timer=()
    rss_file="$(mktemp)"
    if /usr/bin/time -v true >/dev/null 2>&1; then
        timer=(/usr/bin/time -v -o "$rss_file")
    fi

    ${timer[@]+"${timer[@]}"} uv run python -m pmf_concurrency.pipeline \
        dataset="$ds" \
        experiment.device="$DEVICE" \
        ${HYDRA_ARGS[@]+"${HYDRA_ARGS[@]}"} >"$log" 2>&1
    rc=$?
    t1=$(date +%s)

    peak_kb=$(awk '/Maximum resident set size/ {print $NF}' "$rss_file" 2>/dev/null)
    rm -f "$rss_file"
    peak_gb="null"
    [ -n "${peak_kb:-}" ] && peak_gb=$(awk -v k="$peak_kb" 'BEGIN{printf "%.1f", k/1048576}')

    metrics="$PMF_OUTPUT_BASE/${dir}_pmf/structural_metrics.csv"
    rows=0
    [ -f "$metrics" ] && rows=$(( $(wc -l < "$metrics") - 1 ))

    manifest_write "$(cat <<JSON
{"runs": [{
  "dataset": "$ds",
  "output_dir": "$PMF_OUTPUT_BASE/${dir}_pmf",
  "log": "$log",
  "exit_code": $rc,
  "wall_clock_sec": $((t1 - t0)),
  "peak_rss_gb": $peak_gb,
  "structural_metrics_rows": $rows,
  "finished_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}]}
JSON
)"

    if [ "$rc" -ne 0 ]; then
        echo "    FAILED (exit $rc) after $((t1 - t0))s: see $log"
        # Continue: one log running out of memory does not stop the others, and the
        # manifest records what is missing.
        status=1
    else
        rss_note=""
        [ "$peak_gb" != null ] && rss_note=", peak RSS ${peak_gb} GB"
        echo "    done in $((t1 - t0))s${rss_note}, $rows structural rows"
    fi
done

echo
echo "manifest: $manifest"
echo "outputs:  $ROOT/$PMF_OUTPUT_BASE/"
[ "$status" -ne 0 ] && echo "One or more datasets failed: see the manifest." >&2
exit "$status"
