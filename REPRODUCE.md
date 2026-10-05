# Reproducing the results

This page covers rerunning the method from the raw event logs. A map from the paper's
tables and figures to the scripts, and the results themselves, will follow.

## 1. Install

The project uses [uv](https://docs.astral.sh/uv/) on macOS or Linux. On Windows, use WSL
or Docker.

```sh
uv sync
```

Python 3.12 is pinned in `.python-version`, and `uv sync` installs it if needed.
Drawing the models needs [Graphviz](https://graphviz.org/download/) (`dot` on the
`PATH`). Without it the pipeline skips its PDF drawings, and `scripts/render_nets.py`
(section 6) stops.

## 2. Fetch the event logs

```sh
uv run python scripts/fetch_data.py
```

This downloads the four raw logs from Zenodo record
[18327515](https://zenodo.org/records/18327515) (`pmf_data_v1.1.zip`, 58 MB) into
`data/raw/` and checks each against its SHA-256. The logs are not redistributed here.

| Log | File | SHA-256 |
|---|---|---|
| BPI2017 | `BPI2017.xes` | `0cd438617d600e0d7d4c1d666973db090397644a766d48b0187cf6b4eae0688a` |
| BPI2019 | `BPI2019_1.xes` | `7c3668416e786318d9f345e3a614fc0b2cf39cbf523bfb2cd5600e3aae066df2` |
| Sepsis | `Sepsis.xes` | `b501d2063678b9471e3cb1be04d961f924ae5be6ecc8154701bb57a24c7ae0ab` |
| Hospital Billing | `Hospital_Billing.xes` | `b8d7ee7e251223b0d9672102285b29aed64dc5535cc27d4d2ee437ac5be2b5ff` |

## 3. Preprocess

```sh
uv run python scripts/preprocess.py
```

This filters rare variants, adds the start and end markers ▶ and ■, trims both ends of
each log, and writes `data/<File>.xes`, which the pipeline reads.

## 4. Run the pipeline

One run per log and binding-filter setting:

```sh
PMF_OUTPUT_BASE=outputs_tpat0 uv run python -m pmf_concurrency.pipeline \
    dataset=sepsis experiment.device=cuda
PMF_OUTPUT_BASE=outputs_tpatm1 uv run python -m pmf_concurrency.pipeline \
    dataset=sepsis experiment.device=cuda experiment.tPAT=-1.0
```

- `dataset`: `sepsis`, `bpi2017`, `hospital_billing` or `bpi2019`.
- `experiment.tPAT`: the binding filter. `0.0` keeps bindings at or above the mean;
  `-1.0` keeps every binding.
- `experiment.device`: `cuda`, `mps` or `cpu`. Chronos-2 is much faster on a GPU.
- `PMF_OUTPUT_BASE`: the campaign root. A run writes to
  `$PMF_OUTPUT_BASE/<Log>_pmf/`.

A run writes the Chronos-2 forecasts (`forecast_<series>/`), the models of every
evaluation window (`weekly_models/window_<n>_<family>.json`), `structural_metrics.csv`
and `experiment_summary.json`. The output directory must be new or empty.

`scripts/local/run_campaign.sh` runs all four logs in a row and records a manifest:

```sh
PMF_OUTPUT_BASE=outputs_tpatm1 PMF_HYDRA_OVERRIDES="experiment.tPAT=-1.0" \
    bash scripts/local/run_campaign.sh
```

The Chronos-2 weights are pinned to revision `29ec3766d36d6f73f0696f85560a422f50e8498c`
of [`amazon/chronos-2`](https://huggingface.co/amazon/chronos-2) and downloaded on first
use. Runs are byte-reproducible on the same device with OMP and MKL at 8 threads, the
default unless `OMP_NUM_THREADS` or `MKL_NUM_THREADS` is set.

## 5. Evaluate

```sh
PMF_OUTPUT_BASE=outputs_tpat0 bash scripts/local/delegated_scope.sh \
    uv run python scripts/evaluate.py --dataset sepsis \
    --reduce-levels tau --evaluators token_etc alignment er
```

A unit is one window, family, reduction level and evaluator. Each unit runs in a child
process with a budget of 12 GiB and 20 minutes (`--mem-gb` and `--time-min`). The memory
budget is enforced by a cgroup per unit: `memory.max` is the budget and swap is off, so
it caps the memory the unit itself adds. `scripts/local/delegated_scope.sh` starts the
run in a systemd scope that allows this (Linux with cgroup v2 and systemd). Without it
`evaluate.py` refuses to start; `--development` runs anyway, with `RLIMIT_AS` on Linux
or no memory limit elsewhere, and the manifest records which.

Every unit gets a status: `ok`; `unavailable` (finished, but its main metric, F1 or ER,
is missing); `timeout` or `oom` (over budget); `host_oom` (the machine ran out of
memory, so the run is invalid); `error` (an exception); or `crashed` (a signal or an exit
without a result). `failure_category` and `guard_reason` say why, and every row records
the child's exit code, its peak RSS and, under the cgroup, its `memory.peak` and
`memory.events` counts. Without the cgroup, peak RSS includes pages shared with the
parent and is only a rough guide. To check the budget enforcement on a machine, run
`bash scripts/local/delegated_scope.sh uv run python scripts/guard_probe.py`, which
provokes each status once.

Results are appended to `$PMF_OUTPUT_BASE/<Log>_pmf/evaluation.csv` as each unit
finishes, so a stopped run resumes where it left off. `evaluation_manifest.json` records
the log, the models, the budgets, the thread counts, the code (commit and a hash of
uncommitted changes), `uv.lock`, the Python and pm4py versions, the machine, the
concurrency and how the budget is enforced, and a resume with any of them changed stops. ER
always scores the events inside the window, so its rows have mode `window`.

`evaluation_summary.csv` gives, for each metric, its mean, its standard deviation and
`n_<metric>`, the number of units with a finite value out of `n_expected`. The status
counts (`n_ok`, `n_unavailable`, `n_timeout`, ...) and `n_over_budget` (`timeout` and
`oom`) are diagnostics, and `evaluation_failures.csv` counts the units per
`failure_category`. A model that accepts no trace is scored with fitness 0 and no
precision or F1, and counted in `n_empty_model`. `pct_events_unseen` is the share of
events whose activity the model lacks; such a trace never counts as fitting.

To use more cores, run up to three processes at a time, each in its own scope, on its
own windows and output directory, with `--concurrency 3`, and concatenate their
`evaluation.csv` files afterwards. Under the cgroup, three units can use 3 × 12 GiB
(38.7 GB) at most, plus their parents; check that this leaves room on your machine.

```sh
PMF_OUTPUT_BASE=outputs_tpat0 bash scripts/local/delegated_scope.sh \
    uv run python scripts/evaluate.py --dataset sepsis --concurrency 3 \
    --reduce-levels tau --evaluators token_etc --windows 1 2 3 4 --out-dir eval/part1
```

## 6. Draw the models

```sh
uv run python scripts/render_nets.py --output-base outputs_tpat0 --out-dir drawings/tpat0
uv run python scripts/render_nets.py --output-base outputs_tpatm1 --out-dir drawings/tpatm1 \
    --families forecast re_mined
```

Each model is drawn as its workflow net twice, as constructed and after τ-reduction, with
the model's frequencies on the nodes. `drawings/<tag>/<Log>/index.csv` lists every model
with its log, tPAT, window dates, net size before and after reduction and its files. The
Static model is the same in every window and is drawn once. A model without tasks, such
as a Forecast whose counts are all zero, is listed in the index and not drawn. The paper
reports Direct-mined and Static at `tPAT = 0.0` only, hence the family selection for the
second run.

Models and drawings produced by this code are published as the Hugging Face dataset
[pmf-liris/pmf-concurrency-models](https://huggingface.co/datasets/pmf-liris/pmf-concurrency-models),
and the Space [pmf-liris/pmf-concurrency-viewer](https://huggingface.co/spaces/pmf-liris/pmf-concurrency-viewer)
shows the drawings side by side for comparison.

Figure 2 of the paper:

```sh
uv run python scripts/render_nets.py --output-base outputs_tpatm1 \
    --out-dir figures --datasets BPI2017 --families forecast --windows 1 \
    --formats pdf --no-title --no-reduced-labels
```
