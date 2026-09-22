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
Drawing each model as a PDF needs [Graphviz](https://graphviz.org/download/) (`dot` on
the `PATH`). Without it the pipeline skips the drawings.

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
PMF_OUTPUT_BASE=outputs_tpat0 uv run python scripts/evaluate.py --dataset sepsis \
    --reduce-levels tau --evaluators token_etc alignment er
```

A unit is one window, family, reduction level and evaluator. Each unit runs in a child
process with a budget of 12 GB and 20 minutes (`--mem-gb` and `--time-min`). A unit
that exceeds it is recorded as `oom` or `timeout`; that is a result, not missing data.
The memory budget is enforced on Linux only. OMP and MKL default to 8 threads.

Results are appended to `$PMF_OUTPUT_BASE/<Log>_pmf/evaluation.csv` as each unit
finishes, so a stopped run resumes where it left off. `evaluation_manifest.json` records
the log, the models, the budgets and the thread counts, and a resume with any of them
changed stops. A metric that could not be computed is left empty, and
`evaluation_summary.csv` counts the units without a result (`n_intractable`,
`n_unavailable`) next to the mean of each metric.

To use more cores, run up to four processes at a time, each on its own windows and
output directory, and concatenate their `evaluation.csv` files afterwards. Four units
at a time need about 48 GB of memory.

```sh
PMF_OUTPUT_BASE=outputs_tpat0 uv run python scripts/evaluate.py --dataset sepsis \
    --reduce-levels tau --evaluators token_etc --windows 1 2 3 4 --out-dir eval/part1
```
