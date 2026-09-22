"""Tests for scripts/local/run_campaign.sh, the workstation launcher.

The launcher refuses anything that would go wrong silently:

* no default campaign root, so a run never lands in another campaign's tree;
* no existing output directory, because a campaign must start cold.

Each test runs the launcher against a throwaway PMF_ROOT.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "local" / "run_campaign.sh"

#: dataset -> (processed log stem, output directory stem). They differ only for bpi2019,
#: so the launcher reads both from conf/dataset/*.yaml.
DATASETS = {
    "bpi2017": ("BPI2017", "BPI2017"),
    "bpi2019": ("BPI2019_1", "BPI2019"),
    "sepsis": ("Sepsis", "Sepsis"),
    "hospital_billing": ("Hospital_Billing", "Hospital_Billing"),
}

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash is required to run run_campaign.sh"
)


@pytest.fixture
def tree(tmp_path):
    """A throwaway PMF_ROOT with the four processed logs and their real Hydra configs.

    The launcher reads each dataset's log and output paths from the configs, so they are
    not stubbed.
    """
    data = tmp_path / "data"
    data.mkdir()
    conf = tmp_path / "conf" / "dataset"
    conf.mkdir(parents=True)
    for dataset, (log_stem, name) in DATASETS.items():
        (data / f"{log_stem}.xes").write_text("<log/>")
        (conf / f"{dataset}.yaml").write_text(
            f"name: {name}\nlog_path: data/{log_stem}.xes\n"
        )
    return tmp_path


@pytest.fixture
def run(tree):
    """Run the launcher with a recording ``uv`` shim first on PATH."""
    calls = tree / "uv_calls"
    shim_dir = tree / "bin"
    shim_dir.mkdir()
    shim = shim_dir / "uv"
    shim.write_text(f'#!/bin/sh\necho "$@" >> "{calls}"\n')
    shim.chmod(0o755)

    def _run(*args: str, **env_overrides: str):
        env = dict(os.environ)
        for key in (
            "PMF_OUTPUT_BASE",
            "PMF_HYDRA_OVERRIDES",
            "DRY_RUN",
            "TMUX",
        ):
            env.pop(key, None)
        env["PATH"] = f"{shim_dir}{os.pathsep}{env['PATH']}"
        env["PMF_ROOT"] = str(tree)
        env.update(env_overrides)
        proc = subprocess.run(
            ["bash", str(RUNNER), *args],
            capture_output=True,
            text=True,
            env=env,
            cwd=tree,
        )
        proc.uv_calls = calls.read_text() if calls.exists() else ""  # type: ignore[attr-defined]
        return proc

    return _run


# --- what the driver refuses to start --------------------------------------


def test_refuses_without_a_campaign_root(run):
    proc = run()
    assert proc.returncode == 2
    assert "PMF_OUTPUT_BASE is unset" in proc.stderr
    assert proc.uv_calls == ""


def test_refuses_a_root_that_would_not_survive_the_environment(run):
    proc = run(PMF_OUTPUT_BASE="outputs tpatm1,x")
    assert proc.returncode == 2
    assert "use only letters, digits" in proc.stderr
    assert proc.uv_calls == ""


def test_refuses_an_unknown_dataset(run):
    proc = run("bpi2018", PMF_OUTPUT_BASE="outputs_tpatm1")
    assert proc.returncode == 2
    assert "unknown dataset 'bpi2018'" in proc.stderr
    assert "conf/dataset/bpi2018.yaml" in proc.stderr


def test_refuses_a_missing_processed_log(run, tree):
    (tree / "data" / "Sepsis.xes").unlink()
    proc = run("sepsis", PMF_OUTPUT_BASE="outputs_tpatm1")
    assert proc.returncode == 2
    assert "data/Sepsis.xes is missing" in proc.stderr
    assert proc.uv_calls == ""


def test_refuses_a_warm_output_tree_because_the_campaign_runs_cold(run, tree):
    (tree / "outputs_tpatm1" / "Sepsis_pmf").mkdir(parents=True)
    proc = run("sepsis", PMF_OUTPUT_BASE="outputs_tpatm1")
    assert proc.returncode == 2
    assert "already exists" in proc.stderr
    assert "runs cold" in proc.stderr
    assert proc.uv_calls == ""


# --- what the driver would run ---------------------------------------------


def test_dry_run_starts_nothing_and_plans_every_log(run):
    proc = run(
        PMF_OUTPUT_BASE="outputs_tpatm1",
        PMF_HYDRA_OVERRIDES="experiment.tPAT=-1.0",
        DRY_RUN="1",
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.uv_calls == ""
    planned = [ln for ln in proc.stdout.splitlines() if "dataset=" in ln]
    assert len(planned) == len(DATASETS)
    assert {ln.split("dataset=")[1].split()[0] for ln in planned} == set(DATASETS)
    # The campaign's parameters reach every pipeline call, not just the header.
    for line in planned:
        assert "experiment.tPAT=-1.0" in line


def test_cheapest_log_first_so_a_machine_fails_fast(run):
    """Logs run cheapest first, so a machine that cannot cope fails early."""
    proc = run(PMF_OUTPUT_BASE="outputs_tpatm1", DRY_RUN="1")
    order = [
        ln.split("dataset=")[1].split()[0]
        for ln in proc.stdout.splitlines()
        if "dataset=" in ln
    ]
    assert order == ["sepsis", "bpi2017", "hospital_billing", "bpi2019"]


def test_a_subset_runs_only_what_was_named(run):
    proc = run(
        "sepsis",
        "bpi2017",
        PMF_OUTPUT_BASE="outputs_tpatm1",
        DRY_RUN="1",
    )
    assert "dataset=bpi2019" not in proc.stdout
    assert "dataset=hospital_billing" not in proc.stdout


# --- the manifest is the run log -------------------------------------------


def test_the_manifest_records_the_machine_and_the_campaign(run, tree):
    """A shimmed run writes the manifest instead of only planning it."""
    proc = run(
        "sepsis",
        PMF_OUTPUT_BASE="outputs_tpatm1",
        PMF_HYDRA_OVERRIDES="experiment.tPAT=-1.0",
    )
    manifest = json.loads(
        (tree / "outputs_tpatm1" / "campaign_manifest.json").read_text()
    )
    assert "mode" not in manifest
    assert "experiment.tPAT=-1.0" in manifest["hydra_overrides"]
    assert manifest["datasets_requested"] == ["sepsis"]
    # Results are deterministic per device, so the device is recorded.
    assert manifest["device"]
    assert manifest["host"]
    assert manifest["git_commit"]

    run_row = manifest["runs"][0]
    assert run_row["dataset"] == "sepsis"
    assert run_row["exit_code"] == 0
    assert run_row["wall_clock_sec"] >= 0
    # The shim writes no metrics file, so the count is zero.
    assert run_row["structural_metrics_rows"] == 0
    assert proc.returncode == 0, proc.stderr


def test_one_failing_log_does_not_cancel_the_others(run, tree):
    """A log that runs out of memory does not stop the others."""
    shim = tree / "bin" / "uv"
    shim.write_text(
        '#!/bin/sh\ncase "$*" in *dataset=bpi2017*) exit 137 ;; esac\nexit 0\n'
    )
    shim.chmod(0o755)
    proc = run(
        "sepsis",
        "bpi2017",
        PMF_OUTPUT_BASE="outputs_tpatm1",
    )
    assert proc.returncode == 1
    manifest = json.loads(
        (tree / "outputs_tpatm1" / "campaign_manifest.json").read_text()
    )
    by_dataset = {r["dataset"]: r for r in manifest["runs"]}
    assert by_dataset["bpi2017"]["exit_code"] == 137
    assert by_dataset["sepsis"]["exit_code"] == 0


def test_bpi2019_output_dir_is_not_its_log_stem(run, tree):
    """BPI2019's output directory is not named after its log file.

    ``data/BPI2019_1.xes`` writes to ``BPI2019_pmf``. Deriving the directory from the
    log stem would check the wrong path before a cold run.
    """
    (tree / "outputs_tpatm1" / "BPI2019_pmf").mkdir(parents=True)
    proc = run("bpi2019", PMF_OUTPUT_BASE="outputs_tpatm1")
    assert proc.returncode == 2
    assert "outputs_tpatm1/BPI2019_pmf already exists" in proc.stderr
    assert proc.uv_calls == ""


def test_the_manifest_names_the_directory_the_pipeline_writes(run, tree):
    proc = run("bpi2019", PMF_OUTPUT_BASE="outputs_tpatm1")
    assert proc.returncode == 0, proc.stderr
    manifest = json.loads(
        (tree / "outputs_tpatm1" / "campaign_manifest.json").read_text()
    )
    assert manifest["runs"][0]["output_dir"] == "outputs_tpatm1/BPI2019_pmf"
