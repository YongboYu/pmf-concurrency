"""The pipeline and the conformance scripts agree on the campaign root.

The pipeline writes models under ``<base>/<Name>_pmf/`` and the scripts read them
through ``scripts/_dataset.resolve_dataset``. If the two disagreed, a second campaign
could overwrite the first one's models without any error.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _dataset import resolve_dataset  # noqa: E402

from pmf_concurrency.paths import (  # noqa: E402
    DEFAULT_OUTPUT_BASE,
    OUTPUT_BASE_ENV,
    dataset_output_dir,
    output_base,
)


@pytest.fixture(autouse=True)
def _clean_root(monkeypatch):
    """Run from the project root with no inherited campaign root."""
    monkeypatch.delenv(OUTPUT_BASE_ENV, raising=False)
    monkeypatch.chdir(ROOT)


def _pipeline_write_dir(
    monkeypatch, *overrides: str, conf_dir: Path | None = None
) -> Path:
    """The directory ``pipeline.main()`` passes to ``run_pmf_experiment``.

    Uses the real ``conf/config.yaml`` and Hydra entry point, so environment
    interpolation and override precedence are covered.
    """
    from hydra import compose, initialize_config_dir

    from pmf_concurrency import pipeline

    captured: dict = {}
    monkeypatch.setattr(
        pipeline, "run_pmf_experiment", lambda **kw: captured.update(kw)
    )

    with initialize_config_dir(
        config_dir=str(conf_dir or ROOT / "conf"), version_base=None
    ):
        cfg = compose(config_name="config", overrides=list(overrides))
    pipeline.main.__wrapped__(cfg)
    return Path(captured["output_dir"])


def test_default_base_is_outputs():
    assert output_base() == DEFAULT_OUTPUT_BASE == "outputs"
    assert dataset_output_dir("Sepsis") == Path("outputs/Sepsis_pmf")


def test_env_var_sets_the_campaign_root(monkeypatch):
    monkeypatch.setenv(OUTPUT_BASE_ENV, "outputs_tpat0")
    assert output_base() == "outputs_tpat0"
    assert dataset_output_dir("Sepsis") == Path("outputs_tpat0/Sepsis_pmf")


def test_empty_env_var_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv(OUTPUT_BASE_ENV, "")
    assert output_base() == "outputs"


@pytest.mark.parametrize("base", ["outputs", "outputs_tpat0", "outputs_repro/run1"])
def test_write_path_and_read_path_agree(monkeypatch, base):
    """The write path and the read path resolve to the same directory."""
    monkeypatch.setenv(OUTPUT_BASE_ENV, base)
    write_dir = _pipeline_write_dir(monkeypatch, "dataset=sepsis")
    _log, read_dir, weekly_dir = resolve_dataset("sepsis")
    assert write_dir == read_dir == Path(base) / "Sepsis_pmf"
    assert weekly_dir == read_dir / "weekly_models"


def test_hydra_override_beats_the_env_var(monkeypatch):
    """An explicit Hydra override wins over PMF_OUTPUT_BASE."""
    monkeypatch.setenv(OUTPUT_BASE_ENV, "outputs_tpat0")
    write_dir = _pipeline_write_dir(
        monkeypatch, "dataset=sepsis", "experiment.output_base=outputs_repro/run1"
    )
    assert write_dir == Path("outputs_repro/run1/Sepsis_pmf")


def test_env_var_reaches_the_hydra_config(monkeypatch):
    """conf/config.yaml reads PMF_OUTPUT_BASE, so the pipeline needs no flag."""
    monkeypatch.setenv(OUTPUT_BASE_ENV, "outputs_other")
    assert _pipeline_write_dir(monkeypatch, "dataset=sepsis") == Path(
        "outputs_other/Sepsis_pmf"
    )


def test_a_config_file_value_binds_both_paths(monkeypatch, tmp_path):
    """A root set in the config file moves both paths together."""
    conf = tmp_path / "conf"
    shutil.copytree(ROOT / "conf", conf)
    cfg_file = conf / "config.yaml"
    cfg_file.write_text(
        cfg_file.read_text().replace(
            "output_base: ${oc.env:PMF_OUTPUT_BASE,outputs}",
            "output_base: outputs_from_yaml",
        )
    )
    # Dataset configs use relative log paths; only the campaign root matters.
    monkeypatch.chdir(tmp_path)

    write_dir = _pipeline_write_dir(monkeypatch, "dataset=sepsis", conf_dir=conf)
    _log, read_dir, _weekly = resolve_dataset("sepsis")
    assert write_dir == read_dir == Path("outputs_from_yaml/Sepsis_pmf")
