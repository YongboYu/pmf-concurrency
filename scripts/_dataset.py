"""Resolve a dataset key to its paths, the same way the pipeline does.

Paths come from ``conf/dataset/<key>.yaml`` and the campaign root from
``experiment.output_base`` in ``conf/config.yaml`` (``$PMF_OUTPUT_BASE``, default
``outputs``). A Hydra command-line override does not reach the scripts, so campaigns are
selected with the environment variable.

Run the scripts from the project root.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from pmf_concurrency.paths import dataset_output_dir

#: Keys accepted by ``--dataset``, one per conf/dataset/*.yaml.
DATASETS = tuple(
    sorted(
        p.stem
        for p in (Path(__file__).resolve().parents[1] / "conf" / "dataset").glob(
            "*.yaml"
        )
    )
)


def _configured_output_base() -> str:
    """``experiment.output_base`` as ``conf/config.yaml`` resolves it."""
    cfg = OmegaConf.load("conf/config.yaml")
    return str(cfg.experiment.output_base)


def resolve_dataset(key: str) -> tuple[str, Path, Path]:
    """Return ``(log_path, output_dir, weekly_dir)`` for a dataset key.

    ``output_dir`` is ``$PMF_OUTPUT_BASE/<cfg.name>_pmf``, the directory the pipeline
    writes for the same dataset.
    """
    cfg = OmegaConf.load(f"conf/dataset/{key}.yaml")
    log_path = str(cfg.log_path)
    output_dir = dataset_output_dir(str(cfg.name), _configured_output_base())
    return log_path, output_dir, output_dir / "weekly_models"
