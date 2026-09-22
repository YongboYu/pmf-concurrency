"""Where a campaign's outputs are stored.

The pipeline writes models under ``<base>/<Name>_pmf/`` and the conformance scripts read
them from there, often as separate jobs. ``PMF_OUTPUT_BASE`` names the campaign root for
both.

Precedence: ``experiment.output_base=`` on the Hydra command line, then
``$PMF_OUTPUT_BASE``, then ``outputs``. Use the environment variable for any campaign
scored by a later job, since the scripts do not see Hydra overrides.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Campaign root when nothing selects one.
DEFAULT_OUTPUT_BASE = "outputs"

#: Environment variable naming the campaign root for the pipeline and the conformance
#: scripts.
OUTPUT_BASE_ENV = "PMF_OUTPUT_BASE"

#: Suffix of a dataset's output directory.
OUTPUT_DIR_SUFFIX = "_pmf"


def output_base(override: str | None = None) -> str:
    """Campaign root: ``override`` if given, else the environment variable, else
    ``outputs``.
    """
    if override:
        return str(override)
    return os.environ.get(OUTPUT_BASE_ENV) or DEFAULT_OUTPUT_BASE


def dataset_output_dir(name: str, base: str | None = None) -> Path:
    """Return ``<base>/<name>_pmf``, the output directory of a dataset's run."""
    return Path(output_base(base)) / f"{name}{OUTPUT_DIR_SUFFIX}"


def dataset_name_from_output_dir(output_dir: str | Path) -> str:
    """Dataset name of an output directory, the inverse of :func:`dataset_output_dir`.

    Only the ``_pmf`` suffix is removed, so ``Hospital_Billing_pmf`` gives
    ``Hospital_Billing``.
    """
    return Path(os.path.normpath(output_dir)).name.removesuffix(OUTPUT_DIR_SUFFIX)
