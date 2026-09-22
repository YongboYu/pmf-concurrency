"""Scripts resolve ``--dataset`` to the directory the pipeline writes."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _dataset import DATASETS, resolve_dataset  # noqa: E402

EXPECTED = {
    "bpi2017": ("data/BPI2017.xes", ROOT / "outputs/BPI2017_pmf"),
    "bpi2019": ("data/BPI2019_1.xes", ROOT / "outputs/BPI2019_pmf"),
    "sepsis": ("data/Sepsis.xes", ROOT / "outputs/Sepsis_pmf"),
    "hospital_billing": (
        "data/Hospital_Billing.xes",
        ROOT / "outputs/Hospital_Billing_pmf",
    ),
}


@pytest.fixture(autouse=True)
def _chdir_root(monkeypatch):
    """Run from the project root with the default campaign root.

    Other roots are tested in tests/test_output_base.py.
    """
    monkeypatch.delenv("PMF_OUTPUT_BASE", raising=False)
    prev = Path.cwd()
    os.chdir(ROOT)
    yield
    os.chdir(prev)


def test_dataset_keys_match_expected():
    assert set(DATASETS) == set(EXPECTED)


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_resolve_dataset(key):
    exp_log, exp_out = EXPECTED[key]
    log_path, output_dir, weekly_dir = resolve_dataset(key)
    assert log_path == exp_log
    # Compare absolute paths so the result does not depend on the working directory.
    assert (ROOT / output_dir).resolve() == exp_out.resolve()
    assert weekly_dir == output_dir / "weekly_models"
