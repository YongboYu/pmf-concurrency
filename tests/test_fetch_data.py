"""scripts/fetch_data.py accepts a raw log only if its SHA-256 matches the paper's."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import re
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

LOG = b"<log/>\n"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "fetch_data", SCRIPTS / "fetch_data.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules.
    sys.modules["fetch_data"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fetch(tmp_path, monkeypatch):
    """The script with a local bundle holding a one-line Sepsis log."""
    mod = _load_script()
    bundle = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr(mod.DATASETS["sepsis"].zip_member, LOG)
    monkeypatch.setenv("PMF_DATA_ZIP", str(bundle))
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(sys, "argv", ["fetch_data.py", "sepsis"])

    def _expect(digest: str) -> None:
        sepsis = dataclasses.replace(mod.DATASETS["sepsis"], sha256=digest)
        monkeypatch.setitem(mod.DATASETS, "sepsis", sepsis)

    mod.expect = _expect
    return mod


def test_every_dataset_has_a_sha256():
    for ds in _load_script().DATASETS.values():
        assert re.fullmatch(r"[0-9a-f]{64}", ds.sha256), ds.key


def test_a_matching_log_is_extracted(fetch):
    fetch.expect(hashlib.sha256(LOG).hexdigest())

    assert fetch.main() == 0
    assert (fetch.DATA_DIR / "Sepsis.xes").read_bytes() == LOG


def test_a_mismatching_log_is_not_kept(fetch, capsys):
    fetch.expect("0" * 64)

    assert fetch.main() != 0
    assert not (fetch.DATA_DIR / "Sepsis.xes").exists()
    assert not list(fetch.DATA_DIR.glob("*.part"))
    assert "checksum" in capsys.readouterr().out


def test_a_present_log_is_checked_too(fetch, capsys):
    fetch.expect(hashlib.sha256(LOG).hexdigest())
    fetch.DATA_DIR.mkdir()
    (fetch.DATA_DIR / "Sepsis.xes").write_bytes(b"<other/>\n")

    assert fetch.main() != 0
    assert "checksum" in capsys.readouterr().out
