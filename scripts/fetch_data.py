"""Download the four raw event logs used in the paper into ``data/raw/``.

The logs come from the Zenodo record "Process Model Forecasting Datasets" (18327515,
``pmf_data_v1.1.zip``), whose ``raw_logs/`` directory holds the exact logs the paper
used. For BPI2017, BPI2019 and Hospital Billing these differ from the 4TU.ResearchData
exports; the 4TU DOIs are listed for reference only.

Every raw log is checked against its SHA-256 before it is kept, and logs already in
``data/raw/`` are checked too.

Run ``scripts/preprocess.py`` afterwards to produce the ``data/<Filename>.xes`` files
the pipeline reads.

Usage:
    uv run python scripts/fetch_data.py            # all datasets
    uv run python scripts/fetch_data.py bpi2017    # one dataset

The 58 MB bundle is cached at ``data/raw/pmf_data_v1.1.zip``. Set
``PMF_DATA_ZIP=/path/to/pmf_data_v1.1.zip`` to use a local copy.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

ZENODO_RECORD = "18327515"
ZENODO_ZIP_NAME = "pmf_data_v1.1.zip"
ZENODO_URL = (
    f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{ZENODO_ZIP_NAME}/content"
)


@dataclass(frozen=True)
class Dataset:
    key: str
    # Raw log under data/raw/. The pipeline reads the preprocessed copy in data/.
    filename: str
    zip_member: str  # path of the raw log inside the Zenodo bundle
    sha256: str  # of the raw log
    doi: str  # original 4TU landing page, for reference


# Paths inside the bundle (note the lowercase ``sepsis.xes``).
DATASETS: dict[str, Dataset] = {
    "bpi2017": Dataset(
        key="bpi2017",
        filename="BPI2017.xes",
        zip_member="raw_logs/BPI2017.xes",
        sha256="0cd438617d600e0d7d4c1d666973db090397644a766d48b0187cf6b4eae0688a",
        doi="https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b",
    ),
    "bpi2019": Dataset(
        key="bpi2019",
        filename="BPI2019_1.xes",
        zip_member="raw_logs/BPI2019_1.xes",
        sha256="7c3668416e786318d9f345e3a614fc0b2cf39cbf523bfb2cd5600e3aae066df2",
        doi="https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1",
    ),
    "sepsis": Dataset(
        key="sepsis",
        filename="Sepsis.xes",
        zip_member="raw_logs/sepsis.xes",
        sha256="b501d2063678b9471e3cb1be04d961f924ae5be6ecc8154701bb57a24c7ae0ab",
        doi="https://doi.org/10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460",
    ),
    "hospital_billing": Dataset(
        key="hospital_billing",
        filename="Hospital_Billing.xes",
        zip_member="raw_logs/Hospital_Billing.xes",
        sha256="b8d7ee7e251223b0d9672102285b29aed64dc5535cc27d4d2ee437ac5be2b5ff",
        doi="https://doi.org/10.4121/uuid:76c46b83-c930-4798-a1c9-4be94dfeb741",
    ),
}


def _ensure_bundle() -> Path:
    """Return the path of the Zenodo bundle, downloading it once if needed."""
    override = os.environ.get("PMF_DATA_ZIP")
    if override:
        path = Path(override).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"PMF_DATA_ZIP points at a missing file: {path}")
        return path

    cached = DATA_DIR / ZENODO_ZIP_NAME
    if cached.exists():
        return cached

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_suffix(cached.suffix + ".part")
    print(f"  downloading bundle {ZENODO_URL}")
    with urllib.request.urlopen(ZENODO_URL) as resp, open(tmp, "wb") as out:  # noqa: S310
        shutil.copyfileobj(resp, out)
    tmp.replace(cached)
    print(f"  cached -> {cached}")
    return cached


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(ds: Dataset, path: Path) -> bool:
    """True if ``path`` holds the raw log the paper used."""
    actual = _sha256(path)
    if actual != ds.sha256:
        print(f"[{ds.key}] checksum mismatch for {path}")
        print(f"  expected {ds.sha256}\n  got      {actual}")
        return False
    return True


def _extract(zf: zipfile.ZipFile, ds: Dataset, target: Path) -> bool:
    """Extract one raw log, keeping it only if its checksum matches."""
    print(f"[{ds.key}] extracting {ds.zip_member}")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    with zf.open(ds.zip_member) as src, open(tmp, "wb") as out:
        shutil.copyfileobj(src, out)
    if not _verified(ds, tmp):
        tmp.unlink()
        return False
    tmp.replace(target)
    print(f"  -> {target}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch the paper's raw event logs.")
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=[*DATASETS.keys()],
        help="datasets to fetch (default: all)",
    )
    args = parser.parse_args()
    keys = args.datasets or list(DATASETS)

    ok = True
    pending: list[Dataset] = []
    for k in keys:
        ds = DATASETS[k]
        target = DATA_DIR / ds.filename
        if not target.exists():
            pending.append(ds)
        elif _verified(ds, target):
            print(f"[{k}] already present: {target}")
        else:
            ok = False

    if pending:
        bundle = _ensure_bundle()
        with zipfile.ZipFile(bundle) as zf:
            members = set(zf.namelist())
            for ds in pending:
                if ds.zip_member not in members:
                    print(f"[{ds.key}] missing {ds.zip_member} in bundle, skipping")
                    return 1
                ok = _extract(zf, ds, DATA_DIR / ds.filename) and ok

    if not ok:
        print("\nSome logs do not match the paper's checksums. Delete them and rerun.")
        return 1
    print("\nAll requested logs are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
