"""Family names, and readers that reject any other name."""

from pathlib import Path

import pandas as pd
import pytest

from pmf_concurrency.families import (
    ALL_FAMILIES,
    FAMILY_LABELS,
    check_families,
    family_from_path,
    model_json_path,
)
from pmf_concurrency.structural_metrics import (
    STRUCTURAL_METRICS_COLUMNS,
    read_structural_metrics,
)


def test_every_family_has_a_label():
    assert set(FAMILY_LABELS) == set(ALL_FAMILIES)
    assert FAMILY_LABELS["persistence"] == "Persistence"
    assert FAMILY_LABELS["expanding"] == "Expanding"


def test_the_history_families_are_known_and_other_names_are_not():
    check_families(["static", "persistence", "expanding"], "test")
    with pytest.raises(ValueError, match="naive"):
        check_families(["persistence", "naive"], "test")


def test_family_from_path_keeps_underscored_names_whole():
    assert family_from_path(Path("window_3_direct_mined.json")) == "direct_mined"
    assert family_from_path(Path("window_3_re_mined.json")) == "re_mined"


def test_model_json_path_names_the_window_and_family(tmp_path):
    assert model_json_path(tmp_path, 1, "static") == tmp_path / "window_1_static.json"


def test_structural_metrics_reader_rejects_unknown_families(tmp_path):
    rows = [
        dict.fromkeys(STRUCTURAL_METRICS_COLUMNS, 0)
        | {"dataset": "BPI2017", "tPAT": 0.0, "window": 1, "model_family": family}
        for family in ("forecast", "actual")
    ]
    path = tmp_path / "structural_metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    with pytest.raises(ValueError, match="actual"):
        read_structural_metrics(str(path))
