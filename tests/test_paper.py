"""scripts/paper.py: table cells, and what the highlighting marks as changed.

A value is marked when its printed digits differ from the submitted table's, whatever
padding or highlighting either side carries; a value that prints alike is left plain.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from pmf_concurrency.results import Pooled, ResultsError

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


@pytest.fixture(scope="module")
def paper():
    spec = importlib.util.spec_from_file_location("paper", SCRIPTS / "paper.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["paper"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_a_cell_pads_its_mean_and_sd_to_fixed_widths(paper):
    cell = paper._cell(Pooled(85, 85, 3.04, 0.512), 1, 5, 1, 4)
    assert cell == ("\\phantom{00}3.0", "0.5\\phantom{0}")
    assert paper._math(cell, None) == "$\\phantom{00}3.0_{\\pm0.5\\phantom{0}}$"


def test_fewer_than_two_values_print_a_dagger(paper):
    assert paper._cell(Pooled(55, 1, 0.5, float("nan")), 3, 5, 3)[0] == paper.DAGGER


@pytest.mark.parametrize(
    ("submitted", "printed"),
    [
        ("$\\phantom{0}87.8_{\\pm13.2}$", "$\\phantom{0}87.8_{\\pm13.2}$"),
        (
            "$\\chg{\\phantom{0}87.8}_{\\pm\\chg{13.2}}$",
            "$\\phantom{0}87.8_{\\pm13.2}$",
        ),
        ("$\\phantom{0}99.6_{\\pm13.2}$", "$\\rev{\\phantom{0}87.8}_{\\pm13.2}$"),
        (
            "$\\phantom{0}87.8_{\\pm0.2\\phantom{0}}$",
            "$\\phantom{0}87.8_{\\pm\\rev{13.2}}$",
        ),
    ],
)
def test_only_the_part_that_changed_is_marked(paper, submitted, printed):
    assert paper._math(("\\phantom{0}87.8", "13.2"), submitted) == printed


def test_a_new_dagger_is_marked_and_a_kept_one_is_not(paper):
    assert paper._math((paper.DAGGER, None), "$^{\\dagger}$") == paper.DAGGER
    assert (
        paper._math((paper.DAGGER, None), "$0.5_{\\pm0.1}$") == "\\rev{$^{\\dagger}$}"
    )


def test_table1_cells_compare_without_padding(paper):
    assert paper._text("5.1", "\\phantom{0}5.1") == "5.1"
    assert paper._text("16{,}009", "16{,}009") == "16{,}009"
    assert paper._text("135", "137") == "\\rev{135}"


def test_a_range_collapses_when_both_ends_print_alike(paper):
    assert paper._range(0.7215, 0.7249, 2) == "0.72"
    assert paper._range(0.61, 0.65, 2) == "0.61--0.65"


def _manuscript(tmp_path: Path, rows2: int = 24) -> Path:
    d = tmp_path / "submitted"
    d.mkdir()
    rows = "\n".join(
        f"    {label} & 1 & 2 \\\\" for label in ("BPI2017", "BPI2019", "Sepsis")
    )
    (d / "main.tex").write_text(
        "\\begin{table*}\\label{tab:logs}\n\\begin{tabular}{lrr}\n"
        f"{rows}\n    Hosp.\\ Bill. & 3 & \\phantom{{0}}4 \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )
    body = ["    \\multirow{6}{*}{BPI2017}"] + [
        f"      & Forecast & $0.0$ & $0.{i}$ & $1/1$ \\\\" for i in range(rows2)
    ]
    (d / "tab-quality-rows-structure.tex").write_text("\n".join(body) + "\n")
    return d


def test_the_submitted_rows_are_read_by_log_label_and_row(paper, tmp_path):
    table1, table2 = paper.submitted_rows(_manuscript(tmp_path))
    assert table1["Hosp.\\ Bill."] == ["3", "\\phantom{0}4"]
    assert table2[5] == ["$0.5$", "$1/1$"]


def test_a_submitted_table_of_another_shape_is_refused(paper, tmp_path):
    with pytest.raises(ResultsError, match="cannot find"):
        paper.submitted_rows(_manuscript(tmp_path, rows2=23))


def test_a_row_the_submitted_table_lacks_is_marked_whole(paper):
    assert (
        paper._math(("0.912", "0.031"), paper.ABSENT)
        == "$\\rev{0.912}_{\\pm\\rev{0.031}}$"
    )
    assert paper._math((paper.DAGGER, None), paper.ABSENT) == "\\rev{$^{\\dagger}$}"
    assert paper._text("Persistence", paper.ABSENT) == "\\rev{Persistence}"


def test_a_submitted_row_lines_up_around_the_f1_columns_it_lacks(paper):
    n_submitted = paper.TABLE2_CELLS - len(paper.TABLE2_F1_CELLS)
    old = [f"$0.{i}$" for i in range(n_submitted)]
    cells = paper._submitted([old], 0, False)
    assert len(cells) == paper.TABLE2_CELLS
    assert [cells[i] for i in paper.TABLE2_F1_CELLS] == [paper.ABSENT] * len(
        paper.TABLE2_F1_CELLS
    )
    assert [c for c in cells if c != paper.ABSENT] == old


def _history_units(empty: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Alignment units of every log: two windows, five families at tPAT 0."""
    precision = {
        "forecast": 0.9,
        "direct_mined": 0.8,
        "static": 0.5,
        "persistence": 0.7,
        "expanding": 0.5,
    }
    rows = []
    for key in ("bpi2017", "bpi2019", "sepsis", "hospital_billing"):
        for window in (1, 2):
            for family, p in precision.items():
                f1 = {"forecast": 0.9, "persistence": 0.7, "expanding": 0.6}.get(family)
                reason = ""
                if key == "bpi2017" and family == "persistence":
                    p, f1 = 0.85, 0.95 if window == 1 else f1
                if key == "hospital_billing" and family == "expanding":
                    p += 0.02
                if key == "sepsis" and family == "forecast" and window == 2:
                    p, f1, reason = float("nan"), float("nan"), "empty model"
                if family == empty and key == "sepsis":
                    reason = "empty model"
                rows.append((key, 0.0, window, family, p, f1, reason))
            rows.append((key, -1.0, window, "forecast", 0.1, 0.1, ""))
    units = pd.DataFrame(
        rows,
        columns=[
            "dataset",
            "tpat",
            "window",
            "model_type",
            "precision",
            "f1",
            "guard_reason",
        ],
    ).assign(evaluator="alignment", guard_status="ok")
    history = units["model_type"].isin(("persistence", "expanding"))
    return units[~history], units[history]


def test_history_numbers_compare_with_the_families_they_derive_from(paper):
    numbers = paper.history_numbers(*_history_units())
    assert numbers["ExpandingStaticPrecisionMaxDiff"] == "0.020"
    assert numbers["PersistenceLessPreciseLogs"] == "three"
    assert numbers["ForecastOverPersistenceWins"] == "6"
    assert numbers["ForecastOverPersistenceWindows"] == "7"
    assert numbers["ForecastOverExpandingWins"] == "7"
    assert numbers["ForecastOverExpandingWindows"] == "7"


def test_an_empty_history_net_is_refused(paper):
    with pytest.raises(ResultsError, match="empty"):
        paper.history_numbers(*_history_units(empty="persistence"))


def _outcome_units(evaluator: str, statuses: list[tuple[str, str]]) -> pd.DataFrame:
    """Units of one evaluator from (status, reason) pairs; an empty net fits 0."""
    rows = []
    for status, reason in statuses:
        empty = reason == "empty model"
        scored = status == "ok" and not empty
        rows.append(
            {
                "evaluator": evaluator,
                "guard_status": status,
                "guard_reason": reason,
                "log_fitness": 0.0 if empty else (0.8 if scored else float("nan")),
                "precision": 0.6 if scored else float("nan"),
                "f1": 2 * 0.8 * 0.6 / 1.4 if scored else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def test_zero_filled_f1_counts_only_an_empty_net_as_zero(paper):
    units = _outcome_units(
        "alignment", [("ok", ""), ("ok", "empty model"), ("oom", "")]
    )
    f1 = paper.with_f1_zero(units)["f1_zero"]
    assert f1.iloc[0] == pytest.approx(0.96 / 1.4)
    assert f1.iloc[1] == 0.0
    assert f1.isna().iloc[2]


def test_an_empty_net_that_fits_is_refused(paper):
    units = _outcome_units("alignment", [("ok", "empty model")])
    units.loc[0, "log_fitness"] = 0.5
    with pytest.raises(ResultsError, match="empty net"):
        paper.with_f1_zero(units)
