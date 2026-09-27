"""scripts/render_nets.py: drawing the weekly models.

The index names each model's log and tPAT and points at its drawings. Only the arc
that closes a loop is freed from the left-to-right ranking, so the source stays at the
left edge. An empty model is indexed without a drawing, and a selection that matches
nothing stops with an error instead of succeeding silently.
"""

from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from pmf_concurrency.petri_net import causal_net_to_wf_net

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
needs_dot = pytest.mark.skipif(shutil.which("dot") is None, reason="needs Graphviz")


def _load():
    # Imported by name so the process pool can pickle render_model.
    return importlib.import_module("render_nets")


def _node(node_id: str, is_binding: bool = False) -> dict:
    return {"id": node_id, "data": {"label": node_id, "is_binding": is_binding}}


def _edge(source: str, target: str) -> dict:
    return {"source": source, "target": target, "data": {"frequency": 1.0}}


def _loop_model() -> dict:
    """▶ -> A -> B -> ■, with B able to return to A."""
    bindings = ["o_▶_0", "i_A_0", "i_A_1", "o_A_0", "i_B_0", "o_B_0", "o_B_1", "i_■_0"]
    return {
        "metadata": {
            "type": "sliding_re_mined",
            "dataset": "Toy",
            "window": 1,
            "start_date": "20200101",
            "end_date": "20200107",
        },
        "graph": {
            "directed": True,
            "nodes": [_node(t) for t in ["▶", "A", "B", "■"]]
            + [_node(b, is_binding=True) for b in bindings],
            "edges": [
                _edge("▶", "o_▶_0"),
                _edge("o_▶_0", "i_A_0"),
                _edge("i_A_0", "A"),
                _edge("A", "o_A_0"),
                _edge("o_A_0", "i_B_0"),
                _edge("i_B_0", "B"),
                _edge("B", "o_B_0"),
                _edge("o_B_0", "i_■_0"),
                _edge("i_■_0", "■"),
                _edge("B", "o_B_1"),
                _edge("o_B_1", "i_A_1"),
                _edge("i_A_1", "A"),
            ],
        },
    }


def _empty_model(window: int) -> dict:
    model = _loop_model()
    model["metadata"]["window"] = window
    model["graph"] = {"directed": True, "nodes": [], "edges": []}
    return model


def _campaign(tmp_path: Path, models: dict[str, dict]) -> Path:
    weekly = tmp_path / "base" / "Toy_pmf" / "weekly_models"
    weekly.mkdir(parents=True)
    (weekly.parent / "structural_metrics.csv").write_text("dataset,tPAT\nToy,-1.0\n")
    for name, model in models.items():
        (weekly / name).write_text(json.dumps(model))
    return tmp_path / "base"


def _run(mod, monkeypatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["render_nets.py", *argv])
    mod.main()


def test_only_the_arc_closing_a_loop_is_freed():
    mod = _load()
    explicit = causal_net_to_wf_net(_loop_model())
    arcs = [(a["source"], a["target"]) for a in explicit["arcs"]]
    assert mod._loop_back_arcs(explicit["places"], arcs) == {
        ("tI::A::1::i_A_1", "pI::A")
    }

    dot = mod.net_dot(explicit, None, None, arcs)
    assert dot.count("constraint=false") == 1
    assert "rank=source" in dot and "rank=sink" in dot


def test_small_forecast_counts_are_not_printed_as_zero():
    mod = _load()
    assert mod._freq(0.16) == "0.16"
    assert mod._freq(0.004) == "&lt;0.01"
    assert mod._freq(12.0) == "12"


@needs_dot
def test_models_are_drawn_and_an_empty_one_is_only_indexed(tmp_path, monkeypatch):
    mod = _load()
    base = _campaign(
        tmp_path,
        {
            "window_1_re_mined.json": _loop_model(),
            "window_2_re_mined.json": _empty_model(2),
        },
    )
    out = tmp_path / "out"
    _run(
        mod,
        monkeypatch,
        "--output-base",
        str(base),
        "--out-dir",
        str(out),
        "--jobs",
        "1",
    )

    index = pd.read_csv(out / "Toy" / "index.csv")
    assert list(index["log"]) == ["Toy", "Toy"]
    assert list(index["tpat"]) == [-1.0, -1.0]
    assert list(index["window"]) == [1, 2]
    assert list(index["start_date"]) == ["2020-01-01", "2020-01-01"]
    assert list(index["empty"]) == [False, True]
    drawn = sorted(p.name for p in (out / "Toy" / "re_mined").iterdir())
    assert drawn == ["window_1-expanded.svg", "window_1-reduced.svg"]
    for f in index["files"].iloc[0].split():
        assert (out / "Toy" / f).is_file()
    assert (
        "tPAT = &#45;1.0"
        in (out / "Toy" / "re_mined" / "window_1-expanded.svg").read_text()
    )


@needs_dot
def test_a_selection_of_only_empty_models_still_writes_the_index(tmp_path, monkeypatch):
    mod = _load()
    base = _campaign(tmp_path, {"window_2_forecast.json": _empty_model(2)})
    out = tmp_path / "out"
    _run(
        mod,
        monkeypatch,
        "--output-base",
        str(base),
        "--out-dir",
        str(out),
        "--jobs",
        "1",
    )

    index = pd.read_csv(out / "Toy" / "index.csv")
    assert list(index["empty"]) == [True]


@needs_dot
@pytest.mark.parametrize(
    "extra, message",
    [
        (["--windows", "99"], "no models match"),
        (["--datasets", "Other"], "no weekly models"),
    ],
)
def test_a_selection_that_matches_nothing_fails(tmp_path, monkeypatch, extra, message):
    mod = _load()
    base = _campaign(tmp_path, {"window_1_re_mined.json": _loop_model()})
    with pytest.raises(SystemExit) as exc:
        _run(mod, monkeypatch, "--output-base", str(base), "--out-dir", "x", *extra)
    assert message in str(exc.value)


@needs_dot
def test_a_missing_output_base_fails(tmp_path, monkeypatch):
    mod = _load()
    with pytest.raises(SystemExit) as exc:
        _run(
            mod, monkeypatch, "--output-base", str(tmp_path / "nope"), "--out-dir", "x"
        )
    assert "is not a directory" in str(exc.value)

    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit) as exc:
        _run(
            mod, monkeypatch, "--output-base", str(tmp_path / "empty"), "--out-dir", "x"
        )
    assert "no <Log>_pmf directories" in str(exc.value)
