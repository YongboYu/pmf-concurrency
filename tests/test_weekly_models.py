import json
from pathlib import Path
from typing import ClassVar

import pandas as pd

from pmf_concurrency import weekly_models
from pmf_concurrency.structural_metrics import (
    STRUCTURAL_METRICS_FILENAME,
    read_structural_metrics,
    stats_from_bindings,
    stats_from_fodina,
)


class _FakeDependencyGraph:
    DG: ClassVar[set[tuple[int, int]]] = {(0, 1)}


class _FakePatternMiner:
    """Stand-in carrying the arcs and IB/OB mappings the measurement reads.

    One output AND binding and one input singleton, so the Direct-mined row is not all
    zeros.
    """

    dg = _FakeDependencyGraph()
    IB: ClassVar[dict[int, set[frozenset[int]]]] = {1: {frozenset({0})}}
    OB: ClassVar[dict[int, set[frozenset[int]]]] = {0: {frozenset({0, 1})}}


def test_build_weekly_models_emits_direct_mined_family(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "BPI2017_pmf"
    output_dir.mkdir()

    index = pd.to_datetime(["2021-01-01"])
    test_data = {
        "A": pd.DataFrame({"A": [1.0]}, index=index),
        "AB": pd.DataFrame({"A→B": [1.0]}, index=index),
        "ABA": pd.DataFrame(index=index),
        "IB": pd.DataFrame({"B:{A}": [1.0]}, index=index),
        "OB": pd.DataFrame({"A:{B}": [1.0]}, index=index),
    }
    forecasts = {series_type: {} for series_type in test_data}
    test_df = pd.DataFrame(
        {
            "case:concept:name": ["case-1", "case-1"],
            "concept:name": ["A", "B"],
            "time:timestamp": pd.to_datetime(
                ["2021-01-01T09:00:00", "2021-01-01T10:00:00"]
            ),
            "date": [pd.Timestamp("2021-01-01").date()] * 2,
        }
    )

    def fake_build_causal_net_from_forecast(*_args, **_kwargs):
        # The fourth element is the structural statistics record: one AND binding out of
        # two, so the row is not all zeros.
        return (
            {"▶": {"■"}, "■": set()},
            set(),
            {},
            stats_from_bindings(
                tasks=2,
                arcs=1,
                IB={"■": {frozenset({"▶"})}},
                OB={"▶": {frozenset({"■", "X"})}},
                repair_arcs_added=0,
                filter_bindings_removed=0,
                singleton_readds_orphaned=0,
                singleton_readds_uncovered=0,
                singleton_fallback_fires=0,
            ),
        )

    def fake_causal_graph_to_json(
        _graph, _binding_nodes, metadata, _edge_freq=None, _binding_freq=None, **_kwargs
    ):
        return {
            "metadata": metadata,
            "graph": {"directed": True, "nodes": [], "edges": []},
        }

    def fake_mine_fodina(_traces, *_args, **_kwargs):
        # Direct-mined is measured from Fodina's own IB/OB, so the fake carries them.
        return _FakePatternMiner()

    def fake_fodina_to_json(_pm, metadata):
        return {
            "metadata": metadata,
            "graph": {"directed": True, "nodes": [], "edges": []},
        }

    def fake_render_petri_json_to_pdf(_json_path_or_data, pdf_path):
        Path(pdf_path).write_text("pdf")
        return True

    monkeypatch.setattr(
        weekly_models,
        "build_causal_net_from_forecast",
        fake_build_causal_net_from_forecast,
    )
    monkeypatch.setattr(
        weekly_models, "causal_graph_to_json", fake_causal_graph_to_json
    )
    monkeypatch.setattr(weekly_models, "mine_fodina", fake_mine_fodina)
    monkeypatch.setattr(weekly_models, "fodina_to_json", fake_fodina_to_json)
    monkeypatch.setattr(
        weekly_models, "render_petri_json_to_pdf", fake_render_petri_json_to_pdf
    )

    weekly_models.build_weekly_models(
        forecasts,
        test_data,
        str(output_dir),
        0.5,
        0.5,
        0.5,
        0.0,
        horizon=1,
        test_df=test_df,
    )

    weekly_dir = output_dir / "weekly_models"
    direct_mined_json_path = weekly_dir / "window_1_direct_mined.json"
    direct_mined_pdf_path = weekly_dir / "window_1_direct_mined.pdf"

    assert direct_mined_json_path.exists()
    assert direct_mined_pdf_path.exists()
    assert (weekly_dir / "window_1_forecast.json").exists()
    assert (weekly_dir / "window_1_re_mined.json").exists()
    assert not (output_dir / "direct_mined_model.json").exists()
    assert not (output_dir / "direct_mined_model.pdf").exists()

    direct_mined_json = json.loads(direct_mined_json_path.read_text())
    assert direct_mined_json["metadata"]["type"] == "sliding_direct_mined"
    assert direct_mined_json["metadata"]["method"] == "fodina_original"
    assert direct_mined_json["metadata"]["traces"] == 1


def test_build_weekly_models_emits_structural_metrics(tmp_path, monkeypatch) -> None:
    """One structural-metrics row per evaluation window and family, in one CSV.

    The Static model is mined once and reused, so its row repeats unchanged in every
    window.
    """
    output_dir = tmp_path / "BPI2017_pmf"
    output_dir.mkdir()

    index = pd.to_datetime(["2021-01-01", "2021-01-02"])
    test_data = {
        "A": pd.DataFrame({"A": [1.0, 1.0]}, index=index),
        "AB": pd.DataFrame({"A→B": [1.0, 1.0]}, index=index),
        "ABA": pd.DataFrame(index=index),
        "IB": pd.DataFrame({"B:{A}": [1.0, 1.0]}, index=index),
        "OB": pd.DataFrame({"A:{B}": [1.0, 1.0]}, index=index),
    }
    forecasts = {series_type: {} for series_type in test_data}
    test_df = pd.DataFrame(
        {
            "case:concept:name": ["case-1", "case-1", "case-2", "case-2"],
            "concept:name": ["A", "B", "A", "B"],
            "time:timestamp": pd.to_datetime(
                [
                    "2021-01-01T09:00:00",
                    "2021-01-01T10:00:00",
                    "2021-01-02T09:00:00",
                    "2021-01-02T10:00:00",
                ]
            ),
            "date": [pd.Timestamp("2021-01-01").date()] * 2
            + [pd.Timestamp("2021-01-02").date()] * 2,
        }
    )

    monkeypatch.setattr(
        weekly_models,
        "build_causal_net_from_forecast",
        lambda *_a, **_k: (
            {"▶": {"■"}, "■": set()},
            set(),
            {},
            stats_from_bindings(
                tasks=2,
                arcs=1,
                IB={"■": {frozenset({"▶"})}},
                OB={"▶": {frozenset({"■", "X"})}},
                repair_arcs_added=1,
                filter_bindings_removed=3,
                singleton_readds_orphaned=2,
                singleton_readds_uncovered=0,
                singleton_fallback_fires=0,
            ),
        ),
    )
    monkeypatch.setattr(
        weekly_models,
        "causal_graph_to_json",
        lambda _g, _a, metadata, *_x, **_k: {
            "metadata": metadata,
            "graph": {"directed": True, "nodes": [], "edges": []},
        },
    )
    monkeypatch.setattr(
        weekly_models,
        "mine_fodina",
        lambda *_a, **_k: _FakePatternMiner(),
    )
    monkeypatch.setattr(
        weekly_models,
        "fodina_to_json",
        lambda _pm, metadata: {
            "metadata": metadata,
            "graph": {"directed": True, "nodes": [], "edges": []},
        },
    )
    monkeypatch.setattr(
        weekly_models,
        "render_petri_json_to_pdf",
        lambda _p, pdf: Path(pdf).write_text("pdf"),
    )

    static_stats = stats_from_fodina(_FakePatternMiner())
    weekly_models.build_weekly_models(
        forecasts,
        test_data,
        str(output_dir),
        0.5,
        0.5,
        0.5,
        0.0,
        horizon=1,
        test_df=test_df,
        static=weekly_models.StaticModel(
            {"metadata": {"period": "p", "traces": 2}, "graph": {}}, static_stats
        ),
    )

    metrics_path = output_dir / STRUCTURAL_METRICS_FILENAME
    assert metrics_path.exists(), "one structural-metrics CSV per campaign"

    metrics = read_structural_metrics(str(metrics_path))
    assert set(metrics["window"]) == {1, 2}
    for window in (1, 2):
        families = set(metrics.loc[metrics["window"] == window, "model_family"])
        assert families == {"forecast", "re_mined", "direct_mined", "static"}

    # The reconstruction families carry the repair and filter counters apart.
    reconstructed = metrics[metrics["model_family"].isin(["forecast", "re_mined"])]
    assert (reconstructed["repair_arcs_added"] == 1).all()
    assert (reconstructed["filter_bindings_removed"] == 3).all()
    assert (reconstructed["singleton_readds"] == 2).all()
    assert (reconstructed["and_bindings"] == 1).all()
    assert (reconstructed["and_share_pct"] == 50.0).all()

    # The Fodina families report the same measurements and leave the process counters
    # missing rather than zero.
    fodina = metrics[metrics["model_family"].isin(["direct_mined", "static"])]
    assert (fodina["and_bindings"] == 1).all()
    assert (fodina["bindings"] == 2).all()
    assert fodina["repair_arcs_added"].isna().all()

    # The Static model measures the same in every window.
    training = metrics[metrics["model_family"] == "static"]
    assert len(training) == 2
    assert training["and_share_pct"].nunique() == 1
