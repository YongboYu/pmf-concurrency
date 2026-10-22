import json

from pmf_concurrency import rendering
from pmf_concurrency.petri_net import json_to_petri_net
from pmf_concurrency.rendering import generate_petri_net_dot


def _node(node_id: str, *, is_binding: bool = False, label: str | None = None) -> dict:
    return {
        "id": node_id,
        "data": {
            "label": node_id if label is None else label,
            "is_binding": is_binding,
        },
    }


def _edge(source: str, target: str, frequency: float = 1.0) -> dict:
    return {"source": source, "target": target, "data": {"frequency": frequency}}


def _alternative_input_model() -> dict:
    return {
        "metadata": {"type": "re_mined"},
        "graph": {
            "directed": True,
            "nodes": [
                _node("▶"),
                _node("A"),
                _node("B"),
                _node("C"),
                _node("■"),
                _node("o_▶_0", is_binding=True),
                _node("o_▶_1", is_binding=True),
                _node("i_C_0", is_binding=True),
                _node("i_C_1", is_binding=True),
                _node("o_C_0", is_binding=True),
            ],
            "edges": [
                _edge("▶", "o_▶_0"),
                _edge("o_▶_0", "A"),
                _edge("▶", "o_▶_1"),
                _edge("o_▶_1", "B"),
                _edge("A", "i_C_0"),
                _edge("i_C_0", "C"),
                _edge("B", "i_C_1"),
                _edge("i_C_1", "C"),
                _edge("C", "o_C_0"),
                _edge("o_C_0", "■"),
            ],
        },
    }


def _and_output_model() -> dict:
    return {
        "metadata": {"type": "forecasted"},
        "graph": {
            "directed": True,
            "nodes": [
                _node("▶"),
                _node("A"),
                _node("B"),
                _node("C"),
                _node("■"),
                _node("o_▶_0", is_binding=True),
                _node("o_A_0", is_binding=True),
                _node("i_B_0", is_binding=True),
                _node("i_C_0", is_binding=True),
                _node("o_B_0", is_binding=True),
                _node("o_C_0", is_binding=True),
            ],
            "edges": [
                _edge("▶", "o_▶_0"),
                _edge("o_▶_0", "A"),
                _edge("A", "o_A_0"),
                _edge("o_A_0", "i_B_0"),
                _edge("o_A_0", "i_C_0"),
                _edge("i_B_0", "B"),
                _edge("i_C_0", "C"),
                _edge("B", "o_B_0"),
                _edge("o_B_0", "■"),
                _edge("C", "o_C_0"),
                _edge("o_C_0", "■"),
            ],
        },
    }


def test_expansion_converts_alternative_input_bindings_to_xor_join() -> None:
    net, initial_marking, final_marking = json_to_petri_net(_alternative_input_model())

    start_activity = next(t for t in net.transitions if t.name == "act::▶")
    assert start_activity.label == "▶"
    assert {arc.source.name for arc in start_activity.in_arcs} == {"pI::▶"}
    assert {arc.target.name for arc in start_activity.out_arcs} == {"pO::▶"}

    activity_c = next(t for t in net.transitions if t.name == "act::C")
    assert {arc.source.name for arc in activity_c.in_arcs} == {"pI::C"}
    assert {arc.target.name for arc in activity_c.out_arcs} == {"pO::C"}

    end_activity = next(t for t in net.transitions if t.name == "act::■")
    assert end_activity.label == "■"
    assert {arc.source.name for arc in end_activity.in_arcs} == {"pI::■"}
    assert {arc.target.name for arc in end_activity.out_arcs} == {"pO::■"}

    input_taus = sorted(
        (t for t in net.transitions if t.name.startswith("tI::C::")),
        key=lambda t: t.name,
    )
    assert len(input_taus) == 2
    assert {
        frozenset(arc.source.name for arc in tau.in_arcs) for tau in input_taus
    } == {
        frozenset({"pD::A::C"}),
        frozenset({"pD::B::C"}),
    }
    assert {
        frozenset(arc.target.name for arc in tau.out_arcs) for tau in input_taus
    } == {frozenset({"pI::C"})}

    assert {place.name for place in initial_marking} == {"pI::▶"}
    assert {place.name for place in final_marking} == {"pO::■"}


def test_expansion_converts_multi_target_output_binding_to_single_tau() -> None:
    net, _, _ = json_to_petri_net(_and_output_model())

    output_taus = sorted(
        (t for t in net.transitions if t.name.startswith("tO::A::")),
        key=lambda t: t.name,
    )
    assert len(output_taus) == 1

    tau = output_taus[0]
    assert {arc.source.name for arc in tau.in_arcs} == {"pO::A"}
    assert {arc.target.name for arc in tau.out_arcs} == {
        "pD::A::B",
        "pD::A::C",
    }


def test_renderer_uses_expanded_places_and_binding_taus() -> None:
    dot = generate_petri_net_dot(_alternative_input_model())

    assert "act_u003a_u003a_u25b6 [label=<<b>▶</b>>" in dot
    assert "act_u003a_u003a_u25a0 [label=<<b>■</b>>" in dot
    assert 'act_u003a_u003a_u25b6 [label="", shape=box, style=filled' not in dot
    assert 'act_u003a_u003a_u25a0 [label="", shape=box, style=filled' not in dot
    assert "pI_u003a_u003aC" in dot
    assert "pD_u003a_u003aA_u003a_u003aC" in dot
    assert "tI_u003a_u003aC_u003a_u003a0_u003a_u003ai_C_0" in dot
    assert "act_u003a_u003aC" in dot
    assert 'label=<<font color="#555555" point-size="8">1</font>>' in dot
    assert "IB=1" in dot
    assert "OB=1" in dot
    assert "fixedsize=true" in dot
    assert "nodesep=0.05;" in dot

    # The DOT is text and the model stays JSON-serializable.
    json.dumps({"dot": dot})


def test_renderer_uses_weekly_title_for_sliding_models() -> None:
    model = _alternative_input_model()
    model["metadata"] = {
        "type": "sliding_forecast",
        "dataset": "BPI2017",
        "window": 1,
        "start_date": "20210101",
        "end_date": "20210107",
    }

    dot = generate_petri_net_dot(model)

    assert "BPI2017_forecast_window1_20210101_20210107" in dot
    assert "dependency place" not in dot


def test_renderer_uses_weekly_title_for_direct_mined_models() -> None:
    model = _alternative_input_model()
    model["metadata"] = {
        "type": "sliding_direct_mined",
        "dataset": "BPI2017",
        "window": 1,
        "start_date": "20210101",
        "end_date": "20210107",
    }

    dot = generate_petri_net_dot(model)

    assert "BPI2017_direct_mined_window1_20210101_20210107" in dot
    assert "dependency place" not in dot


def test_renderer_uses_weekly_title_for_expanding_models() -> None:
    model = _alternative_input_model()
    model["metadata"] = {
        "type": "sliding_expanding",
        "dataset": "BPI2017",
        "window": 2,
        "start_date": "20210102",
        "end_date": "20210108",
    }

    dot = generate_petri_net_dot(model)

    assert "BPI2017_expanding_window2_20210102_20210108" in dot
    assert "dependency place" not in dot


def test_without_graphviz_no_pdf_is_drawn_and_the_run_warns_once(
    monkeypatch, capsys, tmp_path
) -> None:
    def _dot_called(*_args, **_kwargs):
        raise AssertionError("dot was called")

    monkeypatch.setattr(rendering.shutil, "which", lambda _name: None)
    monkeypatch.setattr(rendering.subprocess, "run", _dot_called)
    rendering._graphviz_available.cache_clear()
    try:
        for n in range(2):
            pdf = tmp_path / f"{n}.pdf"
            assert not rendering.render_petri_json_to_pdf(
                _alternative_input_model(), str(pdf)
            )
            assert not pdf.exists()
    finally:
        rendering._graphviz_available.cache_clear()
    assert capsys.readouterr().out.count("Graphviz") == 1
