from pmf_concurrency.causal_net import fodina_to_json, mine_fodina


def test_fodina_to_json_preserves_separate_binding_frequencies() -> None:
    traces = [["▶", "A", "B", "■"], ["▶", "A", "C", "■"]]
    pm = mine_fodina(traces, tD=0.5, tL1L=0.5, tL2L=0.5, tPAT=0.0)
    assert pm is not None

    model = fodina_to_json(pm, {"type": "re_mined"})

    binding_nodes = [
        node["data"]
        for node in model["graph"]["nodes"]
        if node["data"].get("is_binding", False)
    ]
    assert binding_nodes
    assert any("ib_frequency" in node for node in binding_nodes)
    assert any("ob_frequency" in node for node in binding_nodes)

    binding_edges = [
        edge["data"]
        for edge in model["graph"]["edges"]
        if "ib_frequency" in edge.get("data", {})
        or "ob_frequency" in edge.get("data", {})
    ]
    assert binding_edges
