"""Serialized nets do not depend on the hash seed.

Token replay and ER depend on the order in which nodes and edges are written. The
serializer iterates sets and dicts, so these tests check that the output is sorted, and
that it is byte-identical across ``PYTHONHASHSEED`` values.
"""

from __future__ import annotations

import subprocess  # nosec B404
import sys
from pathlib import Path

from pm4py.objects.petri_net.obj import PetriNet

from pmf_concurrency.causal_net import causal_graph_to_json

_SRC = str(Path(__file__).resolve().parent.parent / "src")


def _sample_graph() -> tuple[dict[str, set[str]], set[str]]:
    """A small causal graph with unsorted keys and multi-element target sets."""
    graph: dict[str, set[str]] = {
        "C": {"o_C_0"},
        "A": {"o_A_0", "o_A_1"},
        "o_A_1": {"i_C_0"},
        "o_A_0": {"i_B_0"},
        "i_B_0": {"B"},
        "B": {"o_B_0"},
        "o_B_0": {"i_C_0"},
        "i_C_0": {"C"},
        "o_C_0": set(),
    }
    binding_nodes = {"o_A_0", "o_A_1", "o_B_0", "i_B_0", "i_C_0", "o_C_0"}
    return graph, binding_nodes


def test_causal_graph_to_json_emits_sorted_nodes_and_edges() -> None:
    graph, binding_nodes = _sample_graph()
    data = causal_graph_to_json(graph, binding_nodes, {"type": "test"})

    node_ids = [n["id"] for n in data["graph"]["nodes"]]
    assert node_ids == sorted(node_ids)

    edges = [(e["source"], e["target"]) for e in data["graph"]["edges"]]
    assert edges == sorted(edges)


_HASHSEED_SNIPPET = """
import json
from pmf_concurrency.causal_net import causal_graph_to_json

graph = {
    "C": {"o_C_0"},
    "A": {"o_A_0", "o_A_1"},
    "o_A_1": {"i_C_0"},
    "o_A_0": {"i_B_0"},
    "i_B_0": {"B"},
    "B": {"o_B_0"},
    "o_B_0": {"i_C_0"},
    "i_C_0": {"C"},
    "o_C_0": set(),
}
binding_nodes = {"o_A_0", "o_A_1", "o_B_0", "i_B_0", "i_C_0", "o_C_0"}
data = causal_graph_to_json(graph, binding_nodes, {"type": "test"})
print(json.dumps(data, sort_keys=False))
"""


def _run_with_hashseed(seed: str) -> str:
    env = {"PYTHONHASHSEED": seed, "PYTHONPATH": _SRC}
    proc = subprocess.run(  # nosec B603
        [sys.executable, "-c", _HASHSEED_SNIPPET],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return proc.stdout


def test_causal_graph_to_json_is_hashseed_independent() -> None:
    """The JSON is byte-identical under two PYTHONHASHSEED values."""
    out0 = _run_with_hashseed("0")
    out1 = _run_with_hashseed("1")
    assert out0 == out1


def test_make_pm4py_deterministic_uses_name_based_hash() -> None:
    """After the patch, pm4py places and transitions hash by name, not by address.

    This makes set iteration order, and so token replay and ER, the same in every
    process.
    """
    from pmf_concurrency.reproducibility import make_pm4py_deterministic

    make_pm4py_deterministic()
    p1, p2 = PetriNet.Place("p_x"), PetriNet.Place("p_x")
    t1, t2 = PetriNet.Transition("t_x", "X"), PetriNet.Transition("t_x", None)
    assert hash(p1) == hash(p2)
    assert hash(t1) == hash(t2)
    # Distinct objects are still not equal.
    assert p1 is not p2


def test_canonical_state_key_is_a_total_order_for_incomparable_states() -> None:
    """The canonical key orders frozenset states totally.

    Plain ``sorted()`` on frozensets uses the subset order, which leaves incomparable
    states in hash order.
    """
    from pmf_concurrency.entropic_relevance import _canonical_state_key

    # Neither state contains the other.
    s1 = frozenset({frozenset({("p1", 1)})})
    s2 = frozenset({frozenset({("p2", 1)})})
    assert _canonical_state_key(s1) != _canonical_state_key(s2)
    # The result does not depend on input order.
    assert sorted([s1, s2], key=_canonical_state_key) == sorted(
        [s2, s1], key=_canonical_state_key
    )
