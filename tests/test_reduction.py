"""The τ-reduction preserves the visible language.

Each test enumerates the prefixes and accepted traces of a small expanded WF-net
before and after reduction and requires them to be equal. The nets cover chains, AND
split and join, and XOR choice.
"""

from __future__ import annotations

from typing import Any

from pm4py.objects.petri_net.obj import Marking, PetriNet

from pmf_concurrency.entropic_relevance import _enabled, _fire, _marking_key
from pmf_concurrency.petri_net import json_to_petri_net
from pmf_concurrency.reduction import net_stats, reduce_net, reduce_tau_series

# ---------------------------------------------------------------------------
# Brute-force visible-language enumeration
# ---------------------------------------------------------------------------


def _tau_closure_markings(net: PetriNet, markings: list[Marking]) -> dict[Any, Marking]:
    out = {_marking_key(m): m for m in markings}
    stack = list(markings)
    while stack:
        m = stack.pop()
        for t in _enabled(net, m):
            if t.label is not None:
                continue
            m2 = _fire(m, t)
            k = _marking_key(m2)
            if k not in out:
                out[k] = m2
                stack.append(m2)
    return out


def visible_language(
    net: PetriNet, im: Marking, fm: Marking, max_len: int
) -> tuple[set[tuple[str, ...]], set[tuple[str, ...]]]:
    """Visible firing prefixes up to ``max_len``, and those that can reach the final
    marking.
    """
    fm_key = _marking_key(fm)
    prefixes: set[tuple[str, ...]] = set()
    accepted: set[tuple[str, ...]] = set()

    frontier: dict[tuple[str, ...], dict[Any, Marking]] = {
        (): _tau_closure_markings(net, [im])
    }
    for depth in range(max_len + 1):
        next_frontier: dict[tuple[str, ...], dict[Any, Marking]] = {}
        for trace, closure in frontier.items():
            prefixes.add(trace)
            if fm_key in closure:
                accepted.add(trace)
            if depth == max_len:
                continue
            for m in closure.values():
                for t in _enabled(net, m):
                    if t.label is None:
                        continue
                    m2 = _fire(m, t)
                    nt = (*trace, t.label)
                    next_frontier.setdefault(nt, {})[_marking_key(m2)] = m2
        frontier = {
            tr: _tau_closure_markings(net, list(ms.values()))
            for tr, ms in next_frontier.items()
        }
    return prefixes, accepted


# ---------------------------------------------------------------------------
# Causal-net JSON fixtures
# ---------------------------------------------------------------------------


def _causal_json(edges: list[tuple[str, str]], bindings: set[str]) -> dict[str, Any]:
    nodes = sorted({n for e in edges for n in e})
    return {
        "metadata": {"type": "sliding_forecast"},
        "graph": {
            "directed": True,
            "nodes": [
                {"id": n, "data": {"label": n, "is_binding": n in bindings}}
                for n in nodes
            ],
            "edges": [{"source": s, "target": t, "data": {}} for s, t in edges],
        },
    }


LINEAR = _causal_json(
    [("▶", "n0"), ("n0", "a"), ("a", "n1"), ("n1", "b"), ("b", "n2"), ("n2", "■")],
    bindings={"n0", "n1", "n2"},
)

AND_SPLIT_JOIN = _causal_json(
    [
        ("▶", "n0"),
        ("n0", "a"),
        ("a", "n1"),
        ("n1", "b"),
        ("n1", "c"),  # AND-split: one output binding covering {b, c}
        ("b", "n2"),
        ("c", "n2"),  # AND-join: one input binding of ■ covering {b, c}
        ("n2", "■"),
    ],
    bindings={"n0", "n1", "n2"},
)

XOR_CHOICE = _causal_json(
    [
        ("▶", "n0"),
        ("n0", "a"),
        ("a", "n1"),
        ("n1", "b"),  # binding 1: a → b
        ("a", "n2"),
        ("n2", "c"),  # binding 2: a → c (XOR between n1/n2)
        ("b", "n3"),
        ("n3", "■"),
        ("c", "n4"),
        ("n4", "■"),
    ],
    bindings={"n0", "n1", "n2", "n3", "n4"},
)

FIXTURES = {"linear": LINEAR, "and": AND_SPLIT_JOIN, "xor": XOR_CHOICE}


def _language_preserved(model_json: dict[str, Any], level: str, max_len: int = 8):
    ref_net, ref_im, ref_fm = json_to_petri_net(model_json)
    ref_lang = visible_language(ref_net, ref_im, ref_fm, max_len)

    net, im, fm = json_to_petri_net(model_json)
    net, im, fm, stats = reduce_net(net, im, fm, level=level)
    red_lang = visible_language(net, im, fm, max_len)

    assert red_lang[0] == ref_lang[0], "visible prefix language changed"
    assert red_lang[1] == ref_lang[1], "accepted (FM-reaching) language changed"
    return stats


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_linear_chain_taus_collapse():
    stats = _language_preserved(LINEAR, "tau")
    assert stats["silent_transitions_after"] < stats["silent_transitions_before"]
    # A pure linear chain needs no τ at all.
    assert stats["silent_transitions_after"] == 0


def test_and_split_join_language_preserved():
    stats = _language_preserved(AND_SPLIT_JOIN, "tau")
    assert stats["silent_transitions_after"] < stats["silent_transitions_before"]


def test_xor_choice_language_preserved():
    stats = _language_preserved(XOR_CHOICE, "tau")
    assert stats["silent_transitions_after"] < stats["silent_transitions_before"]


def test_tau_implicit_language_preserved():
    for name, fixture in FIXTURES.items():
        stats = _language_preserved(fixture, "tau_implicit")
        assert stats["reduction_level"] == "tau_implicit", name


def test_marked_places_survive():
    for fixture in FIXTURES.values():
        net, im, fm = json_to_petri_net(fixture)
        net, im, fm, _ = reduce_net(net, im, fm, level="tau")
        for p in list(im.keys()) + list(fm.keys()):
            assert p in net.places


def test_no_duplicate_arcs_after_reduction():
    for fixture in FIXTURES.values():
        net, im, fm = json_to_petri_net(fixture)
        net, im, fm, _ = reduce_net(net, im, fm, level="tau")
        pairs = [(a.source.name, a.target.name) for a in net.arcs]
        assert len(pairs) == len(set(pairs))


def test_reduction_is_idempotent():
    net, im, fm = json_to_petri_net(AND_SPLIT_JOIN)
    reduce_tau_series(net, im, fm)
    once = net_stats(net)
    reduce_tau_series(net, im, fm)
    assert net_stats(net) == once


def test_level_none_is_passthrough():
    net, im, fm = json_to_petri_net(AND_SPLIT_JOIN)
    before = net_stats(net)
    net, im, fm, stats = reduce_net(net, im, fm, level="none")
    assert net_stats(net) == before
    assert stats["silent_transitions_after"] == stats["silent_transitions_before"]


def test_visible_transitions_never_removed():
    for fixture in FIXTURES.values():
        net, im, fm = json_to_petri_net(fixture)
        visible_before = {t.label for t in net.transitions if t.label is not None}
        net, im, fm, _ = reduce_net(net, im, fm, level="tau")
        visible_after = {t.label for t in net.transitions if t.label is not None}
        assert visible_before == visible_after
