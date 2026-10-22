"""Draw the two panels of Figure 2: one window's WF-net before and after reduction.

Both panels come from one model JSON through one DOT writer:

* visible transition: activity name over its frequency;
* binding τ: black bar tagged ``IB=n`` (green) or ``OB=n`` (blue);
* dependency place: the forecast succession count of its arc;
* source place green, sink place a pink double circle.

The reduction keeps the names of the nodes it keeps, so panel (b) looks up each
surviving node in the expansion. Panel (b) is drawn without frequencies unless
``--annotate-reduced`` is passed.

``scripts/paper.py build`` calls :func:`draw` on the net its results directory names.
Standalone, from the project root::

    uv run python scripts/net_pair_figure.py \\
        --json outputs_tpatm1/BPI2017_pmf/weekly_models/window_1_forecast.json \\
        --out-dir figures

Writes ``fig-w1-expanded`` and ``fig-w1-reduced`` as DOT and PDF and prints the node
counts quoted in the caption. The DOT files are deterministic; the PDFs carry
Graphviz's creation date.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from pmf_concurrency.petri_net import (
    _as_float,
    causal_net_to_wf_net,
    json_to_petri_net,
)
from pmf_concurrency.reduction import reduce_tau_series
from pmf_concurrency.rendering import _escape_html, _freq_label, _safe_id

#: Binding-tau tag colours, matching the expansion's own rendering.
IB_COLOR = "#0B6E4F"
OB_COLOR = "#1D4ED8"


def _freq(freq: float) -> str:
    """Format a forecast frequency without rounding small counts to zero.

    Counts below 1 get two decimals, and counts below 0.01 print as ``<0.01`` (HTML-
    escaped, since every label is a Graphviz HTML label).
    """
    if freq >= 1:
        return _freq_label(freq)
    if freq >= 0.01:
        return f"{freq:.2f}"
    return "&lt;0.01"


def _pair_succession_frequencies(data: dict[str, Any]) -> dict[tuple[str, str], float]:
    """Forecast succession count of each dependency arc, keyed ``(source, target)``.

    The expansion labels a dependency place with the largest frequency of any edge that
    maps to it, and an edge into a multi-member binding carries the sum over its
    members. Only edges between exactly one source and one target task carry the pair's
    own count, so only those are used.
    """
    graph = data["graph"]
    meta = {n["id"]: n.get("data", {}) for n in graph["nodes"]}
    binding = {nid for nid, d in meta.items() if d.get("is_binding", False)}
    out_adj: dict[str, list[str]] = defaultdict(list)
    in_adj: dict[str, list[str]] = defaultdict(list)
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    for e in graph["edges"]:
        out_adj[e["source"]].append(e["target"])
        in_adj[e["target"]].append(e["source"])
        edges[(e["source"], e["target"])] = e.get("data", {})

    def visible(start: str, adj: dict[str, list[str]]) -> set[str]:
        seen: set[str] = set()
        found: set[str] = set()
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for nxt in adj.get(cur, []):
                if nxt in binding:
                    stack.append(nxt)
                else:
                    found.add(nxt)
        return found

    pair_freq: dict[tuple[str, str], float] = {}
    for (src, tgt), d in edges.items():
        weight = _as_float(d.get("ab_frequency") or d.get("frequency"))
        if weight <= 0:
            continue
        srcs = {src} if src not in binding else visible(src, in_adj)
        tgts = {tgt} if tgt not in binding else visible(tgt, out_adj)
        if len(srcs) == 1 and len(tgts) == 1:
            pair = (next(iter(srcs)), next(iter(tgts)))
            pair_freq[pair] = max(pair_freq.get(pair, 0.0), weight)
    return pair_freq


def _relabel_dependency_places(explicit: dict[str, Any], data: dict[str, Any]) -> int:
    """Set each dependency place's frequency to its pair's own count; return how many
    changed.
    """
    lookup = {
        f"pD::{src}::{tgt}": w
        for (src, tgt), w in _pair_succession_frequencies(data).items()
    }
    changed = 0
    for pid, meta in explicit["places"].items():
        if meta.get("kind") != "dependency":
            continue
        new = lookup.get(pid, 0.0)
        if abs(_as_float(meta.get("frequency")) - new) > 1e-6:
            changed += 1
        meta["frequency"] = new
    return changed


def _dot(
    explicit: dict[str, Any],
    keep_places: set[str] | None,
    keep_transitions: set[str] | None,
    arcs: list[tuple[str, str]],
    annotate: bool = True,
) -> str:
    """DOT for the nodes of ``explicit`` named by ``keep_*``.

    ``None`` keeps every node (panel a); a name set keeps the nodes that survived the
    reduction (panel b). ``arcs`` is separate because the reduction also rewires. With
    ``annotate=False`` no frequencies are printed.
    """
    places = {
        pid: meta
        for pid, meta in explicit["places"].items()
        if keep_places is None or pid in keep_places
    }
    transitions = {
        tid: meta
        for tid, meta in explicit["transitions"].items()
        if keep_transitions is None or tid in keep_transitions
    }

    lines = [
        "digraph PetriNet {",
        "  rankdir=LR;",
        "  nodesep=0.05;",
        "  node [fontname=Arial, fontsize=10];",
        "  edge [fontname=Arial, fontsize=9];",
        "",
    ]

    for place_id, meta in places.items():
        safe_id = _safe_id(place_id)
        if meta.get("is_source", False):
            lines.append(
                f'  {safe_id} [label="", shape=circle, style=filled, '
                f'fillcolor="#90EE90", width=0.35, height=0.35];'
            )
            continue
        if meta.get("is_sink", False):
            lines.append(
                f'  {safe_id} [label="", shape=doublecircle, style=filled, '
                f'fillcolor="#FFB6C1", width=0.35, height=0.35];'
            )
            continue
        freq = _as_float(meta.get("frequency"))
        if annotate and meta.get("kind") == "dependency" and freq > 0:
            lbl = f'<<font color="#555555" point-size="8">{_freq(freq)}</font>>'
            lines.append(
                f"  {safe_id} [label={lbl}, shape=circle, style=filled, "
                f"fillcolor=white, width=0.35, height=0.35];"
            )
        else:
            lines.append(
                f'  {safe_id} [label="", shape=circle, style=filled, '
                f"fillcolor=white, width=0.35, height=0.35];"
            )

    tag_nodes: list[tuple[str, str, str, str]] = []
    for transition_id, meta in transitions.items():
        safe_id = _safe_id(transition_id)
        freq = _as_float(meta.get("frequency"))
        if meta.get("invisible", False):
            lines.append(
                f'  {safe_id} [label="", shape=box, style=filled, '
                f"fillcolor=black, width=0.15, height=0.4, fixedsize=true];"
            )
            kind = meta.get("kind")
            if annotate and freq > 0 and kind in ("tau_input", "tau_output"):
                prefix, color = (
                    ("IB", IB_COLOR) if kind == "tau_input" else ("OB", OB_COLOR)
                )
                tag_nodes.append(
                    (
                        _safe_id(f"{transition_id}::label"),
                        safe_id,
                        f"{prefix}={_freq(freq)}",
                        color,
                    )
                )
            continue

        escaped = _escape_html(str(meta.get("label", transition_id)))
        if annotate and freq > 0:
            lbl = (
                f"<<b>{escaped}</b>"
                f'<br/><font color="gray" point-size="8">{_freq(freq)}</font>>'
            )
        else:
            lbl = f"<<b>{escaped}</b>>"
        lines.append(
            f"  {safe_id} [label={lbl}, shape=box, "
            f'style="filled,rounded", fillcolor=white];'
        )

    lines.append("")
    for label_id, _anchor, text, color in tag_nodes:
        lines.append(
            f"  {label_id} [label=<{text}>, shape=plain, fontsize=8, margin=0, "
            f'fontcolor="{color}"];'
        )
    if tag_nodes:
        lines.append("")
        for label_id, anchor, _text, _color in tag_nodes:
            lines.append(f"  {{ rank=same; {anchor}; {label_id}; }}")
            lines.append(
                f"  {anchor} -> {label_id} [style=invis, weight=100, minlen=0];"
            )
        lines.append("")

    for src, tgt in arcs:
        lines.append(f"  {_safe_id(src)} -> {_safe_id(tgt)};")

    lines.append("}")
    return "\n".join(lines)


def _write(dot: str, out_dir: Path, name: str) -> None:
    """``<name>.dot`` and the PDF Graphviz renders from it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.dot").write_text(dot + "\n")
    subprocess.run(
        ["dot", "-Tpdf", "-o", str(out_dir / f"{name}.pdf")],
        input=dot.encode(),
        check=True,
    )


def draw(
    data: dict[str, Any],
    out_dir: Path,
    stem: str = "fig-w1",
    annotate_reduced: bool = False,
) -> dict[str, int]:
    """Draw both panels of one model into ``out_dir``; return the caption's counts."""
    explicit = causal_net_to_wf_net(data)
    _relabel_dependency_places(explicit, data)
    expanded_arcs = [(a["source"], a["target"]) for a in explicit["arcs"]]
    _write(_dot(explicit, None, None, expanded_arcs), out_dir, f"{stem}-expanded")

    net, im, fm = json_to_petri_net(data)
    reduce_tau_series(net, im, fm)
    reduced_arcs = sorted((a.source.name, a.target.name) for a in net.arcs)
    _write(
        _dot(
            explicit,
            {p.name for p in net.places},
            {t.name for t in net.transitions},
            reduced_arcs,
            annotate=annotate_reduced,
        ),
        out_dir,
        f"{stem}-reduced",
    )
    return {
        "expanded_places": len(explicit["places"]),
        "expanded_transitions": len(explicit["transitions"]),
        "expanded_silent": sum(
            1 for m in explicit["transitions"].values() if m.get("invisible")
        ),
        "expanded_arcs": len(expanded_arcs),
        "reduced_places": len(net.places),
        "reduced_transitions": len(net.transitions),
        "reduced_silent": sum(1 for t in net.transitions if t.label is None),
        "reduced_arcs": len(reduced_arcs),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", type=Path, required=True, help="weekly model JSON")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--stem", default="fig-w1", help="basename prefix for both panels")
    ap.add_argument(
        "--annotate-reduced",
        action="store_true",
        help="print frequencies on panel (b) too (default: structure only)",
    )
    args = ap.parse_args()
    counts = draw(
        json.loads(args.json.read_text()),
        args.out_dir,
        args.stem,
        args.annotate_reduced,
    )
    print(
        f"(a) expanded: {counts['expanded_places']} places, "
        f"{counts['expanded_transitions']} transitions of which "
        f"{counts['expanded_silent']} silent, {counts['expanded_arcs']} arcs"
    )
    print(
        f"(b) reduced : {counts['reduced_places']} places, "
        f"{counts['reduced_transitions']} transitions of which "
        f"{counts['reduced_silent']} silent, {counts['reduced_arcs']} arcs"
    )


if __name__ == "__main__":
    main()
