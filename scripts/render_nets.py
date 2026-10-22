"""Draw the weekly models as workflow nets, before and after τ-reduction.

Each model is drawn twice from one DOT writer:

    expanded  the WF-net as constructed from the causal net
    reduced   the same net after language-preserving τ-reduction

Labels show the model's frequencies. A visible transition shows its activity count, a
binding τ is a black bar tagged ``IB=n`` (green) or ``OB=n`` (blue), and a dependency
place shows the succession count of its pair. Forecast counts below 1 keep two decimals,
so small forecasts are not printed as 0. The source place is green and the sink place is
a pink double circle. Each title names the log, family, window dates and the run's tPAT,
read from ``structural_metrics.csv``.

The Static model is mined once from the training period, so when every window holds the
same Static net it is drawn once and indexed without a window. A model with no tasks,
such as a Forecast whose counts are all zero, is listed in the index and not drawn.

Usage, from the project root:
    uv run python scripts/render_nets.py --output-base outputs_tpat0 \
        --out-dir drawings/tpat0
    uv run python scripts/render_nets.py --output-base outputs_tpatm1 \
        --out-dir drawings/tpatm1 --families forecast re_mined

Figure 2 of the paper:
    uv run python scripts/render_nets.py --output-base outputs_tpatm1 \
        --out-dir figures --datasets BPI2017 --families forecast --windows 1 \
        --formats pdf --no-title --no-reduced-labels

Outputs, under --out-dir:
    <Log>/<family>/window_<n>-expanded.<fmt>, window_<n>-reduced.<fmt>
    <Log>/static/static-expanded.<fmt>, static-reduced.<fmt>
    <Log>/index.csv   one row per model: log, tPAT, family, window, dates, net sizes
                      and the drawn files, relative to the index

Needs Graphviz (``dot`` on the PATH).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess  # nosec B404
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from pmf_concurrency.families import (
    ALL_FAMILIES,
    FAMILY_LABELS,
    MODEL_FAMILIES,
    model_json_path,
)
from pmf_concurrency.paths import OUTPUT_DIR_SUFFIX, dataset_name_from_output_dir
from pmf_concurrency.petri_net import _as_float, causal_net_to_wf_net, json_to_petri_net
from pmf_concurrency.reduction import reduce_tau_series
from pmf_concurrency.rendering import _escape_html, _freq_label, _safe_id

_SIZE_COLUMNS = [
    "places",
    "transitions",
    "silent",
    "arcs",
    "reduced_places",
    "reduced_transitions",
    "reduced_silent",
    "reduced_arcs",
]

IB_COLOR = "#0B6E4F"
OB_COLOR = "#1D4ED8"


def _freq(freq: float) -> str:
    """Format a frequency; counts below 1 keep two decimals, below 0.01 print ``<0.01``."""
    if freq >= 1:
        return _freq_label(freq)
    if freq >= 0.01:
        return f"{freq:.2f}"
    return "&lt;0.01"


def _pair_frequencies(data: dict[str, Any]) -> dict[tuple[str, str], float]:
    """Succession count of each dependency arc, keyed ``(source, target)``.

    The WF-net expansion labels a dependency place with the largest frequency of any
    edge mapped to it, and an edge into a multi-member binding carries the sum over its
    members. Only edges between exactly one source task and one target task carry the
    pair's own count, so only those are used.
    """
    graph = data["graph"]
    binding = {
        n["id"] for n in graph["nodes"] if n.get("data", {}).get("is_binding", False)
    }
    out_adj: dict[str, list[str]] = defaultdict(list)
    in_adj: dict[str, list[str]] = defaultdict(list)
    for e in graph["edges"]:
        out_adj[e["source"]].append(e["target"])
        in_adj[e["target"]].append(e["source"])

    def tasks_through(start: str, adj: dict[str, list[str]]) -> set[str]:
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

    pairs: dict[tuple[str, str], float] = {}
    for e in graph["edges"]:
        d = e.get("data", {})
        weight = _as_float(d.get("ab_frequency") or d.get("frequency"))
        if weight <= 0:
            continue
        src, tgt = e["source"], e["target"]
        srcs = {src} if src not in binding else tasks_through(src, in_adj)
        tgts = {tgt} if tgt not in binding else tasks_through(tgt, out_adj)
        if len(srcs) == 1 and len(tgts) == 1:
            pair = (next(iter(srcs)), next(iter(tgts)))
            pairs[pair] = max(pairs.get(pair, 0.0), weight)
    return pairs


def _label_dependency_places(explicit: dict[str, Any], data: dict[str, Any]) -> None:
    """Set each dependency place's frequency to its pair's own succession count."""
    lookup = {f"pD::{s}::{t}": w for (s, t), w in _pair_frequencies(data).items()}
    for pid, meta in explicit["places"].items():
        if meta.get("kind") == "dependency":
            meta["frequency"] = lookup.get(pid, 0.0)


def net_dot(
    explicit: dict[str, Any],
    keep_places: set[str] | None,
    keep_transitions: set[str] | None,
    arcs: list[tuple[str, str]],
    annotate: bool = True,
    title: str | None = None,
) -> str:
    """DOT for the nodes of ``explicit`` named by ``keep_*``, joined by ``arcs``.

    ``None`` keeps every node. Arcs are passed separately because the reduction rewires
    them.
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

    lines = ["digraph PetriNet {", "  rankdir=LR;", "  nodesep=0.05;"]
    if title:
        lines += [
            '  labelloc="t";',
            "  fontname=Arial;",
            f'  label=<<font point-size="12"><b>{_escape_html(title)}</b></font>>;',
        ]
    lines += [
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
            label = f'<<font color="#555555" point-size="8">{_freq(freq)}</font>>'
        else:
            label = '""'
        lines.append(
            f"  {safe_id} [label={label}, shape=circle, style=filled, "
            f"fillcolor=white, width=0.35, height=0.35];"
        )

    tags: list[tuple[str, str, str, str]] = []
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
                tags.append(
                    (
                        _safe_id(f"{transition_id}::label"),
                        safe_id,
                        f"{prefix}={_freq(freq)}",
                        color,
                    )
                )
            continue

        name = _escape_html(str(meta.get("label", transition_id)))
        if annotate and freq > 0:
            label = (
                f"<<b>{name}</b>"
                f'<br/><font color="gray" point-size="8">{_freq(freq)}</font>>'
            )
        else:
            label = f"<<b>{name}</b>>"
        lines.append(
            f"  {safe_id} [label={label}, shape=box, "
            f'style="filled,rounded", fillcolor=white];'
        )

    lines.append("")
    for tag_id, _anchor, text, color in tags:
        lines.append(
            f"  {tag_id} [label=<{text}>, shape=plain, fontsize=8, margin=0, "
            f'fontcolor="{color}"];'
        )
    if tags:
        lines.append("")
        for tag_id, anchor, _text, _color in tags:
            lines.append(f"  {{ rank=same; {anchor}; {tag_id}; }}")
            lines.append(f"  {anchor} -> {tag_id} [style=invis, weight=100, minlen=0];")
        lines.append("")

    # Graphviz breaks loops by reversing arcs in declaration order, which can pull a
    # place inside a loop to the left edge. Arcs that close a loop back towards the
    # source do not constrain the ranking, so the net reads from source to sink.
    back = _loop_back_arcs(places, arcs)
    for pid, meta in places.items():
        if meta.get("is_source") or meta.get("is_sink"):
            rank = "source" if meta.get("is_source") else "sink"
            lines.append(f"  {{ rank={rank}; {_safe_id(pid)}; }}")
    for src, tgt in arcs:
        attrs = " [constraint=false]" if (src, tgt) in back else ""
        lines.append(f"  {_safe_id(src)} -> {_safe_id(tgt)}{attrs};")
    lines.append("}")
    return "\n".join(lines)


def _loop_back_arcs(
    places: dict[str, Any], arcs: list[tuple[str, str]]
) -> set[tuple[str, str]]:
    """Arcs that close a loop: the target is no further from the source than the
    origin, and the target can reach the origin again.
    """
    succ: dict[str, list[str]] = defaultdict(list)
    for src, tgt in arcs:
        succ[src].append(tgt)

    frontier = [pid for pid, meta in places.items() if meta.get("is_source")]
    depth = dict.fromkeys(frontier, 0)
    while frontier:
        nxt = []
        for node in frontier:
            for tgt in succ[node]:
                if tgt not in depth:
                    depth[tgt] = depth[node] + 1
                    nxt.append(tgt)
        frontier = nxt

    reach: dict[str, set[str]] = {}

    def reachable(start: str) -> set[str]:
        if start not in reach:
            seen = {start}
            stack = [start]
            while stack:
                for nxt in succ[stack.pop()]:
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            reach[start] = seen
        return reach[start]

    return {
        (src, tgt)
        for src, tgt in arcs
        if src in depth
        and tgt in depth
        and depth[tgt] <= depth[src]
        and src in reachable(tgt)
    }


def _write(dot: str, stem: Path, formats: list[str]) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = stem.with_name(f"{stem.name}.{fmt}")
        subprocess.run(  # nosec B603 B607
            ["dot", f"-T{fmt}", "-o", str(path)], input=dot.encode(), check=True
        )
        written.append(path.name)
    return written


def _iso(date: Any) -> str:
    """``20161022`` as ``2016-10-22``; other forms are returned unchanged."""
    d = str(date)
    return f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 and d.isdigit() else d


def _title(meta: dict[str, Any], family: str, label: str | None) -> str:
    parts = [str(meta.get("dataset", "")), FAMILY_LABELS.get(family, family)]
    if family == "static":
        parts.append(f"trained {meta.get('training_period', '')}")
    else:
        parts.append(
            f"window {meta.get('window')}, "
            f"{_iso(meta.get('start_date', ''))} to "
            f"{_iso(meta.get('end_date', ''))}"
        )
    if label:
        parts.append(label)
    return "  |  ".join(p for p in parts if p)


def render_model(
    model_path: Path,
    log: str,
    tpat: float | None,
    family: str,
    stem: Path,
    formats: list[str],
    title: bool,
    reduced_labels: bool,
    label: str | None,
) -> dict[str, Any]:
    """Draw one model expanded and reduced; return its index row."""
    data = json.loads(model_path.read_text())
    meta = data.get("metadata", {})
    if label is None and tpat is not None:
        label = f"tPAT = {tpat}"
    row: dict[str, Any] = {
        "log": log,
        "tpat": tpat,
        "family": family,
        "window": None if stem.name == "static" else meta.get("window"),
        "start_date": None if stem.name == "static" else _iso(meta.get("start_date")),
        "end_date": None if stem.name == "static" else _iso(meta.get("end_date")),
        "source": model_path.name,
    }
    tasks = [
        n for n in data["graph"]["nodes"] if not n.get("data", {}).get("is_binding")
    ]
    if not tasks:
        row["empty"] = True
        return row

    explicit = causal_net_to_wf_net(data)
    _label_dependency_places(explicit, data)
    head = _title(meta, family, label) if title else None

    arcs = [(a["source"], a["target"]) for a in explicit["arcs"]]
    expanded = _write(
        net_dot(explicit, None, None, arcs, title=head),
        stem.with_name(f"{stem.name}-expanded"),
        formats,
    )

    net, im, fm = json_to_petri_net(data)
    reduce_tau_series(net, im, fm)
    # The reduced net holds its arcs in a set, so fix their order: arcs kept from the
    # expansion in its order, then the rewired ones sorted. Graphviz's layout depends
    # on the order, and the drawing must be the same on every run.
    position = {arc: i for i, arc in enumerate(arcs)}
    reduced_arcs = sorted(
        ((a.source.name, a.target.name) for a in net.arcs),
        key=lambda arc: (position.get(arc, len(position)), arc),
    )
    reduced = _write(
        net_dot(
            explicit,
            {p.name for p in net.places},
            {t.name for t in net.transitions},
            reduced_arcs,
            annotate=reduced_labels,
            title=head,
        ),
        stem.with_name(f"{stem.name}-reduced"),
        formats,
    )

    row.update(
        empty=False,
        places=len(explicit["places"]),
        transitions=len(explicit["transitions"]),
        silent=sum(1 for m in explicit["transitions"].values() if m.get("invisible")),
        arcs=len(arcs),
        reduced_places=len(net.places),
        reduced_transitions=len(net.transitions),
        reduced_silent=sum(1 for t in net.transitions if t.label is None),
        reduced_arcs=len(reduced_arcs),
        files=" ".join(f"{family}/{name}" for name in expanded + reduced),
    )
    return row


def _tpat(output_dir: Path) -> float | None:
    """The run's binding filter, from its ``structural_metrics.csv`` if present."""
    metrics = output_dir / "structural_metrics.csv"
    if not metrics.is_file():
        return None
    values = pd.read_csv(metrics, usecols=["tPAT"])["tPAT"].unique()
    if len(values) != 1:
        sys.exit(f"{metrics} mixes tPAT values {sorted(values)}")
    return float(values[0])


def _windows(weekly: Path, family: str) -> list[int]:
    return sorted(
        int(p.stem.split("_")[1]) for p in weekly.glob(f"window_*_{family}.json")
    )


def _same_graph(paths: list[Path]) -> bool:
    graphs = {
        json.dumps(json.loads(p.read_text())["graph"], sort_keys=True) for p in paths
    }
    return len(graphs) == 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output-base", required=True, help="campaign root holding <Log>_pmf/"
    )
    ap.add_argument("--out-dir", required=True, help="directory the drawings go to")
    ap.add_argument("--datasets", nargs="+", help="log names; default every <Log>_pmf")
    ap.add_argument(
        "--families", nargs="+", default=list(MODEL_FAMILIES), choices=ALL_FAMILIES
    )
    ap.add_argument("--windows", nargs="+", type=int, help="default every window")
    ap.add_argument(
        "--formats", nargs="+", default=["svg"], choices=["svg", "pdf", "png"]
    )
    ap.add_argument(
        "--label", help="appended to each title; default the run's tPAT setting"
    )
    ap.add_argument("--no-title", dest="title", action="store_false")
    ap.add_argument(
        "--no-reduced-labels",
        dest="reduced_labels",
        action="store_false",
        help="draw the reduced net without frequencies",
    )
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()

    if not shutil.which("dot"):
        sys.exit("Graphviz `dot` is not on the PATH")

    base = Path(args.output_base)
    if not base.is_dir():
        sys.exit(f"--output-base {base} is not a directory")
    datasets = args.datasets or sorted(
        dataset_name_from_output_dir(p) for p in base.glob(f"*{OUTPUT_DIR_SUFFIX}")
    )
    if not datasets:
        sys.exit(f"no <Log>{OUTPUT_DIR_SUFFIX} directories in {base}")
    out_dir = Path(args.out_dir)

    jobs: list[tuple[str, tuple]] = []
    for ds in datasets:
        weekly = base / f"{ds}{OUTPUT_DIR_SUFFIX}" / "weekly_models"
        if not weekly.is_dir():
            sys.exit(f"no weekly models in {weekly}")
        tpat = _tpat(weekly.parent)
        for family in args.families:
            windows = [
                w
                for w in _windows(weekly, family)
                if not args.windows or w in args.windows
            ]
            paths = [model_json_path(weekly, w, family) for w in windows]
            if family == "static" and paths and _same_graph(paths):
                paths, stems = paths[:1], [out_dir / ds / family / "static"]
            else:
                stems = [out_dir / ds / family / f"window_{w}" for w in windows]
            for path, stem in zip(paths, stems, strict=True):
                jobs.append(
                    (
                        ds,
                        (
                            path,
                            ds,
                            tpat,
                            family,
                            stem,
                            args.formats,
                            args.title,
                            args.reduced_labels,
                            args.label,
                        ),
                    )
                )

    if not jobs:
        sys.exit("no models match the selected datasets, families and windows")

    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [(ds, pool.submit(render_model, *a)) for ds, a in jobs]
        for ds, fut in futures:
            rows[ds].append(fut.result())

    for ds, ds_rows in rows.items():
        index = pd.DataFrame(ds_rows)
        sizes = [c for c in ["window", *_SIZE_COLUMNS] if c in index]
        index[sizes] = index[sizes].astype("Int64")
        index = index.sort_values(["family", "window"])
        (out_dir / ds).mkdir(parents=True, exist_ok=True)
        index.to_csv(out_dir / ds / "index.csv", index=False)
        empty = int(index["empty"].sum())
        print(f"{ds}: {len(index) - empty} drawn, {empty} empty")


if __name__ == "__main__":
    main()
