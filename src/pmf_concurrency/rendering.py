"""Graphviz drawings of the WF-nets.

A green source circle and a pink sink double circle hold the initial and final markings;
the start and end activities stay visible.
"""

import re
import shutil
import subprocess  # nosec B404
from functools import cache
from typing import Any

from .petri_net import causal_net_to_wf_net


def _safe_id(name: str) -> str:
    """Convert any string to a valid Graphviz node identifier."""
    s = "".join(
        ch if re.match(r"[a-zA-Z0-9_]", ch) else f"_u{ord(ch):04x}" for ch in name
    )
    return ("_" + s if s and s[0].isdigit() else s) or "_empty"


def _freq_label(freq: float) -> str:
    return str(int(freq)) if freq == int(freq) else f"{freq:.0f}"


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _binding_tau_label_text(freq: float, label_prefix: str) -> str:
    return f"{label_prefix}={_freq_label(freq)}"


#: Per-window families whose rendered net carries a window title.
_TITLED_FAMILIES = {"forecast", "re_mined", "direct_mined"}


def _weekly_model_title(metadata: dict[str, Any]) -> str | None:
    model_type = str(metadata.get("type", ""))
    if not model_type.startswith("sliding_"):
        return None
    family = model_type.removeprefix("sliding_")
    if family not in _TITLED_FAMILIES:
        return None
    dataset = str(metadata.get("dataset", "")).strip()
    window = metadata.get("window")
    start_date = str(metadata.get("start_date", "")).strip()
    end_date = str(metadata.get("end_date", "")).strip()
    if dataset and window is not None and start_date and end_date:
        return f"{dataset}_{family}_window{window}_{start_date}_{end_date}"
    return None


def _wf_net_graph_label(metadata: dict[str, Any]) -> str:
    if title := _weekly_model_title(metadata):
        return f'<<font point-size="10"><b>{_escape_html(title)}</b></font>>'

    return (
        '<<font point-size="10">'
        '<font color="#555555">AB</font>: dependency place  '
        '<font color="#0B6E4F">IB</font>: input binding tau  '
        '<font color="#1D4ED8">OB</font>: output binding tau'
        "</font>>"
    )


def generate_petri_net_dot(data: dict[str, Any]) -> str:
    """Graphviz DOT for a causal-net JSON, expanded into its WF-net."""
    explicit = causal_net_to_wf_net(data)
    binding_label_nodes: list[tuple[str, str, str, str]] = []

    lines: list[str] = [
        "digraph PetriNet {",
        "  rankdir=LR;",
        "  nodesep=0.05;",
        '  labelloc="t";',
        f"  label={_wf_net_graph_label(data.get('metadata', {}))};",
        "  node [fontname=Arial, fontsize=10];",
        "  edge [fontname=Arial, fontsize=9];",
        "",
    ]

    for place_id, meta in explicit["places"].items():
        safe_id = _safe_id(place_id)
        freq = meta.get("frequency", 0.0)
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
        if meta.get("kind") == "dependency" and freq > 0:
            lbl = f'<<font color="#555555" point-size="8">{_freq_label(freq)}</font>>'
            lines.append(
                f"  {safe_id} [label={lbl}, shape=circle, style=filled, "
                f"fillcolor=white, width=0.35, height=0.35];"
            )
        else:
            lines.append(
                f'  {safe_id} [label="", shape=circle, style=filled, '
                f"fillcolor=white, width=0.35, height=0.35];"
            )

    for transition_id, meta in explicit["transitions"].items():
        safe_id = _safe_id(transition_id)
        if meta.get("invisible", False):
            freq = meta.get("frequency", 0.0)
            lines.append(
                f'  {safe_id} [label="", shape=box, style=filled, '
                f"fillcolor=black, width=0.15, height=0.4, fixedsize=true];"
            )
            if meta.get("kind") == "tau_input" and freq > 0:
                binding_label_nodes.append(
                    (
                        _safe_id(f"{transition_id}::label"),
                        safe_id,
                        _binding_tau_label_text(freq, "IB"),
                        "#0B6E4F",
                    )
                )
            elif meta.get("kind") == "tau_output" and freq > 0:
                binding_label_nodes.append(
                    (
                        _safe_id(f"{transition_id}::label"),
                        safe_id,
                        _binding_tau_label_text(freq, "OB"),
                        "#1D4ED8",
                    )
                )
            continue

        raw_label = str(meta.get("label", transition_id))
        escaped = _escape_html(raw_label)
        freq = meta.get("frequency", 0.0)
        if freq > 0:
            lbl = (
                f"<<b>{escaped}</b>"
                f'<br/><font color="gray" point-size="8">{_freq_label(freq)}</font>>'
            )
        else:
            lbl = f"<<b>{escaped}</b>>"
        lines.append(
            f"  {safe_id} [label={lbl}, shape=box, "
            f'style="filled,rounded", fillcolor=white];'
        )

    lines.append("")

    for label_id, _anchor_id, label_text, color in binding_label_nodes:
        lines.append(
            f'  {label_id} [label="{label_text}", shape=plain, fontsize=8, margin=0, '
            f'fontcolor="{color}"];'
        )

    if binding_label_nodes:
        lines.append("")
        for label_id, anchor_id, _label_text, _color in binding_label_nodes:
            lines.append(f"  {{ rank=same; {anchor_id}; {label_id}; }}")
            lines.append(
                f"  {anchor_id} -> {label_id} [style=invis, weight=100, minlen=0];"
            )

        lines.append("")

    for arc in explicit["arcs"]:
        src = _safe_id(arc["source"])
        tgt = _safe_id(arc["target"])
        lines.append(f"  {src} -> {tgt};")

    lines.append("}")
    return "\n".join(lines)


@cache
def _graphviz_available() -> bool:
    """True if Graphviz ``dot`` is on PATH. Without it the run warns once."""
    if shutil.which("dot"):
        return True
    print("  Warning: Graphviz `dot` is not on PATH, so no model PDFs are drawn")
    return False


def render_petri_json_to_pdf(model_json: dict[str, Any], pdf_path: str) -> bool:
    """Draw a model as a Petri-net PDF with Graphviz ``dot``; False if that fails."""
    if not _graphviz_available():
        return False
    try:
        result = subprocess.run(  # nosec B603 B607
            ["dot", "-Tpdf", "-o", pdf_path],
            input=generate_petri_net_dot(model_json),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as e:
        print(f"  Warning: Failed to render Petri net PDF: {e}")
        return False
    if result.returncode != 0:
        print(f"  Warning: Graphviz returned non-zero: {result.stderr[:200]}")
    return result.returncode == 0
