"""The model families compared in every evaluation window.

The four Table 2 families are the pipeline's family set and the default everywhere.
Persistence and Expanding are reference models Fodina mines from the history before
each window, built apart from the pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

#: Family keys of Table 2, in the order the paper presents them.
MODEL_FAMILIES = ("forecast", "re_mined", "direct_mined", "static")

#: Reference families Fodina mines from the history before each window.
HISTORY_FAMILIES = ("persistence", "expanding")

#: Every known family, in paper order with the history families last.
ALL_FAMILIES = (*MODEL_FAMILIES, *HISTORY_FAMILIES)

#: Display names for tables and figures.
FAMILY_LABELS = {
    "forecast": "Forecast",
    "re_mined": "Re-mined",
    "direct_mined": "Direct-mined",
    "static": "Static",
    "persistence": "Persistence",
    "expanding": "Expanding",
}


def check_families(names: Iterable[object], source: str) -> None:
    """Raise if ``names`` holds anything other than the known family keys."""
    unknown = sorted({str(name) for name in names} - set(ALL_FAMILIES))
    if unknown:
        raise ValueError(
            f"{source}: unknown model families {', '.join(unknown)} "
            f"(expected {', '.join(ALL_FAMILIES)})"
        )


def family_from_path(path: Path) -> str:
    """Family key of a ``window_<n>_<family>.json`` file."""
    return path.stem.split("_", 2)[2]


def model_json_path(weekly_dir: Path, window: int, family: str) -> Path:
    """Path of a window's model file."""
    return weekly_dir / f"window_{window}_{family}.json"
