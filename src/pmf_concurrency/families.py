"""The four model families compared in every evaluation window."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

#: Family keys, in the order the paper presents them.
MODEL_FAMILIES = ("forecast", "re_mined", "direct_mined", "static")

#: Display names for tables and figures.
FAMILY_LABELS = {
    "forecast": "Forecast",
    "re_mined": "Re-mined",
    "direct_mined": "Direct-mined",
    "static": "Static",
}


def check_families(names: Iterable[object], source: str) -> None:
    """Raise if ``names`` holds anything other than the four family keys."""
    unknown = sorted({str(name) for name in names} - set(MODEL_FAMILIES))
    if unknown:
        raise ValueError(
            f"{source}: unknown model families {', '.join(unknown)} "
            f"(expected {', '.join(MODEL_FAMILIES)})"
        )


def family_from_path(path: Path) -> str:
    """Family key of a ``window_<n>_<family>.json`` file."""
    return path.stem.split("_", 2)[2]


def model_json_path(weekly_dir: Path, window: int, family: str) -> Path:
    """Path of a window's model file."""
    return weekly_dir / f"window_{window}_{family}.json"
