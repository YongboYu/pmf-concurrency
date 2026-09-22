"""Settings that make pipeline outputs bit-reproducible.

Three measures, none relying on ``PYTHONHASHSEED``:

1. Iteration over sets and dicts is sorted wherever output order matters
   (``causal_net.causal_graph_to_json`` and the ER automaton in
   ``entropic_relevance``).
2. pm4py hashes places, transitions and arcs by memory address, so set
   iteration order changes between processes. :func:`make_pm4py_deterministic`
   hashes them by name instead.
3. The Chronos-2 random state is fixed and torch uses deterministic kernels.

Setting ``PYTHONHASHSEED`` here only affects child processes, since Python fixes the
hash seed at start-up.
"""

from __future__ import annotations

import os
import random
import zlib

import numpy as np
import torch


def make_pm4py_deterministic() -> None:
    """Make pm4py Petri-net objects hash by name.

    This fixes the order in which token replay, ETC precision and the ER automaton
    iterate places and transitions. Equality stays identity-based, so distinct objects
    may share a hash but are never equal. Nets loaded by
    :func:`petri_net.json_to_petri_net` have unique names.
    """
    from pm4py.objects.petri_net.obj import PetriNet

    def _stable(value: object) -> int:
        return zlib.crc32(str(value).encode("utf-8"))

    PetriNet.Place.__hash__ = lambda self: _stable(self.name)
    PetriNet.Transition.__hash__ = lambda self: _stable(self.name)
    PetriNet.Arc.__hash__ = lambda self: _stable(
        (getattr(self.source, "name", None), getattr(self.target, "name", None))
    )


def enable_determinism(seed: int = 42) -> None:
    """Seed all random generators, fix pm4py hashing and enable deterministic torch.

    Call once at the start of every entry point that produces numbers.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    make_pm4py_deterministic()
