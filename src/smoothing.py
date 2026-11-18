"""Temporal smoothing (BONUS).

Even with caching, a single noisy window can pick a different placement from its
neighbours. We apply a tiny majority filter per field: if a window disagrees
with both neighbours and those neighbours agree, snap it to them. This removes
1-window flicker without smearing real transitions.
"""

from __future__ import annotations

from typing import List

from .schema import PlacementDecision


def smooth(decisions: List[PlacementDecision]) -> List[PlacementDecision]:
    if len(decisions) < 3:
        return decisions
    out = [d.model_copy() for d in decisions]
    for field in ("caption", "broll", "icon"):
        for i in range(1, len(out) - 1):
            prev = getattr(decisions[i - 1], field)
            cur = getattr(decisions[i], field)
            nxt = getattr(decisions[i + 1], field)
            if cur != prev and prev == nxt:
                setattr(out[i], field, prev)
    return out
