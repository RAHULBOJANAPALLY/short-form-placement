"""6-row x 4-column occupancy grid.

The grid is the compact, LLM-friendly summary of "where is the subject".
Rows go top (r1) to bottom (r6); columns go left (c1) to right (c4).
"""

from __future__ import annotations

from typing import Iterable, List, Set

from .schema import Box

ROWS = 6
COLS = 4


def cells_for_box(box: Box, min_overlap: float = 0.10) -> Set[str]:
    """Return grid cells a normalized box overlaps by more than `min_overlap`
    of the cell's area."""
    x, y, w, h = box
    occupied: Set[str] = set()
    cell_area = (1.0 / COLS) * (1.0 / ROWS)
    for r in range(ROWS):
        for c in range(COLS):
            cx0, cx1 = c / COLS, (c + 1) / COLS
            cy0, cy1 = r / ROWS, (r + 1) / ROWS
            ix = max(0.0, min(x + w, cx1) - max(x, cx0))
            iy = max(0.0, min(y + h, cy1) - max(y, cy0))
            if (ix * iy) / cell_area > min_overlap:
                occupied.add(f"r{r + 1}c{c + 1}")
    return occupied


def occupied_cells(boxes: Iterable[Box]) -> List[str]:
    """Union of occupied cells over several boxes, sorted for stable output."""
    out: Set[str] = set()
    for b in boxes:
        out |= cells_for_box(b)
    return sorted(out, key=lambda s: (int(s[1]), int(s[3])))
