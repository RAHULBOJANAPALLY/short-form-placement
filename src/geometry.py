"""Canonical on-screen geometry for each placement name.

These rectangles are used ONLY to draw the overlay so a human can eyeball that
the LLM's choice actually avoids the subject. The decision itself stays with the
LLM; this module just renders the named zone. Boxes are [x,y,w,h] normalized.
"""

from __future__ import annotations

from typing import Dict, List

CAPTION_GEOM: Dict[str, List[float]] = {
    "bottom_pill": [0.08, 0.82, 0.84, 0.10],
    "top_pill": [0.08, 0.07, 0.84, 0.10],
    "center_pill": [0.08, 0.45, 0.84, 0.10],
}

BROLL_GEOM: Dict[str, List[float]] = {
    "full": [0.0, 0.0, 1.0, 1.0],
    "top40": [0.0, 0.0, 1.0, 0.40],
    "bottom60": [0.0, 0.40, 1.0, 0.60],
    "pip_top": [0.30, 0.05, 0.40, 0.22],
    "pip_corner_tr": [0.62, 0.05, 0.33, 0.18],
    "side_right_40": [0.60, 0.0, 0.40, 1.0],
}

ICON_GEOM: Dict[str, List[float]] = {
    "top_left": [0.05, 0.05, 0.12, 0.07],
    "top_right": [0.83, 0.05, 0.12, 0.07],
    "bottom_left": [0.05, 0.88, 0.12, 0.07],
    "bottom_right": [0.83, 0.88, 0.12, 0.07],
}
