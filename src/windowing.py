"""Group per-frame detections into contiguous time windows.

Window length default = 500 ms. Reasoning (see README): captions and B-roll
don't need to move faster than ~2x/second to track a talking head, and shorter
windows just add flicker and LLM calls; longer windows lag real motion. 500 ms
(~15 frames @30fps) is a stable middle ground inside the 300-700 ms range.

Per window we compute:
  - faceBoxNorm: median face box if the face is present in most frames, else None
    (host briefly left frame -> omit, per spec).
  - handBoxesNorm: hands from the frame nearest the window midpoint (a clean,
    explainable representative for the sidecar).
  - occupied grid cells: UNION of face + every hand across the whole window, so
    the placement layer is conservative and avoids anything the hands swept past.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import List, Optional

from .grid import occupied_cells
from .schema import Box, FrameDetection


@dataclass
class WindowAgg:
    start_ms: int
    end_ms: int
    face_box: Optional[Box]
    hand_boxes: List[Box]
    occupied: List[str]


def _median_box(boxes: List[Box]) -> Box:
    return [round(statistics.median(c), 4) for c in zip(*boxes)]


def build_windows(
    detections: List[FrameDetection],
    fps: float,
    duration_ms: int,
    window_ms: int = 500,
) -> List[WindowAgg]:
    n_windows = max(1, -(-duration_ms // window_ms))  # ceil division
    windows: List[WindowAgg] = []

    for i in range(n_windows):
        start_ms = i * window_ms
        end_ms = min((i + 1) * window_ms, duration_ms)

        start_f = int(start_ms / 1000.0 * fps)
        end_f = max(start_f + 1, int(end_ms / 1000.0 * fps))
        frames = detections[start_f:end_f] or detections[start_f : start_f + 1]

        # Face: present in the majority of frames -> median box, else omit.
        face_boxes = [f.face_box for f in frames if f.face_box is not None]
        face_box: Optional[Box] = None
        if frames and len(face_boxes) >= max(1, len(frames) // 2):
            face_box = _median_box(face_boxes)

        # Hands for the sidecar: nearest-to-midpoint frame that actually has hands.
        mid = len(frames) // 2
        order = sorted(range(len(frames)), key=lambda k: abs(k - mid))
        hand_boxes: List[Box] = []
        for k in order:
            if frames[k].hand_boxes:
                hand_boxes = frames[k].hand_boxes
                break

        # Occupancy: conservative union of face + all hands across the window.
        union_boxes: List[Box] = []
        if face_box is not None:
            union_boxes.append(face_box)
        for f in frames:
            union_boxes.extend(f.hand_boxes)

        windows.append(
            WindowAgg(
                start_ms=start_ms,
                end_ms=end_ms,
                face_box=face_box,
                hand_boxes=hand_boxes,
                occupied=occupied_cells(union_boxes),
            )
        )

    return windows
