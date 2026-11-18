"""Render the annotated video.

Draws, per frame:
  - red box    around the detected face
  - yellow box around each detected hand
  - green pill in the recommended caption zone (semi-transparent)
  - blue rect  in the recommended B-roll zone (semi-transparent)
  - small corner label: timestamp + chosen placement names

OpenCV's VideoWriter drops audio, so we write a video-only temp file and then
use ffmpeg to copy the *original* audio back (so audio matches the input).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from bisect import bisect_right
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from .geometry import BROLL_GEOM, CAPTION_GEOM, ICON_GEOM
from .schema import FrameDetection, PlacementDecision, Window

RED = (0, 0, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 200, 0)
BLUE = (255, 120, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def _px(box, w, h) -> Tuple[int, int, int, int]:
    x, y, bw, bh = box
    return int(x * w), int(y * h), int(bw * w), int(bh * h)


def _filled(frame, box, color, w, h, alpha=0.35):
    x, y, bw, bh = _px(box, w, h)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)


def _outline(frame, box, color, w, h, thick=2):
    x, y, bw, bh = _px(box, w, h)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, thick)


def _label(frame, lines: List[str], w, h):
    pad = 8
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, w / 1400.0)
    th = int(max(1, scale * 2))
    sizes = [cv2.getTextSize(t, font, scale, th)[0] for t in lines]
    box_w = max(s[0] for s in sizes) + pad * 2
    line_h = max(s[1] for s in sizes) + 6
    box_h = line_h * len(lines) + pad
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (box_w, box_h), BLACK, -1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
    for i, t in enumerate(lines):
        y = pad + line_h * (i + 1) - 4
        cv2.putText(frame, t, (pad, y), font, scale, WHITE, th, cv2.LINE_AA)


def annotate(
    input_path: str,
    output_path: str,
    detections: List[FrameDetection],
    windows: List[Window],
    fps: float,
) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(detections)

    starts = [win.startMs for win in windows]

    def window_for_ms(ms: int) -> Window:
        i = bisect_right(starts, ms) - 1
        return windows[max(0, min(i, len(windows) - 1))]

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))

    idx = 0
    pbar = tqdm(total=total, desc="Annotating", unit="f")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ms = int(idx / fps * 1000)
        win = window_for_ms(ms)
        plc: PlacementDecision = win.suggestedPlacements

        # Recommended zones first (so detection boxes draw on top of them).
        _filled(frame, BROLL_GEOM[plc.broll], BLUE, w, h, alpha=0.25)
        _filled(frame, CAPTION_GEOM[plc.caption], GREEN, w, h, alpha=0.40)
        _outline(frame, ICON_GEOM[plc.icon], WHITE, w, h, thick=2)

        # Live detections for this exact frame.
        det = detections[idx] if idx < len(detections) else None
        if det is not None:
            if det.face_box is not None:
                _outline(frame, det.face_box, RED, w, h, thick=3)
            for hb in det.hand_boxes:
                _outline(frame, hb, YELLOW, w, h, thick=2)

        _label(
            frame,
            [
                f"t={ms / 1000:5.2f}s",
                f"cap:{plc.caption}",
                f"broll:{plc.broll}",
                f"icon:{plc.icon}",
            ],
            w,
            h,
        )

        writer.write(frame)
        idx += 1
        pbar.update(1)
    pbar.close()
    writer.release()
    cap.release()

    _mux_audio(tmp_path, input_path, output_path)
    os.remove(tmp_path)


def _mux_audio(video_only: str, original: str, out_path: str) -> None:
    """Copy the original audio onto the annotated video. The trailing '?' on the
    audio map makes it a no-op if the source has no audio track."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it (e.g. `apt install ffmpeg` "
            "or `brew install ffmpeg`) — it is required to keep the audio."
        )
    cmd = [
        "ffmpeg", "-y",
        "-i", video_only,
        "-i", original,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
