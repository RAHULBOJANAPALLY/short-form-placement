"""Per-frame face + hand detection on CPU using MediaPipe.

We detect on *every* frame because the annotated video needs an accurate
red/yellow box on each frame anyway; the window aggregation later reuses the
same per-frame results, so detection only runs once.

Results are cached to disk keyed by a hash of the video bytes + detection
params, so a rerun on the same file skips detection entirely (BONUS: caching).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import List, Optional

import cv2
import mediapipe as mp
from tqdm import tqdm

from .schema import Box, FrameDetection

mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands


def _video_signature(path: str, face_conf: float, hand_conf: float) -> str:
    """Hash the file contents + params so the cache key is stable per input."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    h.update(f"|face={face_conf}|hand={hand_conf}|v2".encode())
    return h.hexdigest()[:16]


def _hand_box_from_landmarks(landmarks) -> Box:
    """Tight normalized bbox around the 21 hand landmarks, padded a little."""
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad = 0.02
    x0 = max(0.0, x0 - pad)
    y0 = max(0.0, y0 - pad)
    x1 = min(1.0, x1 + pad)
    y1 = min(1.0, y1 + pad)
    return [round(x0, 4), round(y0, 4), round(x1 - x0, 4), round(y1 - y0, 4)]


def detect_video(
    path: str,
    cache_dir: str = ".cache",
    use_cache: bool = True,
    face_conf: float = 0.5,
    hand_conf: float = 0.5,
) -> List[FrameDetection]:
    """Run (or load cached) per-frame detection. Returns one entry per frame."""
    os.makedirs(cache_dir, exist_ok=True)
    sig = _video_signature(path, face_conf, hand_conf)
    cache_path = os.path.join(cache_dir, f"detections_{sig}.json")

    if use_cache and os.path.exists(cache_path):
        with open(cache_path) as f:
            raw = json.load(f)
        return [FrameDetection(**d) for d in raw]

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    detections: List[FrameDetection] = []
    # model_selection=1 = full-range model, better for faces filling the frame.
    with mp_face.FaceDetection(
        model_selection=1, min_detection_confidence=face_conf
    ) as face, mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=hand_conf,
        min_tracking_confidence=0.5,
    ) as hands:
        idx = 0
        pbar = tqdm(total=total, desc="Detecting", unit="f")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            fd = FrameDetection(frame_idx=idx)

            fr = face.process(rgb)
            if fr.detections:
                # Largest detection = the host.
                best = max(
                    fr.detections,
                    key=lambda d: d.location_data.relative_bounding_box.width
                    * d.location_data.relative_bounding_box.height,
                )
                bb = best.location_data.relative_bounding_box
                x = max(0.0, bb.xmin)
                y = max(0.0, bb.ymin)
                w = min(1.0 - x, bb.width)
                h = min(1.0 - y, bb.height)
                fd.face_box = [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]

            hr = hands.process(rgb)
            if hr.multi_hand_landmarks:
                fd.hand_boxes = [
                    _hand_box_from_landmarks(lm) for lm in hr.multi_hand_landmarks
                ]

            detections.append(fd)
            idx += 1
            pbar.update(1)
        pbar.close()
    cap.release()

    if use_cache:
        with open(cache_path, "w") as f:
            json.dump([d.__dict__ for d in detections], f)

    return detections
